# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .span_utils import generalized_temporal_iou, span_cxw_to_xx

from .matcher import build_matcher
from .transformer import build_transformer
from .position_encoding import build_position_encoding
from .misc import accuracy
from .adapter import GMRExistenceAdapter, existence_loss
from models.moment_detr_gmr.dual_grounding import (
    TemporalDualGrounding,
    dual_grounding_losses,
)
from models.moment_detr_gmr.hierarchical_counter import (
    HierarchicalMomentCounter,
    hierarchical_counter_losses,
)
import numpy as np
def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class QDDETR(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0,
                 use_saliency=False, use_exist_head=False, exist_hidden_dim=None,
                 use_dual_grounding=False, dual_num_phrases=3, dual_num_dummies=3,
                 dual_slot_iterations=1, dual_gate_init=-4.0, dual_nheads=8,
                 dual_max_text_len=32, use_hierarchical_counter=False,
                 counter_dropout=0.1, counter_detach_scores=True,
                 use_quality_head=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.use_saliency = bool(use_saliency)
        self.exist_head = (
            GMRExistenceAdapter(hidden_dim, exist_hidden_dim)
            if use_exist_head else None
        )
        self.dual_grounding = (
            TemporalDualGrounding(
                hidden_dim=hidden_dim,
                nheads=int(dual_nheads),
                dropout=float(transformer.t2v_encoder.layers[0].dropout.p),
                num_phrases=int(dual_num_phrases),
                num_dummies=int(dual_num_dummies),
                max_text_len=int(dual_max_text_len),
                phrase_slot_iterations=int(dual_slot_iterations),
                gate_init=float(dual_gate_init),
            )
            if use_dual_grounding else None
        )
        if self.dual_grounding is not None and self.use_saliency:
            raise ValueError(
                "DualGround is only supported by the MR-only path; rolled negative "
                "queries must not reuse video features conditioned on the positive query."
            )
        self.hierarchical_counter = (
            HierarchicalMomentCounter(
                hidden_dim=hidden_dim,
                dropout=float(counter_dropout),
                detach_query_scores=bool(counter_detach_scores),
            )
            if use_hierarchical_counter else None
        )
        self.counter_residual_exist = self.hierarchical_counter is not None and self.exist_head is not None
        if self.counter_residual_exist:
            nn.init.zeros_(self.hierarchical_counter.exist_head.weight)
            nn.init.zeros_(self.hierarchical_counter.exist_head.bias)
        self.quality_embed = MLP(hidden_dim, hidden_dim, 1, 3) if use_quality_head else None
        if self.quality_embed is not None:
            # A constant quality factor preserves the parent query ordering.
            nn.init.zeros_(self.quality_embed.layers[-1].weight)
            nn.init.zeros_(self.quality_embed.layers[-1].bias)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None,
                src_aud_mask=None, src_txt_semantic_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        semantic_text_mask = (
            src_txt_mask if src_txt_semantic_mask is None else src_txt_semantic_mask
        )
        dual_output = None
        if self.dual_grounding is not None:
            # DualGround conditions the positive video tokens after the QD
            # input projections and before the upstream video/text concat.
            dual_output = self.dual_grounding(
                video=src_vid,
                video_mask=src_vid_mask,
                text=src_txt,
                text_mask=semantic_text_mask,
            )
            src_vid = dual_output.video
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)

        # for global token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]
        
        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        if self.quality_embed is not None:
            out['pred_quality_logits'] = self.quality_embed(hs[-1]).squeeze(-1)
        base_exist_logits = None
        if self.exist_head is not None:
            base_exist_logits = self.exist_head(hs[-1])
            out['pred_exist_logits'] = base_exist_logits

        # QD-DETR's transformer returns its second-stage video memory only;
        # retain the projected text tokens as the counter's textual memory.
        txt_mem = src_txt
        vid_mem = memory
        if dual_output is not None:
            out.update({
                'dual_phrase_attention': dual_output.phrase_attention,
                'dual_phrase_eos': dual_output.phrase_eos,
                'dual_text_eos': dual_output.text_eos,
                'dual_sentence_eos_attention': dual_output.sentence_eos_attention,
                'dual_sentence_gate': dual_output.sentence_gate,
                'dual_phrase_gate': dual_output.phrase_gate,
            })
        if self.hierarchical_counter is not None:
            counter_output = self.hierarchical_counter(
                decoder_queries=hs[-1],
                pred_logits=outputs_class[-1],
                text_memory=txt_mem,
                text_mask=semantic_text_mask,
                video_memory=vid_mem,
                video_mask=src_vid_mask,
            )
            if self.counter_residual_exist:
                counter_output['pred_counter_exist_delta'] = counter_output['pred_exist_logits']
                counter_output['pred_exist_logits'] = (
                    base_exist_logits + counter_output['pred_counter_exist_delta']
                )
            out.update(counter_output)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))
            
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        if self.use_saliency:
            # Retain the upstream negative-pair saliency branch when explicitly
            # requested. Soccer-GMR is MR-only, so its default skips this second
            # transformer pass and does not fabricate highlight annotations.
            src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
            src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
            src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
            mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()
            mask_neg = torch.cat([mask_, mask_neg], dim=1)
            src_neg = torch.cat([src_, src_neg], dim=1)
            _, _, memory_neg, memory_global_neg = self.transformer(
                src_neg, ~mask_neg, self.query_embed.weight, pos.clone(), video_length=video_length
            )
            vid_mem_neg = memory_neg[:, :src_vid.shape[1]]
            scale = np.sqrt(self.hidden_dim)
            out["saliency_scores"] = torch.sum(
                self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1
            ) / scale
            out["saliency_scores_neg"] = torch.sum(
                self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1
            ) / scale
            out["video_mask"] = src_vid_mask
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, positive_count_weights=None,
                 dual_dqa_scale=0.3, dual_eos_temperature=0.07,
                 counter_contrastive_temperature=0.1,
                 mask_null_vmr_loss=False):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        if positive_count_weights is None:
            positive_count_weights = torch.ones(4, dtype=torch.float32)
        self.register_buffer(
            'positive_count_weights',
            torch.as_tensor(positive_count_weights, dtype=torch.float32),
        )
        self.dual_dqa_scale = float(dual_dqa_scale)
        self.dual_eos_temperature = float(dual_eos_temperature)
        self.counter_contrastive_temperature = float(counter_contrastive_temperature)
        self.mask_null_vmr_loss = bool(mask_null_vmr_loss)
        
        # for tvsum,
        self.use_matcher = use_matcher

    def _vmr_positive_mask(self, targets, batch_size, device):
        """Return samples allowed to contribute localization-side losses."""
        if not self.mask_null_vmr_loss:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if "exist_label" not in targets:
            raise ValueError(
                "mask_null_vmr_loss requires targets['exist_label']"
            )
        positive = targets["exist_label"].to(device=device).reshape(-1) > 0.5
        if positive.numel() != batch_size:
            raise ValueError(
                "exist_label batch size does not match model output: "
                f"{positive.numel()} != {batch_size}"
            )
        return positive

    @staticmethod
    def _keep_positive_indices(indices, positive):
        return [
            (source, target) if bool(positive[index].item())
            else (source[:0], target[:0])
            for index, (source, target) in enumerate(indices)
        ]

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        if src_spans.shape[0] == 0:
            zero = outputs['pred_spans'].sum() * 0.0
            return {'loss_span': zero, 'loss_giou': zero}
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, src_logits.shape[0], src_logits.device
            )[:, None].expand_as(loss_ce)
            loss_label = (loss_ce * positive).sum() / positive.sum().clamp_min(1)
        else:
            loss_label = loss_ce.mean()
        losses = {'loss_label': loss_label}

        if log and idx[0].numel() > 0:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        elif log:
            losses['class_error'] = src_logits.sum() * 0.0
        return losses

    def loss_exist(self, outputs, targets, indices, log=True):
        if 'pred_exist_logits' not in outputs or 'exist_label' not in targets:
            return {}
        return {'loss_exist': existence_loss(outputs['pred_exist_logits'], targets['exist_label'])}

    def loss_quality(self, outputs, targets, indices, log=True):
        del log
        logits = outputs['pred_quality_logits']
        quality_targets = torch.zeros_like(logits)
        weights = torch.full_like(logits, self.eos_coef)
        source_index = self._get_src_permutation_idx(indices)
        if source_index[0].numel() > 0:
            src_spans = outputs['pred_spans'][source_index]
            target_spans = torch.cat([
                target['spans'][target_indices]
                for target, (_, target_indices) in zip(targets['span_labels'], indices)
            ], dim=0)
            src_xx = span_cxw_to_xx(src_spans).clamp(0, 1)
            target_xx = span_cxw_to_xx(target_spans).clamp(0, 1)
            intersection = (
                torch.minimum(src_xx[:, 1], target_xx[:, 1])
                - torch.maximum(src_xx[:, 0], target_xx[:, 0])
            ).clamp_min(0)
            union = (
                (src_xx[:, 1] - src_xx[:, 0]).clamp_min(0)
                + (target_xx[:, 1] - target_xx[:, 0]).clamp_min(0)
                - intersection
            ).clamp_min(1e-6)
            quality_targets[source_index] = (intersection / union).detach().clamp(0, 1)
            weights[source_index] = 1.0
        raw = F.binary_cross_entropy_with_logits(logits, quality_targets, reduction='none')
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, logits.shape[0], logits.device
            )[:, None]
            weights = weights * positive
        return {'loss_quality': (raw * weights).sum() / weights.sum().clamp_min(1.0)}

    def loss_dual(self, outputs, targets, indices=None, log=True):
        del indices, log
        sample_mask = None
        if self.mask_null_vmr_loss:
            attention = outputs['dual_phrase_attention']
            sample_mask = self._vmr_positive_mask(
                targets, attention.shape[0], attention.device
            )
        return dual_grounding_losses(
            outputs,
            dqa_scale=self.dual_dqa_scale,
            temperature=self.dual_eos_temperature,
            sample_mask=sample_mask,
        )

    def loss_counter(self, outputs, targets, indices=None, log=True):
        del indices, log
        return hierarchical_counter_losses(
            outputs,
            targets,
            positive_count_weights=self.positive_count_weights,
            contrastive_temperature=self.counter_contrastive_temperature,
        )

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        # print(loss_saliency, loss_rank_contrastive)
        # loss_saliency = loss_saliency + loss_rank_contrastive
        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        # loss_saliency = loss_rank_contrastive
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        valid = num_pos > 0
        if self.mask_null_vmr_loss:
            valid = valid & self._vmr_positive_mask(
                targets, logits.shape[0], logits.device
            )
        if valid.any():
            loss_nce = -pos_term / num_pos.clamp_min(1) + neg_term
            loss_nce = loss_nce[valid].mean()
        else:
            loss_nce = logits.sum() * 0.0
        losses = {"loss_contrastive_align": loss_nce}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        valid = num_pos > 0
        if self.mask_null_vmr_loss:
            valid = valid & self._vmr_positive_mask(
                targets, logits.shape[0], logits.device
            )
        if valid.any():
            loss_nce = -pos_term / num_pos.clamp_min(1) + neg_term
            loss_nce = loss_nce[valid].mean()
        else:
            loss_nce = logits.sum() * 0.0
        losses = {"loss_contrastive_align": loss_nce}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "exist": self.loss_exist,
            "quality": self.loss_quality,
            "dual": self.loss_dual,
            "counter": self.loss_counter,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            if self.mask_null_vmr_loss:
                indices = self._keep_positive_indices(
                    indices,
                    self._vmr_positive_mask(
                        targets, outputs['pred_logits'].shape[0],
                        outputs['pred_logits'].device,
                    ),
                )
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    if self.mask_null_vmr_loss:
                        indices = self._keep_positive_indices(
                            indices,
                            self._vmr_positive_mask(
                                targets, aux_outputs['pred_logits'].shape[0],
                                aux_outputs['pred_logits'].device,
                            ),
                        )
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]    
                # for loss in self.losses:
                for loss in losses_target:
                    if loss in {"saliency", "exist", "quality", "dual", "counter"}:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = QDDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            max_v_l=args.max_v_l,
            use_saliency=getattr(args, 'use_saliency', False),
            use_exist_head=getattr(args, 'use_exist_head', False),
            exist_hidden_dim=getattr(args, 'exist_hidden_dim', None),
            use_dual_grounding=bool(getattr(args, 'use_dual_grounding', False)),
            dual_num_phrases=int(getattr(args, 'dual_num_phrases', 3)),
            dual_num_dummies=int(getattr(args, 'dual_num_dummies', 3)),
            dual_slot_iterations=int(getattr(args, 'dual_slot_iterations', 1)),
            dual_gate_init=float(getattr(args, 'dual_gate_init', -4.0)),
            dual_nheads=int(getattr(args, 'dual_nheads', args.nheads)),
            dual_max_text_len=int(getattr(args, 'max_q_l', 32)),
            use_hierarchical_counter=bool(getattr(args, 'use_hierarchical_counter', False)),
            counter_dropout=float(getattr(args, 'counter_dropout', 0.1)),
            counter_detach_scores=bool(getattr(args, 'counter_detach_scores', True)),
            use_quality_head=bool(getattr(args, 'use_quality_head', False)),
        )
    else:
        model = QDDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            max_v_l=args.max_v_l,
            use_saliency=getattr(args, 'use_saliency', False),
            use_exist_head=getattr(args, 'use_exist_head', False),
            exist_hidden_dim=getattr(args, 'exist_hidden_dim', None),
            use_dual_grounding=bool(getattr(args, 'use_dual_grounding', False)),
            dual_num_phrases=int(getattr(args, 'dual_num_phrases', 3)),
            dual_num_dummies=int(getattr(args, 'dual_num_dummies', 3)),
            dual_slot_iterations=int(getattr(args, 'dual_slot_iterations', 1)),
            dual_gate_init=float(getattr(args, 'dual_gate_init', -4.0)),
            dual_nheads=int(getattr(args, 'dual_nheads', args.nheads)),
            dual_max_text_len=int(getattr(args, 'max_q_l', 32)),
            use_hierarchical_counter=bool(getattr(args, 'use_hierarchical_counter', False)),
            counter_dropout=float(getattr(args, 'counter_dropout', 0.1)),
            counter_detach_scores=bool(getattr(args, 'counter_detach_scores', True)),
            use_quality_head=bool(getattr(args, 'use_quality_head', False)),
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef}
    if getattr(args, 'use_saliency', False):
        weight_dict["loss_saliency"] = args.lw_saliency
    if getattr(args, 'use_exist_head', False):
        weight_dict["loss_exist"] = args.exist_loss_coef
    if getattr(args, 'use_quality_head', False):
        weight_dict["loss_quality"] = float(getattr(args, 'quality_loss_coef', 1.0))
    if getattr(args, 'use_dual_grounding', False):
        weight_dict.update({
            "loss_dual_dqa": float(getattr(args, 'dual_dqa_loss_coef', 0.05)),
            "loss_dual_eos": float(getattr(args, 'dual_eos_loss_coef', 0.1)),
        })
    if getattr(args, 'use_hierarchical_counter', False):
        weight_dict.update({
            "loss_count": float(getattr(args, 'count_loss_coef', 1.0)),
            "loss_count_ordinal": float(getattr(args, 'count_ordinal_loss_coef', 0.25)),
            "loss_count_contrastive": float(getattr(args, 'count_contrastive_loss_coef', 0.05)),
            "loss_count_consistency": float(getattr(args, 'count_consistency_loss_coef', 0.05)),
        })
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f'_{i}': v for k, v in weight_dict.items()
                if k not in {
                    "loss_saliency", "loss_exist", "loss_quality",
                    "loss_dual_dqa", "loss_dual_eos", "loss_count",
                    "loss_count_ordinal", "loss_count_contrastive",
                    "loss_count_consistency",
                }
            })
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels']
    if getattr(args, 'use_saliency', False):
        losses.append('saliency')
    if getattr(args, 'use_exist_head', False) and not getattr(args, 'use_hierarchical_counter', False):
        losses.append('exist')
    if getattr(args, 'use_quality_head', False):
        losses.append('quality')
    if getattr(args, 'use_dual_grounding', False):
        losses.append('dual')
    if getattr(args, 'use_hierarchical_counter', False):
        losses.append('counter')
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
        positive_count_weights=getattr(args, 'positive_count_weights', None),
        dual_dqa_scale=float(getattr(args, 'dual_dqa_scale', 0.3)),
        dual_eos_temperature=float(getattr(args, 'dual_eos_temperature', 0.07)),
        counter_contrastive_temperature=float(
            getattr(args, 'counter_contrastive_temperature', 0.1)
        ),
        mask_null_vmr_loss=bool(
            getattr(args, 'mask_null_vmr_loss', False)
        ),
    )
    criterion.to(device)
    return model, criterion
