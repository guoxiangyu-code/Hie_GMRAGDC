"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.moment_detr_gmr.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx
from models.moment_detr_gmr.position_encoding import build_position_encoding
from models.moment_detr_gmr.matcher import build_matcher
from models.moment_detr_gmr.misc import accuracy
from models.moment_detr_gmr.moment_transformer import build_transformer
from models.moment_detr_gmr.gmr_adapter import GMRAdapter, compute_existence_loss
from models.moment_detr_gmr.dual_grounding import (
    TemporalDualGrounding,
    dual_grounding_losses,
)
from models.moment_detr_gmr.hierarchical_counter import (
    HierarchicalMomentCounter,
    hierarchical_counter_losses,
)
from models.moment_detr_gmr.learned_selector import (
    IndependentZeroVerifier,
    PairwiseSameEventHead,
    independent_zero_loss,
    pairwise_same_event_loss,
)

class MomentDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, max_v_l=75, span_loss_type="l1",
                 use_txt_pos=False, n_input_proj=2, aud_dim=0, use_exist_head=False, exist_pool="max",
                 use_dual_grounding=False, dual_num_phrases=3, dual_num_dummies=3,
                 dual_slot_iterations=1, dual_gate_init=-4.0, dual_nheads=8,
                 dual_max_text_len=77,
                 use_hierarchical_counter=False, counter_dropout=0.1,
                 counter_detach_scores=True, use_quality_head=False,
                 use_independent_zero_head=False, use_pairwise_head=False,
                 selector_dropout=0.1, pairwise_detach_inputs=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
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
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
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

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

        # Optional GMR Adapter for query-video existence prediction.
        self.use_exist_head = bool(use_exist_head)
        self.exist_pool = str(exist_pool)
        if self.use_exist_head:
            self.exist_head = GMRAdapter(hidden_dim, hidden_dim, pool=self.exist_pool)
        else:
            self.exist_head = None

        self.use_dual_grounding = bool(use_dual_grounding)
        if self.use_dual_grounding:
            self.dual_grounding = TemporalDualGrounding(
                hidden_dim=hidden_dim,
                nheads=int(dual_nheads),
                dropout=transformer.encoder.layers[0].dropout.p,
                num_phrases=int(dual_num_phrases),
                num_dummies=int(dual_num_dummies),
                max_text_len=int(dual_max_text_len),
                phrase_slot_iterations=int(dual_slot_iterations),
                gate_init=float(dual_gate_init),
            )
        else:
            self.dual_grounding = None

        self.use_hierarchical_counter = bool(use_hierarchical_counter)
        if self.use_hierarchical_counter:
            self.hierarchical_counter = HierarchicalMomentCounter(
                hidden_dim=hidden_dim,
                dropout=float(counter_dropout),
                detach_query_scores=bool(counter_detach_scores),
            )
            # When a released GMR adapter is available, retain it as the
            # calibrated existence prior and learn only a zero-initialized
            # residual from the richer counter representation.  This makes a
            # warm-start exactly reproduce the parent model at step zero.
            self.counter_residual_exist = self.use_exist_head
            if self.counter_residual_exist:
                nn.init.zeros_(self.hierarchical_counter.exist_head.weight)
                nn.init.zeros_(self.hierarchical_counter.exist_head.bias)
        else:
            self.hierarchical_counter = None
            self.counter_residual_exist = False

        self.use_independent_zero_head = bool(use_independent_zero_head)
        self.use_pairwise_head = bool(use_pairwise_head)
        if (self.use_independent_zero_head or self.use_pairwise_head) \
                and self.hierarchical_counter is None:
            raise ValueError(
                "independent zero/pairwise selection heads require the hierarchical counter"
            )
        self.zero_verifier_head = (
            IndependentZeroVerifier(hidden_dim, dropout=float(selector_dropout))
            if self.use_independent_zero_head else None
        )
        self.pairwise_same_event_head = (
            PairwiseSameEventHead(
                hidden_dim,
                dropout=float(selector_dropout),
                detach_inputs=bool(pairwise_detach_inputs),
            )
            if self.use_pairwise_head else None
        )

        self.use_quality_head = bool(use_quality_head)
        if self.use_quality_head:
            self.quality_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            # Constant sigmoid quality preserves the parent DETR ranking until
            # the newly introduced head has learned a localization signal.
            nn.init.zeros_(self.quality_embed.layers[-1].weight)
            nn.init.zeros_(self.quality_embed.layers[-1].bias)
        else:
            self.quality_embed = None

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
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
        dual_output = None
        if self.dual_grounding is not None:
            dual_output = self.dual_grounding(
                video=src_vid,
                video_mask=src_vid_mask,
                text=src_txt,
                text_mask=src_txt_mask,
            )
            src_vid = dual_output.video
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        if self.quality_embed is not None:
            out["pred_quality_logits"] = self.quality_embed(hs[-1]).squeeze(-1)

        base_exist_logits = None
        if self.exist_head is not None:
            base_exist_logits = self.exist_head(hs[-1])
            out["pred_exist_logits"] = base_exist_logits
            out["pred_gate_logits"] = base_exist_logits

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        saliency_scores = self.saliency_proj(vid_mem).squeeze(-1)

        if dual_output is not None:
            out.update({
                "dual_phrase_attention": dual_output.phrase_attention,
                "dual_phrase_eos": dual_output.phrase_eos,
                "dual_text_eos": dual_output.text_eos,
                "dual_sentence_eos_attention": dual_output.sentence_eos_attention,
                "dual_sentence_gate": dual_output.sentence_gate,
                "dual_phrase_gate": dual_output.phrase_gate,
            })

        if self.hierarchical_counter is not None:
            counter_output = self.hierarchical_counter(
                decoder_queries=hs[-1],
                pred_logits=outputs_class[-1],
                text_memory=txt_mem,
                text_mask=src_txt_mask,
                video_memory=vid_mem,
                video_mask=src_vid_mask,
            )
            counter_exist_logits = counter_output.pop("pred_exist_logits")
            counter_output["pred_counter_exist_logits"] = counter_exist_logits
            if self.counter_residual_exist and not self.use_independent_zero_head:
                counter_output["pred_counter_exist_delta"] = counter_exist_logits
                counter_output["pred_exist_logits"] = base_exist_logits + counter_exist_logits
            elif base_exist_logits is not None:
                # The new two-stage variant keeps stage one independent.  Its
                # second decision signal is pred_zero_logits below, not another
                # residual added to the same existence score.
                counter_output["pred_exist_logits"] = base_exist_logits
            else:
                counter_output["pred_exist_logits"] = counter_exist_logits
            out.update(counter_output)

            if self.zero_verifier_head is not None:
                out["pred_zero_logits"] = self.zero_verifier_head(
                    counter_output["counter_representation"],
                    outputs_class[-1],
                    outputs_coord[-1],
                    out.get("pred_quality_logits"),
                )
            if self.pairwise_same_event_head is not None:
                out["pred_same_event_logits"] = self.pairwise_same_event_head(
                    hs[-1], outputs_class[-1], outputs_coord[-1], vid_mem,
                    src_vid_mask, saliency_scores,
                )

        out["saliency_scores"] = saliency_scores  # (bsz, L_vid)

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, span_loss_type, max_v_l,
                 saliency_margin=1, positive_count_weights=None,
                 dual_dqa_scale=0.3, dual_eos_temperature=0.07,
                 counter_contrastive_temperature=0.1,
                 zero_positive_query_weight=1.0,
                 pair_assignment_iou=0.3, pair_ambiguity_margin=0.05,
                 pair_positive_weight=1.0, pair_hard_negative_weight=2.0,
                 mask_null_vmr_loss=False):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        if positive_count_weights is None:
            positive_count_weights = torch.ones(4, dtype=torch.float32)
        self.register_buffer(
            "positive_count_weights",
            torch.as_tensor(positive_count_weights, dtype=torch.float32),
        )
        self.dual_dqa_scale = float(dual_dqa_scale)
        self.dual_eos_temperature = float(dual_eos_temperature)
        self.counter_contrastive_temperature = float(counter_contrastive_temperature)
        self.zero_positive_query_weight = float(zero_positive_query_weight)
        self.pair_assignment_iou = float(pair_assignment_iou)
        self.pair_ambiguity_margin = float(pair_ambiguity_margin)
        self.pair_positive_weight = float(pair_positive_weight)
        self.pair_hard_negative_weight = float(pair_hard_negative_weight)
        self.mask_null_vmr_loss = bool(mask_null_vmr_loss)

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
        # Empty GT batches contribute zero localization loss.
        if idx[0].numel() == 0:
            z = outputs["pred_spans"].sum() * 0.0
            return {"loss_span": z, "loss_giou": z}
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
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

        if log:
            if idx[0].numel() > 0:
                losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
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
        return {"loss_saliency": loss_saliency}

    def loss_exist(self, outputs, targets, indices=None, log=True):
        """Existence loss: whether this query-video pair contains any relevant moment.
        targets should contain key "exist_label": float tensor of shape (bsz,) with values in {0,1}.
        """
        return {"loss_exist": compute_existence_loss(outputs, targets)}

    def loss_quality(self, outputs, targets, indices, log=True):
        """Calibrate each query score to its matched temporal IoU.

        Unmatched queries receive quality zero, which also suppresses duplicate
        predictions that lose the one-to-one Hungarian assignment.
        """
        del log
        logits = outputs["pred_quality_logits"]
        quality_targets = torch.zeros_like(logits)
        weights = torch.full_like(logits, self.eos_coef)
        source_index = self._get_src_permutation_idx(indices)
        if source_index[0].numel() > 0:
            src_spans = outputs["pred_spans"][source_index]
            target_spans = torch.cat([
                target["spans"][target_indices]
                for target, (_, target_indices) in zip(targets["span_labels"], indices)
            ], dim=0)
            # Match inference geometry: predicted windows are clipped to the
            # valid normalized video interval before score calibration.
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
            matched_iou = (intersection / union).detach().clamp(0, 1)
            quality_targets[source_index] = matched_iou
            weights[source_index] = 1.0
        raw = F.binary_cross_entropy_with_logits(logits, quality_targets, reduction="none")
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, logits.shape[0], logits.device
            )[:, None]
            weights = weights * positive
        return {"loss_quality": (raw * weights).sum() / weights.sum().clamp_min(1.0)}

    def loss_dual(self, outputs, targets, indices=None, log=True):
        del indices, log
        sample_mask = None
        if self.mask_null_vmr_loss:
            attention = outputs["dual_phrase_attention"]
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

    def loss_zero(self, outputs, targets, indices=None, log=True):
        del indices, log
        return {"loss_zero": independent_zero_loss(
            outputs, targets,
            positive_query_weight=self.zero_positive_query_weight,
        )}

    def loss_pairwise(self, outputs, targets, indices=None, log=True):
        del indices, log
        return {"loss_pairwise": pairwise_same_event_loss(
            outputs,
            targets,
            assignment_iou=self.pair_assignment_iou,
            ambiguity_margin=self.pair_ambiguity_margin,
            positive_weight=self.pair_positive_weight,
            hard_negative_weight=self.pair_hard_negative_weight,
        )}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "exist": self.loss_exist,
            "quality": self.loss_quality,
            "dual": self.loss_dual,
            "counter": self.loss_counter,
            "zero": self.loss_zero,
            "pairwise": self.loss_pairwise,
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

        # Match predictions to ground-truth windows.
        indices = self.matcher(outputs_without_aux, targets)
        if self.mask_null_vmr_loss:
            indices = self._keep_positive_indices(
                indices,
                self._vmr_positive_mask(
                    targets, outputs['pred_logits'].shape[0],
                    outputs['pred_logits'].device,
                ),
            )

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if self.mask_null_vmr_loss:
                    indices = self._keep_positive_indices(
                        indices,
                        self._vmr_positive_mask(
                            targets, aux_outputs['pred_logits'].shape[0],
                            aux_outputs['pred_logits'].device,
                        ),
                    )
                for loss in self.losses:
                    if loss in {"saliency", "exist", "quality", "dual", "counter", "zero", "pairwise"}:
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
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        use_exist_head=bool(getattr(args, "use_exist_head", False)),
        exist_pool=str(getattr(args, "exist_pool", "max")),
        use_dual_grounding=bool(getattr(args, "use_dual_grounding", False)),
        dual_num_phrases=int(getattr(args, "dual_num_phrases", 3)),
        dual_num_dummies=int(getattr(args, "dual_num_dummies", 3)),
        dual_slot_iterations=int(getattr(args, "dual_slot_iterations", 1)),
        dual_gate_init=float(getattr(args, "dual_gate_init", -4.0)),
        dual_nheads=int(getattr(args, "dual_nheads", args.nheads)),
        dual_max_text_len=int(getattr(args, "max_q_l", 77)),
        use_hierarchical_counter=bool(getattr(args, "use_hierarchical_counter", False)),
        counter_dropout=float(getattr(args, "counter_dropout", 0.1)),
        counter_detach_scores=bool(getattr(args, "counter_detach_scores", True)),
        use_quality_head=bool(getattr(args, "use_quality_head", False)),
        use_independent_zero_head=bool(getattr(args, "use_independent_zero_head", False)),
        use_pairwise_head=bool(getattr(args, "use_pairwise_head", False)),
        selector_dropout=float(getattr(args, "selector_dropout", 0.1)),
        pairwise_detach_inputs=bool(getattr(args, "pairwise_detach_inputs", True)),
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']

    # Add existence supervision when the adapter is enabled.
    if bool(getattr(args, "use_exist_head", False)):
        weight_dict["loss_exist"] = float(getattr(args, "exist_loss_coef", 1.0))
        if not bool(getattr(args, "use_hierarchical_counter", False)):
            losses.append("exist")

    if bool(getattr(args, "use_quality_head", False)):
        weight_dict["loss_quality"] = float(getattr(args, "quality_loss_coef", 1.0))
        losses.append("quality")

    if bool(getattr(args, "use_dual_grounding", False)):
        weight_dict.update({
            "loss_dual_dqa": float(getattr(args, "dual_dqa_loss_coef", 0.05)),
            "loss_dual_eos": float(getattr(args, "dual_eos_loss_coef", 0.1)),
        })
        losses.append("dual")

    if bool(getattr(args, "use_hierarchical_counter", False)):
        weight_dict.update({
            "loss_exist": float(getattr(args, "exist_loss_coef", 1.0)),
            "loss_count": float(getattr(args, "count_loss_coef", 1.0)),
            "loss_count_ordinal": float(getattr(args, "count_ordinal_loss_coef", 0.25)),
            "loss_count_contrastive": float(getattr(args, "count_contrastive_loss_coef", 0.05)),
            "loss_count_consistency": float(getattr(args, "count_consistency_loss_coef", 0.05)),
        })
        losses.append("counter")

    if bool(getattr(args, "use_independent_zero_head", False)):
        weight_dict["loss_zero"] = float(getattr(args, "zero_loss_coef", 1.0))
        losses.append("zero")

    if bool(getattr(args, "use_pairwise_head", False)):
        weight_dict["loss_pairwise"] = float(getattr(args, "pairwise_loss_coef", 1.0))
        losses.append("pairwise")

    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l, saliency_margin=args.saliency_margin,
        positive_count_weights=getattr(args, "positive_count_weights", None),
        dual_dqa_scale=float(getattr(args, "dual_dqa_scale", 0.3)),
        dual_eos_temperature=float(getattr(args, "dual_eos_temperature", 0.07)),
        counter_contrastive_temperature=float(
            getattr(args, "counter_contrastive_temperature", 0.1)
        ),
        zero_positive_query_weight=float(
            getattr(args, "zero_positive_query_weight", 1.0)
        ),
        pair_assignment_iou=float(getattr(args, "pair_assignment_iou", 0.3)),
        pair_ambiguity_margin=float(getattr(args, "pair_ambiguity_margin", 0.05)),
        pair_positive_weight=float(getattr(args, "pair_positive_weight", 1.0)),
        pair_hard_negative_weight=float(
            getattr(args, "pair_hard_negative_weight", 2.0)
        ),
        mask_null_vmr_loss=bool(
            getattr(args, "mask_null_vmr_loss", False)
        ),
    )

    criterion.to(device)
    return model, criterion
