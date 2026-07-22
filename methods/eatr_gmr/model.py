# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
EaTR
"""
import torch
import torch.nn.functional as F
from torch import nn

from .misc import inverse_sigmoid
from .layers import MLP, LinearLayer
from .adapter import GMRExistenceAdapter
from .dual_grounding import TemporalDualGrounding
from .hierarchical_counter import HierarchicalMomentCounter


class EaTR(nn.Module):
    """ This is the EaTR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2,
                 query_dim=2, use_exist_head=False, exist_hidden_dim=None,
                 use_quality_head=False, use_dual_grounding=False,
                 dual_num_phrases=3, dual_num_dummies=3,
                 dual_slot_iterations=1, dual_gate_init=-4.0, dual_nheads=8,
                 dual_max_text_len=77, use_hierarchical_counter=False,
                 counter_dropout=0.1, counter_detach_scores=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         EaTR can detect in a single video.
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
        
        # model
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj

        hidden_dim = transformer.d_model
        self.num_dec_layers = transformer.dec_layers 

        # query
        self.num_queries = num_queries
        self.query_dim = query_dim

        # prediction
        self.max_v_l = max_v_l

        # loss
        self.span_loss_type = span_loss_type
        self.aux_loss = aux_loss

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # encoder
        # saliency prediction
        self.saliency_proj = nn.Linear(hidden_dim, 1)

        # decoder
        # span prediction
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2

        self.event_span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        nn.init.constant_(self.event_span_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.event_span_embed.layers[-1].bias.data, 0)

        self.moment_span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        nn.init.constant_(self.moment_span_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.moment_span_embed.layers[-1].bias.data, 0)
        # foreground classification
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground

        # Soccer-GMR parallel existence branch.  It consumes only the final
        # moment-decoder slots and does not alter EaTR's backbone computation.
        self.exist_head = (
            GMRExistenceAdapter(hidden_dim, exist_hidden_dim)
            if use_exist_head else None
        )

        self.use_quality_head = bool(use_quality_head)
        self.quality_embed = None
        if self.use_quality_head:
            self.quality_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            # A constant quality probability preserves the parent's query
            # ranking at step zero. The branch receives gradients immediately.
            nn.init.zeros_(self.quality_embed.layers[-1].weight)
            nn.init.zeros_(self.quality_embed.layers[-1].bias)

        self.use_dual_grounding = bool(use_dual_grounding)
        self.dual_grounding = None
        if self.use_dual_grounding:
            self.dual_grounding = TemporalDualGrounding(
                hidden_dim=hidden_dim,
                nheads=int(dual_nheads),
                dropout=float(transformer.encoder.layers[0].dropout.p),
                num_phrases=int(dual_num_phrases),
                num_dummies=int(dual_num_dummies),
                max_text_len=int(dual_max_text_len),
                phrase_slot_iterations=int(dual_slot_iterations),
                gate_init=float(dual_gate_init),
            )

        self.use_hierarchical_counter = bool(use_hierarchical_counter)
        self.hierarchical_counter = None
        self.counter_residual_exist = False
        if self.use_hierarchical_counter:
            self.hierarchical_counter = HierarchicalMomentCounter(
                hidden_dim=hidden_dim,
                dropout=float(counter_dropout),
                detach_query_scores=bool(counter_detach_scores),
            )
            self.counter_residual_exist = self.exist_head is not None
            if self.counter_residual_exist:
                # Preserve the calibrated parent existence logit exactly.
                nn.init.zeros_(self.hierarchical_counter.exist_head.weight)
                nn.init.zeros_(self.hierarchical_counter.exist_head.bias)

        # iterative anchor update
        self.transformer.decoder.moment_span_embed = self.moment_span_embed
        self.transformer.decoder.event_span_embed = self.event_span_embed

        # loss
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)
 
    def generate_pseudo_event(self, src_vid, src_vid_mask):
        bsz, L_src, _ = src_vid.size()

        norm_vid = src_vid / (src_vid.norm(dim=2, keepdim=True)+1e-8)
        tsm = torch.bmm(norm_vid, norm_vid.transpose(1,2))
        mask = torch.tensor([[1., 1., 0., -1., -1.],
                             [1., 1., 0., -1., -1.],
                             [0., 0., 0., 0., 0.],
                             [-1., -1., 0., 1., 1.],
                             [-1., -1., 0., 1., 1.]], device=src_vid.device)
        mask_size = mask.size(0)
        mask = mask.view(1,mask_size,mask_size)
        pad_tsm = nn.ZeroPad2d(mask_size//2)(tsm)
        score = torch.diagonal(F.conv2d(pad_tsm.unsqueeze(1), mask.unsqueeze(1)).squeeze(1), dim1=1,dim2=2)  # [bsz,L_src]
        # average score as threshold
        tau = score.mean(1).unsqueeze(1).repeat(1,L_src)
        # fill the start, end indices with the max score
        L_vid = torch.count_nonzero(src_vid_mask,1)
        st_ed = torch.cat([torch.zeros_like(L_vid).unsqueeze(1), L_vid.unsqueeze(1)-1], dim=-1)
        batch_indices = torch.arange(score.size(0), device=score.device).unsqueeze(1)
        score[batch_indices, st_ed] = 100
        # adjacent point removal and thresholding
        score_r = torch.roll(score,1,-1)
        score_l = torch.roll(score,-1,-1)
        bnds = torch.where((score_r<=score) & (score_l<=score) & (tau<=score), 1., 0.)

        bnd_indices = bnds.nonzero()
        temp = torch.roll(bnd_indices, 1, 0)
        center = (bnd_indices + temp) / 2
        width = bnd_indices - temp
        bnd_spans = torch.cat([center, width[:,1:]], dim=-1)
        pseudo_event_spans = [bnd_spans[bnd_spans[:,0] == i,:][:,1:]/L_vid[i] for i in range(bsz)]

        return pseudo_event_spans

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask):
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
        pseudo_event_spans = self.generate_pseudo_event(src_vid, src_vid_mask)  # comment the line for computational cost check

        src_vid = self.input_vid_proj(src_vid)

        src_txt = self.input_txt_proj(src_txt)  # (bsz, L_txt, d)
        dual_output = None
        # HieA2M injection point: after unimodal projections and before EaTR's
        # video/text concatenation and event-aware transformer.
        if self.dual_grounding is not None:
            dual_output = self.dual_grounding(
                video=src_vid,
                video_mask=src_vid_mask,
                text=src_txt,
                text_mask=src_txt_mask,
            )
            src_vid = dual_output.video
        src_txt_global = torch.max(src_txt, dim=1)[0]

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt, src_txt_mask) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        # (#layers+1, bsz, #queries, d), (bsz, L_vid+L_txt, d), (#layers, bsz, #queries, query_dim)
        hs, memory, reference = self.transformer(src, ~mask, pos, src_vid, pos_vid, ~src_vid_mask.bool(), src_txt_global)
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        event_tmp = self.event_span_embed(hs[0])
        event_outputs_coord = event_tmp.sigmoid()

        tmp = self.moment_span_embed(hs[-self.num_dec_layers:])
        tmp[..., :self.query_dim] += reference_before_sigmoid[-self.num_dec_layers:]
        outputs_coord = tmp.sigmoid()

        outputs_class = self.class_embed(hs[-self.num_dec_layers:])  # (#layers, batch_size, #queries, #classes)

        final_moment_queries = hs[-1]
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        if self.quality_embed is not None:
            out['pred_quality_logits'] = self.quality_embed(final_moment_queries).squeeze(-1)

        base_exist_logits = None
        if self.exist_head is not None:
            base_exist_logits = self.exist_head(final_moment_queries)
            out['pred_exist_logits'] = base_exist_logits

        out['pseudo_event_spans'] = pseudo_event_spans  # comment the line for computational cost check
        out['pred_event_spans'] = event_outputs_coord

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
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
                decoder_queries=final_moment_queries,
                pred_logits=outputs_class[-1],
                text_memory=txt_mem,
                text_mask=src_txt_mask,
                video_memory=vid_mem,
                video_mask=src_vid_mask,
            )
            if self.counter_residual_exist:
                counter_output['pred_counter_exist_delta'] = counter_output[
                    'pred_exist_logits'
                ]
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

        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [{'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[1:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        
        return out
