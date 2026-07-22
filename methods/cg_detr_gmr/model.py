# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
CG-DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .span_utils import generalized_temporal_iou, span_cxw_to_xx

from .matcher import build_matcher
from .transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from .position_encoding import build_position_encoding
from .misc import accuracy
from .adapter import GMRExistenceAdapter, existence_loss
from .hiea2m import CGPhraseGrounding
from models.moment_detr_gmr.dual_grounding import dual_grounding_losses
from models.moment_detr_gmr.hierarchical_counter import (
    HierarchicalMomentCounter,
    hierarchical_counter_losses,
    inverse_sqrt_positive_count_weights,
)
import numpy as np
import copy

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res

class CGDETR(nn.Module):
    """ CG DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0,
                 use_saliency=False, use_cg_aux=True, use_exist_head=False,
                 exist_hidden_dim=None, use_quality_head=False,
                 use_phrase_grounding=False, phrase_num_phrases=3,
                 phrase_slot_iterations=1, phrase_gate_init=-4.0,
                 use_hierarchical_counter=False, counter_dropout=0.1,
                 counter_detach_scores=True, args=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         CG-DETR can detect in a single video.
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
        self.args=args
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
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
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
        self.use_cg_aux = bool(use_cg_aux)
        self.exist_head = (
            GMRExistenceAdapter(hidden_dim, exist_hidden_dim)
            if use_exist_head else None
        )
        self.quality_embed = MLP(hidden_dim, hidden_dim, 1, 3) if use_quality_head else None
        if self.quality_embed is not None:
            nn.init.zeros_(self.quality_embed.layers[-1].weight)
            nn.init.zeros_(self.quality_embed.layers[-1].bias)

        self.phrase_grounding = (
            CGPhraseGrounding(
                hidden_dim=hidden_dim,
                nheads=transformer.nhead,
                dropout=float(args.dropout),
                num_phrases=int(phrase_num_phrases),
                max_text_len=int(args.max_q_l),
                slot_iterations=int(phrase_slot_iterations),
                gate_init=float(phrase_gate_init),
            )
            if use_phrase_grounding else None
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
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(args.total_prompts, hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.moment_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.moment_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        self.sent_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.sent_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        self.txt_proj_linear = LinearLayer(txt_dim, hidden_dim, layer_norm=True)

        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)

        scls_encoder_layer = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        scls_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.scls_encoder = TransformerEncoder(scls_encoder_layer, args.sent_layers, scls_encoder_norm)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid=None, qid=None,
                src_aud=None, src_aud_mask=None, targets=None):
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

        ## For discovering real negative samples
        if self.use_saliency and vid is not None: ## optional upstream QV branch
            _count = [v.count('_') for v in vid]
            if self.args.dset_name == 'hl':
                _position_to_cut = [find_nth(v, '_', _count[i]-1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]

        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        phrase_output = None
        if self.phrase_grounding is not None:
            phrase_output = self.phrase_grounding(
                video=src_vid,
                video_mask=src_vid_mask,
                text=src_txt,
                text_mask=src_txt_mask,
            )
            src_vid = phrase_output.video
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)

        ### insert dummy token in front of txt
        txt_dummy = self.dummy_rep_token.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1)
        mask_txt = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1)

        pos_dummy = self.dummy_rep_pos.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1)
        src_txt_dummy = src_txt_dummy.permute(1, 0, 2)  # (L, batch_size, d)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)   # (L, batch_size, d)

        memory = self.txtproj_encoder(src_txt_dummy, src_key_padding_mask=~(src_txt_mask_dummy.bool()), pos=pos_txt_dummy)  # (L, batch_size, d)
        dummy_token = memory[:self.args.num_dummies].permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)  # (L, batch_size, d)

        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1)
        mask_txt_dummy = torch.tensor([[True]*self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)

        # Input : Concat video, dummy, txt
        src = torch.cat([src_vid, src_txt_dummy], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)




        ### sentence token
        smask_ = torch.tensor([[True]]).to(mask.device).repeat(src_txt_mask.shape[0], 1)
        smask = torch.cat([smask_, src_txt_mask.bool()], dim=1)
        ssrc_ = self.sent_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        ssrc = torch.cat([ssrc_, src_txt], dim=1)
        spos_ = self.sent_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        spos = torch.cat([spos_, pos_txt], dim=1)
        ### dummy sentence token
        smaskd = torch.cat([smask_, mask_txt_dummy.bool()], dim=1)
        ssrcd = torch.cat([ssrc_, dummy_token], dim=1)
        sposd = torch.cat([spos_, pos_dummy], dim=1)

        cg_targets = targets if self.use_cg_aux else None
        if cg_targets is not None: # train with legal MR-window-derived CG auxiliaries
            mmask_ = torch.tensor([[True]]).to(mask.device).repeat(src_vid_mask.shape[0], 1)
            mmask = torch.cat([mmask_, src_vid_mask.bool()], dim=1)
            moment_mask_ = torch.clamp(cg_targets["relevant_clips"], 0, 1).bool()
            moment_mask = torch.cat([mmask_, moment_mask_], dim=1)
            mmask = mmask * moment_mask

            msrc_ = self.moment_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_vid.shape[0], 1, 1)
            msrc = torch.cat([msrc_, src_vid], dim=1)
            mpos_ = self.moment_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_vid.shape[0], 1, 1)
            mpos = torch.cat([mpos_, pos_vid], dim=1)


            ### for Not moment token ####
            nmmask_ = torch.tensor([[True]]).to(mask.device).repeat(src_vid_mask.shape[0], 1)
            nmmask = torch.cat([nmmask_, src_vid_mask.bool()], dim=1)
            nmoment_mask_ = ~(torch.clamp(cg_targets["relevant_clips"], 0, 1).bool())
            nmoment_mask = torch.cat([nmmask_, nmoment_mask_], dim=1)
            nmmask = nmmask * nmoment_mask

            nmsrc_ = self.moment_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_vid.shape[0], 1, 1)
            nmsrc = torch.cat([nmsrc_, src_vid], dim=1)
            nmpos_ = self.moment_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_vid.shape[0], 1, 1)
            nmpos = torch.cat([nmpos_, pos_vid], dim=1)
            ###########
        else:
            moment_mask_ = None

        # for t2vidavg sal token
        vidsrc_ = src_vid.new_zeros((len(src_vid), 1, self.hidden_dim))
        for i in range(len(src_vid)):
            vidsrc_[i] = src_vid[i][:src_vid_mask.sum(1)[i].long()].mean(0).clone().detach()

        video_length = src_vid.shape[1]
        if cg_targets is not None: ## train
            ssrc = ssrc.permute(1, 0, 2)  # (L, batch_size, d)
            spos = spos.permute(1, 0, 2)  # (L, batch_size, d)
            smemory = self.scls_encoder(ssrc, src_key_padding_mask=~smask, pos=spos)  # (L, batch_size, d)
            sentence_txt, smemory_words = smemory[0], smemory[1:] # sentence_txt : (batch_size, d)

            ssrcd = ssrcd.permute(1, 0, 2)  # (L, batch_size, d)
            sposd = sposd.permute(1, 0, 2)  # (L, batch_size, d)
            smemoryd = self.scls_encoder(ssrcd, src_key_padding_mask=~smaskd, pos=sposd)  # (L, batch_size, d)
            sentence_dummy, smemory_words_dummy = smemoryd[0], smemoryd[1:]

            txt_dummy_proj = torch.cat([smemory_words_dummy, smemory_words], dim=0)

            hs, reference, memory, memory_global, attn_weights, memory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length, moment_idx=cg_targets["relevant_clips"], msrc=msrc, mpos=mpos, mmask=~mmask, nmsrc=nmsrc, nmpos=nmpos, nmmask=~nmmask,
                                                                                                                  ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
            moment2txt_similarity = torch.matmul(mmemory_frames.permute(1, 0, 2), txt_dummy_proj.permute(1, 2, 0))
            nmoment2txt_similarity = torch.matmul(nmmemory_frames.permute(1, 0, 2), txt_dummy_proj.permute(1, 2, 0))
        else: ## inference
            sentence_dummy, sentence_txt, moment2txt_similarity, nmoment2txt_similarity = None, None, None, None
            hs, reference, memory, memory_global, attn_weights, memory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length,
                                                                                                                  ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
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

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if phrase_output is not None:
            out.update({
                'dual_phrase_attention': phrase_output.phrase_attention,
                'dual_phrase_eos': phrase_output.phrase_eos,
                'dual_text_eos': phrase_output.text_eos,
                'dual_phrase_gate': phrase_output.phrase_gate,
            })
        if self.hierarchical_counter is not None:
            counter_output = self.hierarchical_counter(
                decoder_queries=hs[-1],
                pred_logits=outputs_class[-1],
                # CG's final encoder memory is video-only; its projected text
                # tokens are the non-duplicated language view for the counter.
                text_memory=src_txt,
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

        if self.use_saliency and vid is not None: ## optional upstream QV branch
            ### Neg Pairs ###
            neg_vid = ori_vid[1:] + ori_vid[:1]
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid)).to(src_txt_dummy.device)
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:

                src_txt_dummy_neg = torch.cat([src_txt_dummy[1:], src_txt_dummy[0:1]], dim=0)
                src_txt_mask_dummy_neg = torch.cat([src_txt_mask_dummy[1:], src_txt_mask_dummy[0:1]], dim=0)
                src_dummy_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
                mask_dummy_neg = torch.cat([src_vid_mask, src_txt_mask_dummy_neg], dim=1).bool()
                pos_neg = pos.clone()  # since it does not use actual content

                mask_dummy_neg = mask_dummy_neg[real_neg_mask]
                src_dummy_neg = src_dummy_neg[real_neg_mask]
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_dummy_neg = src_txt_mask_dummy_neg[real_neg_mask]

                _, _, memory_neg, memory_global_neg, attn_weights_neg, _, _, _, _ = self.transformer(src_dummy_neg, ~mask_dummy_neg, self.query_embed.weight, pos_neg, video_length=video_length,
                                                                                               ctxtoken=vidsrc_[real_neg_mask], gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask[real_neg_mask].sum(1).long())
                vid_mem_neg = memory_neg[:, :src_vid.shape[1]]
                out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
                out["src_txt_mask_neg"] = src_txt_mask_dummy_neg

                out["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.args.num_dummies:] * (src_txt_mask_dummy_neg[:, self.args.num_dummies:].unsqueeze(1).repeat(1, video_length, 1))).sum(2)
                out["t2vattnvalues_neg"] = torch.clamp(out["t2vattnvalues_neg"], 0, 1)
            else:
                out["saliency_scores_neg"] = None
                out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = real_neg_mask
        else:
            out["saliency_scores_neg"] = None
            out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = None


        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["memory_moment"] = memory_moment
        out["nmmemory_moment"] = nmmemory_moment

        ## sentence token embeeded with text / dummy
        out["sentence_txt"] = sentence_txt
        out["sentence_dummy"] = sentence_dummy
        out["moment2txt_similarity"] = moment2txt_similarity
        out["nmoment2txt_similarity"] = nmoment2txt_similarity
        out["cate_attn_weights"] = attn_weights
        out["moment_mask"] = moment_mask_
        out["txt_mask"] = src_txt_mask_dummy


        out["t2vattnvalues"] = (attn_weights[:,:,self.args.num_dummies:] * (src_txt_mask.unsqueeze(1).repeat(1, video_length, 1))).sum(2) # (batch_size, L_vid, L_txt) / (batch_size, L_txt)
        out["t2vattnvalues"] = torch.clamp(out["t2vattnvalues"], 0, 1)
        out["dummy_tokens"] = dummy_token
        out["global_rep_tokens"] = self.global_rep_token


        if cg_targets is not None:
            out["src_vid"] = mmemory_frames.permute(1, 0, 2) * moment_mask_.unsqueeze(2) + nmmemory_frames.permute(1, 0, 2) * (~(moment_mask_.unsqueeze(2).bool())).float()
        else:
            out["src_vid"] = None

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

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, positive_count_weights=None,
                 phrase_dqa_scale=0.3, phrase_eos_temperature=0.07,
                 counter_contrastive_temperature=0.1, args=None):
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
        self.args=args
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
        
        # for tvsum,
        self.use_matcher = use_matcher

        # moment sentence contrastive
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none').to(self.args.device)
        self.bce_criterion = nn.BCELoss(reduction='none')
        if positive_count_weights is None:
            positive_count_weights = torch.ones(4, dtype=torch.float32)
        self.register_buffer(
            'positive_count_weights',
            torch.as_tensor(positive_count_weights, dtype=torch.float32),
        )
        self.phrase_dqa_scale = float(phrase_dqa_scale)
        self.phrase_eos_temperature = float(phrase_eos_temperature)
        self.counter_contrastive_temperature = float(counter_contrastive_temperature)
        self.mask_null_vmr_loss = bool(
            getattr(self.args, 'mask_null_vmr_loss', False)
        )

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
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        # Neg pair loss
        if outputs["saliency_scores_neg"] is not None: ## When batch size is not 1 (negative pair exists)
            vid_token_mask = outputs["video_mask"]
            real_neg_mask = outputs["real_neg_mask"]
            saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
            loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

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

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair * 0.
            else:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
                
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_mask"]
            # Neg pair loss

            if outputs["t2vattnvalues_neg"] is not None:
                saliency_scores_neg = outputs["t2vattnvalues_neg"].clone()  # (N, L)
                loss_neg_pair_attn = (- torch.log(1. - saliency_scores_neg) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat(
                [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (
                        1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive_attn = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive_attn = loss_rank_contrastive_attn + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive_attn / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn * 0 + loss_saliency_attn
            else:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn + loss_saliency_attn
            loss_saliency += (loss_saliency_attn * self.args.lw_wattn)
            
        else: ## when batch size == 1
            vid_token_mask = outputs["video_mask"]
            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
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

            loss_saliency = loss_saliency + loss_rank_contrastive
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_mask"]
            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
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
            loss_rank_contrastive_attn = loss_rank_contrastive / 12

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_saliency_attn
            loss_saliency += (loss_saliency_attn * self.args.lw_wattn)
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_moment_sentence(self, outputs, targets, indices, log=True):
        positive = targets["positive_mask"].bool()
        if outputs["memory_moment"] is not None and positive.any():
            # CG's moment-side losses are defined only when an annotated moment
            # exists.  Filtering before the in-batch contrastive matrices keeps
            # null queries from becoming false positive pairs.
            moment_token = outputs["memory_moment"][positive]
            nmmemory_moment = outputs["nmmemory_moment"][positive]
            sentence_token = outputs["sentence_txt"].squeeze(1)[positive]
            sentence_dummy = outputs["sentence_dummy"].squeeze(1)[positive]

            moment_logits = F.normalize(moment_token, dim=1)
            nmoment_logits = F.normalize(nmmemory_moment, dim=1)
            sentence_logits = F.normalize(sentence_token, dim=1)
            dummy_logits = F.normalize(sentence_dummy, dim=1)

            similarity_matrix = torch.matmul(moment_logits, sentence_logits.T) # B B
            nsimilarity_matrix = torch.matmul(nmoment_logits, sentence_logits.T) # B B
            similarity_matrix = torch.cat([similarity_matrix, nsimilarity_matrix], dim=1)
            labels = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
            nlabels = torch.zeros_like(nsimilarity_matrix)
            labels = torch.cat([labels, nlabels], dim=1).max(dim=1)[1]

            loss_ms_align = self.criterion(similarity_matrix, labels)

            dummy_similarity_matrix = torch.matmul(moment_logits, dummy_logits.T)
            dummy_nsimilarity_matrix = torch.matmul(nmoment_logits, dummy_logits.T)
            dummy_similarity_matrix = torch.cat([dummy_similarity_matrix, dummy_nsimilarity_matrix], dim=1)
            dummy_labels = (~torch.eye(
                similarity_matrix.shape[0], device=similarity_matrix.device, dtype=torch.bool
            )).float()
            dummy_nlabels = torch.ones_like(nsimilarity_matrix)
            dummy_labels = torch.cat([dummy_labels, dummy_nlabels], dim=1).max(dim=1)[1]

            dummy_loss_ms_align = self.criterion(dummy_similarity_matrix, dummy_labels)
            loss_ms_align += dummy_loss_ms_align
            video_mask = outputs['video_mask'][positive]
            src_vid = outputs['src_vid'][positive]  # [positive, L_vid, D_vid]
            moment_mask_ = torch.clamp(targets["relevant_clips"][positive], 0, 1)

            momtokcls_pred = torch.matmul(moment_token.unsqueeze(1), src_vid.permute(0, 2, 1))  # bsz 1 L_vid
            momtokcls_label = moment_mask_
            momtokcls_logit = torch.sigmoid(momtokcls_pred)
            frame_bce = self.bce_criterion(
                momtokcls_logit.reshape(-1), momtokcls_label.reshape(-1)
            ) * video_mask.reshape(-1)
            loss_ms_align += frame_bce.sum() / video_mask.sum().clamp_min(1.0)

        else:
            loss_ms_align = outputs["pred_spans"].sum() * 0.0
        return {"loss_ms_align": loss_ms_align}
        #

    def loss_moment2txt_sim_distill(self, outputs, targets, indices, log=True):
        positive = targets["positive_mask"].bool()
        if outputs["moment2txt_similarity"] is not None and positive.any():
            moment2txt_similarity = outputs["moment2txt_similarity"][positive]
            moment_mask = outputs["moment_mask"][positive].to(dtype=torch.float32)

            attn_weights = outputs["cate_attn_weights"][positive]
            b, L_vid, L_txt = attn_weights.size()
            loss_distill = self.kld_criterion(
                torch.log(attn_weights + 1e-6).reshape(b * L_vid, -1),
                torch.softmax(moment2txt_similarity, dim=-1).clone().detach().reshape(b * L_vid, -1)).mean(1) * moment_mask.reshape(-1)
            loss_distill = loss_distill.sum() / moment_mask.sum().clamp_min(1.0)

        else:
            loss_distill = outputs["pred_spans"].sum() * 0.0
        return {"loss_distill": loss_distill}

    def loss_orthogonal_dummy(self, outputs, targets, indices, log=True):
        dummy_tokens = outputs["dummy_tokens"]  # (n_dum, dim)
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, dummy_tokens.shape[0], dummy_tokens.device
            )
            if not positive.any():
                return {
                    "loss_orthogonal_dummy": outputs["pred_spans"].sum() * 0.0
                }
            dummy_tokens = dummy_tokens[positive]
        if dummy_tokens.size(1) != 1:
            dummy_tokens_norm = F.normalize(dummy_tokens, dim=2, eps=1e-6)
            dummy_tokens_sim = torch.matmul(dummy_tokens_norm, dummy_tokens_norm.permute(0, 2, 1).detach())
            for i in range(len(dummy_tokens_sim)):
                dummy_tokens_sim[i].fill_diagonal_(0)
            loss_dummy_ortho = dummy_tokens_sim.abs().mean()
        else:
            loss_dummy_ortho=0.
        global_tokens = outputs["global_rep_tokens"]

        global_tokens_norm = F.normalize(global_tokens, dim=1, eps=1e-6)
        global_tokens_sim = torch.matmul(global_tokens_norm, global_tokens_norm.permute(1, 0).detach())
        for i in range(len(global_tokens_sim)):
            global_tokens_sim.fill_diagonal_(0)
        loss_dummy_ortho += global_tokens_sim.abs().mean()
        return {"loss_orthogonal_dummy": loss_dummy_ortho}


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

    def loss_existence(self, outputs, targets, indices, log=True):
        if "pred_exist_logits" not in outputs:
            return {"loss_exist": outputs["pred_spans"].sum() * 0.0}
        return {
            "loss_exist": existence_loss(
                outputs["pred_exist_logits"], targets["exist_label"]
            )
        }

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

    def loss_phrase_grounding(self, outputs, targets, indices=None, log=True):
        del indices, log
        sample_mask = None
        if self.mask_null_vmr_loss:
            attention = outputs['dual_phrase_attention']
            sample_mask = self._vmr_positive_mask(
                targets, attention.shape[0], attention.device
            )
        return dual_grounding_losses(
            outputs,
            dqa_scale=self.phrase_dqa_scale,
            temperature=self.phrase_eos_temperature,
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

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "ms_align": self.loss_contrastive_moment_sentence,
            "distill": self.loss_moment2txt_sim_distill,
            "orthogonal_dummy": self.loss_orthogonal_dummy,
            "existence": self.loss_existence,
            "quality": self.loss_quality,
            "phrase_grounding": self.loss_phrase_grounding,
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
                    losses_target = ["saliency", "ms_align", "distill", "orthogonal_dummy"]
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    if "ms_align" == loss:
                        continue
                    if "distill" == loss:
                        continue
                    if "orthogonal_dummy" == loss:
                        continue
                    if "existence" == loss:
                        continue
                    if "quality" == loss:
                        continue
                    if "phrase_grounding" == loss:
                        continue
                    if "counter" == loss:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
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

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
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

    if args.a_feat_dir is None:
        model = CGDETR(
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
            use_saliency=args.use_saliency,
            use_cg_aux=args.use_cg_aux,
            use_exist_head=args.use_exist_head,
            exist_hidden_dim=args.exist_hidden_dim,
            use_quality_head=args.use_quality_head,
            use_phrase_grounding=args.use_phrase_grounding,
            phrase_num_phrases=args.phrase_num_phrases,
            phrase_slot_iterations=args.phrase_slot_iterations,
            phrase_gate_init=args.phrase_gate_init,
            use_hierarchical_counter=args.use_hierarchical_counter,
            counter_dropout=args.counter_dropout,
            counter_detach_scores=args.counter_detach_scores,
            args=args
        )
    else:
        model = CGDETR(
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
            use_saliency=args.use_saliency,
            use_cg_aux=args.use_cg_aux,
            use_exist_head=args.use_exist_head,
            exist_hidden_dim=args.exist_hidden_dim,
            use_quality_head=args.use_quality_head,
            use_phrase_grounding=args.use_phrase_grounding,
            phrase_num_phrases=args.phrase_num_phrases,
            phrase_slot_iterations=args.phrase_slot_iterations,
            phrase_gate_init=args.phrase_gate_init,
            use_hierarchical_counter=args.use_hierarchical_counter,
            counter_dropout=args.counter_dropout,
            counter_detach_scores=args.counter_detach_scores,
            args=args
        )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
    }
    losses = ['spans', 'labels']
    if args.use_cg_aux:
        weight_dict.update({
            "loss_ms_align": args.lw_ms_align,
            "loss_distill": args.lw_distill,
            "loss_orthogonal_dummy": args.lw_distill,
        })
        losses += ['ms_align', 'distill', 'orthogonal_dummy']
    if args.use_exist_head and not args.use_hierarchical_counter:
        weight_dict["loss_exist"] = args.exist_loss_coef
        losses += ['existence']
    if args.use_quality_head:
        weight_dict["loss_quality"] = args.quality_loss_coef
        losses += ['quality']
    if args.use_phrase_grounding:
        weight_dict.update({
            "loss_dual_dqa": args.phrase_dqa_loss_coef,
            "loss_dual_eos": args.phrase_eos_loss_coef,
        })
        losses += ['phrase_grounding']
    if args.use_hierarchical_counter:
        weight_dict.update({
            "loss_exist": args.exist_loss_coef,
            "loss_count": args.count_loss_coef,
            "loss_count_ordinal": args.count_ordinal_loss_coef,
            "loss_count_contrastive": args.count_contrastive_loss_coef,
            "loss_count_consistency": args.count_consistency_loss_coef,
        })
        losses += ['counter']
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
        losses += ["contrastive_align"]

    if args.aux_loss:
        auxiliary_keys = {"loss_span", "loss_giou", "loss_label", "loss_contrastive_align"}
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                key + f'_{i}': value
                for key, value in weight_dict.items() if key in auxiliary_keys
            })
        weight_dict.update(aux_weight_dict)
        
    # For highlight detection datasets
    use_matcher = not (args.dset_name in ['youtube_uni', 'tvsum'])
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
        positive_count_weights=inverse_sqrt_positive_count_weights(
            args.positive_count_class_counts
        ),
        phrase_dqa_scale=args.phrase_dqa_scale,
        phrase_eos_temperature=args.phrase_eos_temperature,
        counter_contrastive_temperature=args.counter_contrastive_temperature,
        args=args
    )
    criterion.to(device)
    return model, criterion
