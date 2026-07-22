"""Construct the official EaTR backbone and null-safe MR-only criterion."""

from __future__ import annotations

from .config import EaTRConfig
from .criterion import SetCriterion
from .hierarchical_counter import inverse_sqrt_positive_count_weights
from .matcher import HungarianEventMatcher, HungarianMatcher
from .model import EaTR
from .position_encoding import PositionEmbeddingSine, TrainablePositionalEncoding
from .transformer import Transformer


def build_model(config: EaTRConfig):
    config.validate()
    video_position = PositionEmbeddingSine(config.hidden_dim, normalize=True)
    text_position = TrainablePositionalEncoding(
        max_position_embeddings=config.max_q_l,
        hidden_size=config.hidden_dim,
        dropout=config.input_dropout,
    )
    transformer = Transformer(
        d_model=config.hidden_dim,
        nhead=config.nheads,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        return_intermediate_dec=True,
        query_dim=2,
        num_queries=config.num_queries,
        num_iteration=config.num_slot_iter,
    )
    model = EaTR(
        transformer=transformer,
        position_embed=video_position,
        txt_position_embed=text_position,
        txt_dim=config.text_dim,
        vid_dim=config.video_dim,
        num_queries=config.num_queries,
        input_dropout=config.input_dropout,
        aux_loss=config.aux_loss,
        contrastive_align_loss=False,
        max_v_l=config.max_v_l,
        span_loss_type="l1",
        use_txt_pos=config.use_txt_pos,
        n_input_proj=config.n_input_proj,
        query_dim=2,
        use_exist_head=config.use_exist_head,
        exist_hidden_dim=config.exist_hidden_dim,
        use_quality_head=config.use_quality_head,
        use_dual_grounding=config.use_dual_grounding,
        dual_num_phrases=config.dual_num_phrases,
        dual_num_dummies=config.dual_num_dummies,
        dual_slot_iterations=config.dual_slot_iterations,
        dual_gate_init=config.dual_gate_init,
        dual_nheads=config.nheads if config.dual_nheads is None else config.dual_nheads,
        dual_max_text_len=config.max_q_l,
        use_hierarchical_counter=config.use_hierarchical_counter,
        counter_dropout=config.counter_dropout,
        counter_detach_scores=config.counter_detach_scores,
    )

    matcher = HungarianMatcher(
        cost_class=config.set_cost_class,
        cost_span=config.set_cost_span,
        cost_giou=config.set_cost_giou,
    )
    event_matcher = HungarianEventMatcher(
        cost_span=config.set_cost_span,
        cost_giou=config.set_cost_giou,
    )
    weights = {
        "loss_span": config.span_loss_coef,
        "loss_giou": config.giou_loss_coef,
        "loss_label": config.label_loss_coef,
        "loss_event_span": config.event_coef * config.span_loss_coef,
        "loss_event_giou": config.event_coef * config.giou_loss_coef,
    }
    if config.use_exist_head:
        weights["loss_exist"] = config.exist_loss_coef
    if config.use_quality_head:
        weights["loss_quality"] = config.quality_loss_coef
    if config.use_dual_grounding:
        weights.update({
            "loss_dual_dqa": config.dual_dqa_loss_coef,
            "loss_dual_eos": config.dual_eos_loss_coef,
        })
    if config.use_hierarchical_counter:
        weights.update({
            "loss_exist": config.exist_loss_coef,
            "loss_count": config.count_loss_coef,
            "loss_count_ordinal": config.count_ordinal_loss_coef,
            "loss_count_contrastive": config.count_contrastive_loss_coef,
            "loss_count_consistency": config.count_consistency_loss_coef,
        })
    if config.aux_loss:
        for layer_index in range(config.dec_layers - 1):
            for name in ("loss_span", "loss_giou", "loss_label"):
                weights[f"{name}_{layer_index}"] = weights[name]

    criterion = SetCriterion(
        matcher=matcher,
        event_matcher=event_matcher,
        weight_dict=weights,
        eos_coef=config.eos_coef,
        aux_loss=config.aux_loss,
        use_quality_head=config.use_quality_head,
        use_dual_grounding=config.use_dual_grounding,
        use_hierarchical_counter=config.use_hierarchical_counter,
        mask_null_vmr_loss=config.mask_null_vmr_loss,
        dual_dqa_scale=config.dual_dqa_scale,
        dual_eos_temperature=config.dual_eos_temperature,
        counter_contrastive_temperature=config.counter_contrastive_temperature,
        positive_count_weights=inverse_sqrt_positive_count_weights(
            list(config.positive_count_class_counts)
        ),
    )
    return model, criterion
