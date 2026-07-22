"""Serializable configuration for the isolated EaTR/EaTR-GMR implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EaTRConfig:
    video_dim: int = 2818  # SlowFast 2304 + CLIP 512 + TEF 2
    text_dim: int = 512
    hidden_dim: int = 256
    nheads: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    input_dropout: float = 0.5
    num_queries: int = 10
    num_slot_iter: int = 3
    n_input_proj: int = 2
    max_q_l: int = 32
    max_v_l: int = 75
    use_txt_pos: bool = False
    aux_loss: bool = True
    use_exist_head: bool = False
    exist_hidden_dim: int | None = None
    use_quality_head: bool = False
    use_dual_grounding: bool = False
    use_hierarchical_counter: bool = False
    mask_null_vmr_loss: bool = False

    quality_loss_coef: float = 1.0
    quality_score_alpha: float = 0.5
    diversity_lambda: float = 0.0

    dual_num_phrases: int = 3
    dual_num_dummies: int = 3
    dual_slot_iterations: int = 1
    dual_gate_init: float = -4.0
    dual_nheads: int | None = None
    dual_dqa_scale: float = 0.3
    dual_eos_temperature: float = 0.07
    dual_dqa_loss_coef: float = 0.05
    dual_eos_loss_coef: float = 0.1

    counter_dropout: float = 0.1
    counter_detach_scores: bool = True
    count_loss_coef: float = 1.0
    count_ordinal_loss_coef: float = 0.25
    count_contrastive_loss_coef: float = 0.05
    count_consistency_loss_coef: float = 0.05
    counter_contrastive_temperature: float = 0.1
    positive_count_class_counts: tuple[int, int, int, int] = (1423, 565, 117, 31)
    count_exist_threshold: float = 0.4
    count_confidence_threshold: float = 0.55
    window_score_threshold: float = 0.1

    set_cost_span: float = 10.0
    set_cost_giou: float = 1.0
    set_cost_class: float = 4.0
    span_loss_coef: float = 10.0
    giou_loss_coef: float = 1.0
    label_loss_coef: float = 4.0
    event_coef: float = 3.0
    eos_coef: float = 0.1
    exist_loss_coef: float = 1.0

    def validate(self) -> None:
        if self.hidden_dim % self.nheads:
            raise ValueError("hidden_dim must be divisible by nheads")
        if self.n_input_proj not in (1, 2, 3):
            raise ValueError("n_input_proj must be 1, 2, or 3")
        if self.enc_layers < 1 or self.dec_layers < 1:
            raise ValueError("EaTR requires at least one encoder and decoder layer")
        if self.video_dim < 3 or self.text_dim < 1:
            raise ValueError("invalid feature dimensions")
        if len(self.positive_count_class_counts) != 4 or any(
            int(value) <= 0 for value in self.positive_count_class_counts
        ):
            raise ValueError("positive_count_class_counts must contain four positive counts")
        dual_heads = self.nheads if self.dual_nheads is None else self.dual_nheads
        if self.use_dual_grounding and self.hidden_dim % dual_heads:
            raise ValueError("hidden_dim must be divisible by dual_nheads")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "EaTRConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})
