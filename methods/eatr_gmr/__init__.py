"""Official-EaTR-derived Soccer-GMR and unified HieA2M variants."""

from .build import build_model
from .config import EaTRConfig
from .dataset import SoccerGMRDataset, collate_fn, video_id_to_feature_stem
from .variants import VARIANT_FLAGS, apply_variant

__all__ = [
    "EaTRConfig",
    "SoccerGMRDataset",
    "build_model",
    "collate_fn",
    "video_id_to_feature_stem",
    "VARIANT_FLAGS",
    "apply_variant",
]
