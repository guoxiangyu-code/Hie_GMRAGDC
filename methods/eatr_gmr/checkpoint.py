"""Checkpoint structure detection and safe parent-to-child initialization."""

from __future__ import annotations

from dataclasses import replace

from .config import EaTRConfig
from .variants import variant_from_flags


OPTIONAL_PREFIXES = (
    "exist_head.",
    "quality_embed.",
    "dual_grounding.",
    "hierarchical_counter.",
)


def detect_state_structure(state_dict: dict) -> dict[str, object]:
    flags = {
        "exist": any(name.startswith("exist_head.") for name in state_dict),
        "quality": any(name.startswith("quality_embed.") for name in state_dict),
        "dual": any(name.startswith("dual_grounding.") for name in state_dict),
        "counter": any(name.startswith("hierarchical_counter.") for name in state_dict),
    }
    flags["variant"] = variant_from_flags(**flags)
    return flags


def config_from_checkpoint(checkpoint: dict) -> tuple[EaTRConfig, dict[str, object]]:
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint model state must be a mapping")
    config = EaTRConfig.from_dict(checkpoint.get("config", {}))
    structure = detect_state_structure(state_dict)
    config = replace(
        config,
        use_exist_head=bool(structure["exist"]),
        use_quality_head=bool(structure["quality"]),
        use_dual_grounding=bool(structure["dual"]),
        use_hierarchical_counter=bool(structure["counter"]),
    )
    return config, structure


def load_parent_state(model, parent_state: dict) -> list[str]:
    """Load a parent state while allowing only newly introduced branch keys."""
    incompatible = model.load_state_dict(parent_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"parent checkpoint has unexpected keys: {incompatible.unexpected_keys}"
        )
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith(OPTIONAL_PREFIXES)
    ]
    if invalid_missing:
        raise RuntimeError(f"parent checkpoint misses backbone keys: {invalid_missing}")
    return list(incompatible.missing_keys)
