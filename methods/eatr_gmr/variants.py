"""Canonical EaTR/HieA2M variant definitions."""

from __future__ import annotations

from dataclasses import replace

from .config import EaTRConfig


VARIANT_FLAGS = {
    "eatr": (False, False, False, False),
    "eatr_gmr": (True, False, False, False),
    "eatr_quality": (True, True, False, False),
    "eatr_dual": (True, False, True, False),
    "eatr_quality_dual": (True, True, True, False),
    "eatr_counter": (True, False, False, True),
    "eatr_hiea2m": (True, True, True, True),
}


def apply_variant(config: EaTRConfig, variant: str) -> EaTRConfig:
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"unknown EaTR variant: {variant}")
    exist, quality, dual, counter = VARIANT_FLAGS[variant]
    return replace(
        config,
        use_exist_head=exist,
        use_quality_head=quality,
        use_dual_grounding=dual,
        use_hierarchical_counter=counter,
    )


def variant_from_flags(*, exist: bool, quality: bool, dual: bool, counter: bool) -> str:
    flags = (bool(exist), bool(quality), bool(dual), bool(counter))
    for name, expected in VARIANT_FLAGS.items():
        if flags == expected:
            return name
    return "eatr_custom"
