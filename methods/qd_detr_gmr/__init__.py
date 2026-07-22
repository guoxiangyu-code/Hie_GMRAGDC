"""Soccer-GMR adapter for the official QD-DETR implementation."""

from .adapter import GMRExistenceAdapter
from .model import QDDETR, build_model

__all__ = ["GMRExistenceAdapter", "QDDETR", "build_model"]
