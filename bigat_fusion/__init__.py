"""BiGAT-Fusion package."""

from .data import load_dataset
from .model import BiGATFusionModel

__all__ = ["BiGATFusionModel", "load_dataset"]
