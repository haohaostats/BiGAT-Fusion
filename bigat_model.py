#!/usr/bin/env python3
"""Compatibility imports for BiGAT-Fusion model classes."""

from bigat_fusion.layers import BiGATLayer, GATLayer, ResidualMoEDecoder
from bigat_fusion.model import BiGATFusionModel

__all__ = ["BiGATFusionModel", "BiGATLayer", "GATLayer", "ResidualMoEDecoder"]
