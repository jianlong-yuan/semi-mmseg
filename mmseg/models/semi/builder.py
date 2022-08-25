# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.models.builder import MODELS
ALGORITHMS = MODELS
SEMILOSS = MODELS
ARCHITECTURES = MODELS


def build_algorithm(cfg):
    """Build compressor."""
    return ALGORITHMS.build(cfg)

def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return SEMILOSS.build(cfg)
