# Copyright (c) OpenMMLab. All rights reserved.

from .algorithms import *  # noqa: F401,F403
from .architectures import *  # noqa: F401,F403
from .builder import (ALGORITHMS, ARCHITECTURES, SEMILOSS, build_algorithm, build_architecture,
                      build_loss)
from .losses import *  # noqa: F401,F403
from mmseg.models import *  # noqa: F401,F403

__all__ = [
    'ALGORITHMS', 'ARCHITECTURES', 'SEMILOSS', 'build_architecture',
    'build_algorithm', 'build_loss'
]
