"""Neural building blocks for circular 16-electrode sEMG models and baselines."""

from .aemg import AEMGEncoder, AEMGPretrainingModel
from .bendr import BendrEncoder, BendrLikeContrastiveLoss, BendrPretrainingModel
from .cbramod import CBraModEncoder, CBraModPretrainingModel
from .patch_encoder import MultiScalePatchEncoder
from .ring_criss_cross import RingCrissCrossBlock, RingCrissCrossEncoder
from .ring_attention import RingMultiheadAttention, signed_ring_offsets
from .supervised_tcn import SupervisedTCN

__all__ = [
    "AEMGEncoder",
    "AEMGPretrainingModel",
    "BendrEncoder",
    "BendrLikeContrastiveLoss",
    "BendrPretrainingModel",
    "CBraModEncoder",
    "CBraModPretrainingModel",
    "MultiScalePatchEncoder",
    "RingCrissCrossBlock",
    "RingCrissCrossEncoder",
    "RingMultiheadAttention",
    "signed_ring_offsets",
    "SupervisedTCN",
]
