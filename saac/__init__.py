"""
Saliency-Aware Adaptive Compression (SAAC)
A non-uniform image compression framework that prioritizes semantic importance.
"""

from .compressor import SaacCompressor
from .avif_encoder import AVIFEncoder

__version__ = "2.1.0"
__all__ = ['SaacCompressor', 'AVIFEncoder']

