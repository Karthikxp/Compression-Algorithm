"""
Saliency-Aware Adaptive Compression (SAAC)
A non-uniform image compression framework that prioritizes semantic importance.
"""

from .compressor import SaacCompressor
from .compressor_intelligent import IntelligentSaacCompressor

__version__ = "2.0.0"
__all__ = ['SaacCompressor', 'IntelligentSaacCompressor']

