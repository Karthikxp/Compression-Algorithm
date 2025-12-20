"""
Visual Saliency Detection
Layer 2: The "Eye-Catcher" - Identifies regions that naturally attract human attention.
"""

import numpy as np
import cv2
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyDetector:
    """
    Visual saliency detector to identify eye-catching regions.
    Uses multiple methods: deep learning-based and classical approaches.
    """
    
    def __init__(self, method: str = 'spectral', device: str = 'cpu'):
        """
        Initialize the saliency detector.
        
        Args:
            method: 'spectral', 'fine_grained', or 'u2net'
                   - 'spectral': Fast, frequency-domain method
                   - 'fine_grained': OpenCV's fine-grained saliency
                   - 'u2net': Deep learning (requires model download)
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        self.u2net_model = None
        
        if method == 'u2net':
            self._load_u2net()
        
        print(f"✓ SaliencyDetector initialized with method '{method}' on {device}")
    
    def _load_u2net(self):
        """Load U2-Net model for high-quality saliency detection."""
        try:
            # Try to load U2-Net model (simplified version)
            # In production, you'd download pretrained weights
            print("  Loading U2-Net model...")
            # Placeholder - in real implementation, load pretrained weights
            self.u2net_model = None  # Would load actual model here
            print("  Warning: U2-Net model not fully implemented. Using spectral method as fallback.")
        except Exception as e:
            print(f"  Warning: Could not load U2-Net: {e}")
            print("  Falling back to spectral method.")
            self.method = 'spectral'
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect salient regions in the image.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Saliency map (H, W) with values in [0, 1], where 1 = highly salient
        """
        if self.method == 'spectral':
            return self._spectral_residual(image)
        elif self.method == 'fine_grained':
            return self._fine_grained(image)
        elif self.method == 'u2net' and self.u2net_model is not None:
            return self._u2net_saliency(image)
        else:
            # Default fallback
            return self._spectral_residual(image)
    
    def _spectral_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Fast spectral residual saliency detection.
        Based on frequency domain analysis - fast and effective.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply FFT
        dft = np.fft.fft2(gray.astype(np.float32))
        magnitude = np.abs(dft)
        phase = np.angle(dft)
        
        # Compute spectral residual
        log_magnitude = np.log(magnitude + 1e-8)
        smoothed = cv2.GaussianBlur(log_magnitude, (3, 3), 0)
        residual = log_magnitude - smoothed
        
        # Reconstruct
        saliency_fft = np.exp(residual) * np.exp(1j * phase)
        saliency = np.abs(np.fft.ifft2(saliency_fft))
        
        # Smooth and normalize
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def _fine_grained(self, image: np.ndarray) -> np.ndarray:
        """
        OpenCV's fine-grained saliency detection.
        """
        try:
            # Use OpenCV's built-in saliency detector
            saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
            success, saliency_map = saliency_detector.computeSaliency(image)
            
            if success:
                # Normalize to [0, 1]
                saliency_map = (saliency_map - saliency_map.min()) / \
                              (saliency_map.max() - saliency_map.min() + 1e-8)
                return saliency_map
            else:
                print("  Warning: Fine-grained saliency failed, using spectral method.")
                return self._spectral_residual(image)
        except AttributeError:
            print("  Warning: Fine-grained saliency not available, using spectral method.")
            return self._spectral_residual(image)
    
    def _u2net_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Deep learning-based saliency using U2-Net.
        Placeholder for full implementation.
        """
        # In a full implementation, this would:
        # 1. Preprocess image
        # 2. Run through U2-Net model
        # 3. Post-process output
        # For now, fallback to spectral method
        return self._spectral_residual(image)
    
    def detect_with_threshold(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Detect salient regions and return binary mask.
        
        Args:
            image: Input image (H, W, C)
            threshold: Threshold for binarization (0-1)
            
        Returns:
            Binary mask where 1 = salient region
        """
        saliency = self.detect(image)
        return (saliency > threshold).astype(np.uint8)
    
    def detect_multi_scale(self, image: np.ndarray, scales: list = [0.5, 1.0, 1.5]) -> np.ndarray:
        """
        Multi-scale saliency detection for better robustness.
        
        Args:
            image: Input image
            scales: List of scale factors
            
        Returns:
            Fused saliency map
        """
        h, w = image.shape[:2]
        saliency_maps = []
        
        for scale in scales:
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(image, (new_w, new_h))
            
            # Detect saliency
            sal = self.detect(scaled_img)
            
            # Resize back to original size
            sal = cv2.resize(sal, (w, h))
            saliency_maps.append(sal)
        
        # Average all scales
        fused_saliency = np.mean(saliency_maps, axis=0)
        
        # Normalize
        fused_saliency = (fused_saliency - fused_saliency.min()) / \
                        (fused_saliency.max() - fused_saliency.min() + 1e-8)
        
        return fused_saliency

