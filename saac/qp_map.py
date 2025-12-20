"""
Quantization Parameter (QP) Map Generator
Combines all detection layers to create a unified compression map.
"""

import numpy as np
import cv2
from typing import Dict, Optional


class QPMapGenerator:
    """
    Generates QP (Quantization Parameter) maps for HEVC encoding.
    Lower QP = higher quality, Higher QP = more compression.
    
    QP Range for HEVC: 0-51
    - 0-18: Visually lossless
    - 18-28: High quality
    - 28-40: Medium quality
    - 40-51: Low quality (aggressive compression)
    """
    
    def __init__(self,
                 person_qp: int = 10,
                 saliency_qp: int = 25,
                 background_qp: int = 51,
                 blend_mode: str = 'priority'):
        """
        Initialize QP map generator.
        
        Args:
            person_qp: QP for detected people/objects (default: 10, near lossless)
            saliency_qp: QP for salient regions (default: 25, high quality)
            background_qp: QP for background regions (default: 51, max compression)
            blend_mode: 'priority' (take minimum QP) or 'weighted' (blend QPs)
        """
        self.person_qp = person_qp
        self.saliency_qp = saliency_qp
        self.background_qp = background_qp
        self.blend_mode = blend_mode
        
        # Validate QP values
        for qp_val, name in [(person_qp, 'person_qp'), 
                             (saliency_qp, 'saliency_qp'),
                             (background_qp, 'background_qp')]:
            if not 0 <= qp_val <= 51:
                raise ValueError(f"{name} must be in range [0, 51], got {qp_val}")
        
        print(f"✓ QPMapGenerator initialized")
        print(f"  Person QP: {person_qp}, Saliency QP: {saliency_qp}, Background QP: {background_qp}")
    
    def generate(self,
                 image_shape: tuple,
                 object_mask: Optional[np.ndarray] = None,
                 saliency_map: Optional[np.ndarray] = None,
                 segmentation_masks: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Generate QP map by combining all detection layers.
        
        Args:
            image_shape: (H, W) or (H, W, C) of the original image
            object_mask: Binary mask from object detection (0 or 1)
            saliency_map: Saliency map with values in [0, 1]
            segmentation_masks: Dictionary of semantic segmentation masks
            
        Returns:
            QP map (H, W) with values in [0, 51]
        """
        if len(image_shape) == 3:
            h, w = image_shape[:2]
        else:
            h, w = image_shape
        
        # Start with background QP everywhere
        qp_map = np.full((h, w), self.background_qp, dtype=np.float32)
        
        # Layer 3: Semantic segmentation (if available)
        if segmentation_masks is not None:
            qp_map = self._apply_segmentation(qp_map, segmentation_masks)
        
        # Layer 2: Visual saliency (if available)
        if saliency_map is not None:
            qp_map = self._apply_saliency(qp_map, saliency_map)
        
        # Layer 1: Object detection (highest priority - if available)
        if object_mask is not None:
            qp_map = self._apply_objects(qp_map, object_mask)
        
        # Apply smoothing for better visual quality at boundaries
        qp_map = self._smooth_qp_map(qp_map)
        
        # Ensure valid range and convert to int
        qp_map = np.clip(qp_map, 0, 51).astype(np.uint8)
        
        return qp_map
    
    def _apply_segmentation(self, qp_map: np.ndarray, 
                           segmentation_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply semantic segmentation to QP map."""
        # QP values for different semantic categories
        semantic_qps = {
            'sky': 48,        # Very high compression for sky
            'water': 45,      # High compression for water
            'road': 42,       # High compression for roads
            'vegetation': 38, # Moderate-high compression for vegetation
            'building': 35,   # Moderate compression for buildings
            'unknown': self.background_qp
        }
        
        for category, mask in segmentation_masks.items():
            if category in semantic_qps:
                qp = semantic_qps[category]
                if self.blend_mode == 'priority':
                    # Take minimum QP (higher quality)
                    qp_map[mask > 0] = np.minimum(qp_map[mask > 0], qp)
                else:
                    # Weighted blend
                    qp_map[mask > 0] = (qp_map[mask > 0] + qp) / 2
        
        return qp_map
    
    def _apply_saliency(self, qp_map: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
        """Apply visual saliency to QP map."""
        # Map saliency [0, 1] to QP range
        # High saliency (1.0) -> saliency_qp (high quality)
        # Low saliency (0.0) -> background_qp (low quality)
        saliency_qp_map = self.background_qp - (self.background_qp - self.saliency_qp) * saliency_map
        
        if self.blend_mode == 'priority':
            # Take minimum QP where saliency is significant
            qp_map = np.minimum(qp_map, saliency_qp_map)
        else:
            # Weighted blend based on saliency strength
            weight = saliency_map
            qp_map = qp_map * (1 - weight) + saliency_qp_map * weight
        
        return qp_map
    
    def _apply_objects(self, qp_map: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """Apply object detection to QP map (highest priority)."""
        # Objects always get the highest quality (lowest QP)
        qp_map[object_mask > 0] = self.person_qp
        return qp_map
    
    def _smooth_qp_map(self, qp_map: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        Smooth QP map to avoid harsh transitions.
        Harsh transitions can create visible artifacts at boundaries.
        """
        # Apply Gaussian blur for smooth transitions
        smoothed = cv2.GaussianBlur(qp_map, (kernel_size, kernel_size), 0)
        return smoothed
    
    def downsample_to_macroblocks(self, qp_map: np.ndarray, 
                                  macroblock_size: int = 16) -> np.ndarray:
        """
        Downsample QP map to macroblock resolution.
        HEVC operates on macroblocks (typically 16x16 or 32x32 pixels).
        
        Args:
            qp_map: Full resolution QP map (H, W)
            macroblock_size: Size of macroblocks (default: 16)
            
        Returns:
            Downsampled QP map (H//macroblock_size, W//macroblock_size)
        """
        h, w = qp_map.shape
        mb_h = h // macroblock_size
        mb_w = w // macroblock_size
        
        # Resize to macroblock resolution using area interpolation (averaging)
        mb_qp_map = cv2.resize(qp_map.astype(np.float32), 
                               (mb_w, mb_h), 
                               interpolation=cv2.INTER_AREA)
        
        # For each macroblock, take the MINIMUM QP (highest quality)
        # This ensures important regions are not accidentally over-compressed
        mb_qp_map_min = np.zeros((mb_h, mb_w), dtype=np.float32)
        
        for i in range(mb_h):
            for j in range(mb_w):
                y1, y2 = i * macroblock_size, min((i+1) * macroblock_size, h)
                x1, x2 = j * macroblock_size, min((j+1) * macroblock_size, w)
                mb_qp_map_min[i, j] = np.min(qp_map[y1:y2, x1:x2])
        
        return mb_qp_map_min.astype(np.uint8)
    
    def save_qp_map(self, qp_map: np.ndarray, output_path: str, 
                    macroblock_size: int = 16):
        """
        Save QP map in format suitable for FFmpeg.
        
        Args:
            qp_map: Full or macroblock resolution QP map
            output_path: Path to save the QP map file
            macroblock_size: Macroblock size (for downsampling if needed)
        """
        # Check if already at macroblock resolution
        h, w = qp_map.shape
        
        # If map is large, downsample to macroblock resolution
        if h > 500 or w > 500:  # Likely full resolution
            mb_qp_map = self.downsample_to_macroblocks(qp_map, macroblock_size)
        else:
            mb_qp_map = qp_map
        
        # Save as raw binary file (int8 format for FFmpeg)
        mb_qp_map.astype(np.int8).tofile(output_path)
        
        print(f"  QP map saved: {output_path} (shape: {mb_qp_map.shape})")
    
    def visualize_qp_map(self, qp_map: np.ndarray) -> np.ndarray:
        """
        Create a colorized visualization of the QP map.
        
        Args:
            qp_map: QP map (H, W)
            
        Returns:
            Colored visualization (H, W, 3) in BGR format
        """
        # Normalize to [0, 255]
        normalized = ((51 - qp_map) / 51 * 255).astype(np.uint8)
        
        # Apply colormap (RAINBOW: red=high quality, blue=low quality)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def get_statistics(self, qp_map: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about the QP map.
        
        Returns:
            Dictionary with mean, min, max, and quality distribution
        """
        stats = {
            'mean_qp': float(np.mean(qp_map)),
            'min_qp': int(np.min(qp_map)),
            'max_qp': int(np.max(qp_map)),
            'median_qp': float(np.median(qp_map)),
            'high_quality_percent': float(np.sum(qp_map <= 20) / qp_map.size * 100),
            'medium_quality_percent': float(np.sum((qp_map > 20) & (qp_map <= 35)) / qp_map.size * 100),
            'low_quality_percent': float(np.sum(qp_map > 35) / qp_map.size * 100)
        }
        
        return stats

