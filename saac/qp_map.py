"""
QP Map Generator
Combines scene intent, prominence, and saliency for smart compression.
"""

import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple

from .intent_rules import IntentRuleEngine
from .detectors.prominence import ProminenceCalculator


class QPMapGenerator:
    """
    Generates QP (Quantization Parameter) maps using intelligent weight calculation.
    
    New Flow:
    1. Scene intent provides base weights
    2. Prominence boosts important objects
    3. Saliency fills in the gaps
    4. Semantic segmentation adjusts background
    """
    
    def __init__(self,
                 base_qp: int = 51,
                 high_quality_qp: int = 10,
                 mid_quality_qp: int = 30,
                 blend_mode: str = 'priority'):
        """
        Initialize intelligent QP map generator.
        
        Args:
            base_qp: QP for unimportant regions (default: 51, max compression)
            high_quality_qp: QP for critical regions (default: 10, near lossless)
            mid_quality_qp: QP for moderately important regions (default: 30)
            blend_mode: 'priority' (take minimum QP) or 'weighted' (blend QPs)
        """
        self.base_qp = base_qp
        self.high_quality_qp = high_quality_qp
        self.mid_quality_qp = mid_quality_qp
        self.blend_mode = blend_mode
        
        # Initialize sub-components
        self.intent_engine = IntentRuleEngine()
        self.prominence_calc = ProminenceCalculator(
            size_threshold=0.15,  # 15% of image
            center_radius=0.3,    # 30% from center
            prominence_boost=1.0
        )
        
        print(f"✓ QPMapGenerator initialized")
        print(f"  QP range: {high_quality_qp} (best) to {base_qp} (most compressed)")
    
    def generate(self,
                 image_shape: tuple,
                 scene: str,
                 detections: List[Dict],
                 saliency_map: Optional[np.ndarray] = None,
                 segmentation_masks: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Generate intelligent QP map using the new flow.
        
        Args:
            image_shape: (H, W) or (H, W, C)
            scene: Scene category (e.g., 'restaurant', 'landscape')
            detections: List of detection dicts with 'bbox', 'mask', 'class_name', etc.
            saliency_map: Optional saliency map [0, 1]
            segmentation_masks: Optional semantic segmentation masks
            
        Returns:
            QP map (H, W) with values in [0, 51]
        """
        if len(image_shape) == 3:
            h, w = image_shape[:2]
        else:
            h, w = image_shape
        
        print(f"\n  Scene detected: {scene}")
        print(f"  {self.intent_engine.INTENT_PROFILES[scene]['description']}")
        
        # Step 1: Set scene intent
        self.intent_engine.set_scene(scene)
        
        # Step 2: Calculate prominence for all detections
        detections = self.prominence_calc.calculate_batch_prominence(detections, (h, w))
        
        # Step 3: Build weight map from detections
        weight_map = self._build_weight_map(detections, (h, w))
        
        # Step 4: Apply saliency (fills gaps)
        if saliency_map is not None:
            weight_map = self._apply_saliency(weight_map, saliency_map)
        
        # Step 5: Apply semantic segmentation (background adjustment)
        if segmentation_masks is not None:
            weight_map = self._apply_segmentation(weight_map, segmentation_masks)
        
        # Step 6: Convert weights to QP values
        qp_map = self._weights_to_qp(weight_map)
        
        # Step 7: Smooth transitions
        qp_map = self._smooth_qp_map(qp_map)
        
        # Ensure valid range
        qp_map = np.clip(qp_map, 0, 51).astype(np.uint8)
        
        return qp_map
    
    def _build_weight_map(self, detections: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Build weight map from detections using intent rules + prominence.
        
        Weight Map: 1.0 = maximum quality, 0.0 = maximum compression
        """
        h, w = image_shape
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        print(f"  Processing {len(detections)} detections...")
        
        for det in detections:
            # Get base weight from intent rules
            base_weight = self.intent_engine.get_weight_for_class(
                class_id=det.get('class'),
                class_name=det.get('class_name')
            )
            
            # Apply prominence boost
            prominence = det.get('prominence', {})
            final_weight = self.prominence_calc.apply_prominence_weights(
                base_weight, prominence
            )
            
            # Log prominent objects
            if prominence.get('is_prominent', False):
                print(f"    ⭐ {det.get('class_name', 'object')}: "
                      f"weight {base_weight:.2f} → {final_weight:.2f} "
                      f"(prominent: {prominence.get('prominence_score', 0):.2f})")
            
            # Apply weight to mask or bbox
            if det.get('mask') is not None:
                # Use pixel-perfect segmentation mask
                mask = det['mask']
                if self.blend_mode == 'priority':
                    # Take maximum weight (best quality)
                    weight_map = np.maximum(weight_map, mask.astype(np.float32) * final_weight)
                else:
                    # Weighted blend
                    weight_map += mask.astype(np.float32) * final_weight
            else:
                # Fallback to bounding box
                x1, y1, x2, y2 = det['bbox']
                if self.blend_mode == 'priority':
                    weight_map[y1:y2, x1:x2] = np.maximum(
                        weight_map[y1:y2, x1:x2], final_weight
                    )
                else:
                    weight_map[y1:y2, x1:x2] += final_weight
        
        # Normalize if using weighted blend
        if self.blend_mode == 'weighted':
            weight_map = np.clip(weight_map, 0, 1)
        
        return weight_map
    
    def _apply_saliency(self, weight_map: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
        """
        Apply saliency with MAXIMUM influence - saliency is superior for text/detail detection.
        
        Saliency now DOMINATES the quality allocation, overriding low-importance objects.
        Only high-importance objects (persons, etc.) can override saliency.
        """
        # Full 1.0x saliency contribution (was 0.6x, then 0.9x, now FULL power)
        saliency_weight = saliency_map * 1.0
        
        if self.blend_mode == 'priority':
            # ALWAYS take maximum - saliency can override low-importance objects
            weight_map = np.maximum(weight_map, saliency_weight)
            
            # Additional boost: where saliency is high (>0.7) and object weight is low (<0.5),
            # give saliency even MORE influence
            high_saliency_mask = (saliency_map > 0.7) & (weight_map < 0.5)
            if np.any(high_saliency_mask):
                # Boost high-saliency regions to at least 0.85 weight
                weight_map[high_saliency_mask] = np.maximum(
                    weight_map[high_saliency_mask], 
                    0.85
                )
        else:
            # Weighted mode: Give saliency MAJORITY influence (70%)
            weight_map = 0.3 * weight_map + 0.7 * saliency_weight
            weight_map = np.clip(weight_map, 0, 1)
        
        return weight_map
    
    def _apply_segmentation(self, weight_map: np.ndarray, 
                           segmentation_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust background regions based on semantic segmentation."""
        # Background categories get reduced weight
        background_penalties = {
            'sky': 0.1,        # Very low importance
            'water': 0.15,
            'road': 0.2,
            'vegetation': 0.25,
            'building': 0.3,
            'unknown': 0.0
        }
        
        for category, mask in segmentation_masks.items():
            if category in background_penalties:
                penalty_weight = background_penalties[category]
                # Only reduce weight where it's currently low (background regions)
                background_mask = (weight_map < 0.3) & (mask > 0)
                weight_map[background_mask] = penalty_weight
        
        return weight_map
    
    def _weights_to_qp(self, weight_map: np.ndarray) -> np.ndarray:
        """
        Convert weight map (0-1) to QP map (0-51).
        
        Weight 1.0 → QP 10 (best quality)
        Weight 0.0 → QP 51 (most compression)
        """
        # Linear mapping: QP = base_qp - (base_qp - high_quality_qp) * weight
        qp_map = self.base_qp - (self.base_qp - self.high_quality_qp) * weight_map
        return qp_map.astype(np.float32)
    
    def _smooth_qp_map(self, qp_map: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Smooth QP map to avoid harsh transitions."""
        smoothed = cv2.GaussianBlur(qp_map, (kernel_size, kernel_size), 0)
        return smoothed
    
    def visualize_qp_map(self, qp_map: np.ndarray) -> np.ndarray:
        """Create colorized visualization of QP map."""
        # Normalize to [0, 255] (invert so high quality = bright)
        normalized = ((51 - qp_map) / 51 * 255).astype(np.uint8)
        
        # Apply colormap (JET: red=high quality, blue=low quality)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def get_statistics(self, qp_map: np.ndarray) -> Dict[str, float]:
        """Get statistics about the QP map."""
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

