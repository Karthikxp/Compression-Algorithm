
import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple

from .intent_rules import IntentRuleEngine
from .detectors.prominence import ProminenceCalculator


class QPMapGenerator:
    
    def __init__(self,
                 base_qp: int = 51,
                 high_quality_qp: int = 10,
                 mid_quality_qp: int = 30,
                 blend_mode: str = 'priority'):
        self.base_qp = base_qp
        self.high_quality_qp = high_quality_qp
        self.mid_quality_qp = mid_quality_qp
        self.blend_mode = blend_mode
        
        self.intent_engine = IntentRuleEngine()
        self.prominence_calc = ProminenceCalculator(
            size_threshold=0.15,
            center_radius=0.3,
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
        if len(image_shape) == 3:
            h, w = image_shape[:2]
        else:
            h, w = image_shape
        
        print(f"\n  Scene detected: {scene}")
        print(f"  {self.intent_engine.INTENT_PROFILES[scene]['description']}")
        
        self.intent_engine.set_scene(scene)
        
        detections = self.prominence_calc.calculate_batch_prominence(detections, (h, w))
        
        weight_map = self._build_weight_map(detections, (h, w))
        
        if saliency_map is not None:
            weight_map = self._apply_saliency(weight_map, saliency_map)
        
        if segmentation_masks is not None:
            weight_map = self._apply_segmentation(weight_map, segmentation_masks)
        
        qp_map = self._weights_to_qp(weight_map)
        
        qp_map = self._smooth_qp_map(qp_map)
        
        qp_map = np.clip(qp_map, 0, 51).astype(np.uint8)
        
        return qp_map
    
    def _build_weight_map(self, detections: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        print(f"  Processing {len(detections)} detections...")
        
        for det in detections:
            base_weight = self.intent_engine.get_weight_for_class(
                class_id=det.get('class'),
                class_name=det.get('class_name')
            )
            
            prominence = det.get('prominence', {})
            final_weight = self.prominence_calc.apply_prominence_weights(
                base_weight, prominence
            )
            
            if prominence.get('is_prominent', False):
                print(f"    ⭐ {det.get('class_name', 'object')}: "
                      f"weight {base_weight:.2f} → {final_weight:.2f} "
                      f"(prominent: {prominence.get('prominence_score', 0):.2f})")
            
            if det.get('mask') is not None:
                mask = det['mask']
                if self.blend_mode == 'priority':
                    weight_map = np.maximum(weight_map, mask.astype(np.float32) * final_weight)
                else:
                    weight_map += mask.astype(np.float32) * final_weight
            else:
                x1, y1, x2, y2 = det['bbox']
                if self.blend_mode == 'priority':
                    weight_map[y1:y2, x1:x2] = np.maximum(
                        weight_map[y1:y2, x1:x2], final_weight
                    )
                else:
                    weight_map[y1:y2, x1:x2] += final_weight
        
        if self.blend_mode == 'weighted':
            weight_map = np.clip(weight_map, 0, 1)
        
        return weight_map
    
    def _apply_saliency(self, weight_map: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
        saliency_weight = saliency_map * 1.0
        
        if self.blend_mode == 'priority':
            weight_map = np.maximum(weight_map, saliency_weight)
            
            high_saliency_mask = (saliency_map > 0.7) & (weight_map < 0.5)
            if np.any(high_saliency_mask):
                weight_map[high_saliency_mask] = np.maximum(
                    weight_map[high_saliency_mask], 
                    0.85
                )
        else:
            weight_map = 0.3 * weight_map + 0.7 * saliency_weight
            weight_map = np.clip(weight_map, 0, 1)
        
        return weight_map
    
    def _apply_segmentation(self, weight_map: np.ndarray, 
                           segmentation_masks: Dict[str, np.ndarray]) -> np.ndarray:
        background_penalties = {
            'sky': 0.1,
            'water': 0.15,
            'road': 0.2,
            'vegetation': 0.25,
            'building': 0.3,
            'unknown': 0.0
        }
        
        for category, mask in segmentation_masks.items():
            if category in background_penalties:
                penalty_weight = background_penalties[category]
                background_mask = (weight_map < 0.3) & (mask > 0)
                weight_map[background_mask] = penalty_weight
        
        return weight_map
    
    def _weights_to_qp(self, weight_map: np.ndarray) -> np.ndarray:
        qp_map = self.base_qp - (self.base_qp - self.high_quality_qp) * weight_map
        return qp_map.astype(np.float32)
    
    def _smooth_qp_map(self, qp_map: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        smoothed = cv2.GaussianBlur(qp_map, (kernel_size, kernel_size), 0)
        return smoothed
    
    def visualize_qp_map(self, qp_map: np.ndarray) -> np.ndarray:
        normalized = ((51 - qp_map) / 51 * 255).astype(np.uint8)
        
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def get_statistics(self, qp_map: np.ndarray) -> Dict[str, float]:
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

