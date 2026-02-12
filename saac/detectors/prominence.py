
import numpy as np
import cv2
from typing import Dict, Tuple, List


class ProminenceCalculator:
    
    def __init__(self,
                 size_threshold: float = 0.15,
                 center_radius: float = 0.3,
                 prominence_boost: float = 1.0):
        self.size_threshold = size_threshold
        self.center_radius = center_radius
        self.prominence_boost = prominence_boost
    
    def calculate_prominence(self,
                            bbox: Tuple[int, int, int, int],
                            mask: np.ndarray,
                            image_shape: Tuple[int, int]) -> Dict[str, float]:
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        
        object_area = np.sum(mask > 0) if mask is not None else (x2 - x1) * (y2 - y1)
        image_area = h * w
        area_ratio = object_area / image_area
        
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        img_center_x = w / 2
        img_center_y = h / 2
        
        dx = abs(obj_center_x - img_center_x) / (w / 2)
        dy = abs(obj_center_y - img_center_y) / (h / 2)
        distance_from_center = np.sqrt(dx**2 + dy**2)
        
        centrality = max(0, 1 - distance_from_center)
        
        prominence_score = 0.6 * min(1.0, area_ratio / self.size_threshold) + 0.4 * centrality
        
        is_prominent = (area_ratio >= self.size_threshold) or (centrality >= (1 - self.center_radius))
        
        if area_ratio >= self.size_threshold and centrality >= 0.7:
            prominence_score = 1.0
        
        return {
            'area_ratio': float(area_ratio),
            'centrality': float(centrality),
            'prominence_score': float(np.clip(prominence_score, 0, 1)),
            'is_prominent': bool(is_prominent),
            'distance_from_center': float(distance_from_center)
        }
    
    def calculate_batch_prominence(self,
                                  detections: List[Dict],
                                  image_shape: Tuple[int, int]) -> List[Dict]:
        for detection in detections:
            bbox = detection['bbox']
            mask = detection.get('mask', None)
            
            prominence = self.calculate_prominence(bbox, mask, image_shape)
            detection['prominence'] = prominence
        
        return detections
    
    def apply_prominence_weights(self,
                                base_weight: float,
                                prominence_metrics: Dict[str, float]) -> float:
        if prominence_metrics['is_prominent']:
            boost_factor = 0.5 * prominence_metrics['prominence_score']
            adjusted_weight = min(1.0, base_weight + boost_factor)
        else:
            adjusted_weight = base_weight
        
        return adjusted_weight
    
    def get_central_region_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        cx, cy = w // 2, h // 2
        rx = int(w * self.center_radius)
        ry = int(h * self.center_radius)
        
        Y, X = np.ogrid[:h, :w]
        dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        mask[dist <= 1] = 1
        
        return mask
    
    def visualize_prominence(self,
                            image: np.ndarray,
                            detections: List[Dict]) -> np.ndarray:
        vis = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            prominence = detection.get('prominence', {})
            
            x1, y1, x2, y2 = bbox
            
            if prominence.get('is_prominent', False):
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (255, 0, 0)
                thickness = 2
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            label = detection.get('class_name', 'object')
            score = prominence.get('prominence_score', 0.0)
            text = f"{label}: {score:.2f}"
            
            cv2.putText(vis, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis

