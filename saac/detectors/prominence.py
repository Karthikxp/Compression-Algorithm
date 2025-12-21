"""
Prominence Calculator
Determines if an object is "important" based on size, location, and centrality.
This is the "automatic override" system that boosts important objects.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List


class ProminenceCalculator:
    """
    Calculates object prominence (importance) based on:
    1. Size: Does it take up significant portion of the image?
    2. Location: Is it centrally located?
    3. Focus: Is it in the sharp/focused region?
    """
    
    def __init__(self,
                 size_threshold: float = 0.15,
                 center_radius: float = 0.3,
                 prominence_boost: float = 1.0):
        """
        Initialize prominence calculator.
        
        Args:
            size_threshold: Minimum area fraction (0-1) to be considered prominent
            center_radius: Radius from center (0-1) where objects get boosted
            prominence_boost: Weight multiplier for prominent objects
        """
        self.size_threshold = size_threshold
        self.center_radius = center_radius
        self.prominence_boost = prominence_boost
    
    def calculate_prominence(self,
                            bbox: Tuple[int, int, int, int],
                            mask: np.ndarray,
                            image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate prominence score for an object.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            mask: Binary mask of the object (same size as image)
            image_shape: (height, width) of the image
            
        Returns:
            Dictionary with prominence metrics:
            - area_ratio: 0-1, fraction of image covered
            - centrality: 0-1, how centered the object is
            - prominence_score: 0-1, overall prominence
            - is_prominent: bool, whether object passes prominence threshold
        """
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        
        # Calculate size metrics
        object_area = np.sum(mask > 0) if mask is not None else (x2 - x1) * (y2 - y1)
        image_area = h * w
        area_ratio = object_area / image_area
        
        # Calculate centrality (distance from center)
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        img_center_x = w / 2
        img_center_y = h / 2
        
        # Normalized distance from center (0 = center, 1 = corner)
        dx = abs(obj_center_x - img_center_x) / (w / 2)
        dy = abs(obj_center_y - img_center_y) / (h / 2)
        distance_from_center = np.sqrt(dx**2 + dy**2)
        
        # Centrality score (1 = perfect center, 0 = far corner)
        centrality = max(0, 1 - distance_from_center)
        
        # Combined prominence score
        # Weight: size 60%, centrality 40%
        prominence_score = 0.6 * min(1.0, area_ratio / self.size_threshold) + 0.4 * centrality
        
        # Check if object is prominent
        is_prominent = (area_ratio >= self.size_threshold) or (centrality >= (1 - self.center_radius))
        
        # Additional boost if object is both large AND central
        if area_ratio >= self.size_threshold and centrality >= 0.7:
            prominence_score = 1.0  # Maximum prominence
        
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
        """
        Calculate prominence for multiple detections.
        
        Args:
            detections: List of detection dicts with 'bbox' and optionally 'mask'
            image_shape: (height, width)
            
        Returns:
            List of detection dicts with added prominence metrics
        """
        for detection in detections:
            bbox = detection['bbox']
            mask = detection.get('mask', None)
            
            prominence = self.calculate_prominence(bbox, mask, image_shape)
            detection['prominence'] = prominence
        
        return detections
    
    def apply_prominence_weights(self,
                                base_weight: float,
                                prominence_metrics: Dict[str, float]) -> float:
        """
        Apply prominence boost to base weight.
        
        Args:
            base_weight: Base importance weight from intent rules (0-1)
            prominence_metrics: Prominence metrics from calculate_prominence
            
        Returns:
            Adjusted weight (0-1)
        """
        if prominence_metrics['is_prominent']:
            # Boost the weight based on prominence score
            boost_factor = 0.5 * prominence_metrics['prominence_score']
            adjusted_weight = min(1.0, base_weight + boost_factor)
        else:
            adjusted_weight = base_weight
        
        return adjusted_weight
    
    def get_central_region_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get a mask for the central region of the image.
        Useful for prioritizing central objects.
        
        Args:
            image_shape: (height, width)
            
        Returns:
            Binary mask with 1s in central region
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define central region (based on center_radius parameter)
        cx, cy = w // 2, h // 2
        rx = int(w * self.center_radius)
        ry = int(h * self.center_radius)
        
        # Create elliptical central region
        Y, X = np.ogrid[:h, :w]
        dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        mask[dist <= 1] = 1
        
        return mask
    
    def visualize_prominence(self,
                            image: np.ndarray,
                            detections: List[Dict]) -> np.ndarray:
        """
        Create visualization showing prominence scores.
        
        Args:
            image: Original image (H, W, 3)
            detections: List of detections with prominence metrics
            
        Returns:
            Visualization image
        """
        vis = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            prominence = detection.get('prominence', {})
            
            x1, y1, x2, y2 = bbox
            
            # Color based on prominence (green = prominent, blue = not)
            if prominence.get('is_prominent', False):
                color = (0, 255, 0)  # Green
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with prominence score
            label = detection.get('class_name', 'object')
            score = prominence.get('prominence_score', 0.0)
            text = f"{label}: {score:.2f}"
            
            cv2.putText(vis, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis

