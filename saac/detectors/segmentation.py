"""
Semantic Segmentation
Layer 3: The "Background" - Identifies semantic regions like sky, water, road.
These regions can be aggressively compressed.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional


class SemanticSegmentor:
    """
    Semantic segmentation to identify background elements that can be heavily compressed.
    Uses multiple methods: simple color-based and edge-based segmentation.
    """
    
    # Define semantic categories and their compression priorities
    CATEGORIES = {
        'sky': {'priority': 0, 'description': 'Sky regions - can be heavily compressed'},
        'water': {'priority': 1, 'description': 'Water, lakes, oceans'},
        'road': {'priority': 1, 'description': 'Roads, pavement'},
        'vegetation': {'priority': 2, 'description': 'Trees, grass, plants'},
        'building': {'priority': 3, 'description': 'Buildings, structures'},
        'unknown': {'priority': 4, 'description': 'Unknown regions'}
    }
    
    def __init__(self, method: str = 'simple', device: str = 'cpu'):
        """
        Initialize the semantic segmentor.
        
        Args:
            method: 'simple' (color/edge based) or 'deeplabv3' (deep learning)
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        self.model = None
        
        if method == 'deeplabv3':
            self._load_deeplabv3()
        
        print(f"✓ SemanticSegmentor initialized with method '{method}' on {device}")
    
    def _load_deeplabv3(self):
        """Load DeepLabV3 model (placeholder for full implementation)."""
        try:
            import torchvision
            print("  Loading DeepLabV3 model...")
            # In production, load pretrained model
            # self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
            print("  Warning: DeepLabV3 not fully implemented. Using simple method as fallback.")
        except Exception as e:
            print(f"  Warning: Could not load DeepLabV3: {e}")
            self.method = 'simple'
    
    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment image into semantic categories.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Dictionary mapping category names to binary masks
        """
        if self.method == 'simple':
            return self._simple_segmentation(image)
        elif self.method == 'deeplabv3' and self.model is not None:
            return self._deeplabv3_segmentation(image)
        else:
            return self._simple_segmentation(image)
    
    def _simple_segmentation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simple color and position-based segmentation.
        Fast and effective for common scenarios.
        """
        h, w = image.shape[:2]
        masks = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Sky detection (usually at top, blue-ish)
        sky_mask = self._detect_sky(image, hsv)
        masks['sky'] = sky_mask
        
        # Water detection (blue-ish, often with reflections)
        water_mask = self._detect_water(image, hsv, lab)
        masks['water'] = water_mask
        
        # Road detection (usually gray, at bottom)
        road_mask = self._detect_road(image, hsv)
        masks['road'] = road_mask
        
        # Vegetation detection (green)
        vegetation_mask = self._detect_vegetation(image, hsv)
        masks['vegetation'] = vegetation_mask
        
        # Building detection (straight edges)
        building_mask = self._detect_buildings(image)
        masks['building'] = building_mask
        
        # Unknown (everything else)
        all_known = sky_mask | water_mask | road_mask | vegetation_mask | building_mask
        masks['unknown'] = (~all_known.astype(bool)).astype(np.uint8)
        
        return masks
    
    def _detect_sky(self, image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """Detect sky regions (top of image, blue/white)."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sky is usually in the top 60% of the image
        top_region = hsv[:int(h * 0.6), :]
        
        # Blue sky (H: 100-130, S: 50-255, V: 100-255)
        blue_sky = cv2.inRange(top_region, (100, 50, 100), (130, 255, 255))
        
        # White/gray sky (low saturation, high value)
        white_sky = cv2.inRange(top_region, (0, 0, 180), (180, 50, 255))
        
        # Combine
        sky_top = cv2.bitwise_or(blue_sky, white_sky)
        mask[:int(h * 0.6), :] = sky_top
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _detect_water(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Detect water regions (blue, often with low texture)."""
        h, w = image.shape[:2]
        
        # Water is usually blue (H: 90-130)
        water_color = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        
        # Water often has low texture variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        variance = cv2.Laplacian(blur, cv2.CV_64F).var()
        
        # Combine color and texture
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        water_mask = cv2.morphologyEx(water_color, cv2.MORPH_CLOSE, kernel)
        
        return water_mask
    
    def _detect_road(self, image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """Detect road regions (gray, at bottom)."""
        h, w = image.shape[:2]
        
        # Road is usually at the bottom 40% of the image
        bottom_region = hsv[int(h * 0.6):, :]
        
        # Gray color (low saturation, medium value)
        gray_mask = cv2.inRange(bottom_region, (0, 0, 40), (180, 60, 180))
        
        # Create full mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.6):, :] = gray_mask
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _detect_vegetation(self, image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """Detect vegetation (green areas)."""
        # Green color (H: 35-85, S: 40-255, V: 40-255)
        vegetation_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask
    
    def _detect_buildings(self, image: np.ndarray) -> np.ndarray:
        """Detect buildings (strong vertical/horizontal edges)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines (buildings have many straight lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=50, maxLineGap=10)
        
        mask = np.zeros_like(gray)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly vertical or horizontal
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 20 or angle > 160 or (80 < angle < 100):
                    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=5)
        
        # Dilate to create regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def get_priority_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a priority mask where lower values = lower priority (more compression).
        
        Returns:
            Priority mask (H, W) with values 0-4
        """
        masks = self.segment(image)
        h, w = image.shape[:2]
        priority_mask = np.full((h, w), 4, dtype=np.uint8)  # Default: unknown
        
        # Assign priorities (lower priority regions will be compressed more)
        for category, mask in masks.items():
            if category in self.CATEGORIES:
                priority = self.CATEGORIES[category]['priority']
                priority_mask[mask > 0] = priority
        
        return priority_mask
    
    def _deeplabv3_segmentation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """DeepLabV3 segmentation (placeholder)."""
        # In full implementation, run DeepLabV3 model
        # For now, fallback to simple method
        return self._simple_segmentation(image)

