"""
Semantic Segmentation
Layer 3: The "Background" - Identifies semantic regions like sky, water, road.
These regions can be aggressively compressed.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
        self.deeplabv3_classes = []
        
        if method == 'deeplabv3':
            if not TORCH_AVAILABLE:
                print("  Warning: PyTorch not available. Falling back to simple method.")
                self.method = 'simple'
            else:
                self._load_deeplabv3()
        
        print(f"✓ SemanticSegmentor initialized with method '{method}' on {device}")
    
    def _load_deeplabv3(self):
        """Load DeepLabV3 model for advanced semantic segmentation."""
        try:
            import torch
            import torchvision.models.segmentation as segmentation
            
            print("  Loading DeepLabV3-ResNet50 model...")
            
            # Load pretrained DeepLabV3 with ResNet50 backbone
            self.model = segmentation.deeplabv3_resnet50(weights='DEFAULT')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # DeepLabV3 trained on PASCAL VOC (21 classes)
            self.deeplabv3_classes = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'
            ]
            
            print("  ✓ DeepLabV3 model loaded successfully (21 classes)")
            
        except Exception as e:
            print(f"  Warning: Could not load DeepLabV3: {e}")
            print("  Falling back to simple method.")
            self.method = 'simple'
            self.model = None
    
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
        """
        DeepLabV3 semantic segmentation with 21 PASCAL VOC classes.
        Maps to our compression categories.
        """
        if self.model is None:
            return self._simple_segmentation(image)
        
        import torch
        import torchvision.transforms as T
        
        h, w = image.shape[:2]
        
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # DeepLabV3 preprocessing
        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((520, 520)),  # Standard size for DeepLabV3
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image_rgb).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get predictions (H, W) with class indices
        predictions = output.argmax(0).cpu().numpy()
        
        # Resize back to original size
        predictions = cv2.resize(predictions.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Map DeepLabV3 classes to our semantic categories
        masks = self._map_deeplabv3_to_categories(predictions, h, w)
        
        return masks
    
    def _map_deeplabv3_to_categories(self, predictions: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
        """
        Map DeepLabV3's 21 PASCAL VOC classes to our 6 compression categories.
        """
        masks = {
            'sky': np.zeros((h, w), dtype=np.uint8),
            'water': np.zeros((h, w), dtype=np.uint8),
            'road': np.zeros((h, w), dtype=np.uint8),
            'vegetation': np.zeros((h, w), dtype=np.uint8),
            'building': np.zeros((h, w), dtype=np.uint8),
            'unknown': np.zeros((h, w), dtype=np.uint8)
        }
        
        # DeepLabV3 doesn't have explicit sky/water/road classes
        # We use position heuristics + class info
        
        # Sky detection (top region that's "background")
        top_region = predictions[:int(h * 0.5), :]
        masks['sky'][top_region == 0] = 1  # Class 0 = background (often sky)
        
        # Vegetation: pottedplant (16)
        masks['vegetation'][predictions == 16] = 1
        
        # Building-related: chair (9), diningtable (11), sofa (18), tvmonitor (20)
        # These are indoor objects, suggesting building interior
        for cls_idx in [9, 11, 18, 20]:
            masks['building'][predictions == cls_idx] = 1
        
        # Vehicles suggest roads nearby: car (7), bus (6), motorbike (14), bicycle (2)
        for cls_idx in [2, 6, 7, 14]:
            # Expand slightly to capture road beneath vehicles
            vehicle_mask = (predictions == cls_idx).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
            expanded = cv2.dilate(vehicle_mask, kernel, iterations=1)
            # Only mark bottom region as road
            bottom_region_mask = np.zeros((h, w), dtype=np.uint8)
            bottom_region_mask[int(h * 0.5):, :] = 1
            masks['road'][expanded & bottom_region_mask > 0] = 1
        
        # Water detection (background in bottom region, not road)
        bottom_bg = predictions[int(h * 0.5):, :] == 0
        masks['water'][int(h * 0.5):, :] = bottom_bg.astype(np.uint8)
        
        # Remove water where road was detected
        masks['water'][masks['road'] > 0] = 0
        
        # Unknown: everything else
        all_known = (masks['sky'] | masks['water'] | masks['road'] | 
                    masks['vegetation'] | masks['building'])
        masks['unknown'] = (~all_known.astype(bool)).astype(np.uint8)
        
        return masks

