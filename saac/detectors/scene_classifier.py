"""
Scene/Intent Classifier
The "Context" - Determines what type of scene this is to apply appropriate rules.
Uses EfficientNet or ResNet for fast scene classification.
"""

import numpy as np
import cv2
from typing import Dict, Tuple
import torch
import torch.nn as nn

try:
    import torchvision.models as models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SceneClassifier:
    """
    Fast scene classifier to determine compression intent.
    Acts like a camera's "mode switch" (Portrait, Landscape, Food, etc.)
    """
    
    # Scene categories mapped to compression intents
    SCENE_MAPPINGS = {
        # Places365 → Our Intent Categories
        'restaurant': 'restaurant',
        'dining_room': 'restaurant',
        'cafeteria': 'restaurant',
        'food_court': 'restaurant',
        
        'mountain': 'landscape',
        'beach': 'landscape',
        'forest_path': 'landscape',
        'sky': 'landscape',
        'field': 'landscape',
        'coast': 'landscape',
        
        'street': 'street',
        'highway': 'street',
        'crosswalk': 'street',
        'parking_lot': 'street',
        
        'office': 'document',
        'library': 'document',
        'classroom': 'document',
        
        'living_room': 'indoor',
        'bedroom': 'indoor',
        'conference_room': 'indoor',
        
        'shop': 'retail',
        'mall': 'retail',
        'supermarket': 'retail',
    }
    
    def __init__(self, method: str = 'efficientnet', device: str = 'cpu'):
        """
        Initialize scene classifier.
        
        Args:
            method: 'efficientnet', 'resnet', or 'simple'
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        self.model = None
        self.places_classes = []
        
        if method in ['efficientnet', 'resnet']:
            if not TORCH_AVAILABLE:
                print("  Warning: PyTorch not available. Using simple heuristics.")
                self.method = 'simple'
            else:
                self._load_model()
        
        print(f"✓ SceneClassifier initialized with method '{self.method}' on {device}")
    
    def _load_model(self):
        """Load pretrained scene classification model."""
        try:
            if self.method == 'efficientnet':
                # EfficientNet-B0 is fast and accurate
                print("  Loading EfficientNet-B0 for scene classification...")
                self.model = models.efficientnet_b0(weights='DEFAULT')
                self.model.eval()
                self.model = self.model.to(self.device)
                print("  ✓ EfficientNet-B0 loaded")
                
            elif self.method == 'resnet':
                # ResNet18 as fallback
                print("  Loading ResNet18 for scene classification...")
                self.model = models.resnet18(weights='DEFAULT')
                self.model.eval()
                self.model = self.model.to(self.device)
                print("  ✓ ResNet18 loaded")
            
            # Load ImageNet class names (we'll use these as proxies)
            self._load_imagenet_classes()
            
        except Exception as e:
            print(f"  Warning: Could not load model: {e}")
            print("  Falling back to simple heuristics.")
            self.method = 'simple'
            self.model = None
    
    def _load_imagenet_classes(self):
        """Load ImageNet class labels."""
        # These are the 1000 ImageNet classes
        # We'll map them to our scene categories
        self.imagenet_to_scene = {
            'restaurant': 'restaurant',
            'pizzeria': 'restaurant',
            'bakery': 'restaurant',
            'street': 'street',
            'highway': 'street',
            'valley': 'landscape',
            'seashore': 'landscape',
            'lakeside': 'landscape',
            'alp': 'landscape',
        }
    
    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify the scene/intent of the image.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            (scene_category, confidence) tuple
            scene_category: One of ['restaurant', 'landscape', 'street', 'document', 'indoor', 'retail', 'general']
            confidence: 0.0 to 1.0
        """
        if self.method == 'simple' or self.model is None:
            return self._classify_simple(image)
        else:
            return self._classify_deep(image)
    
    def _classify_deep(self, image: np.ndarray) -> Tuple[str, float]:
        """Deep learning based scene classification."""
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image_rgb).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top prediction
        top_prob, top_idx = torch.topk(probabilities, 1)
        confidence = float(top_prob[0])
        
        # Map ImageNet class to our scene categories
        # For now, use heuristics (in production, you'd use Places365 model)
        scene = self._infer_scene_from_imagenet(top_idx.item(), image)
        
        return scene, confidence
    
    def _infer_scene_from_imagenet(self, class_idx: int, image: np.ndarray) -> str:
        """Infer scene category from ImageNet prediction and image analysis."""
        # Simple heuristics based on image analysis
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check for sky (blue at top) → landscape
        top_region = hsv[:h//3, :]
        blue_sky = cv2.inRange(top_region, (100, 50, 100), (130, 255, 255))
        sky_ratio = np.sum(blue_sky > 0) / blue_sky.size
        
        if sky_ratio > 0.3:
            return 'landscape'
        
        # Check for road (gray at bottom) → street
        bottom_region = hsv[2*h//3:, :]
        gray_road = cv2.inRange(bottom_region, (0, 0, 40), (180, 60, 180))
        road_ratio = np.sum(gray_road > 0) / gray_road.size
        
        if road_ratio > 0.3:
            return 'street'
        
        # Check for indoor (walls, furniture patterns)
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if edge_ratio > 0.15:
            return 'indoor'
        
        # Default
        return 'general'
    
    def _classify_simple(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Fast heuristic-based scene classification.
        Uses color, position, edge patterns, and composition analysis.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        features = {}
        
        # 1. Sky detection (top region, blue/white)
        top_region = hsv[:h//3, :]
        blue_sky = cv2.inRange(top_region, (100, 50, 100), (130, 255, 255))
        white_sky = cv2.inRange(top_region, (0, 0, 180), (180, 50, 255))
        sky_mask = cv2.bitwise_or(blue_sky, white_sky)
        features['sky_ratio'] = np.sum(sky_mask > 0) / sky_mask.size
        
        # 2. Road detection (bottom region, gray)
        bottom_region = hsv[2*h//3:, :]
        gray_road = cv2.inRange(bottom_region, (0, 0, 40), (180, 60, 180))
        features['road_ratio'] = np.sum(gray_road > 0) / gray_road.size
        
        # 3. Green vegetation (outdoor scenes)
        green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        features['vegetation_ratio'] = np.sum(green > 0) / green.size
        
        # 4. Skin tone detection (people/portraits)
        # HSV ranges for skin tones (works for various skin colors)
        skin1 = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin2 = cv2.inRange(hsv, (0, 10, 60), (25, 150, 255))
        skin_mask = cv2.bitwise_or(skin1, skin2)
        features['skin_ratio'] = np.sum(skin_mask > 0) / skin_mask.size
        
        # Check if skin is in center (portrait indicator)
        center_h_start, center_h_end = h//4, 3*h//4
        center_w_start, center_w_end = w//4, 3*w//4
        center_region = skin_mask[center_h_start:center_h_end, center_w_start:center_w_end]
        features['center_skin_ratio'] = np.sum(center_region > 0) / center_region.size if center_region.size > 0 else 0
        
        # 5. Food colors (warm tones - red, orange, yellow, brown)
        food_red = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        food_orange = cv2.inRange(hsv, (10, 50, 50), (25, 255, 255))
        food_yellow = cv2.inRange(hsv, (25, 50, 50), (35, 255, 255))
        food_brown = cv2.inRange(hsv, (10, 20, 20), (20, 255, 200))
        food_mask = cv2.bitwise_or(cv2.bitwise_or(food_red, food_orange), cv2.bitwise_or(food_yellow, food_brown))
        features['food_ratio'] = np.sum(food_mask > 0) / food_mask.size
        
        # 6. Edge density (indoor scenes have more straight edges)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 7. Blur detection (outdoor landscapes tend to be sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['sharpness'] = laplacian_var
        
        # 8. Color saturation (outdoor scenes tend to be more saturated)
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        
        # Classification logic with priority order
        
        # PORTRAIT: High skin ratio in center, low sky/vegetation
        if (features['center_skin_ratio'] > 0.15 or features['skin_ratio'] > 0.20) and \
           features['sky_ratio'] < 0.30:
            return 'portrait', 0.85
        
        # RESTAURANT: High food colors + moderate skin (people eating)
        if features['food_ratio'] > 0.15 and features['skin_ratio'] > 0.05:
            return 'restaurant', 0.80
        
        # STREET: High road ratio + low sky
        if features['road_ratio'] > 0.30:
            return 'street', 0.80
        
        # STREET: Moderate road with vehicles (gray/black patterns)
        if features['road_ratio'] > 0.20 and features['edge_density'] > 0.12:
            return 'street', 0.75
        
        # INDOOR: High edge density + low sky + low vegetation
        if features['edge_density'] > 0.18 and features['sky_ratio'] < 0.15 and \
           features['vegetation_ratio'] < 0.15:
            return 'indoor', 0.75
        
        # LANDSCAPE: High sky + high vegetation (classic outdoor scene)
        if features['sky_ratio'] > 0.35 and features['vegetation_ratio'] > 0.20:
            return 'landscape', 0.85
        
        # LANDSCAPE: Very high sky ratio
        if features['sky_ratio'] > 0.45:
            return 'landscape', 0.80
        
        # RETAIL/INDOOR: Moderate edges, low outdoor indicators
        if features['edge_density'] > 0.15 and features['sky_ratio'] < 0.10:
            return 'retail', 0.70
        
        # GENERAL: Use as fallback with some intelligence
        # If there's some skin but doesn't fit portrait, likely indoor with people
        if features['skin_ratio'] > 0.10:
            return 'indoor', 0.60
        
        # If moderate vegetation/sky but not enough for landscape
        if features['vegetation_ratio'] > 0.15 or features['sky_ratio'] > 0.20:
            return 'landscape', 0.60
        
        # True fallback
        return 'general', 0.50
    
    def get_scene_description(self, scene: str) -> str:
        """Get human-readable description of scene category."""
        descriptions = {
            'restaurant': 'Restaurant/Dining scene - prioritizing people and food',
            'landscape': 'Outdoor landscape - prioritizing people and foreground',
            'street': 'Street/traffic scene - prioritizing people and vehicles',
            'document': 'Document/text scene - prioritizing text and important marks',
            'indoor': 'Indoor scene - prioritizing people and objects',
            'retail': 'Retail/shopping scene - prioritizing products and people',
            'general': 'General scene - using balanced priorities'
        }
        return descriptions.get(scene, 'Unknown scene type')

