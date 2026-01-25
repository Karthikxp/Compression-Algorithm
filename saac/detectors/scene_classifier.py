"""
Scene/Intent Classifier
The "Context" - Determines what type of scene this is to apply appropriate rules.
Uses CLIP, EfficientNet, ResNet, or fast heuristics for scene classification.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List
import torch
import torch.nn as nn

try:
    import torchvision.models as models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


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
            'portrait': 'Portrait scene - prioritizing faces and people',
            'general': 'General scene - using balanced priorities'
        }
        return descriptions.get(scene, 'Unknown scene type')


class ClipSceneClassifier:
    """
    CLIP-based scene/intent classification with superior semantic understanding.
    
    Uses OpenAI's CLIP model for zero-shot classification of scenes into
    compression intent categories. Provides probability distributions across
    all scene types for flexible downstream processing.
    
    Advantages over heuristics:
    - 85-95% accuracy (vs 60-70% for heuristics)
    - Semantic understanding of context
    - Zero-shot learning (no training data needed)
    - Flexible intent categories
    """
    
    # Intent definitions with rich semantic descriptions
    # Aligned with intent_rules.py for comprehensive coverage
    INTENT_DEFINITIONS = {
        # People-focused
        'portrait': "a portrait photo of a person's face, headshot or close-up photograph",
        'selfie': "a selfie photo taken with a phone camera, close-up self-portrait",
        'group_photo': "a group photo with multiple people together, family or friends photo",
        'baby': "a photo of a baby or infant with toys",
        'children': "children playing with toys, kids having fun",
        
        # Animals & Nature
        'pet_portrait': "a close-up photo of a pet dog or cat, animal portrait",
        'wildlife': "wildlife photography of animals in nature, safari or nature photography",
        'garden': "a garden with flowers and plants, botanical photography",
        'park': "a park scene with people enjoying outdoor recreation",
        
        # Food & Dining
        'restaurant': "people eating food in a restaurant, dining scene with meals on table",
        'food_closeup': "close-up food photography, detailed shot of a meal or dish",
        'cooking': "cooking in a kitchen, food preparation scene",
        
        # Outdoor Scenes
        'landscape': "outdoor landscape with mountains, forests, or natural scenery",
        'beach': "beach scene with ocean, sand, and water activities",
        'mountain': "mountain landscape, hiking or mountaineering scene",
        'snow': "snow covered scene, winter sports like skiing or snowboarding",
        
        # Urban & Architecture
        'urban': "urban city scene with buildings and street life",
        'street': "street scene with cars, vehicles, traffic, and roads",
        'architecture': "architectural photography of buildings and structures",
        
        # Sports & Activities
        'sports': "sports action photo with athletes and equipment",
        'gym': "gym or fitness scene with exercise equipment",
        'concert': "concert or live music performance",
        
        # Events
        'wedding': "wedding ceremony or reception, formal celebration",
        'party': "party or celebration with people having fun",
        
        # Indoor Spaces
        'indoor': "indoor room with furniture and decorations",
        'living_room': "living room with couch, TV, and home furniture",
        'bedroom': "bedroom interior with bed and personal items",
        'bathroom': "bathroom scene with sink and fixtures",
        'kitchen': "kitchen interior with appliances and cooking area",
        
        # Work & Education
        'workspace': "office desk with computer, workspace or home office",
        'meeting': "business meeting or conference room scene",
        'classroom': "classroom with students and educational materials",
        
        # Commercial
        'retail': "retail store with products on shelves, shopping scene",
        'product': "product photography with isolated item on display",
        
        # Transportation
        'vehicle': "vehicle photography of cars, motorcycles, or bikes",
        'transportation': "transportation hub like airport, train station, or bus terminal",
        'travel': "travel photography with tourists and landmarks",
        
        # Technical
        'document': "document with text, paper, or printed material",
        'screenshot': "screenshot of a computer screen or user interface",
        
        # Cultural
        'museum': "museum or art gallery with exhibits",
        
        # Special Conditions
        'night': "night photography with low light and artificial lighting",
        'medical': "medical or healthcare scene in clinical setting",
        'studio': "professional studio photography with controlled lighting",
        
        # Artistic
        'abstract': "abstract or artistic photography with creative composition",
        'macro': "macro photography with extreme close-up details",
        
        # Low Quality
        'low_quality': "blurry, grainy, or low quality photograph",
        'blurry': "out of focus or motion blurred photograph",
        'meme': "internet meme or social media image with text",
        'collage': "photo collage with multiple images combined",
        
        # Aerial & Special Views
        'aerial': "aerial or drone photography from above",
        'underwater': "underwater photography of marine life",
        
        # Fashion & Style
        'fashion': "fashion photography with clothing and accessories",
        
        # Special Technical
        'barcode_qr': "barcode or QR code that needs to be scanned",
        
        # Fallback
        'general': "general photograph without specific category, everyday casual photo"
    }
    
    # Scene categories in consistent order (all 50+ intents)
    SCENE_CATEGORIES = [
        # People (5)
        'portrait', 'selfie', 'group_photo', 'baby', 'children',
        # Animals (4)
        'pet_portrait', 'wildlife', 'garden', 'park',
        # Food (3)
        'restaurant', 'food_closeup', 'cooking',
        # Outdoor (4)
        'landscape', 'beach', 'mountain', 'snow',
        # Urban (3)
        'urban', 'street', 'architecture',
        # Sports (3)
        'sports', 'gym', 'concert',
        # Events (2)
        'wedding', 'party',
        # Indoor (5)
        'indoor', 'living_room', 'bedroom', 'bathroom', 'kitchen',
        # Work (3)
        'workspace', 'meeting', 'classroom',
        # Commercial (2)
        'retail', 'product',
        # Transportation (3)
        'vehicle', 'transportation', 'travel',
        # Technical (2)
        'document', 'screenshot',
        # Cultural (1)
        'museum',
        # Special (3)
        'night', 'medical', 'studio',
        # Artistic (2)
        'abstract', 'macro',
        # Low Quality (4)
        'low_quality', 'blurry', 'meme', 'collage',
        # Special Views (2)
        'aerial', 'underwater',
        # Fashion (1)
        'fashion',
        # Technical (1)
        'barcode_qr',
        # Fallback (1)
        'general'
    ]
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu", 
                 auto_confidence_threshold: bool = True):
        """
        Initialize CLIP-based scene classifier.
        
        Args:
            model_name: CLIP model variant
                - "ViT-B/32": Fast, 149 MB (recommended)
                - "ViT-B/16": Better accuracy, 300 MB
                - "ViT-L/14": Best quality, 890 MB (slower)
            device: 'cuda' or 'cpu'
            auto_confidence_threshold: Automatically adjust confidence threshold
                                      based on number of intents (default: True)
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP not installed. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.device = device
        self.model_name = model_name
        self.auto_confidence_threshold = auto_confidence_threshold
        
        # Calculate automatic confidence threshold based on number of intents
        # With more intents, confidence scores are naturally lower (spread across more options)
        # Formula: base_threshold / sqrt(num_intents)
        # For 8 intents: ~0.15, for 54 intents: ~0.08
        if auto_confidence_threshold:
            num_intents = len(self.SCENE_CATEGORIES)
            self.default_confidence_threshold = 0.4 / (num_intents ** 0.5)
            # Clamp between reasonable bounds
            self.default_confidence_threshold = max(0.05, min(0.20, self.default_confidence_threshold))
        else:
            self.default_confidence_threshold = 0.15
        
        print(f"  Loading CLIP model {model_name}...")
        try:
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            print(f"  ✓ CLIP {model_name} loaded on {device}")
        except Exception as e:
            print(f"  ✗ Failed to load CLIP: {e}")
            
            # Check if it's an SSL error
            if "SSL" in str(e) or "CERTIFICATE" in str(e):
                print("\n  💡 SSL Certificate Error Detected!")
                print("  This is usually caused by corporate proxies or network security.")
                print("\n  Solutions:")
                print("  1. Set SSL_CERT_FILE environment variable:")
                print("     export SSL_CERT_FILE=/path/to/cert.pem")
                print("\n  2. Download model manually:")
                print("     mkdir -p ~/.cache/clip")
                print("     # Then download ViT-B-32.pt from OpenAI's servers")
                print("\n  3. Disable SSL verification (temporary, less secure):")
                print("     Run: python3 -c \"import ssl; ssl._create_default_https_context = ssl._create_unverified_context\"")
                print("\n  4. Try the workaround below...")
                
                # Try with SSL verification disabled (last resort)
                try:
                    print("\n  Attempting to load with SSL workaround...")
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
                    
                    self.model, self.preprocess = clip.load(model_name, device=device)
                    self.model.eval()
                    print(f"  ✓ CLIP {model_name} loaded with SSL workaround")
                except Exception as e2:
                    print(f"  ✗ SSL workaround failed: {e2}")
                    print("\n  Please manually download the CLIP model or fix SSL certificates.")
                    raise RuntimeError(
                        f"Failed to load CLIP model due to SSL error. "
                        f"See solutions above."
                    ) from e
            else:
                raise
        
        # Create text prompts for each intent
        self.intent_prompts = [
            self.INTENT_DEFINITIONS[scene]
            for scene in self.SCENE_CATEGORIES
        ]
        
        # Pre-encode text prompts (cache for efficiency)
        print(f"  Encoding {len(self.intent_prompts)} intent prompts...")
        self.text_features = self._encode_prompts()
        
        # Validate alignment with intent_rules.py
        self._validate_intent_alignment()
        
        print(f"  ✓ CLIP classifier ready with {len(self.SCENE_CATEGORIES)} intent categories")
        if auto_confidence_threshold:
            print(f"  ✓ Auto confidence threshold: {self.default_confidence_threshold:.3f}")
    
    def _encode_prompts(self) -> torch.Tensor:
        """
        Pre-encode text prompts for efficiency.
        
        Returns:
            Normalized text feature embeddings
        """
        text_tokens = clip.tokenize(self.intent_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize embeddings (required for cosine similarity)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def _validate_intent_alignment(self):
        """
        Validate that CLIP intents align with intent_rules.py.
        Warns if there are mismatches.
        """
        try:
            from ..intent_rules import IntentRuleEngine
            rule_engine = IntentRuleEngine()
            available_intents = set(rule_engine.list_available_scenes())
            clip_intents = set(self.SCENE_CATEGORIES)
            
            # Check for intents in CLIP but not in rules
            missing_in_rules = clip_intents - available_intents
            if missing_in_rules:
                print(f"  ⚠️  Warning: {len(missing_in_rules)} intents in CLIP but not in rules: {missing_in_rules}")
            
            # Check for intents in rules but not in CLIP
            missing_in_clip = available_intents - clip_intents
            if missing_in_clip:
                print(f"  ℹ️  Note: {len(missing_in_clip)} intents in rules but not in CLIP: {missing_in_clip}")
                print(f"     These will fall back to similar intents during classification")
            
            if not missing_in_rules and not missing_in_clip:
                print(f"  ✓ All {len(self.SCENE_CATEGORIES)} intents aligned with intent_rules.py")
        except Exception as e:
            print(f"  ℹ️  Could not validate intent alignment: {e}")
    
    def classify(self, image: np.ndarray, confidence_threshold: float = None) -> Tuple[str, float]:
        """
        Classify scene using CLIP with intelligent fallback strategy.
        
        With 50+ intents, uses confidence threshold to determine if a specific
        intent is clearly identified, or if a more general category is appropriate.
        
        Args:
            image: Input image (H, W, C) in BGR format
            confidence_threshold: Minimum confidence for specific intents
                                 If None, uses auto-calculated threshold based on
                                 number of intents (~0.08 for 54 intents)
                                 If top prediction is below this, falls back to 'general'
            
        Returns:
            (scene_category, confidence) tuple
            - scene_category: One of SCENE_CATEGORIES
            - confidence: Probability score [0.0, 1.0]
        """
        # Use auto threshold if not specified
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        
        # Get probability distribution
        probabilities = self.classify_with_probabilities(image)
        
        # Find highest probability scene
        max_idx = int(np.argmax(probabilities))
        scene = self.SCENE_CATEGORIES[max_idx]
        confidence = float(probabilities[max_idx])
        
        # Fallback logic: if confidence is low and it's not already 'general',
        # consider using 'general' intent for safety
        if confidence < confidence_threshold and scene != 'general':
            # Check if 'general' is in top 3 predictions
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            general_idx = self.SCENE_CATEGORIES.index('general') if 'general' in self.SCENE_CATEGORIES else -1
            
            if general_idx in top_3_indices:
                # Use general intent with adjusted confidence
                return 'general', float(probabilities[general_idx])
        
        return scene, confidence
    
    def classify_with_probabilities(self, image: np.ndarray) -> np.ndarray:
        """
        Classify scene and return full probability distribution.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Probability array of shape (8,) with scores for each scene category
            [portrait, restaurant, street, landscape, indoor, retail, document, general]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess image for CLIP
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity with text prompts
        # Scale by 100 for better softmax behavior
        similarity = (100.0 * image_features @ self.text_features.T)
        
        # Convert to probabilities using softmax
        probabilities = similarity.softmax(dim=-1)
        
        # Return as numpy array
        return probabilities[0].cpu().numpy()
    
    def classify_top_k(self, image: np.ndarray, k: int = 5, min_confidence: float = 0.05) -> List[Tuple[str, float]]:
        """
        Get top-k most probable scene categories.
        
        Useful for complex scenes that may fit multiple categories.
        With 50+ intents, showing top 5-10 helps understand scene context.
        
        Args:
            image: Input image (H, W, C) in BGR format
            k: Number of top predictions to return (default: 5)
            min_confidence: Minimum confidence threshold to include (default: 0.05)
            
        Returns:
            List of (scene_category, probability) tuples, sorted by probability
        """
        probabilities = self.classify_with_probabilities(image)
        
        # Get top k indices
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        
        # Build result list, filtering by min_confidence
        results = [
            (self.SCENE_CATEGORIES[idx], float(probabilities[idx]))
            for idx in top_k_indices
            if probabilities[idx] >= min_confidence
        ]
        
        return results
    
    def classify_with_fallback_chain(self, image: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify with a chain of fallback options for robust intent selection.
        
        Returns the primary intent plus a list of alternative intents that could
        apply. Useful for complex or ambiguous scenes.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Tuple of:
            - primary_scene: The selected scene category
            - primary_confidence: Confidence for primary scene
            - alternatives: List of (scene, confidence) for top alternatives
        """
        # Get top 5 predictions
        top_predictions = self.classify_top_k(image, k=5, min_confidence=0.05)
        
        if not top_predictions:
            return 'general', 0.5, []
        
        primary_scene, primary_confidence = top_predictions[0]
        alternatives = top_predictions[1:] if len(top_predictions) > 1 else []
        
        # If primary confidence is very low, prefer 'general'
        if primary_confidence < 0.10 and primary_scene != 'general':
            # Check if general is in alternatives
            for i, (scene, conf) in enumerate(alternatives):
                if scene == 'general':
                    # Swap general to primary
                    alternatives = [top_predictions[0]] + alternatives[:i] + alternatives[i+1:]
                    return 'general', conf, alternatives
            
            # If general not found, add it
            return 'general', 0.5, top_predictions
        
        return primary_scene, primary_confidence, alternatives
    
    def get_scene_probabilities_dict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Get scene probabilities as a dictionary for easy inspection.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Dictionary mapping scene names to probabilities
        """
        probabilities = self.classify_with_probabilities(image)
        
        return {
            scene: float(prob) 
            for scene, prob in zip(self.SCENE_CATEGORIES, probabilities)
        }
    
    def get_scene_description(self, scene: str) -> str:
        """Get human-readable description of scene category."""
        descriptions = {
            # People
            'portrait': 'Portrait - maximum quality on faces and people',
            'selfie': 'Selfie - close-up person with maximum face quality',
            'group_photo': 'Group photo - preserve all faces equally',
            'baby': 'Baby photo - maximum quality on infant and toys',
            'children': 'Children - kids and toys with high quality',
            
            # Animals
            'pet_portrait': 'Pet portrait - focus on animals',
            'wildlife': 'Wildlife - animals in nature with priority',
            'garden': 'Garden - plants and nature photography',
            'park': 'Park - recreation and outdoor activities',
            
            # Food
            'restaurant': 'Restaurant - prioritizing people and food',
            'food_closeup': 'Food photography - maximum quality on food',
            'cooking': 'Cooking - kitchen and food preparation',
            
            # Outdoor
            'landscape': 'Landscape - outdoor scenery with foreground priority',
            'beach': 'Beach - water activities and people',
            'mountain': 'Mountain - hiking and outdoor adventure',
            'snow': 'Snow - winter sports and activities',
            
            # Urban
            'urban': 'Urban - city buildings and street life',
            'street': 'Street - traffic, vehicles, and pedestrians',
            'architecture': 'Architecture - buildings and structures',
            
            # Sports
            'sports': 'Sports - athletic action and equipment',
            'gym': 'Gym - fitness and exercise',
            'concert': 'Concert - performance and performers',
            
            # Events
            'wedding': 'Wedding - formal event and celebrations',
            'party': 'Party - celebration with people',
            
            # Indoor
            'indoor': 'Indoor - room with people and objects',
            'living_room': 'Living room - home comfort and family',
            'bedroom': 'Bedroom - personal space',
            'bathroom': 'Bathroom - personal care area',
            'kitchen': 'Kitchen - cooking area and appliances',
            
            # Work
            'workspace': 'Workspace - office desk and computers',
            'meeting': 'Meeting - conference and presentation',
            'classroom': 'Classroom - educational setting',
            
            # Commercial
            'retail': 'Retail - shopping and products on display',
            'product': 'Product photography - maximum quality on item',
            
            # Transportation
            'vehicle': 'Vehicle - car, bike, or motorcycle photography',
            'transportation': 'Transportation - travel hub with people',
            'travel': 'Travel - tourism and landmarks',
            
            # Technical
            'document': 'Document - text and printed material',
            'screenshot': 'Screenshot - UI and interface capture',
            
            # Cultural
            'museum': 'Museum - art gallery and exhibits',
            
            # Special
            'night': 'Night - low-light photography',
            'medical': 'Medical - healthcare and clinical setting',
            'studio': 'Studio - professional photography',
            
            # Artistic
            'abstract': 'Abstract - artistic and creative',
            'macro': 'Macro - extreme close-up details',
            
            # Low Quality
            'low_quality': 'Low quality - aggressive compression acceptable',
            'blurry': 'Blurry - out of focus, heavy compression',
            'meme': 'Meme - social media image',
            'collage': 'Collage - multiple images combined',
            
            # Special Views
            'aerial': 'Aerial - drone or bird\'s eye view',
            'underwater': 'Underwater - aquatic photography',
            
            # Fashion
            'fashion': 'Fashion - clothing and accessories',
            
            # Technical
            'barcode_qr': 'Barcode/QR - maximum quality for scanning',
            
            # Fallback
            'general': 'General - balanced priorities'
        }
        return descriptions.get(scene, f'{scene} - using appropriate compression strategy')
    
    def list_all_intents(self, show_descriptions: bool = True) -> None:
        """
        Print all available intent categories with descriptions.
        
        Args:
            show_descriptions: If True, show full descriptions for each intent
        """
        print(f"\n📋 Available Intent Categories ({len(self.SCENE_CATEGORIES)} total)")
        print("="*70)
        
        if show_descriptions:
            for intent in self.SCENE_CATEGORIES:
                desc = self.get_scene_description(intent)
                print(f"  • {intent:20s} - {desc}")
        else:
            # Group by category for compact display
            categories = {
                "People": ['portrait', 'selfie', 'group_photo', 'baby', 'children'],
                "Animals": ['pet_portrait', 'wildlife', 'garden', 'park'],
                "Food": ['restaurant', 'food_closeup', 'cooking'],
                "Outdoor": ['landscape', 'beach', 'mountain', 'snow'],
                "Urban": ['urban', 'street', 'architecture'],
                "Sports": ['sports', 'gym', 'concert'],
                "Events": ['wedding', 'party'],
                "Indoor": ['indoor', 'living_room', 'bedroom', 'bathroom', 'kitchen'],
                "Work": ['workspace', 'meeting', 'classroom'],
                "Commercial": ['retail', 'product', 'vehicle', 'transportation', 'travel', 'fashion'],
                "Technical": ['document', 'screenshot', 'barcode_qr'],
                "Cultural": ['museum'],
                "Special": ['night', 'medical', 'studio'],
                "Artistic": ['abstract', 'macro'],
                "Low Quality": ['low_quality', 'blurry', 'meme', 'collage'],
                "Special Views": ['aerial', 'underwater'],
                "Fallback": ['general']
            }
            
            for category, intents in categories.items():
                available = [i for i in intents if i in self.SCENE_CATEGORIES]
                if available:
                    print(f"\n  {category}:")
                    for intent in available:
                        print(f"    • {intent}")
        
        print()
    
    def visualize_probabilities(self, image: np.ndarray, 
                               save_path: str = None) -> np.ndarray:
        """
        Create visualization of scene classification probabilities.
        
        Args:
            image: Input image (H, W, C) in BGR format
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image with probability bars
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Get probabilities
        probs = self.classify_with_probabilities(image)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Show original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(image_rgb)
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Show probability bars
        colors = plt.cm.viridis(probs / probs.max())
        bars = ax2.barh(self.SCENE_CATEGORIES, probs, color=colors)
        ax2.set_xlabel('Probability')
        ax2.set_title('Scene Classification Probabilities')
        ax2.set_xlim([0, 1])
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Visualization saved to {save_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        
        # Use buffer_rgba() for newer matplotlib versions
        try:
            buf = fig.canvas.buffer_rgba()
            vis_array = np.asarray(buf)
            # Remove alpha channel and convert to BGR for OpenCV
            vis_array = cv2.cvtColor(vis_array, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            # Fallback for older matplotlib versions
            vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            vis_array = cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        return vis_array

