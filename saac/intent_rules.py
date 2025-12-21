"""
Intent-Based Compression Rules
Pre-defined rule profiles for different scene types.
Maps COCO object classes to compression priorities based on scene context.
"""

from typing import Dict, List


class IntentRuleEngine:
    """
    Manages intent-based compression rules.
    Each scene type has a rule profile that defines importance weights for object classes.
    """
    
    # COCO class names (80 classes)
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light',
        10: 'fire_hydrant', 11: 'stop_sign', 12: 'parking_meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports_ball', 33: 'kite', 34: 'baseball_bat',
        35: 'baseball_glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis_racket',
        39: 'bottle', 40: 'wine_glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot_dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted_plant',
        59: 'bed', 60: 'dining_table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell_phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy_bear', 78: 'hair_drier',
        79: 'toothbrush'
    }
    
    # Intent-based rule profiles
    # Weight: 1.0 = maximum quality (QP 10), 0.0 = maximum compression (QP 51)
    INTENT_PROFILES = {
        'restaurant': {
            'description': 'Restaurant/Dining scene - prioritize people and food',
            'weights': {
                # People are always important
                'person': 1.0,
                
                # Food items - high priority
                'bottle': 0.8,
                'wine_glass': 0.8,
                'cup': 0.8,
                'fork': 0.7,
                'knife': 0.7,
                'spoon': 0.7,
                'bowl': 0.8,
                'banana': 0.9,
                'apple': 0.9,
                'sandwich': 0.9,
                'orange': 0.9,
                'broccoli': 0.9,
                'carrot': 0.9,
                'hot_dog': 0.9,
                'pizza': 0.9,
                'donut': 0.9,
                'cake': 0.9,
                
                # Furniture - lower priority
                'chair': 0.3,
                'dining_table': 0.4,
                
                # Default for unlisted items
                'default': 0.2
            }
        },
        
        'landscape': {
            'description': 'Outdoor landscape - prioritize people and foreground subjects',
            'weights': {
                # People and animals - always protect
                'person': 1.0,
                'dog': 0.9,
                'cat': 0.9,
                'bird': 0.8,
                'horse': 0.9,
                'sheep': 0.8,
                'cow': 0.8,
                'elephant': 0.9,
                'bear': 0.9,
                'zebra': 0.9,
                'giraffe': 0.9,
                
                # Vehicles - moderate priority
                'car': 0.5,
                'bicycle': 0.6,
                'motorcycle': 0.6,
                'bus': 0.5,
                'truck': 0.5,
                
                # Objects in nature
                'umbrella': 0.7,
                'backpack': 0.7,
                'kite': 0.8,
                
                # Default for background elements
                'default': 0.1
            }
        },
        
        'street': {
            'description': 'Street/traffic scene - prioritize people, vehicles, and signs',
            'weights': {
                # People - highest priority
                'person': 1.0,
                
                # Vehicles - high priority
                'car': 0.9,
                'bicycle': 0.9,
                'motorcycle': 0.9,
                'bus': 0.9,
                'truck': 0.9,
                'train': 0.9,
                
                # Traffic elements - critical
                'traffic_light': 0.95,
                'stop_sign': 0.95,
                'parking_meter': 0.7,
                
                # Pedestrian items
                'backpack': 0.7,
                'handbag': 0.7,
                'umbrella': 0.6,
                
                # Default
                'default': 0.2
            }
        },
        
        'document': {
            'description': 'Document/text scene - prioritize text and important marks',
            'weights': {
                # In document mode, we rely mainly on saliency
                # Objects rarely appear, but if they do:
                'person': 0.9,
                'book': 1.0,
                'laptop': 0.9,
                'cell_phone': 0.8,
                'keyboard': 0.7,
                'clock': 0.8,
                
                # Default - let saliency handle text
                'default': 0.5
            }
        },
        
        'indoor': {
            'description': 'Indoor scene - prioritize people and interactive objects',
            'weights': {
                # People - always important
                'person': 1.0,
                
                # Pets
                'cat': 0.9,
                'dog': 0.9,
                'bird': 0.8,
                
                # Electronics and media
                'tv': 0.7,
                'laptop': 0.8,
                'cell_phone': 0.8,
                'remote': 0.6,
                'keyboard': 0.7,
                
                # Furniture - moderate
                'couch': 0.4,
                'chair': 0.3,
                'dining_table': 0.4,
                'bed': 0.4,
                
                # Decorative
                'potted_plant': 0.5,
                'vase': 0.6,
                'clock': 0.6,
                'book': 0.6,
                
                # Default
                'default': 0.3
            }
        },
        
        'retail': {
            'description': 'Retail/shopping scene - prioritize products and people',
            'weights': {
                # People - customers are important
                'person': 1.0,
                
                # Products - high priority
                'bottle': 0.9,
                'cup': 0.8,
                'handbag': 0.9,
                'tie': 0.9,
                'suitcase': 0.9,
                'backpack': 0.9,
                'umbrella': 0.8,
                'vase': 0.9,
                'teddy_bear': 0.9,
                'book': 0.9,
                'clock': 0.9,
                'cell_phone': 0.9,
                'laptop': 0.9,
                
                # Furniture - lower priority
                'chair': 0.3,
                'couch': 0.3,
                'bench': 0.3,
                
                # Default
                'default': 0.4
            }
        },
        
        'general': {
            'description': 'General scene - balanced priorities',
            'weights': {
                # People - always important
                'person': 1.0,
                
                # Animals
                'dog': 0.9,
                'cat': 0.9,
                'bird': 0.8,
                
                # Vehicles
                'car': 0.7,
                'bicycle': 0.7,
                'motorcycle': 0.7,
                
                # Common objects
                'book': 0.6,
                'laptop': 0.7,
                'cell_phone': 0.7,
                'clock': 0.6,
                
                # Default - moderate compression
                'default': 0.5
            }
        }
    }
    
    def __init__(self):
        """Initialize the intent rule engine."""
        self.current_profile = None
        self.current_scene = 'general'
    
    def set_scene(self, scene: str):
        """
        Set the current scene and load its rule profile.
        
        Args:
            scene: Scene category (e.g., 'restaurant', 'landscape', 'street')
        """
        if scene in self.INTENT_PROFILES:
            self.current_scene = scene
            self.current_profile = self.INTENT_PROFILES[scene]
        else:
            # Fallback to general profile
            self.current_scene = 'general'
            self.current_profile = self.INTENT_PROFILES['general']
    
    def get_weight_for_class(self, class_id: int = None, class_name: str = None) -> float:
        """
        Get compression weight for an object class.
        
        Args:
            class_id: COCO class ID (0-79)
            class_name: COCO class name (e.g., 'person', 'car')
            
        Returns:
            Weight value (0-1), where 1.0 = maximum quality
        """
        if self.current_profile is None:
            self.set_scene('general')
        
        # Get class name from ID if needed
        if class_name is None and class_id is not None:
            class_name = self.COCO_CLASSES.get(class_id, None)
        
        if class_name is None:
            return self.current_profile['weights'].get('default', 0.5)
        
        # Look up weight in current profile
        weight = self.current_profile['weights'].get(class_name,
                                                     self.current_profile['weights'].get('default', 0.5))
        
        return weight
    
    def get_weights_for_detections(self, detections: List[Dict]) -> List[float]:
        """
        Get weights for a list of detections.
        
        Args:
            detections: List of detection dicts with 'class' or 'class_name'
            
        Returns:
            List of weights corresponding to each detection
        """
        weights = []
        
        for det in detections:
            class_id = det.get('class', None)
            class_name = det.get('class_name', None)
            
            weight = self.get_weight_for_class(class_id, class_name)
            weights.append(weight)
        
        return weights
    
    def get_scene_description(self) -> str:
        """Get description of current scene profile."""
        if self.current_profile:
            return self.current_profile['description']
        return "No scene profile loaded"
    
    def list_available_scenes(self) -> List[str]:
        """Get list of available scene types."""
        return list(self.INTENT_PROFILES.keys())

