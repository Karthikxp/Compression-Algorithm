
from typing import Dict, List


class IntentRuleEngine:
    
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
    
    INTENT_PROFILES = {
        'portrait': {
            'description': 'Portrait/People-focused scene - maximum quality on faces and people',
            'weights': {
                'person': 1.0,
                
                'handbag': 0.7,
                'backpack': 0.7,
                'tie': 0.8,
                'umbrella': 0.6,
                'cell_phone': 0.7,
                'book': 0.6,
                
                'default': 0.1
            }
        },
        
        'selfie': {
            'description': 'Selfie/close-up person - maximum quality on faces and personal items',
            'weights': {
                'person': 1.0,
                'cell_phone': 0.8,
                'handbag': 0.7,
                'sunglasses': 0.7,
                'tie': 0.7,
                'book': 0.6,
                'default': 0.05
            }
        },
        
        'group_photo': {
            'description': 'Group photo/multiple people - preserve all faces equally',
            'weights': {
                'person': 1.0,
                'handbag': 0.6,
                'backpack': 0.6,
                'tie': 0.7,
                'umbrella': 0.5,
                'default': 0.15
            }
        },
        
        'baby': {
            'description': 'Baby/infant photos - maximum quality on baby and toys',
            'weights': {
                'person': 1.0,
                'teddy_bear': 0.95,
                'bottle': 0.9,
                'book': 0.8,
                'bed': 0.5,
                'chair': 0.4,
                'default': 0.2
            }
        },
        
        'children': {
            'description': 'Children playing - prioritize kids and toys',
            'weights': {
                'person': 1.0,
                'teddy_bear': 0.9,
                'kite': 0.9,
                'frisbee': 0.9,
                'sports_ball': 0.9,
                'bicycle': 0.8,
                'skateboard': 0.8,
                'default': 0.2
            }
        },
        
        'pet_portrait': {
            'description': 'Pet portrait - focus on animals',
            'weights': {
                'dog': 1.0,
                'cat': 1.0,
                'bird': 1.0,
                'horse': 1.0,
                'person': 0.8,
                'bowl': 0.7,
                'teddy_bear': 0.6,
                'default': 0.1
            }
        },
        
        'wildlife': {
            'description': 'Wildlife/nature photography - animals are priority',
            'weights': {
                'bird': 1.0,
                'elephant': 1.0,
                'bear': 1.0,
                'zebra': 1.0,
                'giraffe': 1.0,
                'horse': 1.0,
                'sheep': 0.95,
                'cow': 0.95,
                'dog': 0.9,
                'cat': 0.9,
                'person': 0.7,
                'default': 0.05
            }
        },
        
        'restaurant': {
            'description': 'Restaurant/Dining scene - prioritize people and food',
            'weights': {
                'person': 1.0,
                
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
                
                'chair': 0.3,
                'dining_table': 0.4,
                
                'default': 0.2
            }
        },
        
        'food_closeup': {
            'description': 'Food photography/close-up - maximum quality on food',
            'weights': {
                'pizza': 1.0,
                'cake': 1.0,
                'sandwich': 1.0,
                'hot_dog': 1.0,
                'donut': 1.0,
                'banana': 1.0,
                'apple': 1.0,
                'orange': 1.0,
                'broccoli': 1.0,
                'carrot': 1.0,
                'bowl': 0.9,
                'cup': 0.9,
                'bottle': 0.9,
                'wine_glass': 0.9,
                'fork': 0.8,
                'knife': 0.8,
                'spoon': 0.8,
                'dining_table': 0.3,
                'person': 0.5,
                'default': 0.1
            }
        },
        
        'cooking': {
            'description': 'Cooking/kitchen scene - focus on food prep and ingredients',
            'weights': {
                'person': 0.9,
                'knife': 0.9,
                'bowl': 0.9,
                'spoon': 0.8,
                'fork': 0.8,
                'bottle': 0.8,
                'cup': 0.7,
                'oven': 0.8,
                'microwave': 0.7,
                'sink': 0.6,
                'refrigerator': 0.7,
                'banana': 0.9,
                'apple': 0.9,
                'orange': 0.9,
                'broccoli': 0.9,
                'carrot': 0.9,
                'default': 0.3
            }
        },
        
        'landscape': {
            'description': 'Outdoor landscape - prioritize people and foreground subjects',
            'weights': {
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
                
                'car': 0.5,
                'bicycle': 0.6,
                'motorcycle': 0.6,
                'bus': 0.5,
                'truck': 0.5,
                
                'umbrella': 0.7,
                'backpack': 0.7,
                'kite': 0.8,
                
                'default': 0.1
            }
        },
        
        'beach': {
            'description': 'Beach/water scene - people and beach items',
            'weights': {
                'person': 1.0,
                'surfboard': 0.9,
                'boat': 0.8,
                'umbrella': 0.8,
                'kite': 0.8,
                'frisbee': 0.7,
                'sports_ball': 0.7,
                'backpack': 0.6,
                'default': 0.1
            }
        },
        
        'mountain': {
            'description': 'Mountain/hiking scene - people and outdoor gear',
            'weights': {
                'person': 1.0,
                'backpack': 0.9,
                'bicycle': 0.8,
                'skis': 0.9,
                'snowboard': 0.9,
                'dog': 0.9,
                'umbrella': 0.7,
                'default': 0.05
            }
        },
        
        'snow': {
            'description': 'Snow/winter scene - winter sports and people',
            'weights': {
                'person': 1.0,
                'skis': 1.0,
                'snowboard': 1.0,
                'sled': 0.9,
                'dog': 0.9,
                'backpack': 0.7,
                'default': 0.1
            }
        },
        
        'urban': {
            'description': 'Urban/city scene - buildings and street life',
            'weights': {
                'person': 1.0,
                'bicycle': 0.8,
                'car': 0.7,
                'motorcycle': 0.7,
                'bus': 0.7,
                'traffic_light': 0.8,
                'stop_sign': 0.8,
                'bench': 0.5,
                'backpack': 0.7,
                'handbag': 0.7,
                'default': 0.2
            }
        },
        
        'street': {
            'description': 'Street/traffic scene - prioritize people, vehicles, and signs',
            'weights': {
                'person': 1.0,
                
                'car': 0.9,
                'bicycle': 0.9,
                'motorcycle': 0.9,
                'bus': 0.9,
                'truck': 0.9,
                'train': 0.9,
                
                'traffic_light': 0.95,
                'stop_sign': 0.95,
                'parking_meter': 0.7,
                
                'backpack': 0.7,
                'handbag': 0.7,
                'umbrella': 0.6,
                
                'default': 0.2
            }
        },
        
        'architecture': {
            'description': 'Architecture/buildings - preserve structural details',
            'weights': {
                'person': 0.7,
                'clock': 0.9,
                'bench': 0.6,
                'potted_plant': 0.6,
                'car': 0.4,
                'bicycle': 0.5,
                'traffic_light': 0.7,
                'default': 0.3
            }
        },
        
        'sports': {
            'description': 'Sports/athletic scene - action and equipment',
            'weights': {
                'person': 1.0,
                'sports_ball': 1.0,
                'baseball_bat': 0.95,
                'baseball_glove': 0.95,
                'tennis_racket': 0.95,
                'frisbee': 0.9,
                'skateboard': 0.9,
                'surfboard': 0.9,
                'skis': 0.9,
                'snowboard': 0.9,
                'bicycle': 0.8,
                'motorcycle': 0.8,
                'default': 0.2
            }
        },
        
        'gym': {
            'description': 'Gym/fitness scene - people and equipment',
            'weights': {
                'person': 1.0,
                'sports_ball': 0.8,
                'bottle': 0.7,
                'backpack': 0.6,
                'cell_phone': 0.6,
                'default': 0.3
            }
        },
        
        'wedding': {
            'description': 'Wedding/formal event - people and decorations',
            'weights': {
                'person': 1.0,
                'tie': 0.9,
                'handbag': 0.8,
                'vase': 0.8,
                'cake': 0.95,
                'wine_glass': 0.8,
                'bottle': 0.7,
                'chair': 0.4,
                'dining_table': 0.5,
                'default': 0.3
            }
        },
        
        'party': {
            'description': 'Party/celebration - people and party items',
            'weights': {
                'person': 1.0,
                'cake': 0.95,
                'wine_glass': 0.8,
                'bottle': 0.8,
                'cup': 0.8,
                'pizza': 0.9,
                'donut': 0.9,
                'chair': 0.3,
                'dining_table': 0.4,
                'default': 0.3
            }
        },
        
        'concert': {
            'description': 'Concert/performance - performers and instruments',
            'weights': {
                'person': 1.0,
                'cell_phone': 0.6,
                'chair': 0.2,
                'default': 0.2
            }
        },
        
        'document': {
            'description': 'Document/text scene - prioritize text and important marks',
            'weights': {
                'person': 0.9,
                'book': 1.0,
                'laptop': 0.9,
                'cell_phone': 0.8,
                'keyboard': 0.7,
                'clock': 0.8,
                
                'default': 0.5
            }
        },
        
        'screenshot': {
            'description': 'Screenshot/UI capture - preserve all interface elements',
            'weights': {
                'laptop': 0.9,
                'tv': 0.9,
                'cell_phone': 0.9,
                'keyboard': 0.7,
                'mouse': 0.7,
                'person': 0.5,
                'default': 0.6
            }
        },
        
        'workspace': {
            'description': 'Office/workspace - computers and work items',
            'weights': {
                'person': 0.9,
                'laptop': 1.0,
                'keyboard': 0.9,
                'mouse': 0.9,
                'cell_phone': 0.8,
                'book': 0.8,
                'cup': 0.7,
                'clock': 0.7,
                'chair': 0.4,
                'dining_table': 0.4,
                'potted_plant': 0.5,
                'default': 0.3
            }
        },
        
        'meeting': {
            'description': 'Meeting/conference - people and presentation equipment',
            'weights': {
                'person': 1.0,
                'laptop': 0.9,
                'tv': 0.9,
                'chair': 0.3,
                'dining_table': 0.4,
                'cup': 0.6,
                'book': 0.7,
                'cell_phone': 0.7,
                'default': 0.3
            }
        },
        
        'classroom': {
            'description': 'Classroom/educational - students and learning materials',
            'weights': {
                'person': 1.0,
                'book': 0.9,
                'laptop': 0.8,
                'chair': 0.3,
                'dining_table': 0.3,
                'backpack': 0.7,
                'clock': 0.6,
                'default': 0.3
            }
        },
        
        'indoor': {
            'description': 'Indoor scene - prioritize people and interactive objects',
            'weights': {
                'person': 1.0,
                
                'cat': 0.9,
                'dog': 0.9,
                'bird': 0.8,
                
                'tv': 0.7,
                'laptop': 0.8,
                'cell_phone': 0.8,
                'remote': 0.6,
                'keyboard': 0.7,
                
                'couch': 0.4,
                'chair': 0.3,
                'dining_table': 0.4,
                'bed': 0.4,
                
                'potted_plant': 0.5,
                'vase': 0.6,
                'clock': 0.6,
                'book': 0.6,
                
                'default': 0.3
            }
        },
        
        'living_room': {
            'description': 'Living room - family and home comfort',
            'weights': {
                'person': 1.0,
                'dog': 0.9,
                'cat': 0.9,
                'tv': 0.7,
                'remote': 0.6,
                'couch': 0.5,
                'chair': 0.4,
                'potted_plant': 0.6,
                'vase': 0.6,
                'book': 0.7,
                'default': 0.3
            }
        },
        
        'bedroom': {
            'description': 'Bedroom scene - personal space and rest',
            'weights': {
                'person': 1.0,
                'bed': 0.6,
                'cell_phone': 0.8,
                'book': 0.8,
                'laptop': 0.8,
                'clock': 0.7,
                'teddy_bear': 0.7,
                'cat': 0.9,
                'dog': 0.9,
                'default': 0.3
            }
        },
        
        'bathroom': {
            'description': 'Bathroom scene - hygiene and personal care',
            'weights': {
                'person': 1.0,
                'sink': 0.7,
                'toilet': 0.6,
                'toothbrush': 0.7,
                'hair_drier': 0.7,
                'bottle': 0.6,
                'default': 0.3
            }
        },
        
        'kitchen': {
            'description': 'Kitchen scene - cooking area and appliances',
            'weights': {
                'person': 1.0,
                'refrigerator': 0.7,
                'oven': 0.7,
                'microwave': 0.7,
                'sink': 0.6,
                'dining_table': 0.4,
                'chair': 0.3,
                'bottle': 0.6,
                'cup': 0.6,
                'bowl': 0.6,
                'knife': 0.7,
                'default': 0.3
            }
        },
        
        'retail': {
            'description': 'Retail/shopping scene - prioritize products and people',
            'weights': {
                'person': 1.0,
                
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
                
                'chair': 0.3,
                'couch': 0.3,
                'bench': 0.3,
                
                'default': 0.4
            }
        },
        
        'product': {
            'description': 'Product photography - maximum quality on product',
            'weights': {
                'bottle': 1.0,
                'cup': 1.0,
                'handbag': 1.0,
                'tie': 1.0,
                'suitcase': 1.0,
                'backpack': 1.0,
                'umbrella': 1.0,
                'vase': 1.0,
                'cell_phone': 1.0,
                'laptop': 1.0,
                'book': 1.0,
                'clock': 1.0,
                'teddy_bear': 1.0,
                'scissors': 1.0,
                'person': 0.4,
                'default': 0.2
            }
        },
        
        'vehicle': {
            'description': 'Vehicle photography - cars, bikes, motorcycles',
            'weights': {
                'car': 1.0,
                'motorcycle': 1.0,
                'bicycle': 1.0,
                'bus': 1.0,
                'truck': 1.0,
                'train': 1.0,
                'airplane': 1.0,
                'boat': 1.0,
                'person': 0.6,
                'default': 0.1
            }
        },
        
        'transportation': {
            'description': 'Transportation hub - vehicles and people traveling',
            'weights': {
                'person': 1.0,
                'suitcase': 0.9,
                'backpack': 0.9,
                'handbag': 0.8,
                'car': 0.8,
                'bus': 0.8,
                'train': 0.9,
                'airplane': 0.9,
                'bicycle': 0.7,
                'bench': 0.4,
                'clock': 0.7,
                'default': 0.3
            }
        },
        
        'travel': {
            'description': 'Travel/tourism - landmarks and travelers',
            'weights': {
                'person': 1.0,
                'suitcase': 0.8,
                'backpack': 0.8,
                'handbag': 0.7,
                'camera': 0.8,
                'cell_phone': 0.7,
                'umbrella': 0.6,
                'bicycle': 0.6,
                'default': 0.2
            }
        },
        
        'museum': {
            'description': 'Museum/art gallery - exhibits and artwork',
            'weights': {
                'person': 0.7,
                'vase': 0.9,
                'clock': 0.8,
                'book': 0.8,
                'chair': 0.5,
                'bench': 0.5,
                'potted_plant': 0.6,
                'default': 0.4
            }
        },
        
        'garden': {
            'description': 'Garden/outdoor plants - nature and people',
            'weights': {
                'person': 1.0,
                'potted_plant': 0.9,
                'vase': 0.8,
                'bird': 0.9,
                'cat': 0.8,
                'dog': 0.8,
                'bench': 0.5,
                'umbrella': 0.6,
                'default': 0.1
            }
        },
        
        'park': {
            'description': 'Park/playground - recreation and nature',
            'weights': {
                'person': 1.0,
                'dog': 0.9,
                'frisbee': 0.9,
                'sports_ball': 0.9,
                'kite': 0.9,
                'bicycle': 0.8,
                'skateboard': 0.8,
                'bench': 0.5,
                'bird': 0.7,
                'default': 0.2
            }
        },
        
        'night': {
            'description': 'Night/low-light scene - preserve lit subjects',
            'weights': {
                'person': 1.0,
                'car': 0.8,
                'bicycle': 0.7,
                'traffic_light': 0.9,
                'clock': 0.8,
                'cell_phone': 0.8,
                'default': 0.2
            }
        },
        
        'medical': {
            'description': 'Medical/healthcare scene - clinical precision',
            'weights': {
                'person': 1.0,
                'book': 0.8,
                'laptop': 0.9,
                'chair': 0.3,
                'bed': 0.6,
                'clock': 0.7,
                'bottle': 0.7,
                'default': 0.4
            }
        },
        
        'studio': {
            'description': 'Studio/professional photography - controlled lighting',
            'weights': {
                'person': 1.0,
                'chair': 0.5,
                'couch': 0.5,
                'vase': 0.7,
                'potted_plant': 0.6,
                'default': 0.3
            }
        },
        
        'abstract': {
            'description': 'Abstract/artistic - preserve all visual elements',
            'weights': {
                'person': 0.8,
                'default': 0.5
            }
        },
        
        'macro': {
            'description': 'Macro/close-up - maximum detail on subject',
            'weights': {
                'person': 0.9,
                'bird': 1.0,
                'cat': 1.0,
                'dog': 1.0,
                'bottle': 0.9,
                'cup': 0.9,
                'vase': 0.9,
                'potted_plant': 0.9,
                'default': 0.6
            }
        },
        
        'low_quality': {
            'description': 'Low quality/garbage image - aggressive compression acceptable',
            'weights': {
                'person': 0.6,
                'default': 0.05
            }
        },
        
        'blurry': {
            'description': 'Blurry/out of focus - can compress heavily',
            'weights': {
                'person': 0.7,
                'default': 0.1
            }
        },
        
        'meme': {
            'description': 'Meme/social media - preserve text and key elements',
            'weights': {
                'person': 0.8,
                'cat': 0.9,
                'dog': 0.9,
                'default': 0.4
            }
        },
        
        'collage': {
            'description': 'Collage/multiple images - balanced compression',
            'weights': {
                'person': 0.9,
                'default': 0.4
            }
        },
        
        'aerial': {
            'description': 'Aerial/drone view - landscape from above',
            'weights': {
                'person': 0.8,
                'car': 0.6,
                'boat': 0.7,
                'airplane': 0.8,
                'building': 0.5,
                'default': 0.2
            }
        },
        
        'underwater': {
            'description': 'Underwater photography - aquatic subjects',
            'weights': {
                'person': 1.0,
                'bird': 0.9,
                'boat': 0.8,
                'surfboard': 0.8,
                'default': 0.2
            }
        },
        
        'fashion': {
            'description': 'Fashion/clothing - people and accessories',
            'weights': {
                'person': 1.0,
                'handbag': 0.95,
                'tie': 0.95,
                'umbrella': 0.9,
                'suitcase': 0.9,
                'backpack': 0.9,
                'default': 0.2
            }
        },
        
        'barcode_qr': {
            'description': 'Barcode/QR code - maximum quality for scanability',
            'weights': {
                'book': 0.9,
                'bottle': 0.8,
                'cell_phone': 0.7,
                'default': 0.7
            }
        },
        
        'general': {
            'description': 'General scene - balanced priorities',
            'weights': {
                'person': 1.0,
                
                'dog': 0.9,
                'cat': 0.9,
                'bird': 0.8,
                
                'car': 0.7,
                'bicycle': 0.7,
                'motorcycle': 0.7,
                
                'book': 0.6,
                'laptop': 0.7,
                'cell_phone': 0.7,
                'clock': 0.6,
                
                'default': 0.5
            }
        }
    }
    
    def __init__(self):
        self.current_profile = None
        self.current_scene = 'general'
    
    def set_scene(self, scene: str):
        if scene in self.INTENT_PROFILES:
            self.current_scene = scene
            self.current_profile = self.INTENT_PROFILES[scene]
        else:
            self.current_scene = 'general'
            self.current_profile = self.INTENT_PROFILES['general']
    
    def get_weight_for_class(self, class_id: int = None, class_name: str = None) -> float:
        if self.current_profile is None:
            self.set_scene('general')
        
        if class_name is None and class_id is not None:
            class_name = self.COCO_CLASSES.get(class_id, None)
        
        if class_name is None:
            return self.current_profile['weights'].get('default', 0.5)
        
        weight = self.current_profile['weights'].get(class_name,
                                                     self.current_profile['weights'].get('default', 0.5))
        
        return weight
    
    def get_weights_for_detections(self, detections: List[Dict]) -> List[float]:
        weights = []
        
        for det in detections:
            class_id = det.get('class', None)
            class_name = det.get('class_name', None)
            
            weight = self.get_weight_for_class(class_id, class_name)
            weights.append(weight)
        
        return weights
    
    def get_scene_description(self) -> str:
        if self.current_profile:
            return self.current_profile['description']
        return "No scene profile loaded"
    
    def list_available_scenes(self) -> List[str]:
        return list(self.INTENT_PROFILES.keys())

