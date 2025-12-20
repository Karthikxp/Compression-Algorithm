"""
Object/Person Detection using YOLOv8
Layer 1: The "Must-Have" - Detects critical objects like people, vehicles, etc.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import torch


class ObjectDetector:
    """
    Lightweight object detector using YOLOv8 for identifying critical subjects.
    These regions will receive maximum quality (minimum compression).
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'cpu',
                 target_classes: Optional[List[int]] = None):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLOv8 model variant ('yolov8n.pt' for nano, fastest)
            device: 'cuda' or 'cpu'
            target_classes: List of COCO class IDs to detect. None = detect all.
                           Common: [0] = person only, [0, 2, 3, 5, 7] = person, car, motorcycle, bus, truck
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.device = device
            
            # Default to detecting people, vehicles, and animals (high priority objects)
            if target_classes is None:
                self.target_classes = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 
                # 14-23=various animals
            else:
                self.target_classes = target_classes
                
            print(f"✓ ObjectDetector initialized with {model_name} on {device}")
            
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25) -> np.ndarray:
        """
        Detect objects and return a binary mask.
        
        Args:
            image: Input image (H, W, C) in BGR format
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Binary mask (H, W) where 1 = important object, 0 = background
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Run detection
        results = self.model(image, device=self.device, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get class, confidence, and coordinates
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter by class and confidence
                if cls not in self.target_classes or conf < confidence_threshold:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Fill the mask in this region
                mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def detect_with_expansion(self, image: np.ndarray, 
                             confidence_threshold: float = 0.25,
                             expansion_percent: float = 10.0) -> np.ndarray:
        """
        Detect objects and expand bounding boxes by a percentage.
        Useful to capture context around detected objects.
        
        Args:
            image: Input image (H, W, C) in BGR format
            confidence_threshold: Minimum confidence for detection
            expansion_percent: Percentage to expand bounding boxes
            
        Returns:
            Binary mask (H, W) where 1 = important object, 0 = background
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Run detection
        results = self.model(image, device=self.device, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls not in self.target_classes or conf < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Expand bounding box
                box_w, box_h = x2 - x1, y2 - y1
                expand_x = int(box_w * expansion_percent / 200)  # Divide by 200 (half on each side)
                expand_y = int(box_h * expansion_percent / 200)
                
                x1 = max(0, x1 - expand_x)
                y1 = max(0, y1 - expand_y)
                x2 = min(w, x2 + expand_x)
                y2 = min(h, y2 + expand_y)
                
                mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def get_detection_info(self, image: np.ndarray, 
                          confidence_threshold: float = 0.25) -> List[dict]:
        """
        Get detailed information about detections.
        
        Returns:
            List of detection dictionaries with keys: class, confidence, bbox
        """
        detections = []
        results = self.model(image, device=self.device, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls not in self.target_classes or conf < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                detections.append({
                    'class': cls,
                    'class_name': result.names[cls],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detections

