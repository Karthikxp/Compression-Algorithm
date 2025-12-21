"""
Object/Person Detection using YOLOv8-seg
Layer 1: The "Must-Have" - Detects critical objects with pixel-perfect segmentation masks.
Upgraded from bounding boxes to instance segmentation.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
import torch


class ObjectDetectorSeg:
    """
    Lightweight object detector using YOLOv8-seg for pixel-perfect masks.
    These regions will receive maximum quality (minimum compression).
    """
    
    def __init__(self, model_name: str = 'yolov8n-seg.pt', device: str = 'cpu',
                 target_classes: Optional[List[int]] = None):
        """
        Initialize the segmentation detector.
        
        Args:
            model_name: YOLOv8-seg model variant ('yolov8n-seg.pt' for nano)
            device: 'cuda' or 'cpu'
            target_classes: List of COCO class IDs to detect. None = detect all.
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.device = device
            
            # Default to detecting people, vehicles, and animals (high priority objects)
            if target_classes is None:
                self.target_classes = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
                # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 
                # 14-23=various animals
                # 39-51=food items
            else:
                self.target_classes = target_classes
                
            print(f"✓ ObjectDetectorSeg initialized with {model_name} on {device}")
            print(f"  Detecting {len(self.target_classes)} object classes with segmentation")
            
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def detect_with_masks(self, image: np.ndarray, 
                         confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Detect objects and return full detection info with segmentation masks.
        
        Args:
            image: Input image (H, W, C) in BGR format
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detection dictionaries with keys:
            - 'class': class ID
            - 'class_name': human-readable class name
            - 'confidence': detection confidence (0-1)
            - 'bbox': (x1, y1, x2, y2) bounding box
            - 'mask': binary segmentation mask (H, W) or None
            - 'area': number of pixels in mask
        """
        h, w = image.shape[:2]
        detections = []
        
        # Run detection with segmentation
        results = self.model(image, device=self.device, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') else None
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i, box in enumerate(boxes):
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
                
                # Get segmentation mask if available
                mask = None
                area = (x2 - x1) * (y2 - y1)  # Default to bbox area
                
                if masks is not None and i < len(masks):
                    # Extract mask for this detection
                    mask_data = masks[i].data[0].cpu().numpy()
                    
                    # Resize mask to image size if needed
                    if mask_data.shape != (h, w):
                        mask_data = cv2.resize(mask_data, (w, h), 
                                             interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to binary mask
                    mask = (mask_data > 0.5).astype(np.uint8)
                    area = np.sum(mask)
                
                detections.append({
                    'class': cls,
                    'class_name': result.names[cls],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'mask': mask,
                    'area': area
                })
        
        return detections
    
    def detect_with_expansion(self, image: np.ndarray, 
                             confidence_threshold: float = 0.25,
                             expansion_percent: float = 10.0) -> np.ndarray:
        """
        Detect objects and create expanded binary mask.
        
        Args:
            image: Input image (H, W, C) in BGR format
            confidence_threshold: Minimum confidence for detection
            expansion_percent: Percentage to expand masks/boxes
            
        Returns:
            Binary mask (H, W) where 1 = important object, 0 = background
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        detections = self.detect_with_masks(image, confidence_threshold)
        
        for det in detections:
            if det['mask'] is not None:
                # Use segmentation mask
                mask = det['mask']
                
                # Expand mask slightly
                if expansion_percent > 0:
                    kernel_size = max(3, int(min(h, w) * expansion_percent / 200))
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                      (kernel_size, kernel_size))
                    mask = cv2.dilate(mask, kernel, iterations=1)
                
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            else:
                # Fallback to expanded bounding box
                x1, y1, x2, y2 = det['bbox']
                
                # Expand bounding box
                box_w, box_h = x2 - x1, y2 - y1
                expand_x = int(box_w * expansion_percent / 200)
                expand_y = int(box_h * expansion_percent / 200)
                
                x1 = max(0, x1 - expand_x)
                y1 = max(0, y1 - expand_y)
                x2 = min(w, x2 + expand_x)
                y2 = min(h, y2 + expand_y)
                
                combined_mask[y1:y2, x1:x2] = 1
        
        return combined_mask
    
    def get_detection_info(self, image: np.ndarray, 
                          confidence_threshold: float = 0.25) -> List[dict]:
        """
        Get detailed information about detections (for logging/visualization).
        
        Returns:
            List of detection dictionaries with keys: class, confidence, bbox
        """
        detections = self.detect_with_masks(image, confidence_threshold)
        
        # Simplify for logging
        return [{
            'class': d['class'],
            'class_name': d['class_name'],
            'confidence': d['confidence'],
            'bbox': d['bbox']
        } for d in detections]
    
    def visualize_detections(self, image: np.ndarray, 
                            detections: List[Dict],
                            show_masks: bool = True) -> np.ndarray:
        """
        Create visualization of detections with masks.
        
        Args:
            image: Original image
            detections: List of detection dicts from detect_with_masks
            show_masks: Whether to show segmentation masks
            
        Returns:
            Visualization image
        """
        vis = image.copy()
        
        # Generate unique colors for each detection
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(detections), 3), dtype=np.uint8)
        
        for i, det in enumerate(detections):
            color = tuple(map(int, colors[i]))
            
            # Draw segmentation mask
            if show_masks and det['mask'] is not None:
                mask_overlay = np.zeros_like(vis)
                mask_overlay[det['mask'] > 0] = color
                vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis

