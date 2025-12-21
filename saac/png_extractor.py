"""
PNG Extractor
Extracts people/objects as lossless PNG files from the original image.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Optional


class PNGExtractor:
    """
    Extract detected people/objects as lossless PNG files.
    Perfect for universal compatibility and true lossless quality.
    """
    
    def __init__(self):
        """Initialize PNG extractor."""
        pass
    
    def extract_people_as_png(self,
                             image: np.ndarray,
                             detections: List[Dict],
                             output_dir: str,
                             base_name: str,
                             extract_full_image: bool = True,
                             extract_individuals: bool = True) -> Dict[str, any]:
        """
        Extract people regions as lossless PNG files.
        
        Args:
            image: Original image (full resolution, uncompressed)
            detections: List of detected objects with masks
            output_dir: Directory to save PNG files
            base_name: Base filename for outputs
            extract_full_image: Save full image with people-only mask
            extract_individuals: Save individual person crops
        
        Returns:
            Dictionary with extraction statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter for people only
        people = [det for det in detections if det['class'] == 'person']
        
        if len(people) == 0:
            print("  ⚠️ No people detected, skipping PNG extraction")
            return {'people_count': 0, 'files_created': []}
        
        files_created = []
        
        # Extract full image with people mask (lossless)
        if extract_full_image:
            full_png_path = os.path.join(output_dir, f"{base_name}_people_lossless.png")
            people_image = self._extract_people_region(image, people)
            cv2.imwrite(full_png_path, people_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression
            files_created.append(full_png_path)
            print(f"  ✓ Full people extraction: {full_png_path}")
        
        # Extract individual people crops
        if extract_individuals:
            for i, person in enumerate(people):
                person_crop = self._extract_single_person(image, person)
                if person_crop is not None:
                    crop_path = os.path.join(output_dir, f"{base_name}_person_{i+1}.png")
                    cv2.imwrite(crop_path, person_crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    files_created.append(crop_path)
            
            if len(people) > 0:
                print(f"  ✓ Individual extractions: {len(people)} person(s)")
        
        return {
            'people_count': len(people),
            'files_created': files_created,
            'total_files': len(files_created)
        }
    
    def _extract_people_region(self, image: np.ndarray, people: List[Dict]) -> np.ndarray:
        """
        Extract all people from image, make background transparent.
        
        Args:
            image: Original image
            people: List of person detections with masks
        
        Returns:
            RGBA image with people on transparent background
        """
        h, w = image.shape[:2]
        
        # Create combined mask for all people
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for person in people:
            if person.get('mask') is not None:
                mask = person['mask']
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, mask)
        
        # Convert to RGBA
        if image.shape[2] == 3:
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            rgba_image = image.copy()
        
        # Apply mask to alpha channel
        rgba_image[:, :, 3] = combined_mask
        
        return rgba_image
    
    def _extract_single_person(self, image: np.ndarray, person: Dict) -> Optional[np.ndarray]:
        """
        Extract a single person as a cropped RGBA image.
        
        Args:
            image: Original image
            person: Person detection with mask
        
        Returns:
            Cropped RGBA image or None if extraction failed
        """
        if person.get('mask') is None:
            # Fallback to bounding box
            bbox = person.get('bbox')
            if bbox is None:
                return None
            
            x1, y1, x2, y2 = map(int, bbox)
            crop = image[y1:y2, x1:x2].copy()
            
            # Convert to RGBA (no transparency without mask)
            if crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            
            return crop
        
        # Use mask for precise extraction
        mask = person['mask']
        h, w = image.shape[:2]
        
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Find bounding box of mask
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return None
        
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        
        # Add padding
        padding = 10
        y1 = max(0, y1 - padding)
        x1 = max(0, x1 - padding)
        y2 = min(h, y2 + padding)
        x2 = min(w, x2 + padding)
        
        # Crop image and mask
        crop = image[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]
        
        # Convert to RGBA
        if crop.shape[2] == 3:
            crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        else:
            crop_rgba = crop.copy()
        
        # Apply mask to alpha channel
        crop_rgba[:, :, 3] = mask_crop
        
        return crop_rgba
    
    def save_people_comparison(self,
                              original_image: np.ndarray,
                              people: List[Dict],
                              output_path: str) -> None:
        """
        Save a side-by-side comparison of original and extracted people.
        
        Args:
            original_image: Original image
            people: List of person detections
            output_path: Path to save comparison image
        """
        people_image = self._extract_people_region(original_image, people)
        
        # Convert RGBA to BGR for display (with white background)
        h, w = people_image.shape[:2]
        bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        alpha = people_image[:, :, 3:4] / 255.0
        people_rgb = people_image[:, :, :3]
        
        result = (people_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        
        # Create side-by-side comparison
        comparison = np.hstack([original_image, result])
        
        cv2.imwrite(output_path, comparison)
        print(f"  ✓ Comparison saved: {output_path}")

