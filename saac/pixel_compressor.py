"""
Pixel-Level Compression
Selectively degrades image regions based on SAAC quality map.
Output PNG is smaller because background pixels are simplified/blurred.
"""

import numpy as np
import cv2
from typing import Tuple


class PixelCompressor:
    """
    Compresses images by selectively degrading pixels based on importance.
    
    Strategy:
    - High importance (QP 10-20): Keep full quality
    - Medium importance (QP 20-35): Slight blur/downsample
    - Low importance (QP 35-45): Moderate blur/color quantization
    - Background (QP 45-51): Heavy blur/color quantization
    
    Result: PNG with less complex pixel data = smaller file size
    """
    
    def __init__(self):
        """Initialize pixel compressor."""
        print("✓ PixelCompressor initialized")
    
    def compress_by_qp_map(self, 
                          image: np.ndarray, 
                          qp_map: np.ndarray,
                          preserve_edges: bool = True) -> np.ndarray:
        """
        Compress image pixels based on QP map.
        
        Args:
            image: Input image (H, W, 3)
            qp_map: Quality map (H, W) with values 10-51
            preserve_edges: If True, preserve edges even in low-quality zones
            
        Returns:
            Compressed image with selectively degraded regions
        """
        h, w = image.shape[:2]
        output = image.copy()
        
        print("\n[Pixel Compression] Applying selective degradation...")
        
        # Zone 1: Critical (QP 10-20) - Keep full quality
        mask_critical = (qp_map <= 20).astype(np.uint8)
        critical_pct = np.sum(mask_critical > 0) / (h * w) * 100
        print(f"  Critical regions (QP≤20): {critical_pct:.1f}% - Full quality preserved")
        
        # Zone 2: High (QP 21-30) - Slight blur
        mask_high = ((qp_map > 20) & (qp_map <= 30)).astype(np.uint8)
        high_pct = np.sum(mask_high > 0) / (h * w) * 100
        if high_pct > 0:
            print(f"  High quality (QP 21-30): {high_pct:.1f}% - Slight blur")
            output = self._apply_selective_blur(output, mask_high, kernel_size=3)
        
        # Zone 3: Medium (QP 31-40) - Moderate compression
        mask_medium = ((qp_map > 30) & (qp_map <= 40)).astype(np.uint8)
        medium_pct = np.sum(mask_medium > 0) / (h * w) * 100
        if medium_pct > 0:
            print(f"  Medium quality (QP 31-40): {medium_pct:.1f}% - Blur + color reduction")
            output = self._apply_selective_blur(output, mask_medium, kernel_size=7)
            output = self._apply_color_quantization(output, mask_medium, levels=128)
        
        # Zone 4: Low (QP 41-51) - Heavy compression
        mask_low = (qp_map > 40).astype(np.uint8)
        low_pct = np.sum(mask_low > 0) / (h * w) * 100
        if low_pct > 0:
            print(f"  Low quality (QP>40): {low_pct:.1f}% - Heavy blur + color reduction")
            output = self._apply_selective_blur(output, mask_low, kernel_size=15)
            output = self._apply_color_quantization(output, mask_low, levels=64)
        
        # Optional: Preserve edges even in compressed regions
        if preserve_edges:
            output = self._preserve_edges(image, output, qp_map)
        
        return output
    
    def _apply_selective_blur(self, 
                             image: np.ndarray, 
                             mask: np.ndarray, 
                             kernel_size: int) -> np.ndarray:
        """
        Apply blur only to masked regions.
        
        Args:
            image: Input image
            mask: Binary mask (255 = apply blur, 0 = keep original)
            kernel_size: Blur kernel size (must be odd)
            
        Returns:
            Image with selective blur applied
        """
        if np.sum(mask) == 0:
            return image
        
        # Make kernel size odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Blur the entire image
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Blend based on mask
        mask_3ch = cv2.merge([mask, mask, mask])
        output = np.where(mask_3ch > 0, blurred, image)
        
        return output.astype(np.uint8)
    
    def _apply_color_quantization(self,
                                  image: np.ndarray,
                                  mask: np.ndarray,
                                  levels: int) -> np.ndarray:
        """
        Reduce color depth in masked regions.
        
        Args:
            image: Input image
            mask: Binary mask
            levels: Number of color levels per channel (e.g., 64 = 6-bit color)
            
        Returns:
            Image with reduced color depth in masked regions
        """
        if np.sum(mask) == 0:
            return image
        
        output = image.copy()
        
        # Quantize colors
        step = 256 // levels
        quantized = (image // step) * step
        
        # Apply only to masked regions
        mask_3ch = cv2.merge([mask, mask, mask])
        output = np.where(mask_3ch > 0, quantized, image)
        
        return output.astype(np.uint8)
    
    def _preserve_edges(self,
                       original: np.ndarray,
                       compressed: np.ndarray,
                       qp_map: np.ndarray,
                       edge_threshold: int = 30) -> np.ndarray:
        """
        Preserve strong edges even in compressed regions.
        
        Args:
            original: Original image
            compressed: Compressed image
            qp_map: Quality map
            edge_threshold: Canny edge detection threshold
            
        Returns:
            Image with edges preserved
        """
        # Detect edges in original
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Only preserve edges in low-quality regions
        low_quality_mask = (qp_map > 35).astype(np.uint8) * 255
        edges_to_preserve = cv2.bitwise_and(edges, low_quality_mask)
        
        # Blend
        edges_3ch = cv2.merge([edges_to_preserve, edges_to_preserve, edges_to_preserve])
        output = np.where(edges_3ch > 0, original, compressed)
        
        return output.astype(np.uint8)
    
    def compress_aggressive(self,
                          image: np.ndarray,
                          qp_map: np.ndarray) -> np.ndarray:
        """
        More aggressive compression using downsampling + upsampling.
        
        Args:
            image: Input image
            qp_map: Quality map
            
        Returns:
            Aggressively compressed image
        """
        h, w = image.shape[:2]
        output = image.copy()
        
        print("\n[Aggressive Pixel Compression] Using downsample/upsample...")
        
        # Create scale map based on QP
        # QP 10-20: scale 1.0 (full res)
        # QP 20-30: scale 0.8
        # QP 30-40: scale 0.5
        # QP 40-51: scale 0.25
        
        # Split image into quality zones
        zones = [
            (qp_map <= 20, 1.0, "Critical"),
            ((qp_map > 20) & (qp_map <= 30), 0.8, "High"),
            ((qp_map > 30) & (qp_map <= 40), 0.5, "Medium"),
            (qp_map > 40, 0.25, "Low")
        ]
        
        for zone_mask, scale, zone_name in zones:
            zone_mask = zone_mask.astype(np.uint8) * 255
            zone_pct = np.sum(zone_mask > 0) / (h * w) * 100
            
            if zone_pct > 1:  # Only process if >1% of image
                print(f"  {zone_name} zone: {zone_pct:.1f}% at {scale:.2f}x resolution")
                
                if scale < 1.0:
                    # Extract masked region
                    masked_img = cv2.bitwise_and(image, image, mask=zone_mask)
                    
                    # Downsample
                    small_h = int(h * scale)
                    small_w = int(w * scale)
                    downsampled = cv2.resize(masked_img, (small_w, small_h), 
                                            interpolation=cv2.INTER_AREA)
                    
                    # Upsample back
                    upsampled = cv2.resize(downsampled, (w, h), 
                                          interpolation=cv2.INTER_LINEAR)
                    
                    # Apply to output
                    mask_3ch = cv2.merge([zone_mask, zone_mask, zone_mask])
                    output = np.where(mask_3ch > 0, upsampled, output)
        
        return output.astype(np.uint8)
    
    def get_complexity_reduction(self, 
                                original: np.ndarray, 
                                compressed: np.ndarray) -> dict:
        """
        Calculate how much complexity was reduced.
        
        Args:
            original: Original image
            compressed: Compressed image
            
        Returns:
            Dictionary with complexity metrics
        """
        # Calculate edge density (proxy for complexity)
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_comp = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
        
        edges_orig = cv2.Canny(gray_orig, 50, 150)
        edges_comp = cv2.Canny(gray_comp, 50, 150)
        
        edge_density_orig = np.sum(edges_orig > 0) / edges_orig.size
        edge_density_comp = np.sum(edges_comp > 0) / edges_comp.size
        
        # Color complexity (unique colors)
        orig_reshaped = original.reshape(-1, 3)
        comp_reshaped = compressed.reshape(-1, 3)
        
        unique_colors_orig = len(np.unique(orig_reshaped, axis=0))
        unique_colors_comp = len(np.unique(comp_reshaped, axis=0))
        
        return {
            'edge_density_original': edge_density_orig,
            'edge_density_compressed': edge_density_comp,
            'edge_reduction': (1 - edge_density_comp / edge_density_orig) * 100,
            'unique_colors_original': unique_colors_orig,
            'unique_colors_compressed': unique_colors_comp,
            'color_reduction': (1 - unique_colors_comp / unique_colors_orig) * 100
        }
