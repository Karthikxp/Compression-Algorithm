#!/usr/bin/env python3
"""
Basic usage example for SAAC (Saliency-Aware Adaptive Compression)
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from saac import SaacCompressor


def main():
    """Basic compression example."""
    
    print("SAAC - Saliency-Aware Adaptive Compression")
    print("Basic Usage Example\n")
    
    # Initialize compressor
    compressor = SaacCompressor(
        device='cpu',  # Use 'cuda' if you have a GPU
        person_quality=10,      # High quality for people
        saliency_quality=25,    # Medium-high quality for salient regions
        background_quality=51,  # Maximum compression for background
        enable_saliency=True,
        enable_segmentation=True
    )
    
    # Example 1: Compress a single image
    input_image = 'input.jpg'  # Replace with your image path
    output_image = 'output_saac.hevc'
    
    if os.path.exists(input_image):
        print(f"\nCompressing: {input_image}")
        
        stats = compressor.compress_image(
            input_path=input_image,
            output_path=output_image,
            save_visualizations=True,
            visualization_dir='visualizations'
        )
        
        print("\nResults:")
        print(f"  Original:    {stats['original_size_mb']:.2f} MB")
        print(f"  Compressed:  {stats['compressed_size_mb']:.2f} MB")
        print(f"  Ratio:       {stats['compression_ratio']:.2f}x")
        print(f"  Space saved: {stats['space_saved_percent']:.1f}%")
        
    else:
        print(f"\nPlease place an image at: {input_image}")
        print("Or modify this script to point to your image.")


if __name__ == '__main__':
    main()

