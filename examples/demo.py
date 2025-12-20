#!/usr/bin/env python3
"""
Interactive demo script for SAAC
Downloads a sample image and demonstrates the compression.
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from saac import SaacCompressor


def download_sample_image():
    """Download a sample image for testing."""
    url = "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=1920"
    output_path = "demo_input.jpg"
    
    if os.path.exists(output_path):
        print(f"✓ Sample image already exists: {output_path}")
        return output_path
    
    print("Downloading sample image...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Could not download sample image: {e}")
        print("Please provide your own image as 'demo_input.jpg'")
        return None


def run_demo():
    """Run the interactive demo."""
    print("="*70)
    print(" " * 20 + "SAAC INTERACTIVE DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Download a sample image (or use demo_input.jpg)")
    print("  2. Detect objects, saliency, and semantic regions")
    print("  3. Generate a custom quality map")
    print("  4. Compress the image using HEVC")
    print("  5. Show compression statistics and visualizations")
    print("\n")
    
    input("Press Enter to start...")
    
    # Download or check for sample image
    input_path = download_sample_image()
    
    if input_path is None or not os.path.exists(input_path):
        print("\n✗ No input image available. Please provide demo_input.jpg")
        return
    
    # Initialize compressor with verbose output
    print("\n" + "="*70)
    print("Initializing SAAC Compressor")
    print("="*70)
    
    compressor = SaacCompressor(
        device='cpu',  # Change to 'cuda' if you have a GPU
        yolo_model='yolov8n.pt',
        saliency_method='spectral',
        person_quality=10,
        saliency_quality=25,
        background_quality=51,
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='priority'
    )
    
    # Compress the image
    output_path = 'demo_output.hevc'
    vis_dir = 'demo_visualizations'
    
    print("\n" + "="*70)
    print("Starting Compression")
    print("="*70)
    
    stats = compressor.compress_image(
        input_path=input_path,
        output_path=output_path,
        save_visualizations=True,
        visualization_dir=vis_dir
    )
    
    # Display results
    print("\n" + "="*70)
    print("DEMO RESULTS")
    print("="*70)
    print(f"\nInput:           {input_path}")
    print(f"Output:          {output_path}")
    print(f"Visualizations:  {vis_dir}/")
    print(f"\nOriginal size:   {stats['original_size_mb']:.2f} MB")
    print(f"Compressed size: {stats['compressed_size_mb']:.2f} MB")
    print(f"Compression:     {stats['compression_ratio']:.2f}x")
    print(f"Space saved:     {stats['space_saved_percent']:.1f}%")
    print(f"Processing time: {stats['processing_time_seconds']:.1f} seconds")
    
    print(f"\nObjects detected: {stats['detections']}")
    
    qp_stats = stats['qp_statistics']
    print(f"\nQuality Distribution:")
    print(f"  High quality:   {qp_stats['high_quality_percent']:.1f}%")
    print(f"  Medium quality: {qp_stats['medium_quality_percent']:.1f}%")
    print(f"  Low quality:    {qp_stats['low_quality_percent']:.1f}%")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"\nCheck the visualizations in '{vis_dir}' to see:")
    print("  - Object detection overlay")
    print("  - Saliency heatmap")
    print("  - Quality allocation map (QP map)")
    print("\nThe compressed file is significantly smaller while preserving")
    print("important details at high quality!")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

