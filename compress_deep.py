#!/usr/bin/env python3
"""
Compress image using FULL DEEP LEARNING pipeline.
Uses: YOLOv8 + U2-Net + DeepLabV3

This is slower but more accurate than the hybrid method.
"""

import sys
import os
from saac import SaacCompressor


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compress_deep.py <input_image>")
        print("\nThis uses the FULL DEEP LEARNING pipeline:")
        print("  - YOLOv8-nano for object detection")
        print("  - U2-Net for saliency detection (176 MB model)")
        print("  - DeepLabV3 for semantic segmentation (100+ MB model)")
        print("\nFirst run will download models (~300 MB total).")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_deep.hevc"
    
    print(f"Compressing: {input_path}")
    print("="*70)
    print("🧠 Using DEEP LEARNING pipeline")
    print("   This may take longer but produces better quality allocation")
    print("="*70)
    
    # Detect if CUDA is available
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("✓ GPU detected - using CUDA acceleration")
        else:
            print("ℹ️  No GPU detected - using CPU (slower)")
    except:
        device = 'cpu'
        print("ℹ️  Using CPU")
    
    print()
    
    # Create compressor with deep learning methods
    compressor = SaacCompressor(
        device=device,
        yolo_model='yolov8n.pt',
        saliency_method='u2net',           # 🔥 Deep learning saliency
        segmentation_method='deeplabv3',   # 🔥 Deep learning segmentation
        person_quality=10,
        saliency_quality=25,
        background_quality=51,
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='priority'
    )
    
    # Compress
    stats = compressor.compress_image(
        input_path=input_path,
        output_path=output_path,
        save_visualizations=True,
        visualization_dir='visualizations_deep'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("DEEP LEARNING COMPRESSION SUMMARY")
    print("="*70)
    print(f"Input:          {input_path}")
    print(f"Output:         {output_path}")
    print(f"Original size:  {stats['original_size_mb']:.2f} MB")
    print(f"Compressed:     {stats['compressed_size_mb']:.2f} MB")
    print(f"Ratio:          {stats['compression_ratio']:.2f}x")
    print(f"Space saved:    {stats['space_saved_percent']:.1f}%")
    print(f"Processing:     {stats['processing_time_seconds']:.1f}s")
    print(f"Visualizations: visualizations_deep/")
    print("="*70)


if __name__ == '__main__':
    main()


