#!/usr/bin/env python3
"""
SAAC - Saliency-Aware Adaptive Compression

Scene-aware compression with intelligent quality allocation.
Achieves 15-20x compression while preserving important details.
"""

import sys
import os
from saac import SaacCompressor


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compress.py <input_image>")
        print("\n🎯 SAAC - Saliency-Aware Adaptive Compression")
        print("="*70)
        print("Intelligent image compression with scene-aware quality allocation")
        print("")
        print("Features:")
        print("  • Scene Classification - Auto-detects image context")
        print("  • Object Segmentation - Pixel-perfect masks (YOLOv8-seg)")
        print("  • Prominence Boosting - Boosts large/central subjects")
        print("  • Saliency Detection - Finds visually important regions")
        print("  • Smart QP Map - Adaptive quality allocation")
        print("  • HEVC Encoding - 15-20x compression ratio")
        print("")
        print("="*70)
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_compressed.hevc"
    
    print(f"Compressing: {input_path}")
    print("="*70)
    
    # Detect device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("✓ GPU detected - using CUDA acceleration")
        else:
            print("ℹ️  Using CPU")
    except:
        device = 'cpu'
        print("ℹ️  Using CPU")
    
    print()
    
    # Create compressor
    # Note: Use scene_method='clip' for superior scene understanding (85-95% accuracy)
    #       Requires CLIP installation: pip install git+https://github.com/openai/CLIP.git
    compressor = SaacCompressor(
        device=device,
        yolo_model='yolov8n-seg.pt',
        saliency_method='spectral',
        segmentation_method='simple',
        scene_method='simple',  # Options: 'simple', 'clip', 'efficientnet', 'resnet'
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='priority'
    )
    
    # Compress
    stats = compressor.compress_image(
        input_path=input_path,
        output_path=output_path,
        save_visualizations=True,
        visualization_dir='visualizations'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("COMPRESSION SUMMARY")
    print("="*70)
    print(f"Input:          {input_path}")
    print(f"Output:         {output_path}")
    print(f"Scene:          {stats['scene']} (confidence: {stats['scene_confidence']:.2f})")
    print(f"Objects found:  {stats['detections']}")
    print(f"Original size:  {stats['original_size_mb']:.2f} MB")
    print(f"Compressed:     {stats['compressed_size_mb']:.2f} MB")
    print(f"Ratio:          {stats['compression_ratio']:.2f}x")
    print(f"Space saved:    {stats['space_saved_percent']:.1f}%")
    print(f"Processing:     {stats['processing_time_seconds']:.1f}s")
    print(f"Visualizations: visualizations/")
    print("="*70)
    print("\n✨ Check visualizations/ folder for quality maps:")
    print("   • _detections.jpg - Segmentation masks")
    print("   • _prominence.jpg - Prominence scores")
    print("   • _qp_map.jpg - Quality allocation map")
    print("   • _saliency.jpg - Saliency heatmap")
    print("   • _scene.jpg - Detected scene type")


if __name__ == '__main__':
    main()
