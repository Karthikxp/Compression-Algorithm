#!/usr/bin/env python3
"""
Intelligent SAAC Compression (Version 2.0)

New Features:
1. Scene Classification - Automatically detects scene type (restaurant, landscape, street, etc.)
2. Intent-Based Rules - Applies smart compression rules based on scene context
3. YOLOv8-seg - Pixel-perfect segmentation masks instead of bounding boxes
4. Prominence Boosting - Automatically boosts large/central objects
5. Weight Calculation - Combines scene intent + prominence + saliency

This is the "gold standard" compression pipeline!
"""

import sys
import os
from saac import IntelligentSaacCompressor


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compress_intelligent.py <input_image>")
        print("\n🧠 INTELLIGENT SAAC 2.0")
        print("="*70)
        print("This uses the revolutionary scene-aware compression pipeline:")
        print("")
        print("  1. Scene Classification → Determines image context")
        print("  2. YOLOv8-seg → Pixel-perfect object masks")
        print("  3. Prominence Check → Boosts large/central subjects")
        print("  4. Intent Rules → Scene-specific importance weights")
        print("  5. Saliency → Fills in visual attention gaps")
        print("  6. Smart QP Map → Adaptive quality allocation")
        print("  7. HEVC Encoding → Final compression")
        print("")
        print("="*70)
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_intelligent.hevc"
    
    print(f"Compressing: {input_path}")
    print("="*70)
    print("🧠 Using INTELLIGENT SAAC 2.0")
    print("   Scene-Aware + Prominence-Based Compression")
    print("="*70)
    
    # Detect device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("✓ GPU detected - using CUDA acceleration")
        else:
            print("ℹ️  No GPU detected - using CPU")
    except:
        device = 'cpu'
        print("ℹ️  Using CPU")
    
    print()
    
    # Create intelligent compressor
    compressor = IntelligentSaacCompressor(
        device=device,
        yolo_model='yolov8n-seg.pt',  # 🔥 Segmentation model
        saliency_method='spectral',
        segmentation_method='simple',
        scene_method='simple',          # 🔥 Scene classification
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='priority'
    )
    
    # Compress
    stats = compressor.compress_image(
        input_path=input_path,
        output_path=output_path,
        save_visualizations=True,
        visualization_dir='visualizations_intelligent'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("INTELLIGENT COMPRESSION SUMMARY")
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
    print(f"Visualizations: visualizations_intelligent/")
    print("="*70)
    print("\n✨ Check visualizations_intelligent/ to see:")
    print("   - _detections.jpg: Segmentation masks")
    print("   - _prominence.jpg: Prominence scores")
    print("   - _qp_map.jpg: Quality allocation map")
    print("   - _scene.jpg: Detected scene type")


if __name__ == '__main__':
    main()

