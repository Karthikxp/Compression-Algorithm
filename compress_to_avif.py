#!/usr/bin/env python3
"""
SAAC Compression with AVIF Output

Compresses images to AVIF format with content-aware quality allocation.
AVIF offers superior compression (30-50% smaller than JPEG) with wide browser support.
"""

import sys
import os
from saac import SaacCompressor


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compress_to_avif.py <input_image>")
        print("\n🎯 SAAC with AVIF Compression")
        print("="*70)
        print("Content-aware compression using modern AVIF format")
        print("")
        print("AVIF Benefits:")
        print("  ✓ 30-50% smaller than JPEG/WebP at same quality")
        print("  ✓ Wide browser support (Chrome, Firefox, Safari 16+)")
        print("  ✓ Based on AV1 video codec (royalty-free)")
        print("  ✓ Supports HDR and wide color gamut")
        print("  ✓ Perfect for web delivery and archival")
        print("")
        print("Content-Aware Features:")
        print("  • Scene Classification - Auto-detects image context")
        print("  • Object Segmentation - Pixel-perfect masks (YOLOv8-seg)")
        print("  • Prominence Boosting - Boosts large/central subjects")
        print("  • Saliency Detection - Finds visually important regions")
        print("  • Smart QP Map - Adaptive quality allocation")
        print("  • AV1 Encoding - Superior compression with quality preservation")
        print("")
        print("Requirements:")
        print("  • FFmpeg with libaom-av1 or libsvtav1 support")
        print("  • Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        print("")
        print("="*70)
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_compressed.avif"  # AVIF output
    
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
    
    # Check for optional CLIP scene classifier
    use_clip = False
    try:
        import clip
        print("✓ CLIP installed - using advanced scene classification")
        use_clip = True
    except ImportError:
        print("ℹ️  Using simple scene classification (install CLIP for better results)")
    
    print()
    
    # Create compressor with AVIF mode
    print("Initializing SAAC with AVIF encoder...")
    print("-"*70)
    
    compressor = SaacCompressor(
        device=device,
        yolo_model='yolov8n-seg.pt',
        saliency_method='spectral',
        segmentation_method='simple',
        scene_method='clip' if use_clip else 'simple',
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='priority',
        compression_mode='avif'  # ← Use AVIF compression
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
    print("COMPRESSION SUMMARY (AVIF)")
    print("="*70)
    print(f"Input:          {input_path}")
    print(f"Output:         {output_path}")
    print(f"\n🎯 Scene Analysis:")
    print(f"  Primary Scene:  {stats['scene']}")
    print(f"  Confidence:     {stats['scene_confidence']:.1%}")
    
    # If CLIP classifier is available, show probability distribution
    if use_clip and hasattr(compressor.scene_classifier, 'get_scene_probabilities_dict'):
        print(f"\n📊 Scene Probability Distribution:")
        import cv2
        image = cv2.imread(input_path)
        probs_dict = compressor.scene_classifier.get_scene_probabilities_dict(image)
        
        # Show top 4 scenes
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:4]
        for scene, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = "█" * bar_length
            print(f"  {scene:12s} {bar} {prob:5.1%}")
    
    print(f"\n📦 Compression Results:")
    print(f"  Objects found:  {stats['detections']}")
    print(f"  Original size:  {stats['original_size_mb']:.2f} MB")
    print(f"  AVIF size:      {stats['compressed_size_mb']:.2f} MB")
    print(f"  Ratio:          {stats['compression_ratio']:.2f}x")
    print(f"  Space saved:    {stats['space_saved_percent']:.1f}%")
    print(f"  Processing:     {stats['processing_time_seconds']:.1f}s")
    
    print(f"\n📈 Quality Allocation:")
    qp_stats = stats['qp_statistics']
    print(f"  High quality:   {qp_stats['high_quality_percent']:.1f}% (faces, objects)")
    print(f"  Medium quality: {qp_stats['medium_quality_percent']:.1f}% (moderate importance)")
    print(f"  Low quality:    {qp_stats['low_quality_percent']:.1f}% (backgrounds)")
    
    print(f"\n📁 Visualizations: visualizations/")
    print("="*70)
    
    print("\n✨ Check visualizations/ folder for quality maps:")
    print("   • _detections.jpg - Segmentation masks")
    print("   • _prominence.jpg - Prominence scores")
    print("   • _qp_map.jpg - Quality allocation map")
    print("   • _saliency.jpg - Saliency heatmap")
    print("   • _scene.jpg - Detected scene type")
    
    print("\n🌐 AVIF Support:")
    print("   • Browsers: Chrome 85+, Firefox 93+, Safari 16+, Edge 121+")
    print("   • Preview: Can be opened in modern browsers or image viewers")
    print(f"   • Decode: python3 preview_avif.py {output_path}")
    
    print("\n💡 Why AVIF?")
    print("   AVIF uses AV1 codec with content-aware quality allocation to achieve")
    print("   superior compression while preserving important details in faces,")
    print("   objects, and salient regions. Perfect for web delivery!")


if __name__ == '__main__':
    main()
