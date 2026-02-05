#!/usr/bin/env python3
"""
SAAC Format Comparison

Compares different output formats (PNG, AVIF, HEVC) side-by-side
to demonstrate compression ratios and quality preservation.
"""

import sys
import os
from saac import SaacCompressor
import time


def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024*1024):.2f} MB"


def compress_with_format(input_path, output_format, device='cpu'):
    """Compress image with specified format."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Determine output path and compression mode
    if output_format == 'avif':
        output_path = f"{base_name}_compressed.avif"
        compression_mode = 'avif'
    elif output_format == 'png':
        output_path = f"{base_name}_compressed.png"
        compression_mode = 'pixel'
    elif output_format == 'hevc':
        output_path = f"{base_name}_compressed.hevc"
        compression_mode = 'hevc'
    else:
        raise ValueError(f"Unknown format: {output_format}")
    
    print(f"\n{'='*70}")
    print(f"Compressing to {output_format.upper()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Create compressor
    compressor = SaacCompressor(
        device=device,
        yolo_model='yolov8n-seg.pt',
        saliency_method='spectral',
        segmentation_method='simple',
        scene_method='simple',
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='priority',
        compression_mode=compression_mode
    )
    
    # Compress
    stats = compressor.compress_image(
        input_path=input_path,
        output_path=output_path,
        save_visualizations=False
    )
    
    elapsed_time = time.time() - start_time
    
    return {
        'format': output_format.upper(),
        'output_path': output_path,
        'original_size': stats['original_size_bytes'],
        'compressed_size': stats['compressed_size_bytes'],
        'compression_ratio': stats['compression_ratio'],
        'space_saved_percent': stats['space_saved_percent'],
        'processing_time': elapsed_time,
        'scene': stats['scene'],
        'detections': stats['detections']
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 example_comparison.py <input_image> [formats]")
        print("\n📊 SAAC Format Comparison")
        print("="*70)
        print("Compare different compression formats side-by-side")
        print("")
        print("Arguments:")
        print("  <input_image>  Path to input image")
        print("  [formats]      Comma-separated formats (default: avif,png)")
        print("                 Options: avif, png, hevc")
        print("")
        print("Examples:")
        print("  python3 example_comparison.py photo.jpg")
        print("  python3 example_comparison.py photo.jpg avif,png,hevc")
        print("  python3 example_comparison.py photo.jpg avif")
        print("")
        print("="*70)
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Parse formats
    if len(sys.argv) >= 3:
        formats = [f.strip().lower() for f in sys.argv[2].split(',')]
    else:
        formats = ['avif', 'png']  # Default: compare AVIF and PNG
    
    # Validate formats
    valid_formats = ['avif', 'png', 'hevc']
    for fmt in formats:
        if fmt not in valid_formats:
            print(f"Error: Invalid format '{fmt}'. Valid formats: {', '.join(valid_formats)}")
            sys.exit(1)
    
    print("="*70)
    print("SAAC FORMAT COMPARISON")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Formats: {', '.join([f.upper() for f in formats])}")
    
    # Detect device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'
    
    # Compress with each format
    results = []
    for fmt in formats:
        try:
            result = compress_with_format(input_path, fmt, device)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error compressing to {fmt.upper()}: {e}")
            continue
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    if results:
        # Print header
        print(f"\n{'Format':<8} {'Size':<12} {'Ratio':<8} {'Saved':<8} {'Time':<8}")
        print("-" * 50)
        
        # Print results
        for r in results:
            print(f"{r['format']:<8} {format_size(r['compressed_size']):<12} "
                  f"{r['compression_ratio']:>6.2f}x {r['space_saved_percent']:>6.1f}% "
                  f"{r['processing_time']:>6.1f}s")
        
        # Find best compression
        best_ratio = max(results, key=lambda x: x['compression_ratio'])
        fastest = min(results, key=lambda x: x['processing_time'])
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Original size:      {format_size(results[0]['original_size'])}")
        print(f"Scene detected:     {results[0]['scene']}")
        print(f"Objects found:      {results[0]['detections']}")
        print(f"\nBest compression:   {best_ratio['format']} ({best_ratio['compression_ratio']:.2f}x)")
        print(f"Fastest:            {fastest['format']} ({fastest['processing_time']:.1f}s)")
        
        # Format recommendations
        print("\n📝 Recommendations:")
        print("  • AVIF: Best for web delivery (30-50% smaller than JPEG)")
        print("  • PNG: Best for universal compatibility")
        print("  • HEVC: Best for maximum compression (but limited browser support)")
        
        print("\n✨ Output files created:")
        for r in results:
            print(f"  • {r['output_path']}")
        
        print("\n" + "="*70)
    else:
        print("\n❌ No successful compressions")


if __name__ == '__main__':
    main()
