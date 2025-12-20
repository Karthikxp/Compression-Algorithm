#!/usr/bin/env python3
"""Quick script to compress a single image"""

import sys
from saac import SaacCompressor

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'gramps.jpg'
    output_file = input_file.replace('.jpg', '_compressed.hevc')
    
    print(f"Compressing: {input_file}")
    print("="*70)
    
    # Initialize compressor
    compressor = SaacCompressor(
        device='cpu',
        person_quality=10,
        saliency_quality=25,
        background_quality=51,
        enable_saliency=True,
        enable_segmentation=True
    )
    
    # Compress
    stats = compressor.compress_image(
        input_path=input_file,
        output_path=output_file,
        save_visualizations=True,
        visualization_dir='visualizations_new'
    )
    
    print("\n" + "="*70)
    print("COMPRESSION SUMMARY")
    print("="*70)
    print(f"Original:       {stats['original_size_mb']:.2f} MB")
    print(f"Compressed:     {stats['compressed_size_mb']:.2f} MB")
    print(f"Ratio:          {stats['compression_ratio']:.2f}x")
    print(f"Space saved:    {stats['space_saved_percent']:.1f}%")
    print(f"People found:   {stats['detections']}")
    print(f"Processing:     {stats['processing_time_seconds']:.1f}s")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"Visualizations: visualizations_new/")

if __name__ == '__main__':
    main()

