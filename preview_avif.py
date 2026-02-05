#!/usr/bin/env python3
"""
AVIF Preview Tool

Decodes AVIF images back to PNG for preview and editing.
Also displays compression statistics and quality information.
"""

import sys
import os
from saac import AVIFEncoder
import cv2


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 preview_avif.py <input.avif> [output.png]")
        print("\n🖼️  AVIF Preview Tool")
        print("="*70)
        print("Decode AVIF images to PNG for preview and editing")
        print("")
        print("Features:")
        print("  • Decodes AVIF to PNG format")
        print("  • Shows compression statistics")
        print("  • Preserves full quality")
        print("")
        print("Examples:")
        print("  python3 preview_avif.py image_compressed.avif")
        print("  python3 preview_avif.py image_compressed.avif output.png")
        print("")
        print("="*70)
        sys.exit(1)
    
    avif_path = sys.argv[1]
    
    if not os.path.exists(avif_path):
        print(f"Error: File not found: {avif_path}")
        sys.exit(1)
    
    if not avif_path.endswith('.avif'):
        print(f"Warning: File doesn't have .avif extension: {avif_path}")
    
    # Determine output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(avif_path))[0]
        output_path = f"{base_name}_preview.png"
    
    print(f"Decoding AVIF: {avif_path}")
    print("="*70)
    
    # Get AVIF file info
    avif_size = os.path.getsize(avif_path)
    print(f"\n📊 AVIF File Info:")
    print(f"  Path: {avif_path}")
    print(f"  Size: {avif_size / (1024*1024):.2f} MB ({avif_size / 1024:.1f} KB)")
    
    # Try to get image dimensions using OpenCV (if it supports AVIF)
    # Otherwise just decode with FFmpeg
    print(f"\n🔄 Decoding to PNG...")
    
    # Initialize encoder (for decoding)
    encoder = AVIFEncoder()
    
    # Decode AVIF to PNG
    success = encoder.decode_avif_to_png(avif_path, output_path)
    
    if success:
        png_size = os.path.getsize(output_path)
        
        print(f"\n✅ Success!")
        print(f"\n📁 Output:")
        print(f"  Path: {output_path}")
        print(f"  Size: {png_size / (1024*1024):.2f} MB ({png_size / 1024:.1f} KB)")
        
        # Load decoded image to get dimensions
        img = cv2.imread(output_path)
        if img is not None:
            h, w = img.shape[:2]
            print(f"  Resolution: {w}x{h}")
            print(f"  Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
        
        # Compression statistics
        compression_ratio = png_size / avif_size if avif_size > 0 else 0
        print(f"\n📈 Compression Statistics:")
        print(f"  AVIF vs PNG ratio: 1:{compression_ratio:.2f}")
        print(f"  AVIF savings: {(1 - 1/compression_ratio)*100:.1f}% smaller than PNG")
        
        print("\n💡 Tips:")
        print("  • AVIF is designed for efficient storage and web delivery")
        print("  • PNG is lossless but much larger - use for editing")
        print("  • Modern browsers can display AVIF directly (no decoding needed)")
        print("  • For archival: keep AVIF, generate PNG only when editing")
        
        print("\n✨ Preview saved successfully!")
        print("="*70)
    else:
        print(f"\n❌ Failed to decode AVIF file")
        print("\nTroubleshooting:")
        print("  • Make sure FFmpeg is installed with AV1 decoder support")
        print("  • Check if the AVIF file is corrupted")
        print("  • Try: ffmpeg -i input.avif output.png")
        sys.exit(1)


if __name__ == '__main__':
    main()
