#!/usr/bin/env python3
"""
Convert HEVC compressed files back to viewable JPEG format
"""

import subprocess
import sys
import os

def hevc_to_jpg(hevc_file):
    """Convert HEVC file to JPEG for viewing."""
    if not os.path.exists(hevc_file):
        print(f"Error: File not found: {hevc_file}")
        return False
    
    # Generate output filename
    if hevc_file.endswith('.hevc'):
        output_file = hevc_file.replace('.hevc', '_viewable.jpg')
    else:
        output_file = hevc_file + '_viewable.jpg'
    
    print(f"Converting: {hevc_file}")
    print(f"Output:     {output_file}")
    print()
    
    # Convert using FFmpeg
    cmd = ['ffmpeg', '-y', '-i', hevc_file, output_file]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            # Show file sizes
            original_size = os.path.getsize(hevc_file) / 1024
            output_size = os.path.getsize(output_file) / 1024
            
            print("✓ Conversion successful!")
            print()
            print(f"HEVC file:      {original_size:.1f} KB")
            print(f"Viewable JPEG:  {output_size:.1f} KB")
            print()
            print(f"You can now open: {output_file}")
            return True
        else:
            print("✗ Conversion failed")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Conversion timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 view_compressed.py <file.hevc>")
        print()
        print("Examples:")
        print("  python3 view_compressed.py demo_new_compressed.hevc")
        print("  python3 view_compressed.py output.hevc")
        return
    
    hevc_file = sys.argv[1]
    hevc_to_jpg(hevc_file)


if __name__ == '__main__':
    main()

