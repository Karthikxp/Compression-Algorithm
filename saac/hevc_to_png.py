"""
HEVC to PNG Converter
Converts compressed HEVC video to PNG image for easy viewing.
"""

import subprocess
import os
from typing import Optional


class HEVCToPNGConverter:
    """
    Convert HEVC video files to PNG images for easy viewing.
    """
    
    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        """
        Initialize converter.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary
        """
        self.ffmpeg_path = ffmpeg_path
    
    def convert_hevc_to_png(self, 
                           hevc_path: str, 
                           png_path: str,
                           quality: str = 'best') -> bool:
        """
        Convert HEVC file to PNG for viewing.
        
        Args:
            hevc_path: Path to input HEVC file
            png_path: Path to output PNG file
            quality: 'best' (no compression) or 'fast' (some compression)
        
        Returns:
            True if conversion succeeded
        """
        try:
            # Determine PNG compression level
            # 0 = no compression (best quality, larger file)
            # 9 = maximum compression (smaller file, slower)
            compression = 0 if quality == 'best' else 6
            
            cmd = [
                self.ffmpeg_path,
                '-i', hevc_path,
                '-frames:v', '1',  # Extract first frame
                '-compression_level', str(compression),
                '-y',  # Overwrite
                png_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(png_path):
                return True
            else:
                print(f"  ⚠️ HEVC to PNG conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("  ⚠️ HEVC to PNG conversion timed out")
            return False
        except Exception as e:
            print(f"  ⚠️ HEVC to PNG error: {e}")
            return False

