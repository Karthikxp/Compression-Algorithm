"""
HEIC Converter
Converts HEVC video files to HEIC image format with lossless subject preservation.
"""

import subprocess
import os
from typing import Optional


class HEICConverter:
    """
    Convert HEVC files to HEIC (single-frame HEVC in image container).
    Perfect for storing compressed images with lossless subjects.
    """
    
    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        """
        Initialize the HEIC converter.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary (default: 'ffmpeg' in PATH)
        """
        self.ffmpeg_path = ffmpeg_path
    
    def hevc_to_heic(self, 
                     hevc_path: str, 
                     heic_path: str,
                     copy_stream: bool = True) -> bool:
        """
        Convert HEVC file to HEIC format.
        
        Args:
            hevc_path: Path to input HEVC file
            heic_path: Path to output HEIC file
            copy_stream: If True, copy the stream without re-encoding (faster, preserves quality)
        
        Returns:
            True if conversion succeeded, False otherwise
        """
        try:
            if copy_stream:
                # Stream copy - no re-encoding, instant conversion
                cmd = [
                    self.ffmpeg_path,
                    '-i', hevc_path,
                    '-c:v', 'copy',  # Copy video stream without re-encoding
                    '-f', 'hevc',    # Force HEVC format
                    '-frames:v', '1', # Single frame (image)
                    '-y',            # Overwrite output
                    heic_path
                ]
            else:
                # Re-encode (slower, but can adjust quality)
                cmd = [
                    self.ffmpeg_path,
                    '-i', hevc_path,
                    '-c:v', 'libx265',
                    '-frames:v', '1',
                    '-y',
                    heic_path
                ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(heic_path):
                return True
            else:
                print(f"HEIC conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("HEIC conversion timed out")
            return False
        except Exception as e:
            print(f"HEIC conversion error: {e}")
            return False
    
    def png_to_heic(self,
                    png_path: str,
                    heic_path: str,
                    qp_map_path: Optional[str] = None,
                    enable_lossless: bool = True) -> bool:
        """
        Convert PNG directly to HEIC with optional QP map.
        
        Args:
            png_path: Path to input PNG file
            heic_path: Path to output HEIC file
            qp_map_path: Path to QP map file (optional)
            enable_lossless: Enable cu-lossless=1 for QP=0 regions
        
        Returns:
            True if conversion succeeded, False otherwise
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', png_path,
                '-c:v', 'libx265',
                '-frames:v', '1',  # Single frame
                '-pix_fmt', 'yuv420p',
            ]
            
            # Build x265 parameters
            x265_params = ['crf=18', 'aq-mode=4']
            
            if enable_lossless:
                x265_params.append('cu-lossless=1')  # True lossless for QP=0
            
            if qp_map_path and os.path.exists(qp_map_path):
                x265_params.append(f'qpmap={qp_map_path}')
            
            cmd.extend([
                '-x265-params', ':'.join(x265_params),
                '-y',
                heic_path
            ])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(heic_path):
                return True
            else:
                print(f"PNG to HEIC conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("PNG to HEIC conversion timed out")
            return False
        except Exception as e:
            print(f"PNG to HEIC conversion error: {e}")
            return False

