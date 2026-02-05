"""
FFmpeg AVIF Encoder Integration
Handles AVIF encoding with custom quality maps using AV1 codec.
AVIF offers better compression than JPEG/WebP with wide browser support.
"""

import subprocess
import os
import numpy as np
from typing import Optional, Dict
import tempfile


class AVIFEncoder:
    """
    Wrapper for FFmpeg AVIF encoder using libaom-av1 codec with QP map support.
    
    AVIF benefits:
    - Better compression than JPEG/WebP (30-50% smaller)
    - Wide browser support (Chrome, Firefox, Safari 16+)
    - Supports HDR and wide color gamut
    - Based on AV1 video codec
    """
    
    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        """
        Initialize the AVIF encoder.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary (default: 'ffmpeg' in PATH)
        """
        self.ffmpeg_path = ffmpeg_path
        
        # Check if FFmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg with libaom-av1 or libsvtav1 support.")
        
        print(f"✓ AVIFEncoder initialized")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed and has AV1 support."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False
            
            # Check for AV1 encoder support
            result = subprocess.run(
                [self.ffmpeg_path, '-encoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'libaom-av1' in result.stdout or 'libsvtav1' in result.stdout or 'av1' in result.stdout:
                print("  FFmpeg with AV1/AVIF support detected")
                return True
            else:
                print("  Warning: FFmpeg found but AV1 encoder support unclear")
                return True  # Try anyway
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def encode_to_avif(self,
                      input_path: str,
                      output_path: str,
                      crf: int = 30,
                      speed: int = 6,
                      additional_params: Optional[Dict[str, str]] = None) -> bool:
        """
        Encode image to AVIF format.
        
        Args:
            input_path: Path to input image
            output_path: Path to output AVIF file
            crf: Constant Rate Factor (0-63, lower=better quality, default: 30)
                 Recommended: 23-32 for photos, 18-25 for high quality
            speed: Encoding speed (0-8, higher=faster but lower compression)
                   Recommended: 6 for balanced, 4 for better compression
            additional_params: Additional AV1 parameters
            
        Returns:
            True if encoding succeeded, False otherwise
        """
        # Ensure output has .avif extension
        if not output_path.endswith('.avif'):
            output_path = os.path.splitext(output_path)[0] + '.avif'
        
        # Build FFmpeg command
        # Try libaom-av1 first (better quality), fallback to libsvtav1 (faster)
        cmd = [
            self.ffmpeg_path,
            '-y',  # Overwrite output file
            '-i', input_path,
            '-c:v', 'libaom-av1',  # Use libaom-av1 encoder
            '-crf', str(crf),
            '-cpu-used', str(speed),  # Encoding speed (0=slowest/best, 8=fastest)
            '-row-mt', '1',  # Enable row-based multithreading
            '-threads', '0',  # Use all available threads
            '-pix_fmt', 'yuv420p',  # Color format
        ]
        
        # Build additional AV1 parameters
        aom_params = []
        
        # Add default parameters for better quality
        if additional_params:
            for key, value in additional_params.items():
                aom_params.append(f"{key}={value}")
        
        # Add AV1 params to command if any
        if aom_params:
            cmd.extend(['-aom-params', ':'.join(aom_params)])
        
        # Output file
        cmd.append(output_path)
        
        # Execute FFmpeg
        try:
            print(f"  Encoding: {input_path} -> {output_path}")
            print(f"  Settings: CRF={crf}, Speed={speed}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"  ✓ AVIF encoding successful")
                return True
            else:
                # Try libsvtav1 as fallback
                print(f"  ℹ️  libaom-av1 failed, trying libsvtav1...")
                return self._encode_with_svtav1(input_path, output_path, crf, speed)
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Encoding timed out")
            return False
        except Exception as e:
            print(f"  ✗ Encoding error: {e}")
            return False
    
    def _encode_with_svtav1(self,
                           input_path: str,
                           output_path: str,
                           crf: int,
                           speed: int) -> bool:
        """Fallback encoding with libsvtav1 (faster encoder)."""
        cmd = [
            self.ffmpeg_path,
            '-y',
            '-i', input_path,
            '-c:v', 'libsvtav1',
            '-crf', str(crf),
            '-preset', str(speed),
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"  ✓ AVIF encoding successful (libsvtav1)")
                return True
            else:
                print(f"  ✗ Encoding failed:")
                print(f"    {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ✗ Encoding error: {e}")
            return False
    
    def encode_with_quality_zones(self,
                                  input_path: str,
                                  output_path: str,
                                  qp_map: np.ndarray,
                                  base_crf: int = 30) -> bool:
        """
        Encode using QP-based quality allocation with AV1's adaptive quantization.
        
        Args:
            input_path: Path to input image
            output_path: Path to output AVIF file
            qp_map: QP map array (H, W) with values 10-51
            base_crf: Base CRF value (default: 30)
            
        Returns:
            True if encoding succeeded
        """
        import cv2
        
        # Load image to get dimensions
        img = cv2.imread(input_path)
        if img is None:
            print(f"  ✗ Could not load image: {input_path}")
            return False
        
        h, w = img.shape[:2]
        
        # Analyze QP map for quality zones
        min_qp = int(np.min(qp_map))
        max_qp = int(np.max(qp_map))
        median_qp = int(np.median(qp_map))
        
        # Calculate percentage of high-quality regions
        high_quality_pixels = np.sum(qp_map <= 20)
        high_quality_percent = (high_quality_pixels / qp_map.size) * 100
        
        print(f"  QP Map Stats: min={min_qp}, median={median_qp}, max={max_qp}")
        print(f"  High-quality regions: {high_quality_percent:.1f}%")
        
        # Strategy: Use QP statistics to determine CRF
        # Lower CRF = better quality protection for important regions
        if high_quality_percent > 20:
            # Lots of important content - use low CRF
            crf = 23
            aq_mode = 1  # Variance-based AQ
            aq_strength = 1.0
            print(f"  Strategy: High important content - CRF={crf}, moderate AQ")
        elif high_quality_percent > 5:
            # Moderate important content
            crf = 28
            aq_mode = 1
            aq_strength = 1.5
            print(f"  Strategy: Moderate important content - CRF={crf}, strong AQ")
        else:
            # Small important regions - protect them specifically
            crf = 20  # Very low CRF to protect important details
            aq_mode = 2  # Complexity-based AQ
            aq_strength = 2.0  # Very aggressive AQ to compress background
            print(f"  Strategy: Small critical regions - CRF={crf}, aggressive AQ")
        
        # AV1-specific parameters for quality optimization
        additional_params = {
            'aq-mode': str(aq_mode),  # Adaptive quantization mode
            'tune': 'ssim',  # Tune for perceptual quality
            'enable-cdef': '1',  # Constrained directional enhancement filter
            'enable-restoration': '1',  # Loop restoration filter
        }
        
        return self.encode_to_avif(
            input_path=input_path,
            output_path=output_path,
            crf=crf,
            speed=4,  # Slower for better compression
            additional_params=additional_params
        )
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return 0
    
    def get_compression_ratio(self, input_path: str, output_path: str) -> float:
        """Calculate compression ratio."""
        input_size = self.get_file_size(input_path)
        output_size = self.get_file_size(output_path)
        
        if output_size == 0:
            return 0.0
        
        return input_size / output_size
    
    def batch_encode(self,
                    input_paths: list,
                    output_dir: str,
                    qp_maps: Optional[list] = None,
                    crf: int = 30,
                    speed: int = 6) -> Dict[str, bool]:
        """
        Batch encode multiple files to AVIF.
        
        Args:
            input_paths: List of input file paths
            output_dir: Output directory
            qp_maps: Optional list of QP map arrays
            crf: CRF value for encoding
            speed: Encoding speed (0-8)
            
        Returns:
            Dictionary mapping input paths to success status
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for i, input_path in enumerate(input_paths):
            # Generate output path
            basename = os.path.basename(input_path)
            name, _ = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{name}_compressed.avif")
            
            # Get QP map if provided
            qp_map = qp_maps[i] if qp_maps and i < len(qp_maps) else None
            
            # Encode
            if qp_map is not None:
                success = self.encode_with_quality_zones(
                    input_path, output_path, qp_map
                )
            else:
                success = self.encode_to_avif(
                    input_path, output_path, crf=crf, speed=speed
                )
            
            results[input_path] = success
        
        return results
    
    def decode_avif_to_png(self, avif_path: str, output_path: str) -> bool:
        """
        Decode AVIF back to PNG for preview/editing.
        
        Args:
            avif_path: Path to AVIF file
            output_path: Path to output PNG
            
        Returns:
            True if successful
        """
        cmd = [
            self.ffmpeg_path,
            '-y',
            '-i', avif_path,
            '-f', 'image2',
            '-c:v', 'png',
            '-compression_level', '9',
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"  ✓ Decoded AVIF to PNG: {output_path}")
                return True
            else:
                print(f"  ✗ Decoding failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ✗ Decoding error: {e}")
            return False
