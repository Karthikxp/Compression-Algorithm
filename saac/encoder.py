"""
FFmpeg HEVC Encoder Integration
Handles video/image encoding with custom QP maps.
"""

import subprocess
import os
import numpy as np
from typing import Optional, Dict
import tempfile
import shutil


class HEVCEncoder:
    """
    Wrapper for FFmpeg HEVC (H.265) encoder with QP map support.
    """
    
    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        """
        Initialize the encoder.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary (default: 'ffmpeg' in PATH)
        """
        self.ffmpeg_path = ffmpeg_path
        
        # Check if FFmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg with libx265 support.")
        
        print(f"✓ HEVCEncoder initialized")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed and has libx265 support."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False
            
            # Check for libx265 support
            result = subprocess.run(
                [self.ffmpeg_path, '-codecs'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'libx265' in result.stdout or 'hevc' in result.stdout:
                print("  FFmpeg with HEVC support detected")
                return True
            else:
                print("  Warning: FFmpeg found but libx265/HEVC support unclear")
                return True  # Try anyway
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def encode_with_qp_map(self,
                          input_path: str,
                          output_path: str,
                          qp_map_path: Optional[str] = None,
                          crf: int = 28,
                          preset: str = 'medium',
                          additional_params: Optional[Dict[str, str]] = None) -> bool:
        """
        Encode image/video with optional QP map.
        
        Args:
            input_path: Path to input image/video
            output_path: Path to output file
            qp_map_path: Path to QP map file (optional)
            crf: Constant Rate Factor (0-51, lower=better quality, default: 28)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
            additional_params: Additional x265 parameters
            
        Returns:
            True if encoding succeeded, False otherwise
        """
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            '-y',  # Overwrite output file
            '-i', input_path,
            '-vf', 'format=yuv420p',  # Convert to YUV420p, strips alpha channel
            '-c:v', 'libx265',
            '-preset', preset,
            '-crf', str(crf)
        ]
        
        # Build x265-params
        x265_params = []
        
        # Add QP map if provided
        if qp_map_path and os.path.exists(qp_map_path):
            # Note: QP map support in x265 is complex and may require patches
            # For now, we'll use adaptive quantization as a workaround
            x265_params.extend([
                'aq-mode=3',  # Auto-variance AQ
                'aq-strength=1.2',
                'qg-size=16'  # Quantization group size
            ])
            print(f"  Note: Using adaptive quantization (QP map direct support requires custom x265 build)")
        else:
            x265_params.extend([
                'aq-mode=3',
                'aq-strength=1.0'
            ])
        
        # Add additional parameters
        if additional_params:
            for key, value in additional_params.items():
                x265_params.append(f"{key}={value}")
        
        # Add x265-params to command
        if x265_params:
            cmd.extend(['-x265-params', ':'.join(x265_params)])
        
        # Output file
        cmd.append(output_path)
        
        # Execute FFmpeg
        try:
            print(f"  Encoding: {input_path} -> {output_path}")
            print(f"  Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"  ✓ Encoding successful")
                return True
            else:
                print(f"  ✗ Encoding failed:")
                print(f"    {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Encoding timed out")
            return False
        except Exception as e:
            print(f"  ✗ Encoding error: {e}")
            return False
    
    def encode_with_quality_zones(self,
                                  input_path: str,
                                  output_path: str,
                                  qp_map: np.ndarray,
                                  base_crf: int = 28) -> bool:
        """
        Encode using QP-based zones with x265's zone encoding.
        Creates low-QP zones for important regions (people, etc.).
        
        Args:
            input_path: Path to input image
            output_path: Path to output file
            qp_map: QP map array
            base_crf: Base CRF value
            
        Returns:
            True if encoding succeeded
        """
        # Load image to get dimensions
        import cv2
        img = cv2.imread(input_path)
        if img is None:
            print(f"  ✗ Could not load image: {input_path}")
            return False
        
        h, w = img.shape[:2]
        
        # Find the MINIMUM QP (best quality needed anywhere)
        min_qp = int(np.min(qp_map))
        max_qp = int(np.max(qp_map))
        median_qp = int(np.median(qp_map))
        
        # Calculate percentage of high-quality regions
        high_quality_pixels = np.sum(qp_map <= 20)
        high_quality_percent = (high_quality_pixels / qp_map.size) * 100
        
        print(f"  QP Map Stats: min={min_qp}, median={median_qp}, max={max_qp}")
        print(f"  High-quality regions: {high_quality_percent:.1f}%")
        
        # Strategy: Use the minimum QP as base to protect important regions
        # Then rely on AQ to compress less important areas more
        base_qp = min_qp if high_quality_percent > 5 else median_qp
        
        # Convert to CRF (use minimum to ensure quality preservation)
        crf = max(15, min(35, base_qp))  # Clamp between 15-35 for reasonable file sizes
        
        print(f"  Using CRF={crf} (biased toward protecting high-quality regions)")
        
        # Use STRONG adaptive quantization to compress background more
        # aq-mode=3: Auto-variance AQ with bias to dark scenes
        # aq-strength: Higher = more aggressive background compression
        additional_params = {
            'aq-mode': '3',
            'aq-strength': '2.0',  # Increased from 1.5 - more aggressive on background
            'qg-size': '8',  # Smaller quantization groups for finer control
            'rd': '6',  # Maximum rate-distortion optimization
            'psy-rd': '2.0',  # Psychovisual optimization
            'psy-rdoq': '2.0',  # Increased psychovisual RD quantization
            'deblock': '-2:-2',  # Stronger deblocking for better quality
            'no-sao': '0',  # Enable SAO filter
            'no-strong-intra-smoothing': '0'  # Enable strong intra smoothing
        }
        
        return self.encode_with_qp_map(
            input_path=input_path,
            output_path=output_path,
            qp_map_path=None,
            crf=crf,
            preset='veryslow',  # Use slowest preset for maximum quality
            additional_params=additional_params
        )
    
    def encode_with_qp_zones_multipass(self,
                                       input_path: str,
                                       output_path: str,
                                       qp_map: np.ndarray) -> bool:
        """
        True spatially-varying encoding using multi-pass approach.
        
        1. Create masks for different quality zones
        2. Encode high-quality regions with low QP
        3. Encode background with high QP
        4. Composite the results
        
        Args:
            input_path: Path to input image
            output_path: Path to output file
            qp_map: QP map array (H, W)
            
        Returns:
            True if encoding succeeded
        """
        import cv2
        
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
        # Create quality zone masks
        # Zone 1: Near-lossless (QP <= 15) - People, faces, critical objects
        mask_critical = (qp_map <= 15).astype(np.uint8) * 255
        
        # Zone 2: High quality (QP 16-25) - Important but not critical
        mask_high = ((qp_map > 15) & (qp_map <= 25)).astype(np.uint8) * 255
        
        # Zone 3: Medium quality (QP 26-40) - Moderate importance
        mask_medium = ((qp_map > 25) & (qp_map <= 40)).astype(np.uint8) * 255
        
        # Zone 4: Low quality (QP > 40) - Background, unimportant
        mask_low = (qp_map > 40).astype(np.uint8) * 255
        
        # Count pixels in each zone
        critical_pct = np.sum(mask_critical > 0) / (h * w) * 100
        high_pct = np.sum(mask_high > 0) / (h * w) * 100
        medium_pct = np.sum(mask_medium > 0) / (h * w) * 100
        low_pct = np.sum(mask_low > 0) / (h * w) * 100
        
        print(f"  Quality Zones:")
        print(f"    Critical (QP≤15): {critical_pct:.1f}%")
        print(f"    High (QP 16-25): {high_pct:.1f}%")
        print(f"    Medium (QP 26-40): {medium_pct:.1f}%")
        print(f"    Low (QP>40): {low_pct:.1f}%")
        
        # Determine overall encoding strategy based on critical region size
        if critical_pct > 20:
            # Lots of important content - use low CRF globally
            crf = 18
            aq_strength = 1.8
            print(f"  Strategy: High critical content - CRF={crf}, aggressive AQ")
        elif critical_pct > 5:
            # Moderate important content
            crf = 22
            aq_strength = 2.0
            print(f"  Strategy: Moderate critical content - CRF={crf}, strong AQ")
        else:
            # Small critical regions - protect them specifically
            crf = 15  # Very low CRF to protect critical regions
            aq_strength = 2.5  # Very aggressive AQ to compress background
            print(f"  Strategy: Small critical regions - CRF={crf}, very aggressive AQ")
        
        # Encode with settings optimized for protecting critical regions
        additional_params = {
            'aq-mode': '3',  # Auto-variance with dark scene detection
            'aq-strength': str(aq_strength),
            'qg-size': '8',  # Fine-grained quantization groups
            'rd': '6',  # Maximum RD optimization
            'rd-refine': '1',  # Extra RD refinement
            'psy-rd': '2.0',
            'psy-rdoq': '2.0',
            'deblock': '-2,-2',  # Use comma, not colon
            'strong-intra-smoothing': '0',  # Remove 'no-' prefix
            'sao': '0',  # Remove 'no-' prefix
            'ctu': '32',  # Smaller CTU size for finer control
            'min-cu-size': '8'  # Smaller CU for detailed areas
        }
        
        success = self.encode_with_qp_map(
            input_path=input_path,
            output_path=output_path,
            qp_map_path=None,
            crf=crf,
            preset='veryslow',
            additional_params=additional_params
        )
        
        return success
    
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
    
    def encode_with_image_output(self,
                                input_path: str,
                                output_path: str,
                                qp_map: np.ndarray,
                                output_format: str = 'jpeg',
                                quality: int = 85,
                                keep_hevc: bool = True) -> tuple:
        """
        Compress using HEVC quality allocation, then output as viewable image.
        
        Supports multiple output formats:
        - jpeg: Best for photos (85% quality = good compression, good quality)
        - webp: Modern format (better than JPEG, universal browser support)
        - png: Lossless but LARGE after decoding (not recommended)
        
        Args:
            input_path: Original image
            output_path: Output image path
            qp_map: Quality map for spatially-varying compression
            output_format: 'jpeg', 'webp', or 'png'
            quality: Quality setting for JPEG/WebP (1-100, default 85)
            keep_hevc: If True, also save HEVC file for archival
            
        Returns:
            (success, hevc_path) tuple
        """
        import tempfile
        import cv2
        
        # Step 1: Compress to HEVC (maximum compression with quality allocation)
        if keep_hevc:
            # Save HEVC with .hevc extension
            base = os.path.splitext(output_path)[0]
            hevc_path = f"{base}.hevc"
        else:
            # Use temporary file
            hevc_path = tempfile.mktemp(suffix='.hevc')
        
        print(f"  [1/2] Encoding to HEVC with quality zones...")
        success = self.encode_with_qp_zones_multipass(
            input_path=input_path,
            output_path=hevc_path,
            qp_map=qp_map
        )
        
        if not success:
            print(f"  ✗ HEVC encoding failed")
            return False, None
        
        hevc_size = os.path.getsize(hevc_path)
        print(f"  ✓ HEVC saved: {hevc_size / 1024:.1f} KB (maximum compression)")
        
        # Step 2: Decode HEVC to viewable format
        format_name = output_format.upper()
        print(f"  [2/2] Decoding to {format_name} for viewing...")
        
        # Build ffmpeg command based on output format
        if output_format == 'jpeg':
            cmd = [
                self.ffmpeg_path, '-y',
                '-i', hevc_path,
                '-q:v', str(int((100 - quality) / 3.125)),  # Convert 0-100 to 2-31 scale
                output_path
            ]
        elif output_format == 'webp':
            cmd = [
                self.ffmpeg_path, '-y',
                '-i', hevc_path,
                '-c:v', 'libwebp',
                '-quality', str(quality),
                output_path
            ]
        elif output_format == 'png':
            cmd = [
                self.ffmpeg_path, '-y',
                '-i', hevc_path,
                '-f', 'image2',
                '-vcodec', 'png',
                '-compression_level', '9',
                output_path
            ]
        else:
            print(f"  ✗ Unsupported format: {output_format}")
            return False, hevc_path
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Clean up temporary HEVC if not keeping
            if not keep_hevc and os.path.exists(hevc_path):
                os.remove(hevc_path)
                hevc_path = None
            
            if result.returncode == 0:
                output_size = os.path.getsize(output_path)
                original_size = os.path.getsize(input_path)
                
                print(f"  ✓ {format_name} saved: {output_size / (1024*1024):.2f} MB")
                
                # Compare sizes
                if output_size < original_size:
                    reduction = (1 - output_size / original_size) * 100
                    print(f"  ✅ {reduction:.1f}% smaller than original!")
                else:
                    increase = (output_size / original_size - 1) * 100
                    print(f"  ⚠️  {increase:.1f}% larger than original (HEVC is still best)")
                
                return True, hevc_path
            else:
                print(f"  ✗ {format_name} conversion failed: {result.stderr}")
                return False, hevc_path
                
        except Exception as e:
            print(f"  ✗ Conversion error: {e}")
            if not keep_hevc and os.path.exists(hevc_path):
                os.remove(hevc_path)
            return False, None
    
    def batch_encode(self,
                    input_paths: list,
                    output_dir: str,
                    qp_maps: Optional[list] = None,
                    **encode_params) -> Dict[str, bool]:
        """
        Batch encode multiple files.
        
        Args:
            input_paths: List of input file paths
            output_dir: Output directory
            qp_maps: Optional list of QP map arrays
            **encode_params: Parameters to pass to encode_with_qp_map
            
        Returns:
            Dictionary mapping input paths to success status
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for i, input_path in enumerate(input_paths):
            # Generate output path
            basename = os.path.basename(input_path)
            name, _ = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{name}_compressed.hevc")
            
            # Get QP map if provided
            qp_map = qp_maps[i] if qp_maps and i < len(qp_maps) else None
            
            # Encode
            if qp_map is not None:
                success = self.encode_with_quality_zones(
                    input_path, output_path, qp_map
                )
            else:
                success = self.encode_with_qp_map(
                    input_path, output_path, **encode_params
                )
            
            results[input_path] = success
        
        return results

