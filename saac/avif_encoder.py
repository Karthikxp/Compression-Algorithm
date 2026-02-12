
import subprocess
import os
import numpy as np
from typing import Optional, Dict
import tempfile


class AVIFEncoder:
    
    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        self.ffmpeg_path = ffmpeg_path
        
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg with libaom-av1 or libsvtav1 support.")
        
        print(f"✓ AVIFEncoder initialized")
    
    def _check_ffmpeg(self) -> bool:
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False
            
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
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def encode_to_avif(self,
                      input_path: str,
                      output_path: str,
                      crf: int = 30,
                      speed: int = 6,
                      additional_params: Optional[Dict[str, str]] = None) -> bool:
        if not output_path.endswith('.avif'):
            output_path = os.path.splitext(output_path)[0] + '.avif'
        
        cmd = [
            self.ffmpeg_path,
            '-y',
            '-i', input_path,
            '-c:v', 'libaom-av1',
            '-crf', str(crf),
            '-cpu-used', str(speed),
            '-row-mt', '1',
            '-threads', '0',
            '-pix_fmt', 'yuv420p',
        ]
        
        aom_params = []
        
        if additional_params:
            for key, value in additional_params.items():
                aom_params.append(f"{key}={value}")
        
        if aom_params:
            cmd.extend(['-aom-params', ':'.join(aom_params)])
        
        cmd.append(output_path)
        
        try:
            print(f"  Encoding: {input_path} -> {output_path}")
            print(f"  Settings: CRF={crf}, Speed={speed}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"  ✓ AVIF encoding successful")
                return True
            else:
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
        import cv2
        
        img = cv2.imread(input_path)
        if img is None:
            print(f"  ✗ Could not load image: {input_path}")
            return False
        
        h, w = img.shape[:2]
        
        min_qp = int(np.min(qp_map))
        max_qp = int(np.max(qp_map))
        median_qp = int(np.median(qp_map))
        
        person_pixels = np.sum(qp_map <= 10)
        person_percent = (person_pixels / qp_map.size) * 100
        
        background_pixels = np.sum(qp_map >= 35)
        background_percent = (background_pixels / qp_map.size) * 100
        
        print(f"  QP Map Stats: min={min_qp}, median={median_qp}, max={max_qp}")
        print(f"  Person regions (QP≤10): {person_percent:.1f}%")
        print(f"  Background regions (QP≥35): {background_percent:.1f}%")
        
        print(f"  [1/2] ⚠️  APPLYING QP MAP BLUEPRINT TO IMAGE PIXELS...")
        degraded_img = self._apply_qp_map_to_pixels(img, qp_map)
        
        temp_degraded = tempfile.mktemp(suffix='.png')
        cv2.imwrite(temp_degraded, degraded_img)
        
        print(f"  [2/2] Encoding pre-degraded image to AVIF...")
        
        crf = 23
        aq_mode = 1
        aq_strength = 1.0
        
        print(f"  Strategy: QP MAP ENFORCED VIA PIXEL DEGRADATION")
        print(f"  ✓ Persons (QP 0-10): Pristine pixels preserved")
        print(f"  ✗ Background (QP 35-51): Heavily degraded pixels")
        
        additional_params = {
            'aq-mode': str(aq_mode),
            'tune': 'ssim',
            'enable-cdef': '1',
            'enable-restoration': '1',
        }
        
        success = self.encode_to_avif(
            input_path=temp_degraded,
            output_path=output_path,
            crf=crf,
            speed=4,
            additional_params=additional_params
        )
        
        if os.path.exists(temp_degraded):
            os.remove(temp_degraded)
        
        return success
    
    def _apply_qp_map_to_pixels(self, image: np.ndarray, qp_map: np.ndarray) -> np.ndarray:
        import cv2
        
        h, w = image.shape[:2]
        
        if qp_map.shape[0] != h or qp_map.shape[1] != w:
            print(f"        ⚠️  QP map shape {qp_map.shape} != image shape ({h}, {w}), resizing...")
            qp_map = cv2.resize(qp_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
        output = image.copy().astype(np.float32)
        
        print(f"      Applying QP-based pixel degradation:")
        
        mask_pristine = (qp_map <= 10).astype(np.uint8)
        pristine_pct = np.sum(mask_pristine > 0) / (h * w) * 100
        if pristine_pct > 0:
            print(f"        QP 0-10:  {pristine_pct:5.1f}% → PRISTINE (no degradation)")
        
        mask_slight = ((qp_map > 10) & (qp_map <= 20)).astype(np.uint8)
        slight_pct = np.sum(mask_slight > 0) / (h * w) * 100
        if slight_pct > 0:
            print(f"        QP 11-20: {slight_pct:5.1f}% → Slight blur (3x3)")
            blurred = cv2.GaussianBlur(output, (3, 3), 0)
            mask_3ch = cv2.merge([mask_slight, mask_slight, mask_slight])
            output = np.where(mask_3ch > 0, blurred, output)
        
        mask_moderate = ((qp_map > 20) & (qp_map <= 30)).astype(np.uint8)
        moderate_pct = np.sum(mask_moderate > 0) / (h * w) * 100
        if moderate_pct > 0:
            print(f"        QP 21-30: {moderate_pct:5.1f}% → Moderate blur (7x7) + 128 colors")
            blurred = cv2.GaussianBlur(output, (7, 7), 0)
            quantized = (blurred // 2) * 2
            mask_3ch = cv2.merge([mask_moderate, mask_moderate, mask_moderate])
            output = np.where(mask_3ch > 0, quantized, output)
        
        mask_heavy = ((qp_map > 30) & (qp_map <= 40)).astype(np.uint8)
        heavy_pct = np.sum(mask_heavy > 0) / (h * w) * 100
        if heavy_pct > 0:
            print(f"        QP 31-40: {heavy_pct:5.1f}% → HEAVY blur (15x15) + 64 colors")
            blurred = cv2.GaussianBlur(output, (15, 15), 0)
            quantized = (blurred // 4) * 4
            mask_3ch = cv2.merge([mask_heavy, mask_heavy, mask_heavy])
            output = np.where(mask_3ch > 0, quantized, output)
        
        mask_max = (qp_map > 40).astype(np.uint8)
        max_pct = np.sum(mask_max > 0) / (h * w) * 100
        if max_pct > 0:
            print(f"        QP 41-51: {max_pct:5.1f}% → MAXIMUM blur (31x31) + 32 colors (UNIDENTIFIABLE)")
            blurred = cv2.GaussianBlur(output, (31, 31), 0)
            quantized = (blurred // 8) * 8
            mask_3ch = cv2.merge([mask_max, mask_max, mask_max])
            output = np.where(mask_3ch > 0, quantized, output)
        
        return output.astype(np.uint8)
    
    def get_file_size(self, file_path: str) -> int:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return 0
    
    def get_compression_ratio(self, input_path: str, output_path: str) -> float:
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
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for i, input_path in enumerate(input_paths):
            basename = os.path.basename(input_path)
            name, _ = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{name}_compressed.avif")
            
            qp_map = qp_maps[i] if qp_maps and i < len(qp_maps) else None
            
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
