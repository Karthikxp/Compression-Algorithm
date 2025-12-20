"""
Main SAAC Compressor
Integrates all detection layers and encoding to perform saliency-aware compression.
"""

import numpy as np
import cv2
import os
import time
from typing import Optional, Dict, Tuple
import tempfile

from .detectors import ObjectDetector, SaliencyDetector, SemanticSegmentor
from .qp_map import QPMapGenerator
from .encoder import HEVCEncoder


class SaacCompressor:
    """
    Saliency-Aware Adaptive Compression (SAAC) main class.
    
    Combines object detection, saliency detection, and semantic segmentation
    to create a variable-quality compression map, then encodes using HEVC.
    """
    
    def __init__(self,
                 device: str = 'cpu',
                 yolo_model: str = 'yolov8n.pt',
                 saliency_method: str = 'spectral',
                 segmentation_method: str = 'simple',
                 person_quality: int = 10,
                 saliency_quality: int = 25,
                 background_quality: int = 51,
                 enable_saliency: bool = True,
                 enable_segmentation: bool = True,
                 blend_mode: str = 'priority'):
        """
        Initialize the SAAC compressor.
        
        Args:
            device: 'cuda' or 'cpu'
            yolo_model: YOLO model name (e.g., 'yolov8n.pt' for nano)
            saliency_method: 'spectral', 'fine_grained', or 'u2net'
            segmentation_method: 'simple' or 'deeplabv3'
            person_quality: QP for people/objects (0-51, lower=better, default: 10)
            saliency_quality: QP for salient regions (default: 25)
            background_quality: QP for background (default: 51)
            enable_saliency: Enable saliency detection layer
            enable_segmentation: Enable segmentation layer
            blend_mode: 'priority' or 'weighted'
        """
        print("="*60)
        print("Initializing SAAC (Saliency-Aware Adaptive Compression)")
        print("="*60)
        
        self.device = device
        self.enable_saliency = enable_saliency
        self.enable_segmentation = enable_segmentation
        
        # Initialize detectors
        print("\n[1/4] Loading Object Detector...")
        self.object_detector = ObjectDetector(
            model_name=yolo_model,
            device=device
        )
        
        if enable_saliency:
            print("\n[2/4] Loading Saliency Detector...")
            self.saliency_detector = SaliencyDetector(
                method=saliency_method,
                device=device
            )
        else:
            self.saliency_detector = None
            print("\n[2/4] Saliency detection disabled")
        
        if enable_segmentation:
            print("\n[3/4] Loading Semantic Segmentor...")
            self.segmentor = SemanticSegmentor(
                method=segmentation_method,
                device=device
            )
        else:
            self.segmentor = None
            print("\n[3/4] Semantic segmentation disabled")
        
        print("\n[4/4] Initializing QP Map Generator and Encoder...")
        self.qp_generator = QPMapGenerator(
            person_qp=person_quality,
            saliency_qp=saliency_quality,
            background_qp=background_quality,
            blend_mode=blend_mode
        )
        
        self.encoder = HEVCEncoder()
        
        # Statistics from last compression
        self.last_stats = {}
        
        print("\n" + "="*60)
        print("✓ SAAC Ready!")
        print("="*60 + "\n")
    
    def compress_image(self,
                      input_path: str,
                      output_path: str,
                      save_visualizations: bool = False,
                      visualization_dir: Optional[str] = None) -> Dict[str, any]:
        """
        Compress a single image using SAAC.
        
        Args:
            input_path: Path to input image
            output_path: Path to output compressed file
            save_visualizations: Save intermediate visualizations
            visualization_dir: Directory for visualizations (default: same as output)
            
        Returns:
            Dictionary with compression statistics
        """
        print(f"\n{'='*60}")
        print(f"Compressing: {input_path}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load image
        print("\n[Step 1/6] Loading image...")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        h, w = image.shape[:2]
        print(f"  Resolution: {w}x{h}")
        
        # Check if dimensions need padding for HEVC encoding
        needs_padding = (h % 2 != 0) or (w % 2 != 0)
        temp_padded_path = None
        
        if needs_padding:
            # Calculate padded dimensions (make even)
            new_h = h + (h % 2)
            new_w = w + (w % 2)
            print(f"  Note: Padding to {new_w}x{new_h} for HEVC compatibility")
            
            # Pad the image
            padded_image = cv2.copyMakeBorder(
                image, 
                0, new_h - h,  # top, bottom
                0, new_w - w,  # left, right
                cv2.BORDER_REPLICATE
            )
            
            # Save temporary padded image
            temp_padded_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(temp_padded_path, padded_image)
            encoding_input_path = temp_padded_path
        else:
            encoding_input_path = input_path
        
        # Detect objects (Layer 1: Must-Have)
        print("\n[Step 2/6] Detecting objects (people, vehicles, etc.)...")
        object_mask = self.object_detector.detect_with_expansion(
            image, 
            confidence_threshold=0.25,
            expansion_percent=15.0
        )
        detections = self.object_detector.get_detection_info(image)
        print(f"  Found {len(detections)} objects:")
        for det in detections[:5]:  # Show first 5
            print(f"    - {det['class_name']} (confidence: {det['confidence']:.2f})")
        
        # Detect saliency (Layer 2: Eye-Catcher)
        saliency_map = None
        if self.enable_saliency and self.saliency_detector:
            print("\n[Step 3/6] Detecting visual saliency...")
            saliency_map = self.saliency_detector.detect(image)
            salient_pixels = np.sum(saliency_map > 0.5)
            print(f"  Salient pixels: {salient_pixels} ({salient_pixels/image.size*100:.1f}%)")
        else:
            print("\n[Step 3/6] Saliency detection skipped")
        
        # Segment semantics (Layer 3: Background)
        segmentation_masks = None
        if self.enable_segmentation and self.segmentor:
            print("\n[Step 4/6] Performing semantic segmentation...")
            segmentation_masks = self.segmentor.segment(image)
            for category, mask in segmentation_masks.items():
                coverage = np.sum(mask > 0) / mask.size * 100
                if coverage > 1.0:  # Only show significant categories
                    print(f"  {category}: {coverage:.1f}%")
        else:
            print("\n[Step 4/6] Semantic segmentation skipped")
        
        # Generate QP map
        print("\n[Step 5/6] Generating QP map...")
        qp_map = self.qp_generator.generate(
            image_shape=image.shape,
            object_mask=object_mask,
            saliency_map=saliency_map,
            segmentation_masks=segmentation_masks
        )
        
        qp_stats = self.qp_generator.get_statistics(qp_map)
        print(f"  Average QP: {qp_stats['mean_qp']:.1f}")
        print(f"  High quality regions: {qp_stats['high_quality_percent']:.1f}%")
        print(f"  Medium quality regions: {qp_stats['medium_quality_percent']:.1f}%")
        print(f"  Low quality regions: {qp_stats['low_quality_percent']:.1f}%")
        
        # Save visualizations if requested
        if save_visualizations:
            vis_dir = visualization_dir or os.path.dirname(output_path)
            os.makedirs(vis_dir, exist_ok=True)
            self._save_visualizations(
                image, object_mask, saliency_map, 
                segmentation_masks, qp_map, vis_dir,
                os.path.basename(input_path)
            )
        
        # Encode with HEVC
        print("\n[Step 6/6] Encoding with HEVC...")
        try:
            success = self.encoder.encode_with_quality_zones(
                input_path=encoding_input_path,
                output_path=output_path,
                qp_map=qp_map
            )
            
            if not success:
                raise RuntimeError("Encoding failed")
        finally:
            # Clean up temporary padded image if it was created
            if temp_padded_path and os.path.exists(temp_padded_path):
                os.remove(temp_padded_path)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        stats = {
            'input_path': input_path,
            'output_path': output_path,
            'resolution': (w, h),
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'space_saved_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0,
            'processing_time_seconds': elapsed_time,
            'detections': len(detections),
            'qp_statistics': qp_stats
        }
        
        self.last_stats = stats
        
        # Print summary
        print(f"\n{'='*60}")
        print("COMPRESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Original size:      {stats['original_size_mb']:.2f} MB")
        print(f"Compressed size:    {stats['compressed_size_mb']:.2f} MB")
        print(f"Compression ratio:  {compression_ratio:.2f}x")
        print(f"Space saved:        {stats['space_saved_percent']:.1f}%")
        print(f"Processing time:    {elapsed_time:.1f}s")
        print(f"{'='*60}\n")
        
        return stats
    
    def compress_batch(self,
                      input_paths: list,
                      output_dir: str,
                      save_visualizations: bool = False) -> list:
        """
        Compress multiple images.
        
        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            save_visualizations: Save visualizations for each image
            
        Returns:
            List of statistics dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, input_path in enumerate(input_paths):
            print(f"\n{'#'*60}")
            print(f"Processing {i+1}/{len(input_paths)}")
            print(f"{'#'*60}")
            
            # Generate output path
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{name}_saac.hevc")
            
            try:
                stats = self.compress_image(
                    input_path=input_path,
                    output_path=output_path,
                    save_visualizations=save_visualizations,
                    visualization_dir=os.path.join(output_dir, 'visualizations')
                )
                results.append(stats)
            except Exception as e:
                print(f"✗ Error processing {input_path}: {e}")
                results.append({'input_path': input_path, 'error': str(e)})
        
        # Print batch summary
        self._print_batch_summary(results)
        
        return results
    
    def _save_visualizations(self,
                            image: np.ndarray,
                            object_mask: np.ndarray,
                            saliency_map: Optional[np.ndarray],
                            segmentation_masks: Optional[Dict],
                            qp_map: np.ndarray,
                            output_dir: str,
                            base_name: str):
        """Save visualization images."""
        print("\n  Saving visualizations...")
        
        # Object detection overlay
        obj_vis = image.copy()
        obj_vis[object_mask > 0] = obj_vis[object_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_objects.jpg"), obj_vis)
        
        # Saliency map
        if saliency_map is not None:
            sal_vis = (saliency_map * 255).astype(np.uint8)
            sal_vis = cv2.applyColorMap(sal_vis, cv2.COLORMAP_HOT)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_saliency.jpg"), sal_vis)
        
        # QP map visualization
        qp_vis = self.qp_generator.visualize_qp_map(qp_map)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_qp_map.jpg"), qp_vis)
        
        print(f"  ✓ Visualizations saved to {output_dir}")
    
    def _print_batch_summary(self, results: list):
        """Print summary of batch compression."""
        successful = [r for r in results if 'error' not in r]
        failed = len(results) - len(successful)
        
        if len(successful) == 0:
            print("\n✗ All compressions failed")
            return
        
        total_original = sum(r['original_size_mb'] for r in successful)
        total_compressed = sum(r['compressed_size_mb'] for r in successful)
        avg_ratio = sum(r['compression_ratio'] for r in successful) / len(successful)
        total_time = sum(r['processing_time_seconds'] for r in successful)
        
        print(f"\n{'='*60}")
        print("BATCH COMPRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed:  {len(successful)}/{len(results)}")
        if failed > 0:
            print(f"Failed:                  {failed}")
        print(f"Total original size:     {total_original:.2f} MB")
        print(f"Total compressed size:   {total_compressed:.2f} MB")
        print(f"Average compression:     {avg_ratio:.2f}x")
        print(f"Total space saved:       {total_original - total_compressed:.2f} MB")
        print(f"Total processing time:   {total_time:.1f}s")
        print(f"{'='*60}\n")
    
    def get_last_stats(self) -> Dict:
        """Get statistics from the last compression."""
        return self.last_stats

