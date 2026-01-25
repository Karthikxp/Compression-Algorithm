"""
SAAC Compressor
Scene-aware compression with intent classification and prominence-based weighting.

Flow:
1. Scene Classification → Intent Rules
2. YOLOv8-seg → Pixel-perfect masks
3. Prominence Check → Size + Location boosts
4. Weight Calculation → Intent + Prominence
5. Saliency → Fill gaps
6. Segmentation → Background adjustment
7. Adaptive Quantization → HEVC encoding
"""

import numpy as np
import cv2
import os
import time
from typing import Optional, Dict, List

from .detectors import ObjectDetector, SaliencyDetector, SemanticSegmentor, SceneClassifier
from .detectors.scene_classifier import ClipSceneClassifier
from .qp_map import QPMapGenerator
from .encoder import HEVCEncoder
from .pixel_compressor import PixelCompressor


class SaacCompressor:
    """
    Saliency-Aware Adaptive Compression (SAAC).
    
    Features:
    - Scene-aware intent classification
    - Prominence-based object boosting
    - Pixel-perfect segmentation masks
    """
    
    def __init__(self,
                 device: str = 'cpu',
                 yolo_model: str = 'yolov8n-seg.pt',
                 saliency_method: str = 'spectral',
                 segmentation_method: str = 'simple',
                 scene_method: str = 'simple',
                 enable_saliency: bool = True,
                 enable_segmentation: bool = True,
                 blend_mode: str = 'priority',
                 compression_mode: str = 'pixel'):
        """
        Initialize the intelligent SAAC compressor.
        
        Args:
            device: 'cuda' or 'cpu'
            yolo_model: YOLO model ('yolov8n-seg.pt' for segmentation)
            saliency_method: 'spectral', 'fine_grained', or 'u2net'
            segmentation_method: 'simple' or 'deeplabv3'
            scene_method: 'simple', 'efficientnet', 'resnet', or 'clip'
            enable_saliency: Enable saliency detection
            enable_segmentation: Enable semantic segmentation
            blend_mode: 'priority' or 'weighted'
            compression_mode: 'pixel' (PNG output) or 'hevc' (HEVC output)
        """
        print("="*60)
        print("Initializing SAAC")
        if compression_mode == 'pixel':
            print("Mode: Pixel-Level Compression (PNG output)")
        else:
            print("Mode: HEVC Encoding")
        print("="*60)
        
        self.device = device
        self.enable_saliency = enable_saliency
        self.enable_segmentation = enable_segmentation
        self.compression_mode = compression_mode
        
        # Initialize components in order
        print("\n[1/5] Loading Scene Classifier...")
        if scene_method == 'clip':
            # Use CLIP for superior semantic understanding
            self.scene_classifier = ClipSceneClassifier(
                model_name='ViT-B/32',  # Fast 149MB model
                device=device
            )
        else:
            # Use traditional methods (simple, efficientnet, resnet)
            self.scene_classifier = SceneClassifier(
                method=scene_method,
                device=device
            )
        
        print("\n[2/5] Loading Object Detector (Segmentation)...")
        self.object_detector = ObjectDetector(
            model_name=yolo_model,
            device=device
        )
        
        if enable_saliency:
            print("\n[3/5] Loading Saliency Detector...")
            self.saliency_detector = SaliencyDetector(
                method=saliency_method,
                device=device
            )
        else:
            self.saliency_detector = None
            print("\n[3/5] Saliency detection disabled")
        
        if enable_segmentation:
            print("\n[4/5] Loading Semantic Segmentor...")
            self.segmentor = SemanticSegmentor(
                method=segmentation_method,
                device=device
            )
        else:
            self.segmentor = None
            print("\n[4/5] Semantic segmentation disabled")
        
        print("\n[5/5] Initializing QP Generator and Encoder...")
        self.qp_generator = QPMapGenerator(
            base_qp=51,
            high_quality_qp=10,
            mid_quality_qp=30,
            blend_mode=blend_mode
        )
        
        # Compression engines
        if compression_mode == 'hevc':
            print("\n[5/5] Loading HEVC Encoder...")
            self.encoder = HEVCEncoder()
            self.pixel_compressor = None
        else:  # pixel mode
            print("\n[5/5] Loading Pixel Compressor...")
            self.pixel_compressor = PixelCompressor()
            self.encoder = None
        
        # Statistics
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
        Compress image using SAAC.
        
        Args:
            input_path: Path to input image
            output_path: Path to output compressed file
            save_visualizations: Save intermediate visualizations
            visualization_dir: Directory for visualizations
            
        Returns:
            Dictionary with compression statistics
        """
        print(f"\n{'='*60}")
        print(f"Compressing: {input_path}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load image
        print("\n[Step 1/7] Loading image...")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        h, w = image.shape[:2]
        print(f"  Resolution: {w}x{h}")
        
        # Check if dimensions need padding
        needs_padding = (h % 2 != 0) or (w % 2 != 0)
        temp_padded_path = None
        
        if needs_padding:
            new_h = h + (h % 2)
            new_w = w + (w % 2)
            print(f"  Note: Padding to {new_w}x{new_h} for HEVC compatibility")
            
            padded_image = cv2.copyMakeBorder(
                image, 0, new_h - h, 0, new_w - w,
                cv2.BORDER_REPLICATE
            )
            
            import tempfile
            temp_padded_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(temp_padded_path, padded_image)
            encoding_input_path = temp_padded_path
        else:
            encoding_input_path = input_path
        
        # Step 1: Scene Classification
        print("\n[Step 2/7] Classifying scene...")
        scene, scene_confidence = self.scene_classifier.classify(image)
        scene_desc = self.scene_classifier.get_scene_description(scene)
        print(f"  Scene: {scene} (confidence: {scene_confidence:.2f})")
        print(f"  → {scene_desc}")
        
        # Step 2: Object Detection with Segmentation
        print("\n[Step 3/7] Detecting objects with segmentation masks...")
        detections = self.object_detector.detect_with_masks(
            image, 
            confidence_threshold=0.25
        )
        print(f"  Found {len(detections)} objects:")
        for det in detections[:5]:  # Show first 5
            print(f"    - {det['class_name']} (confidence: {det['confidence']:.2f}, "
                  f"area: {det['area']/image.size*100:.1f}%)")
        
        # Step 3: Detect saliency (fills gaps)
        saliency_map = None
        if self.enable_saliency and self.saliency_detector:
            print("\n[Step 4/7] Detecting visual saliency...")
            saliency_map = self.saliency_detector.detect(image)
            salient_pixels = np.sum(saliency_map > 0.5)
            print(f"  Salient pixels: {salient_pixels} ({salient_pixels/image.size*100:.1f}%)")
        else:
            print("\n[Step 4/7] Saliency detection skipped")
        
        # Step 4: Semantic segmentation (background)
        segmentation_masks = None
        if self.enable_segmentation and self.segmentor:
            print("\n[Step 5/7] Performing semantic segmentation...")
            segmentation_masks = self.segmentor.segment(image)
            for category, mask in segmentation_masks.items():
                coverage = np.sum(mask > 0) / mask.size * 100
                if coverage > 1.0:
                    print(f"  {category}: {coverage:.1f}%")
        else:
            print("\n[Step 5/7] Semantic segmentation skipped")
        
        # Step 5: Generate Intelligent QP map
        print("\n[Step 6/7] Generating intelligent QP map...")
        qp_map = self.qp_generator.generate(
            image_shape=image.shape,
            scene=scene,
            detections=detections,
            saliency_map=saliency_map,
            segmentation_masks=segmentation_masks
        )
        
        qp_stats = self.qp_generator.get_statistics(qp_map)
        print(f"  Average QP: {qp_stats['mean_qp']:.1f}")
        print(f"  High quality regions: {qp_stats['high_quality_percent']:.1f}%")
        print(f"  Medium quality regions: {qp_stats['medium_quality_percent']:.1f}%")
        print(f"  Low quality regions: {qp_stats['low_quality_percent']:.1f}%")
        
        # Save visualizations
        if save_visualizations:
            vis_dir = visualization_dir or os.path.dirname(output_path)
            os.makedirs(vis_dir, exist_ok=True)
            self._save_visualizations(
                image, detections, saliency_map,
                segmentation_masks, qp_map, vis_dir,
                os.path.basename(input_path), scene
            )
        
        # Step 6: Compress
        print("\n[Step 7/7] Compressing with smart quality allocation...")
        
        output_ext = os.path.splitext(output_path)[1].lower()
        
        try:
            if self.compression_mode == 'pixel':
                # PIXEL MODE: Selectively degrade pixels, save as PNG
                print("  Mode: Pixel-level compression (PNG output)")
                
                # Apply selective degradation based on QP map
                compressed_pixels = self.pixel_compressor.compress_by_qp_map(
                    image=image,
                    qp_map=qp_map,
                    preserve_edges=True
                )
                
                # Calculate complexity reduction
                complexity = self.pixel_compressor.get_complexity_reduction(
                    original=image,
                    compressed=compressed_pixels
                )
                
                print(f"  Complexity reduced:")
                print(f"    Edge density: {complexity['edge_reduction']:.1f}% reduction")
                print(f"    Unique colors: {complexity['color_reduction']:.1f}% reduction")
                
                # Save as PNG with maximum compression
                print(f"  Saving to PNG with maximum compression level...")
                cv2.imwrite(output_path, compressed_pixels, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                hevc_path = None
                success = os.path.exists(output_path)
                
                if success:
                    output_size = os.path.getsize(output_path)
                    original_size = os.path.getsize(input_path)
                    ratio = original_size / output_size if output_size > 0 else 0
                    
                    print(f"  ✓ PNG saved: {output_size / (1024*1024):.2f} MB")
                    if ratio > 1:
                        print(f"  ✅ {(1 - 1/ratio)*100:.1f}% smaller than original!")
                    else:
                        print(f"  ⚠️  Output is larger (try more aggressive compression)")
                
            else:
                # HEVC MODE: Traditional HEVC encoding
                print("  Mode: HEVC encoding")
                
                if output_ext in ['.png', '.jpg', '.jpeg', '.webp']:
                    # Image output: Compress with HEVC, decode to viewable format
                    output_format = {
                        '.png': 'png',
                        '.jpg': 'jpeg',
                        '.jpeg': 'jpeg',
                        '.webp': 'webp'
                    }[output_ext]
                    
                    print(f"  Output format: {output_format.upper()} + HEVC")
                    success, hevc_path = self.encoder.encode_with_image_output(
                        input_path=encoding_input_path,
                        output_path=output_path,
                        qp_map=qp_map,
                        output_format=output_format,
                        quality=85,
                        keep_hevc=True
                    )
                else:
                    # HEVC output: Direct encoding
                    print("  Output format: HEVC")
                    success = self.encoder.encode_with_qp_zones_multipass(
                        input_path=encoding_input_path,
                        output_path=output_path,
                        qp_map=qp_map
                    )
                    hevc_path = output_path
            
            if not success:
                raise RuntimeError("Compression failed")
                
        finally:
            if temp_padded_path and os.path.exists(temp_padded_path):
                os.remove(temp_padded_path)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        # If PNG output, also track HEVC size
        hevc_size = None
        hevc_ratio = None
        if output_ext == '.png' and hevc_path and os.path.exists(hevc_path):
            hevc_size = os.path.getsize(hevc_path)
            hevc_ratio = original_size / hevc_size if hevc_size > 0 else 0
        
        stats = {
            'input_path': input_path,
            'output_path': output_path,
            'resolution': (w, h),
            'scene': scene,
            'scene_confidence': scene_confidence,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'space_saved_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0,
            'processing_time_seconds': elapsed_time,
            'detections': len(detections),
            'qp_statistics': qp_stats,
            'hevc_path': hevc_path,
            'hevc_size_bytes': hevc_size,
            'hevc_size_mb': hevc_size / (1024 * 1024) if hevc_size else None,
            'hevc_compression_ratio': hevc_ratio,
            'hevc_space_saved_percent': (1 - 1/hevc_ratio) * 100 if hevc_ratio and hevc_ratio > 0 else None
        }
        
        self.last_stats = stats
        
        # Print summary
        print(f"\n{'='*60}")
        print("COMPRESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Scene type:         {scene}")
        print(f"Objects detected:   {len(detections)}")
        print(f"Original size:      {stats['original_size_mb']:.2f} MB")
        
        if output_ext == '.png' and stats['hevc_size_mb']:
            # PNG output - show both formats
            print(f"\n📦 Dual Output:")
            print(f"  PNG (compatible):  {stats['compressed_size_mb']:.2f} MB ({compression_ratio:.2f}x smaller)")
            print(f"  HEVC (archival):   {stats['hevc_size_mb']:.2f} MB ({stats['hevc_compression_ratio']:.2f}x smaller)")
            print(f"\n💡 PNG is {stats['space_saved_percent']:.1f}% smaller than original")
            print(f"   (Contains already-compressed pixel data)")
        else:
            # HEVC only
            print(f"Compressed size:    {stats['compressed_size_mb']:.2f} MB")
            print(f"Compression ratio:  {compression_ratio:.2f}x")
            print(f"Space saved:        {stats['space_saved_percent']:.1f}%")
        
        print(f"\nProcessing time:    {elapsed_time:.1f}s")
        print(f"{'='*60}\n")
        
        return stats
    
    def _save_visualizations(self,
                            image: np.ndarray,
                            detections: List[Dict],
                            saliency_map: Optional[np.ndarray],
                            segmentation_masks: Optional[Dict],
                            qp_map: np.ndarray,
                            output_dir: str,
                            base_name: str,
                            scene: str):
        """Save visualization images."""
        print("\n  Saving visualizations...")
        
        # Detection visualization with segmentation masks
        det_vis = self.object_detector.visualize_detections(image, detections, show_masks=True)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_detections.jpg"), det_vis)
        
        # Prominence visualization
        from .detectors.prominence import ProminenceCalculator
        prom_calc = ProminenceCalculator()
        prom_vis = prom_calc.visualize_prominence(image, detections)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_prominence.jpg"), prom_vis)
        
        # Saliency map
        if saliency_map is not None:
            sal_vis = (saliency_map * 255).astype(np.uint8)
            sal_vis = cv2.applyColorMap(sal_vis, cv2.COLORMAP_HOT)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_saliency.jpg"), sal_vis)
        
        # QP map visualization
        qp_vis = self.qp_generator.visualize_qp_map(qp_map)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_qp_map.jpg"), qp_vis)
        
        # Scene info overlay
        info_vis = image.copy()
        cv2.putText(info_vis, f"Scene: {scene}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_scene.jpg"), info_vis)
        
        print(f"  ✓ Visualizations saved to {output_dir}")
    
    def get_last_stats(self) -> Dict:
        """Get statistics from the last compression."""
        return self.last_stats

