import numpy as np
import cv2
import os
import time
from typing import Optional, Dict, List

from .detectors import ObjectDetector, SaliencyDetector, SemanticSegmentor, SceneClassifier
from .detectors.scene_classifier import ClipSceneClassifier
from .qp_map import QPMapGenerator
from .avif_encoder import AVIFEncoder


class SaacCompressor:
    
    def __init__(self,
                 device: str = 'cpu',
                 yolo_model: str = 'yolov8n-seg.pt',
                 saliency_method: str = 'spectral',
                 segmentation_method: str = 'simple',
                 scene_method: str = 'simple',
                 enable_saliency: bool = True,
                 enable_segmentation: bool = True,
                 blend_mode: str = 'priority'):
        
        print("="*60)
        print("Initializing SAAC - AVIF Encoding")
        print("="*60)
        
        self.device = device
        self.enable_saliency = enable_saliency
        self.enable_segmentation = enable_segmentation
        
        print("\n[1/5] Loading Scene Classifier...")
        if scene_method == 'clip':
            self.scene_classifier = ClipSceneClassifier(
                model_name='ViT-B/32',
                device=device
            )
        else:
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
        
        print("\n[5/5] Initializing QP Generator and AVIF Encoder...")
        self.qp_generator = QPMapGenerator(
            base_qp=51,
            high_quality_qp=10,
            mid_quality_qp=30,
            blend_mode=blend_mode
        )
        
        self.avif_encoder = AVIFEncoder()
        
        self.last_stats = {}
        
        print("\n" + "="*60)
        print("✓ SAAC Ready!")
        print("="*60 + "\n")
    
    def compress_image(self,
                      input_path: str,
                      output_path: str,
                      save_visualizations: bool = False,
                      visualization_dir: Optional[str] = None) -> Dict[str, any]:
        
        print(f"\n{'='*60}")
        print(f"Compressing: {input_path}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        print("\n[Step 1/7] Loading image...")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        h, w = image.shape[:2]
        print(f"  Resolution: {w}x{h}")
        
        print("\n[Step 2/7] Classifying scene...")
        scene, scene_confidence = self.scene_classifier.classify(image)
        scene_desc = self.scene_classifier.get_scene_description(scene)
        print(f"  Scene: {scene} (confidence: {scene_confidence:.2f})")
        print(f"  → {scene_desc}")
        
        print("\n[Step 3/7] Detecting objects with segmentation masks...")
        detections = self.object_detector.detect_with_masks(
            image, 
            confidence_threshold=0.25
        )
        print(f"  Found {len(detections)} objects:")
        for det in detections[:5]:
            print(f"    - {det['class_name']} (confidence: {det['confidence']:.2f}, "
                  f"area: {det['area']/image.size*100:.1f}%)")
        
        saliency_map = None
        if self.enable_saliency and self.saliency_detector:
            print("\n[Step 4/7] Detecting visual saliency...")
            saliency_map = self.saliency_detector.detect(image)
            salient_pixels = np.sum(saliency_map > 0.5)
            print(f"  Salient pixels: {salient_pixels} ({salient_pixels/image.size*100:.1f}%)")
        else:
            print("\n[Step 4/7] Saliency detection skipped")
        
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
        
        if save_visualizations:
            vis_dir = visualization_dir or os.path.dirname(output_path)
            os.makedirs(vis_dir, exist_ok=True)
            self._save_visualizations(
                image, detections, saliency_map,
                segmentation_masks, qp_map, vis_dir,
                os.path.basename(input_path), scene
            )
        
        print("\n[Step 7/7] Compressing with smart quality allocation...")
        
        print("  Mode: AVIF encoding (AV1 codec)")
        
        success = self.avif_encoder.encode_with_quality_zones(
            input_path=input_path,
            output_path=output_path,
            qp_map=qp_map
        )
        
        if not success or not os.path.exists(output_path):
            raise RuntimeError("Compression failed")
        
        output_size = os.path.getsize(output_path)
        original_size = os.path.getsize(input_path)
        ratio = original_size / output_size if output_size > 0 else 0
        
        print(f"  ✓ AVIF saved: {output_size / (1024*1024):.2f} MB")
        print(f"  ✓ Compression ratio: {ratio:.2f}x")
        if ratio > 1:
            print(f"  ✅ {(1 - 1/ratio)*100:.1f}% smaller than original!")
        
        elapsed_time = time.time() - start_time
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
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
            'qp_statistics': qp_stats
        }
        
        self.last_stats = stats
        
        print(f"\n{'='*60}")
        print("COMPRESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Scene type:         {scene}")
        print(f"Objects detected:   {len(detections)}")
        print(f"Original size:      {stats['original_size_mb']:.2f} MB")
        print(f"Compressed size:    {stats['compressed_size_mb']:.2f} MB")
        print(f"Compression ratio:  {compression_ratio:.2f}x")
        print(f"Space saved:        {stats['space_saved_percent']:.1f}%")
        print(f"Processing time:    {elapsed_time:.1f}s")
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
        
        print("\n  Saving visualizations...")
        
        det_vis = self.object_detector.visualize_detections(image, detections, show_masks=True)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_detections.jpg"), det_vis)
        
        from .detectors.prominence import ProminenceCalculator
        prom_calc = ProminenceCalculator()
        prom_vis = prom_calc.visualize_prominence(image, detections)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_prominence.jpg"), prom_vis)
        
        if saliency_map is not None:
            sal_vis = (saliency_map * 255).astype(np.uint8)
            sal_vis = cv2.applyColorMap(sal_vis, cv2.COLORMAP_HOT)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_saliency.jpg"), sal_vis)
        
        qp_vis = self.qp_generator.visualize_qp_map(qp_map)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_qp_map.jpg"), qp_vis)
        
        info_vis = image.copy()
        cv2.putText(info_vis, f"Scene: {scene}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_scene.jpg"), info_vis)
        
        print(f"  ✓ Visualizations saved to {output_dir}")
    
    def get_last_stats(self) -> Dict:
        return self.last_stats

