#!/usr/bin/env python3
"""
Comparison script to benchmark different detection methods.
Compares: Hybrid (fast) vs Deep Learning (accurate) pipelines.
"""

import sys
import os
import time
import cv2
from saac import SaacCompressor


def compare_methods(input_image: str):
    """
    Compare compression results using different detection methods.
    
    Args:
        input_image: Path to input image
    """
    
    if not os.path.exists(input_image):
        print(f"Error: File not found: {input_image}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    
    # Read image for info
    img = cv2.imread(input_image)
    h, w = img.shape[:2]
    
    print("="*80)
    print("SAAC METHOD COMPARISON")
    print("="*80)
    print(f"Input: {input_image}")
    print(f"Resolution: {w}x{h}")
    print(f"Size: {os.path.getsize(input_image) / (1024*1024):.2f} MB")
    print("="*80)
    
    # Configuration for different methods
    methods = [
        {
            'name': 'Hybrid (Fast)',
            'description': 'YOLOv8-nano + Spectral Residual + Color-based',
            'config': {
                'yolo_model': 'yolov8n.pt',
                'saliency_method': 'spectral',
                'segmentation_method': 'simple',
                'device': 'cpu'
            },
            'output': f'{base_name}_hybrid.hevc',
            'vis_dir': f'visualizations_hybrid'
        },
        {
            'name': 'Deep Learning (Accurate)',
            'description': 'YOLOv8-nano + U2-Net + DeepLabV3',
            'config': {
                'yolo_model': 'yolov8n.pt',
                'saliency_method': 'u2net',
                'segmentation_method': 'deeplabv3',
                'device': 'cpu'  # Change to 'cuda' if GPU available
            },
            'output': f'{base_name}_deep.hevc',
            'vis_dir': f'visualizations_deep'
        },
        {
            'name': 'Minimal (Ultra Fast)',
            'description': 'YOLOv8-nano only (no saliency, no segmentation)',
            'config': {
                'yolo_model': 'yolov8n.pt',
                'saliency_method': 'spectral',
                'segmentation_method': 'simple',
                'device': 'cpu',
                'enable_saliency': False,
                'enable_segmentation': False
            },
            'output': f'{base_name}_minimal.hevc',
            'vis_dir': f'visualizations_minimal'
        }
    ]
    
    results = []
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing: {method['name']}")
        print(f"Description: {method['description']}")
        print(f"{'='*80}")
        
        try:
            # Create compressor
            start_time = time.time()
            compressor = SaacCompressor(**method['config'])
            
            # Compress
            stats = compressor.compress_image(
                input_path=input_image,
                output_path=method['output'],
                save_visualizations=True,
                visualization_dir=method['vis_dir']
            )
            
            total_time = time.time() - start_time
            
            # Store results
            results.append({
                'name': method['name'],
                'time': total_time,
                'output_size': stats['compressed_size_mb'],
                'ratio': stats['compression_ratio'],
                'space_saved': stats['space_saved_percent'],
                'output_path': method['output']
            })
            
            print(f"\n✓ {method['name']} completed in {total_time:.1f}s")
            
        except Exception as e:
            print(f"\n✗ {method['name']} failed: {e}")
            results.append({
                'name': method['name'],
                'error': str(e)
            })
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Time (s)':<12} {'Size (MB)':<12} {'Ratio':<10} {'Saved %':<10}")
    print("-"*80)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['name']:<25} {result['time']:<12.1f} "
                  f"{result['output_size']:<12.2f} {result['ratio']:<10.2f} "
                  f"{result['space_saved']:<10.1f}")
        else:
            print(f"{result['name']:<25} FAILED: {result['error']}")
    
    print("="*80)
    
    # Summary
    successful = [r for r in results if 'error' not in r]
    if len(successful) > 0:
        fastest = min(successful, key=lambda x: x['time'])
        best_ratio = max(successful, key=lambda x: x['ratio'])
        
        print(f"\n🏆 Fastest: {fastest['name']} ({fastest['time']:.1f}s)")
        print(f"🏆 Best Compression: {best_ratio['name']} ({best_ratio['ratio']:.2f}x)")
    
    print(f"\n{'='*80}")
    print("Visualizations saved to:")
    for method in methods:
        if os.path.exists(method['vis_dir']):
            print(f"  - {method['vis_dir']}/")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_methods.py <input_image>")
        print("\nExample:")
        print("  python3 compare_methods.py car.jpg")
        sys.exit(1)
    
    input_image = sys.argv[1]
    compare_methods(input_image)


if __name__ == '__main__':
    main()


