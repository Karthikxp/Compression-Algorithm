#!/usr/bin/env python3
"""
Advanced usage examples for SAAC
"""

import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from saac import SaacCompressor


def example_batch_compression():
    """Example: Compress multiple images in a directory."""
    print("="*60)
    print("Example 1: Batch Compression")
    print("="*60)
    
    compressor = SaacCompressor(
        device='cpu',
        person_quality=10,
        saliency_quality=25,
        background_quality=51
    )
    
    # Get all images from a directory
    input_dir = 'images'  # Replace with your directory
    output_dir = 'compressed'
    
    if os.path.exists(input_dir):
        image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
        image_paths.extend(glob.glob(os.path.join(input_dir, '*.png')))
        
        if image_paths:
            print(f"\nFound {len(image_paths)} images")
            
            results = compressor.compress_batch(
                input_paths=image_paths,
                output_dir=output_dir,
                save_visualizations=True
            )
            
            print(f"\nProcessed {len(results)} images")
        else:
            print(f"\nNo images found in {input_dir}")
    else:
        print(f"\nDirectory not found: {input_dir}")
        print("Create the directory and add some images, then run again.")


def example_security_camera():
    """Example: Security camera optimization (people only)."""
    print("\n" + "="*60)
    print("Example 2: Security Camera Mode")
    print("="*60)
    
    # For security cameras: prioritize people detection, aggressive background compression
    compressor = SaacCompressor(
        device='cpu',
        yolo_model='yolov8n.pt',
        person_quality=8,       # Near-lossless for people
        saliency_quality=40,    # Low priority for other salient regions
        background_quality=51,  # Maximum compression for background
        enable_saliency=False,  # Disable saliency (faster)
        enable_segmentation=True
    )
    
    input_image = 'security_feed.jpg'
    output_image = 'security_compressed.hevc'
    
    if os.path.exists(input_image):
        stats = compressor.compress_image(
            input_path=input_image,
            output_path=output_image,
            save_visualizations=True
        )
        
        print(f"\nSecurity feed compression:")
        print(f"  People detected: {stats['detections']}")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")
    else:
        print(f"\nPlace a security camera image at: {input_image}")


def example_photo_storage():
    """Example: Photo storage optimization (balanced quality)."""
    print("\n" + "="*60)
    print("Example 3: Photo Storage Mode")
    print("="*60)
    
    # For photo storage: balanced compression, preserve important details
    compressor = SaacCompressor(
        device='cpu',
        person_quality=12,      # High quality for people
        saliency_quality=20,    # High quality for salient regions
        background_quality=45,  # Moderate compression for background
        enable_saliency=True,
        enable_segmentation=True,
        blend_mode='weighted'   # Smooth transitions
    )
    
    input_image = 'family_photo.jpg'
    output_image = 'family_compressed.hevc'
    
    if os.path.exists(input_image):
        stats = compressor.compress_image(
            input_path=input_image,
            output_path=output_image,
            save_visualizations=True
        )
        
        print(f"\nPhoto storage compression:")
        print(f"  Original: {stats['original_size_mb']:.2f} MB")
        print(f"  Compressed: {stats['compressed_size_mb']:.2f} MB")
        print(f"  Quality preserved: {stats['qp_statistics']['high_quality_percent']:.1f}%")
    else:
        print(f"\nPlace a family photo at: {input_image}")


def example_extreme_compression():
    """Example: Extreme compression for archival."""
    print("\n" + "="*60)
    print("Example 4: Extreme Compression Mode")
    print("="*60)
    
    # Maximum compression while preserving critical content
    compressor = SaacCompressor(
        device='cpu',
        person_quality=15,      # Moderate quality for people
        saliency_quality=35,    # Low-medium quality for salient regions
        background_quality=51,  # Maximum compression for background
        enable_saliency=True,
        enable_segmentation=True
    )
    
    input_image = 'archive_photo.jpg'
    output_image = 'archive_compressed.hevc'
    
    if os.path.exists(input_image):
        stats = compressor.compress_image(
            input_path=input_image,
            output_path=output_image
        )
        
        print(f"\nExtreme compression:")
        print(f"  Achieved: {stats['compression_ratio']:.2f}x compression")
        print(f"  Space saved: {stats['space_saved_percent']:.1f}%")
    else:
        print(f"\nPlace an image at: {input_image}")


def main():
    """Run all examples."""
    print("\nSAAC - Advanced Usage Examples")
    print("="*60)
    print("\nThese examples demonstrate different use cases:")
    print("1. Batch compression of multiple images")
    print("2. Security camera optimization")
    print("3. Photo storage optimization")
    print("4. Extreme compression for archival")
    print("\n")
    
    # Uncomment the examples you want to run:
    
    # example_batch_compression()
    # example_security_camera()
    # example_photo_storage()
    # example_extreme_compression()
    
    print("\nTo run examples, uncomment them in the main() function")
    print("and provide appropriate input images.\n")


if __name__ == '__main__':
    main()

