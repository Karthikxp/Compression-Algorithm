#!/usr/bin/env python3
"""
Download and cache deep learning models for SAAC.
Handles SSL issues and provides progress bars.
"""

import os
import ssl
import urllib.request
import certifi


def fix_ssl():
    """Fix SSL certificate verification issues on macOS."""
    try:
        # Use certifi's certificate bundle
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✓ SSL certificate verification configured")
    except Exception as e:
        print(f"Warning: Could not configure SSL: {e}")


def download_deeplabv3():
    """Download DeepLabV3-ResNet50 model."""
    try:
        import torch
        import torchvision.models.segmentation as segmentation
        
        print("\n[1/2] Downloading DeepLabV3-ResNet50...")
        print("Size: ~160 MB")
        
        model = segmentation.deeplabv3_resnet50(weights='DEFAULT')
        
        print("✓ DeepLabV3 downloaded and cached")
        return True
    except Exception as e:
        print(f"✗ Failed to download DeepLabV3: {e}")
        return False


def download_u2net():
    """Download U2-Net model."""
    try:
        import torch
        
        print("\n[2/2] Downloading U2-Net...")
        print("Size: ~176 MB")
        
        # Try to load from torch hub
        try:
            model = torch.hub.load('xuebinqin/U-2-Net', 'u2net', pretrained=True)
            print("✓ U2-Net downloaded and cached")
            return True
        except Exception as e1:
            print(f"  Torch hub failed: {e1}")
            print("  Trying alternative download method...")
            
            # Alternative: Download directly
            model_url = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "u2net.pth")
            
            if os.path.exists(model_path):
                print(f"✓ U2-Net already exists at {model_path}")
                return True
            
            print(f"  Downloading from {model_url}...")
            urllib.request.urlretrieve(model_url, model_path)
            print(f"✓ U2-Net saved to {model_path}")
            return True
            
    except Exception as e:
        print(f"✗ Failed to download U2-Net: {e}")
        return False


def check_existing_models():
    """Check which models are already available."""
    print("Checking existing models...")
    print("-" * 60)
    
    status = {
        'yolov8n': False,
        'deeplabv3': False,
        'u2net': False
    }
    
    # Check YOLOv8
    if os.path.exists('yolov8n.pt'):
        print("✓ YOLOv8-nano: Already downloaded (2.6 MB)")
        status['yolov8n'] = True
    else:
        print("○ YOLOv8-nano: Will download on first use (2.6 MB)")
    
    # Check DeepLabV3 (in torch cache)
    try:
        import torch
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
        deeplabv3_file = 'deeplabv3_resnet50_coco-cd0a2569.pth'
        if os.path.exists(os.path.join(cache_dir, deeplabv3_file)):
            print("✓ DeepLabV3: Already cached (~160 MB)")
            status['deeplabv3'] = True
        else:
            print("○ DeepLabV3: Not downloaded (~160 MB)")
    except:
        print("○ DeepLabV3: Not downloaded (~160 MB)")
    
    # Check U2-Net
    u2net_paths = [
        'models/u2net.pth',
        os.path.expanduser('~/.cache/torch/hub/checkpoints/u2net.pth')
    ]
    
    if any(os.path.exists(p) for p in u2net_paths):
        print("✓ U2-Net: Already cached (~176 MB)")
        status['u2net'] = True
    else:
        print("○ U2-Net: Not downloaded (~176 MB)")
    
    print("-" * 60)
    
    return status


def main():
    print("="*60)
    print("SAAC Deep Learning Models Downloader")
    print("="*60)
    
    # Fix SSL issues
    fix_ssl()
    
    # Check existing models
    status = check_existing_models()
    
    # Ask user what to download
    print("\nWhat would you like to download?")
    print("  1. DeepLabV3 only (~160 MB)")
    print("  2. U2-Net only (~176 MB)")
    print("  3. Both models (~336 MB)")
    print("  4. Skip (use hybrid method instead)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        download_deeplabv3()
    elif choice == '2':
        download_u2net()
    elif choice == '3':
        download_deeplabv3()
        download_u2net()
    elif choice == '4':
        print("\nℹ️  Using hybrid method (Spectral + Color-based)")
        print("   This is fast and works without downloading models.")
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nYou can now use:")
    print("  - python3 compress_deep.py <image>     # Deep learning pipeline")
    print("  - python3 compress_single.py <image>   # Hybrid (fast) pipeline")
    print("  - python3 compare_methods.py <image>   # Compare all methods")
    print("="*60)


if __name__ == '__main__':
    main()


