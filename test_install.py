#!/usr/bin/env python3
"""
Quick installation test for SAAC
"""

import sys

def test_imports():
    """Test that all required packages are importable."""
    print("Testing package imports...")
    
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics YOLO"),
        ("PIL", "Pillow"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("ffmpeg", "FFmpeg-Python"),
    ]
    
    failed = []
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    return failed


def test_saac():
    """Test that SAAC module loads."""
    print("\nTesting SAAC module...")
    try:
        from saac import SaacCompressor
        print("  ✓ SAAC module loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ SAAC module failed: {e}")
        return False


def test_ffmpeg():
    """Test FFmpeg installation."""
    print("\nTesting FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ {version_line}")
            
            # Check for x265 support
            result = subprocess.run(
                ['ffmpeg', '-codecs'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if 'libx265' in result.stdout or 'hevc' in result.stdout:
                print("  ✓ HEVC/x265 support detected")
            else:
                print("  ⚠ HEVC/x265 support unclear (may still work)")
            
            return True
        else:
            print("  ✗ FFmpeg returned error")
            return False
    except FileNotFoundError:
        print("  ✗ FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"  ✗ FFmpeg test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SAAC Installation Test")
    print("="*60)
    print()
    
    # Test imports
    failed_imports = test_imports()
    
    # Test SAAC
    saac_ok = test_saac()
    
    # Test FFmpeg
    ffmpeg_ok = test_ffmpeg()
    
    # Summary
    print("\n" + "="*60)
    print("INSTALLATION TEST SUMMARY")
    print("="*60)
    
    if failed_imports:
        print(f"✗ Failed imports: {', '.join(failed_imports)}")
        print("  Run: pip3 install -r requirements.txt")
    else:
        print("✓ All Python packages installed")
    
    if saac_ok:
        print("✓ SAAC module working")
    else:
        print("✗ SAAC module has issues")
    
    if ffmpeg_ok:
        print("✓ FFmpeg installed and working")
    else:
        print("✗ FFmpeg not properly installed")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt-get install ffmpeg libx265-dev")
    
    print()
    
    if not failed_imports and saac_ok and ffmpeg_ok:
        print("🎉 All tests passed! SAAC is ready to use!")
        print("\nNext steps:")
        print("  1. Run the demo:       python3 examples/demo.py")
        print("  2. Try basic usage:    python3 examples/basic_usage.py")
        print("  3. Read quick start:   cat QUICKSTART.md")
        return 0
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

