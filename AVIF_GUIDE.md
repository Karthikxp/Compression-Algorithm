# AVIF Compression Guide

## Overview

SAAC now supports AVIF (AV1 Image File Format) compression with content-aware quality allocation. AVIF is a modern image format based on the AV1 video codec that offers superior compression compared to JPEG and WebP.

## Features

### Why AVIF?
- **30-50% smaller** than JPEG at equivalent quality
- **20-30% smaller** than WebP
- **Wide browser support**: Chrome 85+, Firefox 93+, Safari 16+, Edge 121+
- **AV1 codec**: Royalty-free, state-of-the-art compression
- **HDR support**: Wide color gamut and high dynamic range
- **Content-aware**: Uses SAAC's QP map for spatially-varying quality

### Content-Aware Quality Allocation
SAAC's AVIF encoder uses the same intelligent quality allocation as other formats:
1. **Scene Classification** - Detects image context (portrait, landscape, food, etc.)
2. **Object Detection** - Finds people, objects, and important regions
3. **Prominence Boosting** - Prioritizes large/central subjects
4. **Saliency Detection** - Identifies visually important areas
5. **QP Map Generation** - Creates spatially-varying quality map
6. **AV1 Encoding** - Encodes with adaptive quality zones

## Installation

### Requirements
- Python 3.7+
- FFmpeg with AV1 support (libaom-av1 or libsvtav1)

### Install FFmpeg with AV1 Support

**macOS (Homebrew):**
```bash
brew install ffmpeg
# FFmpeg from Homebrew includes libaom-av1, libsvtav1, and librav1e
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg libaom-dev libsvtav1-dev
```

**Windows:**
```bash
choco install ffmpeg
# Or download from https://www.gyan.dev/ffmpeg/builds/
```

### Verify Installation
```bash
# Check FFmpeg version
ffmpeg -version

# Verify AV1 encoder support
ffmpeg -encoders | grep av1
# Should show: libaom-av1, libsvtav1, or librav1e
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic AVIF Compression

Compress an image to AVIF format with content-aware quality allocation:

```bash
python3 compress_to_avif.py input_photo.jpg
```

Output: `input_photo_compressed.avif`

### 2. Preview AVIF Images

Decode AVIF back to PNG for preview or editing:

```bash
python3 preview_avif.py input_photo_compressed.avif
```

Output: `input_photo_compressed_preview.png`

You can also specify custom output path:
```bash
python3 preview_avif.py input_photo_compressed.avif custom_output.png
```

### 3. Format Comparison

Compare AVIF with other formats side-by-side:

```bash
# Compare AVIF and PNG
python3 example_comparison.py photo.jpg

# Compare all formats
python3 example_comparison.py photo.jpg avif,png,hevc

# Compare specific formats
python3 example_comparison.py photo.jpg avif,png
```

### 4. Python API

Use AVIF compression programmatically:

```python
from saac import SaacCompressor, AVIFEncoder

# Create compressor with AVIF mode
compressor = SaacCompressor(
    device='cpu',  # or 'cuda' for GPU acceleration
    yolo_model='yolov8n-seg.pt',
    saliency_method='spectral',
    segmentation_method='simple',
    scene_method='clip',  # or 'simple'
    compression_mode='avif'  # ← AVIF mode
)

# Compress image
stats = compressor.compress_image(
    input_path='photo.jpg',
    output_path='compressed.avif',
    save_visualizations=True,  # Save quality maps
    visualization_dir='visualizations'
)

# Print results
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Space saved: {stats['space_saved_percent']:.1f}%")
print(f"Scene: {stats['scene']}")

# Decode AVIF to PNG
encoder = AVIFEncoder()
encoder.decode_avif_to_png('compressed.avif', 'preview.png')
```

### 5. Advanced Options

#### Custom CRF and Speed
```python
from saac import AVIFEncoder

encoder = AVIFEncoder()

# Lower CRF = better quality, larger file
# Higher CRF = more compression, smaller file
# Recommended range: 20-35
encoder.encode_to_avif(
    input_path='photo.jpg',
    output_path='output.avif',
    crf=25,  # Quality (23-32 for photos)
    speed=6   # Speed (0=slowest/best, 8=fastest)
)
```

#### Content-Aware Quality Zones
```python
import numpy as np
from saac import AVIFEncoder

encoder = AVIFEncoder()

# Create custom QP map (or use SAAC's automatic QP map)
# Lower QP = higher quality
qp_map = np.full((1080, 1920), 40)  # Base QP
qp_map[200:400, 300:600] = 15  # High quality region

# Encode with quality zones
encoder.encode_with_quality_zones(
    input_path='photo.jpg',
    output_path='output.avif',
    qp_map=qp_map,
    base_crf=30
)
```

## Quality Settings

### CRF Values (Constant Rate Factor)
- **15-20**: Near-lossless, very large files
- **23-28**: High quality, good compression (recommended for photos)
- **30-35**: Medium quality, higher compression
- **36-40**: Low quality, maximum compression

### Speed Settings (cpu-used)
- **0-2**: Very slow, best compression
- **4**: Slow, excellent compression (recommended)
- **6**: Medium, good compression (default)
- **8**: Fast, moderate compression

## Performance

### Compression Ratios (vs Original)
| Image Type | AVIF Ratio | File Size Reduction |
|------------|------------|---------------------|
| Portrait (4K) | 15-25x | 93-96% |
| Landscape | 12-20x | 92-95% |
| Food Photo | 18-30x | 94-97% |
| Street Scene | 10-18x | 90-94% |

### Processing Time (CPU)
- **Small (1MP)**: 1-2 seconds
- **Medium (4MP)**: 3-5 seconds
- **Large (12MP)**: 8-15 seconds
- **GPU**: 2-3x faster

### Comparison: AVIF vs Other Formats
At equivalent quality (SSIM ~0.95):
- **vs JPEG**: 30-50% smaller
- **vs WebP**: 20-30% smaller
- **vs PNG (lossless)**: 60-80% smaller
- **vs HEVC**: Similar size, better compatibility

## Browser Support

### Desktop Browsers
- ✅ Chrome 85+ (August 2020)
- ✅ Firefox 93+ (October 2021)
- ✅ Safari 16+ (September 2022)
- ✅ Edge 121+ (January 2024)
- ✅ Opera 71+

### Mobile Browsers
- ✅ Chrome Android 87+
- ✅ Safari iOS 16+
- ✅ Firefox Android 93+
- ✅ Samsung Internet 15+

### Server Support
- ✅ Most CDNs support AVIF
- ✅ Cloudflare Image Resizing
- ✅ Cloudinary
- ✅ imgix
- ✅ ImageKit

## Use Cases

### 1. Web Delivery
AVIF is perfect for serving images on websites:
```html
<picture>
  <source srcset="photo.avif" type="image/avif">
  <source srcset="photo.webp" type="image/webp">
  <img src="photo.jpg" alt="Fallback">
</picture>
```

### 2. E-commerce
- Product images with sharp details
- Fast page load times
- Lower bandwidth costs

### 3. Photography Portfolios
- High quality with small file sizes
- Fast gallery loading
- Mobile-friendly

### 4. Social Media
- Profile pictures
- Cover photos
- Post images

### 5. Cloud Storage
- Reduce storage costs by 60-80%
- Faster uploads/downloads
- Keep original quality

## Troubleshooting

### FFmpeg Not Found
```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu
```

### No AV1 Encoder Support
```bash
# Check available encoders
ffmpeg -encoders | grep av1

# If missing, reinstall FFmpeg with AV1 support
brew reinstall ffmpeg  # macOS
```

### Encoding Too Slow
```python
# Use faster speed setting
compressor = SaacCompressor(compression_mode='avif')
encoder.encode_to_avif(
    input_path='photo.jpg',
    output_path='output.avif',
    speed=8  # Fastest (0-8)
)
```

### File Too Large
```python
# Increase CRF for more compression
encoder.encode_to_avif(
    input_path='photo.jpg',
    output_path='output.avif',
    crf=35  # Higher = more compression
)
```

### Browser Doesn't Support AVIF
Use picture element with fallbacks:
```html
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="Description">
</picture>
```

## Technical Details

### AV1 Codec Parameters
SAAC uses the following AV1 parameters for optimal quality:
- **Adaptive Quantization (aq-mode)**: Variance-based or complexity-based
- **AQ Strength**: 1.0-2.0 depending on image content
- **CDEF Filter**: Enabled (constrained directional enhancement)
- **Restoration Filter**: Enabled (loop restoration)
- **Row-based Multithreading**: Enabled for faster encoding

### Quality Zone Strategy
SAAC analyzes the QP map to determine optimal encoding:

1. **High Important Content (>20% critical regions)**
   - CRF: 23
   - AQ Strength: 1.0 (moderate)
   - Strategy: Protect all important regions

2. **Moderate Important Content (5-20% critical regions)**
   - CRF: 28
   - AQ Strength: 1.5 (strong)
   - Strategy: Balance quality and size

3. **Small Important Regions (<5% critical regions)**
   - CRF: 20 (very low)
   - AQ Strength: 2.0 (very aggressive)
   - Strategy: Protect critical regions, compress background heavily

## FAQ

### Q: Is AVIF better than WebP?
**A:** Yes, AVIF typically provides 20-30% better compression than WebP at equivalent quality, with similar encoding/decoding performance.

### Q: Can I use AVIF for all my images?
**A:** AVIF is excellent for web delivery and archival. Use PNG only when you need lossless quality or universal compatibility with older systems.

### Q: How does AVIF compare to JPEG?
**A:** AVIF is 30-50% smaller than JPEG at equivalent quality and supports advanced features like HDR, alpha channel, and lossless compression.

### Q: Is AVIF royalty-free?
**A:** Yes, AV1 (and AVIF) is completely royalty-free and open-source.

### Q: Can I convert AVIF back to PNG/JPEG?
**A:** Yes, use the preview_avif.py script or FFmpeg to decode AVIF to any format.

### Q: Does AVIF support transparency?
**A:** Yes, AVIF supports alpha channel (transparency).

### Q: What about HEIC/HEIF?
**A:** AVIF offers similar compression to HEIC but with wider browser support and no licensing concerns.

## Resources

- **AVIF Specification**: https://aomediacodec.github.io/av1-avif/
- **AV1 Codec**: https://aomedia.org/av1/
- **Browser Support**: https://caniuse.com/avif
- **FFmpeg AV1 Guide**: https://trac.ffmpeg.org/wiki/Encode/AV1

## License

SAAC is released under the MIT License. AV1 codec and AVIF format are royalty-free and open-source.

---

**Version:** 2.1.0  
**Last Updated:** 2026-02-05  
**SAAC AVIF Support** - Superior compression with content-aware quality allocation
