# 🚀 SAAC Quick Reference

## Commands Cheat Sheet

```bash
# Fast compression (default - 2.6 MB models)
python3 compress_single.py image.jpg

# Deep learning compression (340 MB models, better accuracy)
python3 compress_deep.py image.jpg

# Compare all methods
python3 compare_methods.py image.jpg

# Download models manually
python3 download_models.py
```

---

## Detection Methods Summary

| Layer | Hybrid (Fast) | Deep Learning (Accurate) |
|-------|---------------|--------------------------|
| **Objects** | YOLOv8-nano | YOLOv8-nano |
| **Saliency** | Spectral Residual (FFT) | U2-Net (176 MB) |
| **Segmentation** | Color + Position Rules | DeepLabV3 (160 MB) |
| **Total Size** | 2.6 MB | ~340 MB |
| **Speed** | ⚡⚡⚡ 3-5s | ⚡ 15-20s (CPU) |

---

## Python API

### Basic Usage
```python
from saac import SaacCompressor

# Hybrid mode (fast)
compressor = SaacCompressor(device='cpu')
stats = compressor.compress_image('input.jpg', 'output.hevc')
```

### Deep Learning Mode
```python
# Full DL pipeline
compressor = SaacCompressor(
    saliency_method='u2net',
    segmentation_method='deeplabv3',
    device='cuda'  # or 'cpu'
)
stats = compressor.compress_image('input.jpg', 'output.hevc', 
                                 save_visualizations=True)
```

### Custom Mix
```python
# U2-Net saliency + classical segmentation
compressor = SaacCompressor(
    saliency_method='u2net',
    segmentation_method='simple'
)
```

---

## QP (Quality) Settings

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `person_quality` | 10 | 0-51 | Lower = better quality for people |
| `saliency_quality` | 25 | 0-51 | Quality for eye-catching areas |
| `background_quality` | 51 | 0-51 | Higher = more compression for background |

**Example:**
```python
# Max quality for people, aggressive compression elsewhere
compressor = SaacCompressor(
    person_quality=5,      # Near lossless
    saliency_quality=30,   # Moderate
    background_quality=51  # Maximum compression
)
```

---

## File Structure

```
compression/
├── saac/                      # Core library
│   ├── compressor.py          # Main pipeline
│   ├── encoder.py             # HEVC encoder
│   ├── qp_map.py              # Quality map generator
│   └── detectors/
│       ├── object_detector.py    # YOLOv8
│       ├── saliency_detector.py  # U2-Net / Spectral
│       └── segmentation.py       # DeepLabV3 / Color-based
│
├── compress_single.py         # Fast compression script
├── compress_deep.py           # DL compression script ⭐
├── compare_methods.py         # Benchmark tool ⭐
├── download_models.py         # Model downloader ⭐
│
├── UPGRADE_GUIDE.md           # Full documentation ⭐
├── UPGRADE_SUMMARY.md         # Upgrade details ⭐
└── QUICK_REFERENCE.md         # This file
```

---

## Troubleshooting

### Models won't download
```bash
# Fix SSL on macOS
/Applications/Python\ 3.*/Install\ Certificates.command

# Or use hybrid mode (no download needed)
python3 compress_single.py image.jpg
```

### Out of memory
```python
# Disable heavy layers
compressor = SaacCompressor(
    enable_saliency=False,      # Saves memory
    enable_segmentation=False
)
```

### Too slow
```bash
# Use hybrid mode or GPU
python3 compress_single.py image.jpg  # Fast
# or
CUDA_VISIBLE_DEVICES=0 python3 compress_deep.py image.jpg  # GPU
```

---

## Typical Results

**4K Image (3840×2160):**
- Original: 28.5 MB
- SAAC Compressed: 1.8 MB
- Ratio: **15.8x**
- Faces: Crystal clear
- Background: Heavily compressed

**Processing Time:**
- Hybrid: 3-5s (CPU)
- Deep Learning: 15-20s (CPU), 3-5s (GPU)

---

## For Presentations

**Key Talking Points:**
1. "Three-layer detection system (objects, saliency, semantics)"
2. "Flexible architecture: classical OR deep learning methods"
3. "15-20x compression while preserving critical details"
4. "Production-ready with graceful fallbacks"

**Demo Commands:**
```bash
# Show comparison
python3 compare_methods.py demo_image.jpg

# Show deep learning
python3 compress_deep.py demo_image.jpg

# Open visualizations
open visualizations_deep/demo_image.jpg_qp_map.jpg
```

---

## Resources

- 📖 **Full Guide:** `UPGRADE_GUIDE.md`
- 📄 **Summary:** `UPGRADE_SUMMARY.md`  
- 📦 **Models:** Run `python3 download_models.py`
- 🎓 **Examples:** `examples/` directory

---

**Version:** 2.0.0 (Deep Learning Edition)  
**Updated:** December 21, 2025


