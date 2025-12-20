# SAAC Quick Start Guide

## 🚀 Installation Complete!

All dependencies have been installed. You're ready to start compressing images!

---

## ⚡ 5-Minute Quick Start

### 1. Run the Interactive Demo

The easiest way to see SAAC in action:

```bash
cd /Users/karthikm/compression
python3 examples/demo.py
```

This will:
- Download a sample image
- Detect objects, saliency, and semantic regions
- Compress the image
- Show you the results and visualizations

---

### 2. Compress Your Own Image

**Option A: Using Python**

```python
from saac import SaacCompressor

# Initialize
compressor = SaacCompressor(
    device='cpu',  # Use 'cuda' if you have NVIDIA GPU
    person_quality=10,      # High quality for people (0-51, lower=better)
    saliency_quality=25,    # Medium quality for interesting regions
    background_quality=51   # Maximum compression for background
)

# Compress
compressor.compress_image(
    input_path='your_image.jpg',
    output_path='compressed.hevc',
    save_visualizations=True
)
```

**Option B: Use the Examples**

Edit `examples/basic_usage.py` to point to your image:

```bash
python3 examples/basic_usage.py
```

---

## 🎯 Use Cases & Presets

### Security Camera Mode
```python
compressor = SaacCompressor(
    person_quality=8,       # Crystal clear people
    saliency_quality=40,    # Don't care about other details
    background_quality=51,  # Destroy the background
    enable_saliency=False   # Faster - only detect people
)
```

### Photo Storage Mode
```python
compressor = SaacCompressor(
    person_quality=12,      # High quality faces
    saliency_quality=20,    # Preserve interesting details
    background_quality=45,  # Moderate background compression
    blend_mode='weighted'   # Smooth transitions
)
```

### Extreme Compression Mode
```python
compressor = SaacCompressor(
    person_quality=15,      # Still preserve people
    saliency_quality=35,    # Aggressive everywhere else
    background_quality=51   # Maximum background destruction
)
```

---

## 📊 Understanding QP Values

**QP = Quantization Parameter** (controls quality)

| QP Range | Quality | Use For |
|----------|---------|---------|
| 0-18 | Near-lossless | Critical content (faces, license plates) |
| 18-28 | High quality | Salient regions (mountains, text) |
| 28-40 | Medium quality | Less important areas |
| 40-51 | Heavy compression | Backgrounds, sky, empty space |

---

## 🎨 Visualizations

When you enable `save_visualizations=True`, you get:

1. **`*_objects.jpg`** - Shows detected people/objects in green
2. **`*_saliency.jpg`** - Heatmap of visually interesting regions
3. **`*_qp_map.jpg`** - Color-coded quality allocation map
   - Red = High quality
   - Blue = Heavy compression

---

## 📁 Project Structure

```
compression/
├── saac/                    # Main library
│   ├── compressor.py        # Main compression pipeline
│   ├── detectors/           # Detection modules
│   │   ├── object_detector.py
│   │   ├── saliency_detector.py
│   │   └── segmentation.py
│   ├── qp_map.py           # Quality map generator
│   └── encoder.py          # FFmpeg integration
├── examples/               # Usage examples
│   ├── demo.py            # Interactive demo
│   ├── basic_usage.py     # Simple example
│   └── advanced_usage.py  # Multiple use cases
├── models/                # Model weights (auto-downloaded)
├── requirements.txt       # Dependencies
└── README.md             # Full documentation
```

---

## 🔧 Troubleshooting

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg libx265-dev

# Windows
choco install ffmpeg
```

### "CUDA not available" (Optional - CPU works fine!)
SAAC works perfectly on CPU. GPU just makes it faster for large images.

### Low compression ratio?
- Increase `background_quality` (closer to 51)
- Decrease `saliency_quality` and `person_quality` if acceptable
- Use `blend_mode='weighted'` for more aggressive background compression

### Image quality not good enough?
- Decrease QP values (10 → 5 for people, etc.)
- Enable all layers: `enable_saliency=True`, `enable_segmentation=True`
- Use `blend_mode='priority'` to protect important regions

---

## 🎓 Next Steps

1. **Try the demo**: `python3 examples/demo.py`
2. **Compress your images**: Edit `examples/basic_usage.py`
3. **Batch processing**: See `examples/advanced_usage.py`
4. **Read full docs**: Check `README.md`

---

## 💡 Pro Tips

1. **For batch processing**: Images are processed sequentially, so be patient with large folders
2. **Save visualizations**: They help you understand how SAAC is allocating quality
3. **Experiment with QP values**: Every image is different - tune for your use case
4. **GPU acceleration**: If you have NVIDIA GPU, set `device='cuda'` for 3-5x speedup

---

## 📞 Need Help?

Check the full documentation in `README.md` for:
- Detailed architecture explanation
- API reference
- More advanced examples
- Technical details about QP maps and HEVC encoding

---

**Happy Compressing! 🎉**

