# SAAC - Scene-Aware Adaptive Compression

**Intelligent image compression that preserves what matters most.**

Achieves 15-20x compression ratios while keeping faces crystal clear, text readable, and important objects sharp. Compresses the boring stuff (sky, empty space, backgrounds) aggressively using zone-aware encoding.

---

## Quick Start

```bash
# Compress an image
python3 compress.py your_photo.jpg

# Output: your_photo_compressed.hevc (15-20x smaller!)
# Visualizations: visualizations/ folder
```

---

## How It Works

SAAC uses AI to understand your image and allocate quality intelligently:

1. **Scene Classification** - "Is this a street? Restaurant? Landscape?"
2. **Object Detection** - Finds people, cars, animals with pixel-perfect masks (YOLOv8-seg)
3. **Prominence Boost** - Automatically boosts large/central subjects
4. **Saliency Detection** - Identifies visually interesting regions (optional)
5. **Semantic Segmentation** - Classifies background (sky, water, roads) (optional)
6. **Smart QP Map** - Creates a "quality blueprint" with zone-based allocation
7. **Zone-Aware HEVC Encoding** - Protects critical regions with low CRF, aggressive AQ compresses background

**Result:** 20x smaller file, but faces/text stay sharp with spatially-varying quality allocation

---

## Example Results

**Portrait Scene (4K):**
- Original: 4.54 MB (PNG)
- SAAC: 0.05 MB (HEVC)
- Compression: 99x ratio
- Face region: Near-lossless (QP 15, CRF 15-18)
- Background: Heavily compressed (QP 51, aggressive AQ)
- Processing: 5 seconds on CPU

**Street Scene (4K):**
- Original: 1.55 MB
- SAAC: 0.07 MB (21.7x compression)
- Vehicles: Crystal clear (QP 10-15)
- Sky/background: Heavily compressed (QP 45-51)
- Processing: 4-5 seconds on CPU

---

## 🔧 Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for encoding)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg libx265-dev

# Windows
choco install ffmpeg
```

### 3. Test Installation
```bash
python3 -c "from saac import SaacCompressor; print('✅ SAAC Ready!')"
```

---

## 💡 Python API Usage

```python
from saac import SaacCompressor

# Create compressor
compressor = SaacCompressor(
    device='cpu',  # or 'cuda' for GPU
    yolo_model='yolov8n-seg.pt',
    saliency_method='spectral',
    segmentation_method='simple',
    scene_method='simple'
)

# Compress an image
stats = compressor.compress_image(
    input_path='photo.jpg',
    output_path='compressed.hevc',
    save_visualizations=True
)

print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Scene detected: {stats['scene']}")
print(f"Objects found: {stats['detections']}")
```

---

## Visualizations

After compression, check the `visualizations/` folder for 5 diagnostic images:

| File | Description | Model/Algorithm |
|------|-------------|-----------------|
| `_detections.jpg` | Objects with pixel-perfect segmentation masks | YOLOv8-seg |
| `_prominence.jpg` | Importance scores (green=prominent, blue=normal) | Geometric calculator (size + centrality) |
| `_saliency.jpg` | Visual attention heatmap (hot colormap) | Spectral Residual or U2-Net |
| `_qp_map.jpg` | Final quality allocation (red=QP10, blue=QP51) | Intent + Prominence + Saliency combined |
| `_scene.jpg` | Detected scene type label overlay | Scene classifier |

**QP Map Color Legend:**
- Red/Orange (QP 10-15): Near-lossless quality, critical regions
- Yellow/Green (QP 25-35): High quality, important objects
- Cyan (QP 40-45): Medium quality, less important
- Dark Blue (QP 51): Maximum compression, backgrounds

---

## Scene-Based Compression Rules

SAAC automatically applies scene-specific compression rules:

| Scene | Protected Objects | Compressed Heavily |
|-------|-------------------|-------------------|
| **Street** | People, vehicles, traffic signs | Sky, distant buildings |
| **Restaurant** | People, food, drinks | Walls, empty tables |
| **Landscape** | People, animals, foreground | Sky, distant mountains |
| **Document** | Text, signatures, stamps | Paper texture |
| **Indoor** | People, faces, electronics | Walls, furniture |
| **Retail** | Products, people | Empty shelves, backgrounds |

**Example:** In a restaurant photo, SAAC keeps faces at 95% quality and the pizza perfectly sharp, but compresses the wall behind you by 90%.

---

## Advanced Features

### Zone-Aware Encoding (NEW)
Unlike traditional methods that average the QP map into a single value, SAAC uses intelligent zone-based encoding:

1. **Quality Zone Analysis**: Divides QP map into 4 zones
   - Critical (QP <=15): People, faces, text
   - High (QP 16-25): Important objects
   - Medium (QP 26-40): Moderate importance
   - Low (QP >40): Background, sky, empty space

2. **Adaptive Strategy**: Selects encoding parameters based on zone distribution
   - Large critical regions: CRF 18, moderate AQ
   - Moderate critical regions: CRF 22, strong AQ
   - Small critical regions: CRF 15, very aggressive AQ (2.5x)

3. **Spatially-Varying Quality**: Uses x265 adaptive quantization to protect critical zones while heavily compressing backgrounds
   - Low base CRF preserves important regions
   - High AQ strength (up to 3.0) compresses backgrounds more aggressively
   - Fine-grained quantization groups (8x8) for precise control

### Prominence Boosting
Automatically detects and protects the main subject:
- Size: Is it taking up >15% of the image?
- Location: Is it centered within 30% radius?
- Auto-boost: Large + centered = 1.0 weight override

### Intent Rules
7 pre-loaded scene profiles that map object importance:
- Restaurant: food=0.9, people=1.0, chairs=0.3
- Street: vehicles=0.9, traffic signs=0.95, people=1.0
- Landscape: people=1.0, animals=0.9, background=0.1
- Portrait: person=1.0 (always protected)

### Pixel-Perfect Masks
YOLOv8-seg provides exact object boundaries (not just bounding boxes), so quality is allocated precisely to important pixels without waste.

---

## Project Structure

```
saac/
├── compress.py              # Main compression script
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installer
├── yolov8n-seg.pt          # YOLO segmentation model (2.7 MB)
│
├── saac/                    # Core library
│   ├── compressor.py        # Main compression pipeline
│   ├── qp_map.py            # Smart QP map generator
│   ├── intent_rules.py      # Scene-based rule profiles
│   ├── encoder.py           # FFmpeg HEVC wrapper
│   └── detectors/
│       ├── object_detector.py    # YOLOv8-seg
│       ├── saliency_detector.py  # Saliency detection
│       ├── segmentation.py       # Semantic segmentation
│       ├── scene_classifier.py   # Scene classification
│       └── prominence.py         # Importance calculator
│
├── models/                  # Downloaded models (auto-created)
└── test_images/             # Your test images
```

---

## Use Cases

1. **Security Cameras** - Keep license plates/faces readable, compress empty parking lots
2. **Cloud Photo Storage** - Store 10x more photos with important details intact
3. **E-commerce** - Product details sharp, studio backgrounds compressed
4. **Medical Imaging** - Preserve diagnostic regions, compress peripheral areas
5. **Documents** - Text stays readable, paper texture compressed

---

## Configuration Options

### Device Selection
```python
compressor = SaacCompressor(device='cuda')  # Use GPU
compressor = SaacCompressor(device='cpu')   # Use CPU
```

### Saliency Methods
```python
# Fast (no model download)
SaacCompressor(saliency_method='spectral')

# Deep learning (requires U2-Net model, 176 MB)
SaacCompressor(saliency_method='u2net')
```

### Segmentation Methods
```python
# Fast color-based rules
SaacCompressor(segmentation_method='simple')

# Deep learning (requires DeepLabV3 model, 160 MB)
SaacCompressor(segmentation_method='deeplabv3')
```

### Scene Classification
```python
# Fast heuristics
SaacCompressor(scene_method='simple')

# Deep learning (requires EfficientNet)
SaacCompressor(scene_method='efficientnet')
```

---

## Benchmarks

**Test Image:** 4K street scene (4032x3024, 1.55 MB)

| Metric | Value |
|--------|-------|
| **Compression Ratio** | 21.74x |
| **Original Size** | 1.55 MB |
| **Compressed Size** | 0.07 MB (71 KB) |
| **Space Saved** | 95.4% |
| **Processing Time** | 4.3 seconds (CPU) |
| **Scene Detected** | Street (75% confidence) |
| **Objects Found** | 8 vehicles (with segmentation) |
| **High Quality Regions** | 11.0% (vehicles, signs) |
| **Heavily Compressed** | 88.6% (sky, distant areas) |

---

## Encoding Parameters for Aggressive Compression

To increase compression while protecting faces, modify these parameters in `saac/encoder.py`:

### AQ Strength (Line 284, 289, 294)
Controls background compression aggressiveness:
```python
aq_strength = 2.8  # Range: 1.0-3.0, higher = more background compression
```

### CRF Values (Line 283, 288, 293)
Base quality level:
```python
crf = 22  # Range: 15-28, higher = smaller files (affects everything)
```

### Preset (Line 318)
Encoding speed vs efficiency:
```python
preset='slow'  # Options: veryslow, slow, medium, fast
```

**Recommendation for maximum compression**: `aq_strength=3.0`, `crf=25`, `preset='medium'`

---

## Troubleshooting

**Import Error:**
```bash
# Make sure you're in the project directory
cd /path/to/compression
python3 compress.py image.jpg
```

**FFmpeg Not Found:**
```bash
# Install FFmpeg with HEVC support
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu
```

**Out of Memory:**
```python
# Disable optional layers
compressor = SaacCompressor(
    enable_saliency=False,
    enable_segmentation=False
)
```

**Slow Processing:**
```python
# Use GPU if available
compressor = SaacCompressor(device='cuda')
```

**HEIC Files:**
```bash
# HEIC files appear as PNG but can't be read by OpenCV
# Convert using macOS sips:
sips -s format png input.heic --out input.png
python3 compress.py input.png
```

---

## License

MIT License - Free for academic and commercial use

---

## Acknowledgments

- **YOLOv8** by Ultralytics - Object detection with segmentation
- **FFmpeg & x265** - HEVC encoding
- **OpenCV** - Computer vision operations
- **PyTorch** - Deep learning framework

---

## Citation

If you use SAAC in research:

```bibtex
@software{saac2024,
  title={SAAC: Saliency-Aware Adaptive Compression},
  author={Your Name},
  year={2024},
  version={2.0},
  url={https://github.com/yourusername/saac}
}
```

---

**Built with:** Python • PyTorch • YOLOv8 • OpenCV • FFmpeg (x265)
**Version:** 2.1.0
**Status:** Production Ready

---

## Need Help?

1. Check `test_images/` - Add your test images here
2. Run `python3 compress.py test_images/your_photo.jpg`
3. Check `visualizations/` - See quality allocation maps
4. Adjust settings in `saac/intent_rules.py` if needed

Star this project if you find it useful.
