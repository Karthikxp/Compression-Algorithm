# QP Mapping Pipeline - Complete Breakdown

## Overview

The QP (Quantization Parameter) map is the **final quality allocation blueprint** that determines how much compression each pixel gets. It's built through a **7-step pipeline** where multiple AI models and algorithms contribute their inputs.

**QP Scale:**
- **QP 0** = Lossless (no compression)
- **QP 10** = Near-lossless (very high quality)
- **QP 30** = Medium quality
- **QP 51** = Maximum compression

---

## The 7-Step Pipeline

### Step 1: Scene Classification
**Model:** CLIP or Simple classifier  
**Purpose:** Understand image context  
**Output:** Scene category (e.g., "portrait", "landscape", "restaurant")

**Weight Contribution:** Sets the foundation for ALL object importance  
**Impact:** 🔥🔥🔥🔥🔥 (5/5) - **Most Critical**

**How it works:**
- Classifies the entire image into one of 54 intent categories
- Each category has predefined rules (Intent Profiles) that map object classes to importance weights
- Example for "portrait" scene:
  ```
  person → weight 1.0 (maximum quality)
  handbag → weight 0.7 (high quality)
  chair → weight 0.3 (low quality)
  default → weight 0.1 (maximum compression)
  ```

**Effect on QP Map:**
- A person in a "portrait" scene gets weight 1.0 → QP 10 (or QP 0 in extreme mode)
- Same person in a "landscape" scene still gets weight 1.0 → QP 10
- But a car in "portrait" gets weight 0.1 → QP 51 (compressed)
- Same car in "street" scene gets weight 0.9 → QP 15 (protected)

**Key Point:** Scene classification doesn't directly create pixels in the QP map, but it determines the importance rules for every object detected.

---

### Step 2: Object Detection (YOLOv8-seg)
**Model:** YOLOv8 with segmentation  
**Purpose:** Find objects with pixel-perfect masks  
**Output:** List of detected objects with:
- Class name (person, car, dog, etc.)
- Bounding box
- Pixel-perfect segmentation mask
- Confidence score

**Weight Contribution:** Provides the spatial location for applying scene-based weights  
**Impact:** 🔥🔥🔥🔥 (4/5) - **Critical**

**How it works:**
1. Detects all objects in the image
2. For each object, gets the base weight from Scene Intent Rules (Step 1)
3. Example detection:
   ```
   Object: person
   Scene: portrait
   Base weight from intent rules: 1.0
   Mask: [H, W] binary mask showing exact person pixels
   ```

**Effect on QP Map:**
- For EVERY pixel where the person mask = 1, set weight = 1.0
- This creates a "stencil" in the weight map
- Person pixels → weight 1.0 → QP 10 (or QP 0)
- Non-person pixels → weight 0.0 → QP 51

**Blend Mode:**
- **Priority mode** (default): Takes MAXIMUM weight if pixels overlap
  - If person (1.0) overlaps with dog (0.9), pixel gets 1.0
- **Weighted mode**: Averages overlapping weights
  - Person (1.0) + dog (0.9) = average 0.95

---

### Step 3: Prominence Calculation
**Algorithm:** Geometric analysis  
**Purpose:** Boost large/central objects  
**Output:** Prominence metrics for each detection

**Weight Contribution:** +0.0 to +0.5 boost to base weight  
**Impact:** 🔥🔥🔥 (3/5) - **Significant**

**How it works:**
1. **Size Analysis:**
   - Calculates: `area_ratio = object_pixels / total_pixels`
   - Threshold: 15% (0.15) to be considered "large"
   - Example: Person taking up 20% of image → considered prominent

2. **Centrality Analysis:**
   - Calculates distance from image center
   - `centrality = 1.0 - normalized_distance`
   - Example: Person in center → centrality 1.0, corner → centrality 0.2

3. **Combined Score:**
   ```python
   prominence_score = 0.6 * (area_ratio / 0.15) + 0.4 * centrality
   ```
   - **60% weight** on size
   - **40% weight** on location

4. **Boost Application:**
   ```python
   if is_prominent:
       boost = 0.5 * prominence_score
       final_weight = min(1.0, base_weight + boost)
   ```

**Effect on QP Map:**
- **Small person in corner:**
  - Base weight: 1.0 (from intent)
  - Prominence: NOT prominent
  - Final weight: 1.0 (no change)
  - QP: 10

- **Large central person:**
  - Base weight: 1.0
  - Prominence: YES (area 25%, centrality 0.9)
  - Prominence score: 0.6*(25/15) + 0.4*0.9 = 1.36 (capped at 1.0)
  - Boost: 0.5 * 1.0 = 0.5
  - Final weight: min(1.0, 1.0 + 0.5) = 1.0 (already maxed)
  - QP: 10

- **Car in corner (street scene):**
  - Base weight: 0.9 (from street scene intent)
  - Prominence: NOT prominent (small, far from center)
  - Final weight: 0.9
  - QP: 14

- **Large central car (street scene):**
  - Base weight: 0.9
  - Prominence: YES (area 30%, centrality 0.8)
  - Boost: 0.5 * prominence_score = ~0.5
  - Final weight: min(1.0, 0.9 + 0.5) = 1.0
  - QP: 10 (boosted to maximum!)

**Key Insight:** Prominence can elevate medium-importance objects to high importance if they're the main subject.

---

### Step 4: Saliency Detection
**Model:** Spectral Residual or U2-Net  
**Purpose:** Fill gaps - find visually important regions NOT detected as objects  
**Output:** Saliency map [H, W] with values 0.0-1.0

**Weight Contribution:** Up to 0.6x contribution in empty regions  
**Impact:** 🔥🔥 (2/5) - **Moderate**

**How it works:**
1. Computes visual saliency (where human eyes naturally look)
2. Scaled to 60% contribution: `saliency_weight = saliency_map * 0.6`
3. Applied using priority mode:
   ```python
   weight_map = np.maximum(weight_map, saliency_weight)
   ```

**Effect on QP Map:**
- **Object-covered pixels:** Already have high weight from detections → saliency has NO effect
- **Empty regions:** Have weight 0.0 → saliency fills in
  - Visually interesting texture → saliency 0.8 → weight 0.48 → QP ~31
  - Boring uniform area → saliency 0.1 → weight 0.06 → QP ~48

**Real Example:**
- Portrait with person (weight 1.0) in front of textured wallpaper
- Wallpaper has no objects detected
- Saliency detects interesting patterns → saliency 0.7
- Wallpaper pixels: weight 0.0 → max(0.0, 0.7*0.6) = 0.42 → QP ~30
- Without saliency: wallpaper would be QP 51 (destroyed)

**Key Point:** Saliency prevents total destruction of visually interesting backgrounds.

---

### Step 5: Semantic Segmentation
**Model:** Simple color-based or DeepLabV3  
**Purpose:** Identify background categories (sky, water, road)  
**Output:** Category masks for background elements

**Weight Contribution:** Reduces weight in low-importance background regions  
**Impact:** 🔥 (1/5) - **Minor**

**How it works:**
1. Identifies background categories:
   ```python
   background_penalties = {
       'sky': 0.1,        # Very low importance
       'water': 0.15,
       'road': 0.2,
       'vegetation': 0.25,
       'building': 0.3
   }
   ```

2. Applied to regions with low existing weight:
   ```python
   # Only reduce where weight < 0.3 (already low importance)
   background_mask = (weight_map < 0.3) & (sky_mask > 0)
   weight_map[background_mask] = 0.1  # Force to sky penalty
   ```

**Effect on QP Map:**
- **Sky pixels** (already low weight 0.2) → forced to 0.1 → QP ~47
- **Person pixels** (high weight 1.0) in front of sky → NOT affected
- **Grass** (low weight 0.1) → forced to 0.25 → QP ~44

**Key Point:** Fine-tunes background regions that are already marked as low importance. Doesn't override high-importance objects.

---

### Step 6: Weight → QP Conversion
**Algorithm:** Linear mapping  
**Purpose:** Convert weight map (0-1) to QP map (0-51)

**How it works:**

**Normal Mode:**
```python
QP = 51 - (51 - 10) * weight
QP = 51 - 41 * weight
```
- Weight 1.0 → QP = 51 - 41 = 10
- Weight 0.5 → QP = 51 - 20.5 = 30.5
- Weight 0.0 → QP = 51 - 0 = 51

**EXTREME Person Mode (AVIF):**
```python
QP = 51 - 51 * weight
```
- Weight 1.0 (person) → QP = 0 (lossless!)
- Weight 0.5 → QP = 25.5
- Weight 0.0 (background) → QP = 51

**Effect:** This is the final transformation that converts all the accumulated intelligence into actual compression parameters.

---

### Step 7: Smoothing
**Algorithm:** Gaussian blur  
**Purpose:** Avoid harsh transitions between quality zones

**Weight Contribution:** Blends nearby QP values  
**Impact:** 🔥 (1/5) - **Minor (aesthetic)**

**How it works:**
```python
# Normal mode: 15x15 kernel (heavy smoothing)
qp_map_smoothed = cv2.GaussianBlur(qp_map, (15, 15), 0)

# Extreme mode: 5x5 kernel (minimal smoothing)
qp_map_smoothed = cv2.GaussianBlur(qp_map, (5, 5), 0)
```

**Effect on QP Map:**
- **Before smoothing:** Person pixels QP 10, adjacent background QP 51 (harsh edge)
- **After smoothing:** Person edge QP 10 → 12 → 15 → 20 → 35 → 51 (gradual transition)
- **Extreme mode:** Less smoothing to maintain sharp person/background distinction

**Key Point:** Makes compression artifacts less noticeable at boundaries. In extreme mode, intentionally reduced to maintain visual contrast.

---

## Complete Example Flow

### Input Image: Portrait of person in park

**Step 1: Scene Classification**
- Result: "portrait"
- Intent rules loaded:
  - person: 1.0
  - dog: 0.9
  - tree: 0.3
  - sky: 0.1

**Step 2: Object Detection**
- Detected: 1 person (center, 30% of image)
- Person mask: [H, W] binary mask
- Base weight from intent: 1.0

**Step 3: Prominence**
- Person area: 30% > 15% threshold → PROMINENT
- Person centrality: 0.95 (very centered)
- Prominence score: 1.0 (max)
- Boost: +0.5
- Final weight: min(1.0, 1.0 + 0.5) = 1.0 (capped)

**Step 4: Weight Map After Detections**
```
Person pixels:     weight = 1.0
Background pixels: weight = 0.0
```

**Step 5: Saliency**
- Interesting texture in background: saliency = 0.5
- Applied: weight = max(0.0, 0.5 * 0.6) = 0.3
```
Person pixels:       weight = 1.0 (unchanged)
Interesting texture: weight = 0.3 (filled in)
Boring background:   weight = 0.0 (no saliency)
```

**Step 6: Segmentation**
- Sky detected in boring background
- Sky penalty: 0.1
```
Person pixels:       weight = 1.0
Interesting texture: weight = 0.3
Sky:                 weight = 0.1 (reduced)
```

**Step 7: Convert to QP (Extreme AVIF Mode)**
```
Person pixels:       QP = 51 - 51*1.0 = 0  (LOSSLESS!)
Interesting texture: QP = 51 - 51*0.3 = 36 (medium compression)
Sky:                 QP = 51 - 51*0.1 = 46 (heavy compression)
```

**Step 8: Smooth (minimal in extreme mode)**
- 5x5 blur
```
Person pixels:       QP = 0 → 0 → 2 → 8 → ... (sharp edge maintained)
Interesting texture: QP = 36 (slightly smoothed)
Sky:                 QP = 46 (slightly smoothed)
```

**Final QP Map:**
```
[Person region]        QP 0-5    (lossless - perfect quality)
[Person edge]          QP 5-15   (very high quality)
[Texture background]   QP 30-40  (medium quality)
[Sky]                  QP 45-51  (maximum compression)
```

---

## Weight Contributions Summary

| Component | Contribution Type | Weight Range | Impact Level | When Active |
|-----------|------------------|--------------|--------------|-------------|
| **Scene Intent** | Sets base weight | 0.0 - 1.0 | 🔥🔥🔥🔥🔥 Critical | Always |
| **Object Detection** | Spatial mapping | Uses scene weights | 🔥🔥🔥🔥 Critical | Always |
| **Prominence** | Boost modifier | +0.0 to +0.5 | 🔥🔥🔥 Significant | If large/central |
| **Saliency** | Gap filler | 0.0 - 0.6 | 🔥🔥 Moderate | In empty regions |
| **Segmentation** | Background penalty | Forces 0.1 - 0.3 | 🔥 Minor | Low-weight regions |
| **Smoothing** | Aesthetic blend | Blurs neighbors | 🔥 Minor | Always (final step) |

---

## Multiplicative vs Additive Effects

### Multiplicative (Sequential Pipeline)
Each step builds on the previous:
1. Scene → Sets base weights
2. Detection → Maps weights to pixels
3. Prominence → Boosts weights
4. Saliency → Fills gaps (max operation)
5. Segmentation → Reduces background
6. QP Conversion → Applies formula
7. Smoothing → Blends

**Key Point:** Later stages can only work with what earlier stages provide. You can't boost prominence if detection didn't find the object!

### Priority Mode (Maximum Operation)
```python
weight_map = np.maximum(current_weight, new_weight)
```
- Person weight 1.0 > saliency 0.6 → keeps 1.0
- Empty region 0.0 < saliency 0.6 → updates to 0.6
- Overlapping objects take higher weight

**Effect:** Protects high-importance regions from being downgraded by later stages.

---

## Extreme Person Mode (AVIF) Changes

### Standard Mode
- Scene intent: person 1.0 → others vary
- Prominence: +0.0 to +0.5 boost
- QP conversion: weight 1.0 → QP 10
- Smoothing: 15x15 kernel (smooth transitions)

### Extreme Person Mode
- Scene intent: person 1.0 → **everything else forced to 0.0**
- Prominence: DISABLED (everything non-person is 0.0 anyway)
- QP conversion: weight 1.0 → **QP 0** (lossless!), weight 0.0 → QP 51
- Smoothing: **5x5 kernel** (sharp boundaries)
- AVIF encoding: CRF 10, **AQ strength 4.0** (extremely aggressive background compression)

**Result:** Unidentifiable background, perfect person quality, visible contrast.

---

## Factors That Affect Final Compression

### 1. Scene Classification Accuracy (Biggest Impact)
- Correct scene → correct intent rules → correct base weights
- Wrong scene → wrong priorities → bad compression
- Example: Portrait misclassified as landscape → may under-protect person

### 2. Object Detection Quality
- Missed person → no protection → destroyed in compression
- False positive → wastes quality budget on non-important region
- Poor segmentation mask → imprecise quality boundaries

### 3. Image Composition
- **Centered subjects** → prominence boost → better protection
- **Small subjects** → no prominence boost → rely only on intent
- **Multiple subjects** → weight spreading → each gets less focus

### 4. Scene Complexity
- Many objects → complex weight map → nuanced QP allocation
- Few objects → binary weight map → stark quality differences
- Background texture → saliency fills in → moderate compression
- Uniform background → no saliency → maximum compression

### 5. Compression Mode
- **AVIF Extreme Mode:** Binary person/background split
- **PNG Pixel Mode:** Uses full QP range smoothly
- **HEVC Mode:** Uses full QP range with AQ

---

## How to Interpret QP Map Visualizations

When you see the `_qp_map.jpg` visualization:

**Color Legend:**
- 🔴 Red/Orange (QP 0-15): **Lossless/near-lossless** - persons, faces, critical objects
- 🟡 Yellow/Green (QP 20-35): **High/medium quality** - secondary objects, textures
- 🔵 Cyan/Blue (QP 40-51): **Heavy compression** - backgrounds, sky, unimportant areas

**What to look for:**
- Are person regions red? ✅ Good
- Is background blue? ✅ Good (saves bits for person)
- Are textures yellow/green? ✅ Good (saliency working)
- Is entire image one color? ❌ Bad (no differentiation)

---

## Summary: The Weight Journey

```
Scene "portrait" → person gets base weight 1.0
       ↓
YOLOv8 detects person → creates mask [H,W] with person pixels = 1
       ↓
Prominence checks: 30% area + centered → boost +0.5 → final weight 1.0 (capped)
       ↓
Weight map: person pixels = 1.0, others = 0.0
       ↓
Saliency fills texture areas → updates empty pixels to 0.3
       ↓
Segmentation identifies sky → forces low pixels to 0.1
       ↓
Convert to QP: weight 1.0 → QP 0 (extreme mode) or QP 10 (normal)
       ↓
Smooth with 5x5 blur (extreme) or 15x15 blur (normal)
       ↓
Final QP map: person QP 0-5, texture QP 30-40, sky QP 45-51
       ↓
AVIF encoder: CRF 10 + AQ 4.0 → extreme differential compression
       ↓
Result: Perfect person, destroyed background
```

**Every step matters, but Scene Intent and Object Detection are the foundation. Everything else fine-tunes their output.**
