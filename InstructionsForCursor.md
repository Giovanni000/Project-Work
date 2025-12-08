Perfetto, allora ora facciamo il “secondo step” per Cursor, usando il **JSON `Training/pcl_roi_config.json`** invece del dizionario hardcoded.

Incolla questo prompt in una nuova chat di Cursor (stesso repo del notebook `step2_efficientad_per_connector.ipynb`):

---

### Prompt per Cursor (uso ROI da JSON + mask più forte + debug raw/masked)

````text
We are working on the notebook:

- `step2_efficientad_per_connector.ipynb`

Context (already implemented):
- Teacher–Student ResNet18 EfficientAD-style
- multi-layer features (e.g. layer2/layer3/layer4)
- one model per connector
- spatial weight mask per connector from OK variance
- robust anomaly score based on a fused anomaly map

NOW:
I have created a JSON file with manual PCL ROIs for each connector:

- `Training/pcl_roi_config.json`

with this structure:

```json
{
  "conn1": [y1, y2, x1, x2],
  "conn2": [y1, y2, x1, x2],
  ...
}
````

Coordinates are in image space (0..127).

I want you to:

1. Load this JSON once and use it to override the spatial masks (strong focus on PCL).
2. Make the spatial mask more aggressive (strong suppression outside the ROI).
3. Add a debug function to compare RAW vs MASKED anomaly maps for one OK and one KO image.

Please do NOT change the overall training and scoring logic, only the mask and debug parts.

---

## A) LOAD ROI CONFIG JSON (GLOBAL)

At the top of the notebook (after imports), add:

```python
import json

PCL_ROI_CONFIG_PATH = "Training/pcl_roi_config.json"

try:
    with open(PCL_ROI_CONFIG_PATH, "r") as f:
        PCL_ROI_CONFIG = json.load(f)
    print(f"[ROI] Loaded PCL ROI config from {PCL_ROI_CONFIG_PATH}")
except FileNotFoundError:
    PCL_ROI_CONFIG = {}
    print(f"[ROI] WARNING: {PCL_ROI_CONFIG_PATH} not found. No manual ROIs will be used.")
```

So we have a global dict:

* `PCL_ROI_CONFIG[connector_name] = [y1, y2, x1, x2]` if available.

---

## B) STRONGER SPATIAL MASK + ROI OVERRIDE

Near the mask code, define global mask bounds:

```python
MASK_MIN_WEIGHT = 0.1   # strong suppression outside ROI
MASK_MAX_WEIGHT = 2.0   # strong emphasis inside ROI
```

Then modify `compute_spatial_weight_mask_for_connector(...)` as follows:

1. Keep the existing logic that computes `weights` from OK variance (std → normalized → inverted → scaled).
2. After you obtain the initial `weights` array, add:

```python
# Log the range before ROI override
print(f"[MASK] {connector_name}: variance-based weights range = ({weights.min():.3f}, {weights.max():.3f})")
```

3. Then, if the connector has a manual ROI in `PCL_ROI_CONFIG`, override the mask completely:

```python
if connector_name in PCL_ROI_CONFIG:
    y1, y2, x1, x2 = PCL_ROI_CONFIG[connector_name]
    roi_mask = np.full_like(weights, MASK_MIN_WEIGHT, dtype=np.float32)
    # Clamp coordinates
    h, w = roi_mask.shape
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    roi_mask[y1:y2, x1:x2] = MASK_MAX_WEIGHT
    weights = roi_mask
    print(f"[MASK] Applied manual PCL ROI override for {connector_name} with coords (y1={y1}, y2={y2}, x1={x1}, x2={x2})")
```

4. Finally, save `weights` as before (`spatial_mask_{connector_name}.npy`) and use it everywhere else exactly as you already do.

This way:

* If a ROI is defined in the JSON → ONLY the ROI geometry matters (PCL zone is strongly emphasized, rest heavily suppressed).
* If not defined → fallback to variance-based mask.

---

## C) DEBUG FUNCTION: RAW vs MASKED ANOMALY MAP (OK + KO)

I want a visualization to see how much the mask changes the anomaly maps.

Add a function, for example:

```python
def debug_compare_raw_vs_masked_anomaly_map(
    connector_name,
    csv_path="data/dataset.csv",
    ko_labels=("KO",),
    models_dir="models",
    topk_percent=0.01
):
    """
    - Load one OK and one KO image for the given connector_name.
      * OK: label == "OK"
      * KO: label in ko_labels (ignore 'OCCLUSION' / 'PARTIAL OCCLUSION' etc.)
    - For each image:
        1. Load original RGB image (128x128 for display).
        2. Run Teacher + Student to get multi-layer teacher_feats and student_feats.
        3. Compute fused anomaly map WITHOUT spatial mask (amap_raw).
        4. Compute fused anomaly map WITH spatial mask (amap_masked).
        5. Upsample both maps to image resolution (IMG_SIZE x IMG_SIZE).
        6. Normalize each map separately to [0,1] for visualization.
        7. Compute scalar anomaly scores from amap_raw and amap_masked (using the same top-k or pooling method used in the scoring function).
    - Plot a 2x3 matplotlib figure:
        Row 1: OK image, OK amap_raw, OK amap_masked.
        Row 2: KO image, KO amap_raw, KO amap_masked.
    - Print to console:
        * min/max of the spatial mask used
        * scalar scores (raw vs masked) for OK and KO
    """
```

Implementation hints:

* Reuse the existing fused anomaly map logic you already have (the same code path that `compute_anomaly_score_from_features` uses, but:

  * for `amap_raw`, skip applying the spatial mask.
  * for `amap_masked`, apply it as usual.

* For image loading:

  * use `PIL.Image.open(image_path).convert("RGB")`
  * resize to `(IMG_SIZE, IMG_SIZE)` for consistent visualization.

* For upsampling maps:

  ```python
  amap_raw_up = F.interpolate(amap_raw, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
  amap_masked_up = F.interpolate(amap_masked, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
  ```

* For normalization:

  ```python
  def normalize_map(m):
      m = m - m.min()
      if m.max() > 0:
          m = m / m.max()
      return m
  ```

* For plotting:

  ```python
  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  # row 0: OK
  axes[0,0].imshow(ok_img); axes[0,0].set_title("OK image"); axes[0,0].axis("off")
  axes[0,1].imshow(ok_raw_norm, cmap="jet"); axes[0,1].set_title("OK raw anomaly"); axes[0,1].axis("off")
  axes[0,2].imshow(ok_masked_norm, cmap="jet"); axes[0,2].set_title("OK masked anomaly"); axes[0,2].axis("off")
  # row 1: KO
  ...
  plt.tight_layout()
  ```

At the end of the notebook, add an example cell:

```python
# Example manual debug (I will call this for specific connectors):
# debug_compare_raw_vs_masked_anomaly_map("conn2")
```

---

## D) CONSTRAINTS

* Do NOT modify the main training loop or the threshold computation logic (except that they now use the new spatial masks).
* Assume that the training and scoring pipeline already handles:

  * Teacher–Student
  * multi-layer features
  * spatial mask loading from `spatial_mask_{connector}.npy`.

Just:

* load `Training/pcl_roi_config.json`,
* override the masks based on these ROIs,
* strengthen out-of-ROI suppression (MASK_MIN_WEIGHT),
* and add the debug RAW vs MASKED anomaly visualization.

```

---

Con questo Cursor userà il JSON che hai fatto con il tool delle ROI, applicherà la mask fortissima sul PCL, e ti darà anche il confronto visivo tra mappa grezza e mappa pesata: così capisci subito se ha smesso di fissarsi sui cavi e ha iniziato a guardare dove deve.
```
