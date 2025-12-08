
### Prompt per Cursor – nuovo `crop_connectors_rgb.py` + update notebook

````text
We are working in the connectors anomaly detection repo.

There is already a script:

- `Codice/crop_connectors.py`

that:
- reads aligned images (top view),
- uses a ROI config JSON to crop each connector,
- normalizes the crops,
- and saves grayscale PNGs in `Data/connectors/...` (Conn1, Conn2, etc.).

I want to ADD a NEW script, without breaking the existing grayscale pipeline:

- `Codice/crop_connectors_rgb.py`

that will:
- take the aligned top images from: `Data/aligned_top`   (if the actual path has a slightly different name that already exists in the repo, use that one),
- use the SAME ROI logic as `crop_connectors.py` (same JSON, same iteration over connectors),
- but produce **RGB crops** instead of grayscale,
- and save them in a PARALLEL folder:

- input:  `Data/aligned_top`
- output: `Data/connectors_rgb/Conn1`, `Data/connectors_rgb/Conn2`, ...

with the same file naming convention used in `Data/connectors`.

---

## 1. New script `crop_connectors_rgb.py`

Create a new file `Codice/crop_connectors_rgb.py` by reusing as much as possible from `crop_connectors.py`, but:

1. Keep the same CLI arguments and overall structure:
   - `--input-dir`
   - `--output-dir`
   - `--roi-config`
   - `--margin`
   - logging / progress printing
   - the per-connector loop that builds subfolders like `Conn1`, `Conn2`, etc.

2. Use the same ROI JSON and cropping logic:
   - Reuse the helper that reads ROI config (or reimplement the same logic).
   - Use the same mapping between connector names and ROI entries (e.g. conn1, conn2, etc.).
   - The only difference is how the ROI is normalized and saved (RGB vs grayscale).

3. Implement a new `normalize_roi(img: np.ndarray) -> np.ndarray` with this behavior:
   - Input: `img` is a **BGR** ROI from OpenCV, 3-channel.
   - We want to keep full color information, but slightly normalize illumination on the luminance channel only.
   - Suggested implementation:

   ```python
   def normalize_roi(img: np.ndarray) -> np.ndarray:
       """
       Keep color information, apply light illumination normalization on the ROI
       and return float32 in [0, 1] with 3 channels (BGR).
       """
       # Work in LAB to adjust only luminance
       lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
       l, a, b = cv2.split(lab)

       # Light CLAHE (we already have alignment pre-processing upstream)
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
       l_eq = clahe.apply(l)

       lab_eq = cv2.merge((l_eq, a, b))
       bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

       # Normalize to [0, 1] with 3 channels
       normalized = bgr_eq.astype(np.float32) / 255.0
       return normalized
````

* Then, when saving:

  * convert back to uint8: `(normalized * 255).astype(np.uint8)`
  * and write PNG **in color** (3 channels) with `cv2.imwrite`.

4. The main entry point should allow running:

   ```bash
   python Codice/crop_connectors_rgb.py \
       --input-dir "Data/aligned_top" \
       --output-dir "Data/connectors_rgb" \
       --roi-config "Codice/roi_config.json" \
       --margin 8
   ```

   * Use `margin=8` as default if appropriate (or keep the same default as in `crop_connectors.py`).
   * For each connector ID, create a subfolder under `Data/connectors_rgb` with the SAME naming convention as the grayscale version (`Conn1`, `Conn2`, etc.) so the layout mirrors `Data/connectors`.

5. Do NOT modify `crop_connectors.py`.

   * The grayscale pipeline must remain available.
   * The new RGB pipeline is an alternative living in `crop_connectors_rgb.py`.

---

## 2. Update the notebook to use RGB connectors

Now update the notebook responsible for EfficientAD-style training:

* `step2_efficientad_per_connector.ipynb`

Currently it loads grayscale crops from `Data/connectors/...` and builds datasets / dataloaders accordingly.

I want to switch this notebook to use the NEW RGB crops under:

* `Data/connectors_rgb/...` (same subfolder structure: Conn1, Conn2, …)

Tasks:

1. Locate where the dataset paths are defined:

   * Anywhere the code currently points to `Data/connectors` (or equivalent).
   * Replace these paths to point to `Data/connectors_rgb` instead.

2. Check the image loading code:

   * If it uses OpenCV, make sure images are read in color (e.g. `cv2.IMREAD_COLOR`) and converted to RGB if needed.
   * If it uses PIL, ensure `.convert("RGB")` is applied.
   * Remove any explicit conversion to grayscale or channel squeezing (`unsqueeze(0)`, etc.).

3. Make sure the tensors are **3-channel**:

   * After transforms, the shape for a single image should be `(3, H, W)` instead of `(1, H, W)`.
   * Remove any code that assumes a single channel (e.g. concatenation like `x = x.repeat(3, 1, 1)` to feed a 1-channel image into a ResNet). That should not be needed anymore since we now have true RGB.

4. Normalization parameters:

   * If the code uses a normalization transform with mean/std for grayscale (single value), update it to 3-channel values.
   * For now you can use ImageNet-style normalization as a reasonable default:

   ```python
   transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225],
   )
   ```

   * Apply this after converting to tensor in the usual way.

5. Make sure the EfficientAD / ResNet backbone is still expecting 3-channel input:

   * ResNet18 default expects 3 channels, so as long as we provide `(3, H, W)` it should work without further changes.
   * Remove any workaround that was there just to adapt grayscale to 3-channel (e.g. repeating the single channel).

6. Debug plots:

   * Where the notebook currently visualizes:

     * `OK Raw Anomaly`, `KO Raw Anomaly`,
     * `OK Masked Anomaly`, `KO Masked Anomaly`,
     * and the corresponding input crops,
   * Make sure the image show functions handle RGB correctly (no accidental `cmap="gray"` for the input crops).
   * It is fine to leave anomaly maps as 2D heatmaps with color map, but the input crops should be shown in true color.

7. At the end of the notebook, add/update a short markdown cell that explains:

   * We are now using **RGB connector crops** from `Data/connectors_rgb/ConnX/...`
   * The crops are generated by `Codice/crop_connectors_rgb.py` using LAB + CLAHE on the luminance channel but preserving color.
   * The model now receives 3-channel input, and normalization uses ImageNet-like mean/std.

---

Please:

* First, implement `crop_connectors_rgb.py` as described, reusing as much logic as possible from `crop_connectors.py`.
* Then update the notebook `step2_efficientad_per_connector.ipynb` to read from `Data/connectors_rgb` and handle 3-channel RGB tensors in the whole pipeline.
* Keep everything clean, well commented, and do not break the existing grayscale scripts.

```

