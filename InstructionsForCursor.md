
````text
We have already upgraded `step2_efficientad_per_connector.ipynb` to:

- use a Teacher–Student ResNet18 (Teacher pretrained on ImageNet, frozen; Student random-init, trainable),
- use multi-layer features (e.g., layer2/layer3/layer4),
- compute a fused anomaly map and a robust scalar score (top-k / pooled) per image,
- use a spatial weight mask per connector, computed from OK images and saved as `spatial_mask_{connector}.npy`.

Now I want to add ONLY visualization utilities on top of the existing logic, without changing the core training/scoring pipeline.

Please:

1) Add a simple config at the top of the notebook:

```python
DEBUG_CONNECTOR = "conn1"  # I will change this manually if needed
````

2. Add a function to visualize the spatial mask for a connector:

```python
def visualize_spatial_mask(connector_name=DEBUG_CONNECTOR, models_dir="models"):
    """
    Load spatial_mask_{connector_name}.npy from models_dir
    and show it with matplotlib:
    - grayscale image
    - with a colorbar
    """
```

Implementation hints:

* `mask = np.load(models_dir + f"/spatial_mask_{connector_name}.npy")`
* `plt.imshow(mask, cmap="viridis" or "magma")`
* `plt.colorbar()`
* `plt.title(f"Spatial mask for {connector_name}")`

3. Add a function to compute and visualize an AVERAGE anomaly heatmap over OK images for one connector:

```python
def visualize_average_anomaly_map_for_connector(
    connector_name=DEBUG_CONNECTOR,
    csv_path="data/dataset.csv",
    max_ok_samples=50
):
    """
    For the given connector:
    - load up to max_ok_samples images with label == "OK"
    - for each image:
        * compute the fused anomaly map (before reducing to a scalar score),
          using the EXISTING pipeline: teacher features, student features,
          spatial mask, multi-layer fusion
        * upsample the fused anomaly map to image resolution
    - average all these maps -> average anomaly heatmap
    - visualize:
        * show the average anomaly map as a heatmap with matplotlib
    """
```

Please REUSE the existing helper you have for computing the fused anomaly map (the same logic used internally before you reduce to a scalar score).
If currently you only have a function that returns scalar scores, refactor it slightly so that you can also obtain the fused 2D map for a single image, and use that here.

4. Add a function to visualize anomaly heatmaps on example OK and KO images:

```python
def visualize_example_ok_ko_heatmaps(
    connector_name=DEBUG_CONNECTOR,
    csv_path="data/dataset.csv",
    ko_labels=("KO",),
    models_dir="models"
):
    """
    - Select one OK image for this connector (label == "OK")
    - Select one KO image for this connector (label in ko_labels),
      ignoring occlusion-related labels (like 'OCCLUSION' / 'PARTIAL OCCLUSION').
    - For each selected image:
        * load the original RGB image
        * compute fused anomaly map with the current pipeline
        * upsample the map to the original image resolution
        * normalize to [0,1]
        * visualize:
            - original image
            - anomaly heatmap alone
            - overlay: original image + semi-transparent anomaly heatmap
    """
```

Use matplotlib for visualization, with something like:

* `plt.imshow(image)` for the original
* `plt.imshow(anomaly_map, cmap="jet")` for the heatmap
* overlay example:

  ```python
  plt.imshow(image)
  plt.imshow(anomaly_map_norm, cmap="jet", alpha=0.5)
  ```

5. Make sure none of these visualization functions change the training or threshold logic.
   They should only:

* load existing models (Teacher+Student),
* load the spatial mask,
* compute features and fused anomaly maps using the already implemented code,
* and plot things.

At the end of the notebook, add a small example section:

```python
# Example debug calls (I will run them manually):
# visualize_spatial_mask()
# visualize_average_anomaly_map_for_connector()
# visualize_example_ok_ko_heatmaps()
```

Do not run long training loops automatically; keep these as manual calls for debugging and interpretation.

```

---

Così Cursor usa esattamente quello che ha già implementato e tu alla fine ti ritrovi:

- **la mask per connettore** disegnata  
- **l’average heatmap sugli OK** (capisci dove guarda davvero il modello)  
- **un esempio OK/KO con overlay della heatmap** (vedi se il PCL “si accende” come deve).
::contentReference[oaicite:0]{index=0}
```
