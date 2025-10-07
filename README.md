# BIM to BIM Matching Evoluation

Per-element **3D-IoU** and **3D-Compactness** between two IFCs (GT vs PRED) using **Open3D** CSG with voxel fallback—while preserving full IFC metadata.

## Features
- Load IFCs (keeps `GlobalId`, `Name`, `IfcType`, attributes, and Psets).
- Align PRED→GT
- Pairwise IoU 
- Calculate per-element metrics for **both** GT and PRED, plus dataset means.
- Exports:
  - `metrics.json`
  - `out/metrics_per_gt.csv`
  - `out/metrics_per_pred.csv`
  - `out/metrics_edges.csv`


## Install
```bash
pip install -r requirements.txt
```

## Reference
This implementation follows the component-level metrics proposed in **Liu et al., 2025**, which accompanies this repo as a reference. 
Liu, Y., Huang, H., Gao, G., Ke, Z., Li, S., & Gu, M. (2025). Dataset and benchmark for as-built BIM reconstruction from real-world point cloud. Automation in Construction, 173, 106096.
