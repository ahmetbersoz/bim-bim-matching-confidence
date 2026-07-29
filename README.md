# BIM-to-BIM Matching

Compute per-element **3D-IoU** and **3D-Compactness** metrics between two IFC files (ground truth vs. predicted), using **Object-Oriented Bounding Boxes (OBBs)**. Preserves full IFC metadata throughout the process.

## Features

- Load IFC files preserving `GlobalId`, `Name`, `LongName`, `IfcType`, attributes, and property sets
- Match GT and PRED spaces (`IfcSpace`) by OBB IoU, then compare the elements contained in each matched space
- Global and per-space alignment of PRED to GT with several modes:
  - `icp` — translation-only ICP (X/Y/Z, no rotation)
  - `icp_xy` — XY-plane ICP
  - `icp_xyz_rz` — X/Y/Z translation + rotation around the Z axis only
  - `centroid` — centroid translation
  - `none`
- Optional model-wide ICP refinement derived from a single target space (`--target-space-guid`)
- Exact OBB intersection/union math with inclusion-exclusion for pairwise IoU
- Footprint-based **2D IoU** for matched spaces: XY-projected footprints are rasterized onto a shared occupancy grid (handles non-convex rooms), reported as `space_iou_2d` / `iou2d_footprint` alongside the 3D IoU
- Room-type extraction for spaces (`LongName` → `ObjectType` → `Pset_SpaceCommon`), reported as `gt_room_type` / `pred_room_type` in the outputs
- Floor-area comparison CSV for matched spaces
- Open3D visualization of GT (green) vs. PRED (red) before/after alignment, globally and per space
- Session-scoped outputs: every run writes into its own timestamped directory
- Export results to JSON and CSV formats, plus combined `.ply` meshes per matched space

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy
- open3d >= 0.18.0
- ifcopenshell >= 0.7.0

## Usage

```bash
python main.py --gt <ground_truth.ifc> --pred <predicted.ifc> [options]
```

Running `python main.py` with no arguments uses the `DEFAULT_ARGS` block defined at the bottom of the script (convenient for repeated experiments — edit it to point at your models).

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gt` | Path to ground-truth IFC file (required) | - |
| `--pred` | Path to predicted/reconstructed IFC file (required) | - |
| `--target-space-guid` | GlobalId of a GT `IfcSpace` used to derive a global (model-wide) ICP refinement. Disables per-space alignment | - |
| `--mesh-output-dir` | Base directory for session outputs (meshes, CSVs) | `out` |
| `--floor-area-csv` | CSV path for floor-area comparisons of matched spaces | `out/matched_space_floor_areas.csv` |
| `--align` | Global alignment: `icp`, `icp_xy`, `icp_xyz_rz`, `centroid`, `none` | `centroid` |
| `--space-align` | Per-space alignment for matched PRED spaces: `icp`, `icp_xy`, `icp_xyz_rz`, `centroid`, `none` | `icp` |
| `--visualize` | Show GT (green) vs. PRED (red) before/after global alignment | off |
| `--visualize-per-space` | Show each matched space pair before/after per-space alignment | off |
| `--space-match-thresh` | IoU threshold to accept a GT/PRED space match | `0.5` |
| `--epsilon` | IoU threshold to consider an element correspondence | `0.05` |
| `--footprint-cell` | Grid cell size (m) for rasterizing space footprints in the 2D IoU | `0.05` |
| `--save-json` | Path to save JSON report | - |
| `--save-csv-prefix` | Prefix for CSV output files (written inside the session directory) | `metrics` |
| `--ifc-classes` | IFC classes to include (space- or comma-separated) | `IfcSpace IfcWall IfcWallStandardCase` |
| `--include-unmatched` | Elements in unmatched spaces: `ignore` or `global` (aggregate in one bucket) | `ignore` |
| `--inside-eps` | Tolerance for half-space tests / plane membership | `1e-7` |
| `--ie-cap` | Max K for exact inclusion-exclusion before pairwise approximation | `8` |

### Example

```bash
python main.py \
    --gt ground_truth.ifc \
    --pred reconstructed.ifc \
    --target-space-guid 1UABPD7uD69BRTD38UZ2$k \
    --align centroid \
    --space-align none \
    --space-match-thresh 0.25 \
    --epsilon 0.10 \
    --save-json metrics_obb.json \
    --save-csv-prefix out/metrics_obb \
    --ifc-classes IfcWall IfcWallStandardCase IfcSpace IfcDoor IfcWindow
```

## Output Files

Each run creates a timestamped session directory under `--mesh-output-dir`:

```
out/<timestamp>__pred_<pred_name>__gt_<gt_name>/
├── <prefix>_per_gt.csv        # Per-element metrics for GT elements (incl. room types, 2D footprint IoU for spaces)
├── <prefix>_per_pred.csv      # Per-element metrics for PRED elements (incl. room types, 2D footprint IoU for spaces)
├── <prefix>_edges.csv         # Pairwise IoU values between matched elements (incl. 2D IoU for space pairs)
├── matched_space_floor_areas.csv  # Floor-area comparison for matched spaces (incl. space_iou_2d)
└── meshes/
    └── <match>/gt_space_with_elements.ply, pred_space_with_elements.ply
```

The JSON report (`--save-json`) contains the complete metrics in one file.

## Helper Scripts

The `helpers/` directory contains utility scripts:

- `move_storey.py` - Edit IFC storey structure (rename, remove, reassign elements)
- `rotate_180.py` - Rotate all storeys and elements by 180° around Z-axis
- `print_spaces.py` - List and analyze IfcSpace boundary relationships
- `split_by_storey.py` - Split an IFC model into separate files per storey

## Citation

This implementation is inspired by the evaluation metrics proposed in the following paper:

> Liu, Y., Huang, H., Gao, G., Ke, Z., Li, S., & Gu, M. (2025). Dataset and benchmark for as-built BIM reconstruction from real-world point cloud. *Automation in Construction*, 173, 106096. https://doi.org/10.1016/j.autcon.2025.106096

```bibtex
@article{liu2025dataset,
  title={Dataset and benchmark for as-built BIM reconstruction from real-world point cloud},
  author={Liu, Yudong and Huang, Han and Gao, Ge and Ke, Ziyi and Li, Shengtao and Gu, Ming},
  journal={Automation in Construction},
  volume={173},
  pages={106096},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.autcon.2025.106096}
}
```

## License

See [LICENSE](LICENSE) file for details.
