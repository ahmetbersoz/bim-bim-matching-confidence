# BIM-to-BIM Matching

Compute per-element **3D-IoU** and **3D-Compactness** metrics between two IFC files (ground truth vs. predicted). Preserves full IFC metadata throughout the process.

## Features

- Load IFC files preserving `GlobalId`, `Name`, `IfcType`, attributes, and property sets
- Align predicted model to ground truth (ICP or centroid-based)
- Compute pairwise IoU matrix between elements
- Calculate per-element metrics for both GT and predicted models
- Export results to JSON and CSV formats

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy
- open3d >= 0.18.0
- ifcopenshell >= 0.7.0
- trimesh

## Usage

```bash
python main.py --gt <ground_truth.ifc> --pred <predicted.ifc> [options]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gt` | Path to ground-truth IFC file (required) | - |
| `--pred` | Path to predicted/reconstructed IFC file (required) | - |
| `--align` | Alignment method: `icp`, `centroid`, or `none` | `icp` |
| `--backend` | CSG backend: `auto`, `exact`, or `voxel` | `auto` |
| `--voxel-size` | Voxel pitch in meters for fallback operations | `0.03` |
| `--epsilon` | IoU threshold to consider a correspondence | `0.05` |
| `--save-json` | Path to save JSON report | - |
| `--save-csv-prefix` | Prefix for CSV output files | - |
| `--ifc-classes` | Filter by IFC classes (e.g., `IfcWall IfcDoor`) | all |

### Example

```bash
python main.py \
    --gt ground_truth.ifc \
    --pred reconstructed.ifc \
    --align icp \
    --backend auto \
    --voxel-size 0.05 \
    --epsilon 0.1 \
    --save-json metrics.json \
    --save-csv-prefix out/metrics \
    --ifc-classes IfcWall IfcSlab
```

## Output Files

| File | Description |
|------|-------------|
| `metrics.json` | Complete metrics report in JSON format |
| `<prefix>_per_gt.csv` | Per-element metrics for ground truth elements |
| `<prefix>_per_pred.csv` | Per-element metrics for predicted elements |
| `<prefix>_edges.csv` | Pairwise IoU values between matched elements |

## Helper Scripts

The `helpers/` directory contains utility scripts:

- `move_storey.py` - Edit IFC storey structure (rename, remove, reassign elements)
- `rotate_180.py` - Rotate all storeys and elements by 180° around Z-axis
- `print_spaces.py` - List and analyze IfcSpace boundary relationships

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
