#!/usr/bin/env python3
# ifc_metrics_with_metadata_o3d.py
# Compute per-element 3D-IoU & 3D-Compactness between two IFCs, preserve IFC metadata.
# Refactored to use Open3D wherever possible (mesh ops, voxelization, booleans, transforms, ICP, viz).
# - Loads IFCs with ifcopenshell
# - Aligns PRED -> GT (ICP if available, else centroid translation)
# - Builds pairwise IoU matrix (O3D boolean or voxel fallback)
# - Reports per-element metrics for *both* GT and PRED
# - Preserves metadata (GlobalId, Name, Type, attributes, and Psets)

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------- Dependencies ----------
try:
    import ifcopenshell
    import ifcopenshell.geom
    from ifcopenshell.util import unit as ifc_unit
    try:
        # pset helper is optional but common
        from ifcopenshell.util.element import get_psets as ifc_get_psets
    except Exception:
        ifc_get_psets = None
except Exception:
    print("This script needs ifcopenshell with geometry enabled. Try: pip install ifcopenshell", file=sys.stderr)
    raise

# Kept as a fallback for rare cases; primary ops use Open3D now
try:
    import trimesh
    TRIMESH_OK = True
except Exception:
    TRIMESH_OK = False

try:
    import open3d as o3d
    OPEN3D_OK = True
except Exception:
    OPEN3D_OK = False
    print("This script needs open3d. Try: pip install open3d", file=sys.stderr)
    raise

VERBOSE = True


def log_step(message: str) -> None:
    if VERBOSE:
        print(f"[step] {message}", flush=True)


_LOG_STATE = {
    "boolean_intersection_fallback": False,
    "boolean_union_fallback": False,
    "voxel_union_fallback": False,
}

# ---------- Data structures ----------

@dataclass
class Meta:
    GlobalId: str
    IfcType: str
    Name: str
    ObjectType: Optional[str]
    PredefinedType: Optional[str]
    Tag: Optional[str]
    ExpressID: Optional[int]
    Psets: Optional[Dict[str, Any]]  # nested dict of {Pset: {Prop: value}}

@dataclass
class Comp:
    idx: int                 # index in our arrays
    guid: str
    etype: str
    meta: Meta
    mesh: o3d.geometry.TriangleMesh  # Open3D mesh (primary)
    volume: float
    aabb_min: np.ndarray
    aabb_max: np.ndarray

# ---------- Open3D helpers ----------

def _o3d_mesh_copy(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # deep copy helper (o3d meshes are mutable)
    return o3d.geometry.TriangleMesh(mesh)

def _o3d_concat_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    """Concatenate triangle meshes into a single mesh (index-safe)."""
    if not meshes:
        return o3d.geometry.TriangleMesh()
    all_verts = []
    all_tris = []
    v_offset = 0
    for m in meshes:
        if len(m.vertices) == 0 or len(m.triangles) == 0:
            continue
        V = np.asarray(m.vertices)
        F = np.asarray(m.triangles)
        all_verts.append(V)
        all_tris.append(F + v_offset)
        v_offset += V.shape[0]
    if not all_verts:
        return o3d.geometry.TriangleMesh()
    V = np.vstack(all_verts)
    F = np.vstack(all_tris)
    out = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F)
    )
    return out

def _o3d_clean_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    """Best-effort cleanup akin to trimesh processing."""
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
    except Exception:
        pass  # keep going with best effort

def _o3d_bounds(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    aabb = mesh.get_axis_aligned_bounding_box()
    return np.asarray(aabb.min_bound), np.asarray(aabb.max_bound)

def _o3d_is_watertight(mesh: o3d.geometry.TriangleMesh) -> bool:
    try:
        return mesh.is_watertight()
    except Exception:
        # conservative default
        return False

def _o3d_signed_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """Compute signed volume of a (possibly watertight) triangle mesh via divergence theorem."""
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    # sum over tetrahedra w.r.t. origin: 1/6 * dot(v0, cross(v1, v2))
    cross = np.cross(v1, v2)
    vol = np.einsum('ij,ij->i', v0, cross).sum() / 6.0
    return float(vol)

def _o3d_mesh_volume_or_hull(mesh: o3d.geometry.TriangleMesh) -> float:
    """Positive volume using mesh if watertight, else convex hull volume."""
    if _o3d_is_watertight(mesh):
        return abs(_o3d_signed_volume(mesh))
    try:
        hull, _ = mesh.compute_convex_hull()
        return abs(_o3d_signed_volume(hull))
    except Exception:
        # last resort: zero
        return 0.0

def _o3d_transform_inplace(meshes: List[Comp], T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix T must be 4x4.")
    for c in meshes:
        c.mesh.transform(T)
        c.aabb_min, c.aabb_max = _o3d_bounds(c.mesh)

def _o3d_sample_points(meshes: List[o3d.geometry.TriangleMesh], target_pts: int = 80000) -> o3d.geometry.PointCloud:
    total_area = sum(m.get_surface_area() for m in meshes if len(m.triangles) > 0)
    points = []
    if total_area <= 0:
        if not meshes:
            return o3d.geometry.PointCloud()
        V = np.vstack([np.asarray(m.vertices) for m in meshes if len(m.vertices) > 0]) if meshes else np.zeros((0, 3))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    for m in meshes:
        if len(m.triangles) == 0:
            continue
        n = max(200, int(target_pts * (m.get_surface_area() / total_area)))
        pts = m.sample_points_uniformly(number_of_points=n)
        points.append(np.asarray(pts.points))
    P = np.vstack(points) if points else np.zeros((0, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))

def _o3d_boolean_intersection(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> Optional[o3d.geometry.TriangleMesh]:
    try:
        return a.boolean_intersection(b)
    except Exception:
        if not _LOG_STATE["boolean_intersection_fallback"]:
            log_step("  Open3D boolean_intersection failed; will fall back to voxel IoU where needed")
            _LOG_STATE["boolean_intersection_fallback"] = True
        return None

def _o3d_boolean_union(meshes: List[o3d.geometry.TriangleMesh]) -> Optional[o3d.geometry.TriangleMesh]:
    if not meshes:
        return None
    if len(meshes) == 1:
        return _o3d_mesh_copy(meshes[0])
    try:
        acc = _o3d_mesh_copy(meshes[0])
        for m in meshes[1:]:
            acc = acc.boolean_union(m)
        return acc
    except Exception:
        if not _LOG_STATE["boolean_union_fallback"]:
            log_step("  Open3D boolean_union failed; will fall back to voxel-based union approximation")
            _LOG_STATE["boolean_union_fallback"] = True
        return None

def _o3d_voxelize_indices(mesh: o3d.geometry.TriangleMesh, pitch: float, origin: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Voxelize mesh into a fixed grid defined by (origin, pitch, dims)."""
    # shift mesh so that origin maps to (0,0,0) in voxel grid space
    m = _o3d_mesh_copy(mesh)
    m.translate(-origin, relative=True)
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(m, voxel_size=pitch)
    occ = np.zeros((dims[0], dims[1], dims[2]), dtype=bool)
    if vg is None or len(vg.get_voxels()) == 0:
        return occ
    for v in vg.get_voxels():
        idx = v.grid_index  # IntVector3d
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
            occ[i, j, k] = True
    return occ

def _o3d_boxes_from_indices(indices: np.ndarray, pitch: float, origin: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create a mesh by instancing boxes at occupied voxel indices (fallback for union)."""
    boxes = []
    half = pitch / 2.0
    for (i, j, k) in np.argwhere(indices):
        center = origin + np.array([i + 0.5, j + 0.5, k + 0.5]) * pitch
        box = o3d.geometry.TriangleMesh.create_box(width=pitch, height=pitch, depth=pitch)
        box.translate(center - np.array([half, half, half]), relative=False)
        boxes.append(box)
    return _o3d_concat_meshes(boxes) if boxes else o3d.geometry.TriangleMesh()

# ---------- IFC -> mesh & metadata ----------

def _ifc_length_scale_m(ifc) -> float:
    try:
        return float(ifc_unit.calculate_unit_scale(ifc))
    except Exception:
        return 1.0

def _flatten_psets(psets: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for pset, props in (psets or {}).items():
        if isinstance(props, dict):
            for k, v in props.items():
                flat[f"{pset}:{k}"] = v
        else:
            flat[pset] = props
    return flat

def _product_meta(p) -> Meta:
    guid = getattr(p, "GlobalId", None)
    name = getattr(p, "Name", None)
    etype = p.is_a()
    objtype = getattr(p, "ObjectType", None)
    tag = getattr(p, "Tag", None)
    predefined = None
    try:
        predefined = getattr(p, "PredefinedType", None)
        if isinstance(predefined, ifcopenshell.entity_instance):
            predefined = str(predefined)
    except Exception:
        pass
    expid = None
    try:
        expid = int(p.id())
    except Exception:
        pass
    psets = None
    if ifc_get_psets is not None:
        try:
            psets = ifc_get_psets(p, include_inherited=True, recursive=True)
        except Exception:
            psets = None
    return Meta(
        GlobalId=guid or "",
        IfcType=etype,
        Name=name or "",
        ObjectType=objtype if objtype not in (None, "") else None,
        PredefinedType=predefined if predefined not in (None, "") else None,
        Tag=tag if tag not in (None, "") else None,
        ExpressID=expid,
        Psets=psets
    )

def _shape_to_o3d_mesh(shape, scale_to_m: float) -> Optional[o3d.geometry.TriangleMesh]:
    g = shape.geometry
    verts = getattr(g, "verts", None)
    if verts is None:
        verts = getattr(g, "vertices", None)
    faces = getattr(g, "faces", None)
    if faces is None:
        faces = getattr(g, "indices", None)
    if verts is None or faces is None:
        return None
    V = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    # 4x4 transform (column-major to standard)
    M = np.array(shape.transformation.matrix, dtype=np.float64).reshape(4, 4).T
    Vh = np.c_[V, np.ones((V.shape[0], 1))]
    Vw = (Vh @ M.T)[:, :3] * scale_to_m

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(Vw),
        triangles=o3d.utility.Vector3iVector(F)
    )
    _o3d_clean_mesh(mesh)
    return mesh

def _build_geom_settings(use_python_occ: bool) -> Optional["ifcopenshell.geom.settings"]:
    """Create geometry settings, optionally enabling pythonOCC output."""
    log_step(f"Configuring geometry settings (pythonOCC={'on' if use_python_occ else 'off'})")
    settings = ifcopenshell.geom.settings()
    if use_python_occ:
        try:
            settings.set("USE_PYTHON_OPENCASCADE", True)
            log_step("  pythonOCC enabled for geometry extraction")
        except Exception:
            log_step("  pythonOCC enable failed; will fall back")
            return None
    settings.set("DISABLE_OPENING_SUBTRACTIONS", False)
    settings.set("WELD_VERTICES", True)
    settings.set("APPLY_DEFAULT_MATERIALS", False)
    log_step("  Geometry settings ready")
    return settings

def _type_matches(etype: str, include_types: Optional[List[str]]) -> bool:
    """Return True if etype matches any entry in include_types.

    Matching is case-insensitive and permits entries without the 'Ifc' prefix,
    e.g. 'Wall' will match 'IfcWall'.
    """
    if not include_types:
        return True
    etl = (etype or "").strip().lower()
    for t in include_types:
        if t is None:
            continue
        tl = t.strip().lower()
        if not tl:
            continue
        if tl == etl:
            return True
        # allow user to provide 'Wall' to match 'IfcWall'
        if tl == etl.replace("ifc", ""):
            return True
    return False

def load_ifc_components(path: str, include_types: Optional[List[str]] = None) -> Tuple[List[Comp], float]:
    """Load IfcProduct elements from path and return list of Comp.

    If include_types is provided (e.g. ['IfcWall', 'IfcDoor'] or ['Wall','Door']),
    only products whose type matches the provided list will be processed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    log_step(f"Loading IFC components from {path}")
    ifc = ifcopenshell.open(path)
    schema_attr = getattr(ifc, "schema", None)
    if callable(schema_attr):
        try:
            schema_name = schema_attr()
        except Exception:
            schema_name = "unknown"
    else:
        schema_name = schema_attr if isinstance(schema_attr, str) else "unknown"
    log_step(f"  IFC schema: {schema_name}")
    scale = _ifc_length_scale_m(ifc)
    log_step(f"  Unit scale: {scale:.6f} meters per IFC unit")

    settings = _build_geom_settings(use_python_occ=True)
    using_occ = settings is not None
    if not using_occ:
        log_step("  pythonOCC geometry unavailable; using default triangulation settings")
        settings = _build_geom_settings(use_python_occ=False)
    fallback_settings = None

    comps: List[Comp] = []
    total_products = 0
    geometry_errors = 0
    suppressed_geometry_error_notice = False
    empty_volume_count = 0
    progress_step = 50
    for p in ifc.by_type("IfcProduct"):
        # Filter by representation present
        if not getattr(p, "Representation", None):
            continue

        # Filter by requested IFC classes if provided
        etype = p.is_a()
        if not _type_matches(etype, include_types):
            continue

        total_products += 1
        guid = getattr(p, "GlobalId", "") or "<no guid>"
        try:
            shape = ifcopenshell.geom.create_shape(settings, p)
        except Exception as exc:
            geometry_errors += 1
            if geometry_errors <= 5:
                log_step(f"  Geometry creation failed for {guid} ({etype}): {exc}")
            elif geometry_errors == 6 and not suppressed_geometry_error_notice:
                log_step("  Additional geometry creation failures encountered; suppressing further details")
                suppressed_geometry_error_notice = True
            continue
        mesh = _shape_to_o3d_mesh(shape, scale_to_m=scale)
        if (mesh is None or len(mesh.triangles) == 0) and using_occ:
            if fallback_settings is None:
                log_step("  Initializing fallback geometry settings (pythonOCC off)")
                fallback_settings = _build_geom_settings(use_python_occ=False)
            if fallback_settings is None:
                log_step(f"  Fallback geometry settings unavailable; skipping {guid} ({etype})")
                continue
            log_step(f"  Empty mesh from pythonOCC for {guid} ({etype}); retrying with fallback settings")
            try:
                shape = ifcopenshell.geom.create_shape(fallback_settings, p)
            except Exception as exc:
                log_step(f"  Fallback geometry also failed for {guid} ({etype}): {exc}")
                continue
            mesh = _shape_to_o3d_mesh(shape, scale_to_m=scale)
            if mesh is None or len(mesh.triangles) == 0:
                log_step(f"  No mesh produced for {guid} ({etype}) even after fallback; skipping")
                continue
            settings = fallback_settings
            using_occ = False
            log_step("  Switching to fallback geometry settings for remaining elements")

        if mesh is None or len(mesh.triangles) == 0:
            log_step(f"  Empty mesh for {guid} ({etype}); skipping")
            continue

        vol = _o3d_mesh_volume_or_hull(mesh)
        if vol <= 0.0:
            empty_volume_count += 1
            if empty_volume_count <= 5:
                log_step(f"  Non-positive volume for {guid} ({etype}); skipping")
            elif empty_volume_count == 6:
                log_step("  Additional non-positive volume elements suppressed")
            continue
        aabb_min, aabb_max = _o3d_bounds(mesh)
        meta = _product_meta(p)
        comps.append(Comp(
            idx=len(comps),
            guid=meta.GlobalId,
            etype=meta.IfcType,
            meta=meta,
            mesh=mesh,
            volume=vol,
            aabb_min=aabb_min,
            aabb_max=aabb_max
        ))
        if len(comps) % progress_step == 0:
            log_step(f"  Meshed {len(comps)} components so far")
    if not comps:
        raise RuntimeError(f"No meshable IfcProducts with positive volume found in {os.path.basename(path)}")
    log_step(f"Finished meshing {len(comps)} components (processed {total_products} candidates)")
    return comps, scale

# ---------- Alignment (PRED -> GT) ----------

def _centroid_align(pred_meshes: List[o3d.geometry.TriangleMesh], gt_meshes: List[o3d.geometry.TriangleMesh]) -> np.ndarray:
    pred_all = _o3d_concat_meshes(pred_meshes)
    gt_all = _o3d_concat_meshes(gt_meshes)
    T = np.eye(4)
    T[:3, 3] = (_o3d_bounds(gt_all)[0] + _o3d_bounds(gt_all)[1]) / 2.0 - ((_o3d_bounds(pred_all)[0] + _o3d_bounds(pred_all)[1]) / 2.0)
    return T

def _o3d_pcd_from_meshes(meshes: List[o3d.geometry.TriangleMesh], target_pts: int = 80000):
    return _o3d_sample_points(meshes, target_pts=target_pts)

def rigid_icp_align(pred_meshes: List[o3d.geometry.TriangleMesh], gt_meshes: List[o3d.geometry.TriangleMesh]) -> np.ndarray:
    T0 = _centroid_align(pred_meshes, gt_meshes)
    src = _o3d_pcd_from_meshes(pred_meshes)
    tgt = _o3d_pcd_from_meshes(gt_meshes)
    src.transform(T0)

    gt_all = _o3d_concat_meshes(gt_meshes)
    gmin, gmax = _o3d_bounds(gt_all)
    diag = float(np.linalg.norm(gmax - gmin))
    max_corr = max(0.02 * diag, 0.05)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(reg.transformation)
    return reg.transformation @ T0

# ---------- IoU + Compactness ----------

def _aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    return np.all(a_min <= b_max) and np.all(b_min <= a_max) and np.all(a_max >= b_min) and np.all(b_max >= a_min)

def _boolean_intersection_volume(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> float:
    inter = _o3d_boolean_intersection(a, b)
    if inter is None or len(inter.vertices) == 0 or len(inter.triangles) == 0:
        return 0.0
    _o3d_clean_mesh(inter)
    return _o3d_mesh_volume_or_hull(inter)

def _voxelize_to_grid(mesh: o3d.geometry.TriangleMesh, pitch: float, origin: np.ndarray, dims: np.ndarray) -> np.ndarray:
    return _o3d_voxelize_indices(mesh, pitch, origin, dims)

def _iou_voxel(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh, pitch: float) -> float:
    a_min, a_max = _o3d_bounds(a)
    b_min, b_max = _o3d_bounds(b)
    if not _aabb_overlap(a_min, a_max, b_min, b_max):
        return 0.0
    mn = np.minimum(a_min, b_min)
    mx = np.maximum(a_max, b_max)
    dims = np.ceil((mx - mn) / pitch).astype(int) + 1
    if np.prod(dims) > 400_000_000:
        raise MemoryError(f"Voxel grid too large: dims={tuple(int(d) for d in dims)}")
    occ_a = _voxelize_to_grid(a, pitch, mn, dims)
    occ_b = _voxelize_to_grid(b, pitch, mn, dims)
    inter = np.count_nonzero(occ_a & occ_b)
    union = np.count_nonzero(occ_a | occ_b)
    return float(inter / union) if union > 0 else 0.0

def _iou_exact(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> float:
    a_min, a_max = _o3d_bounds(a)
    b_min, b_max = _o3d_bounds(b)
    if not _aabb_overlap(a_min, a_max, b_min, b_max):
        return 0.0
    va = _o3d_mesh_volume_or_hull(a)
    vb = _o3d_mesh_volume_or_hull(b)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    inter = _boolean_intersection_volume(a, b)
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0

def _iou(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh, backend: str, pitch: float) -> float:
    if backend == "exact":
        return _iou_exact(a, b)
    if backend == "voxel":
        return _iou_voxel(a, b, pitch)
    # auto: try exact; on failure, use voxel
    try:
        return _iou_exact(a, b)
    except Exception:
        return _iou_voxel(a, b, pitch)

def _pairwise_iou(gt: List[Comp], pr: List[Comp], backend: str, pitch: float, eps: float) -> np.ndarray:
    m, n = len(gt), len(pr)
    log_step(f"Computing pairwise IoU matrix ({m}x{n}) using backend='{backend}' (pitch={pitch}, epsilon={eps})")
    M = np.zeros((m, n), dtype=float)
    progress_stride = max(1, min(50, (m // 10) or 5))
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            if not _aabb_overlap(g.aabb_min, g.aabb_max, p.aabb_min, p.aabb_max):
                continue
            v = _iou(g.mesh, p.mesh, backend, pitch)
            M[i, j] = v if v >= eps else 0.0
        if VERBOSE and (m > 0) and (((i + 1) % progress_stride == 0) or (i == m - 1)):
            log_step(f"  IoU progress: processed {i + 1}/{m} GT elements")
    log_step("Pairwise IoU matrix ready")
    return M

def _union_many(meshes: List[o3d.geometry.TriangleMesh], backend: str, pitch: float, ref: Optional[o3d.geometry.TriangleMesh] = None) -> Optional[o3d.geometry.TriangleMesh]:
    if backend in ("exact", "auto"):
        u = _o3d_boolean_union(meshes)
        if u is not None and len(u.triangles) > 0:
            return u
        if backend == "exact":
            # allow raising on exact if union failed, to match original intent
            pass
    # voxel fallback union
    bounds = [ _o3d_bounds(m) for m in meshes ]
    if ref is not None:
        bounds.append(_o3d_bounds(ref))
    mins = np.min(np.stack([b[0] for b in bounds], axis=0), axis=0)
    maxs = np.max(np.stack([b[1] for b in bounds], axis=0), axis=0)
    dims = np.ceil((maxs - mins) / pitch).astype(int) + 1
    if np.prod(dims) > 400_000_000:
        raise MemoryError("Voxel union grid too large")
    if not _LOG_STATE["voxel_union_fallback"]:
        log_step("  Falling back to voxel-based union approximation (Open3D)")
        _LOG_STATE["voxel_union_fallback"] = True
    occ = np.zeros((dims[0], dims[1], dims[2]), dtype=bool)
    for m in meshes:
        occ |= _o3d_voxelize_indices(m, pitch, mins, dims)
    # Build a box-instanced mesh from occupied voxels
    u_mesh = _o3d_boxes_from_indices(occ, pitch, mins)
    _o3d_clean_mesh(u_mesh)
    return u_mesh

# ---------- Metrics aggregation ----------

def compute_all_metrics(
    gt: List[Comp],
    pr: List[Comp],
    backend: str,
    pitch: float,
    eps: float
) -> Dict[str, Any]:
    log_step("Aggregating IoU and compactness metrics")
    M = _pairwise_iou(gt, pr, backend, pitch, eps)
    m, n = M.shape

    # Per-GT
    per_gt = []
    gt_local_compact = []
    gt_ious = []
    log_step("  Computing per-GT aggregates")
    for i, g in enumerate(gt):
        js = [j for j in range(n) if M[i, j] > 0.0]
        local_compact = (1.0 / len(js)) if js else 0.0
        gt_local_compact.append(local_compact)

        pairwise = [{"pred_index": j, "pred_guid": pr[j].guid, "iou": float(M[i, j])} for j in js]
        if js:
            u = _union_many([pr[j].mesh for j in js], backend, pitch, ref=g.mesh)
            iou_union = _iou(u, g.mesh, backend, pitch) if (u is not None and len(u.triangles) > 0) else 0.0
        else:
            iou_union = 0.0
        gt_ious.append(iou_union)

        per_gt.append({
            "gt_index": i,
            "gt_guid": g.guid,
            "gt_ifc_type": g.etype,
            "gt_meta": asdict(g.meta),
            "matches_pred_indices": js,
            "matches_pred_guids": [pr[j].guid for j in js],
            "pairwise_pred": pairwise,
            "iou_union_pred_vs_gt": float(iou_union),
            "local_compactness_gt_to_pred": float(local_compact)
        })

    # Per-PRED
    per_pred = []
    pr_local_compact = []
    pr_ious = []
    log_step("  Computing per-PRED aggregates")
    for j, p in enumerate(pr):
        is_ = [i for i in range(m) if M[i, j] > 0.0]
        local_compact = (1.0 / len(is_)) if is_ else 0.0
        pr_local_compact.append(local_compact)

        pairwise = [{"gt_index": i, "gt_guid": gt[i].guid, "iou": float(M[i, j])} for i in is_]
        if is_:
            u = _union_many([gt[i].mesh for i in is_], backend, pitch, ref=p.mesh)
            iou_union = _iou(u, p.mesh, backend, pitch) if (u is not None and len(u.triangles) > 0) else 0.0
        else:
            iou_union = 0.0
        pr_ious.append(iou_union)

        per_pred.append({
            "pred_index": j,
            "pred_guid": p.guid,
            "pred_ifc_type": p.etype,
            "pred_meta": asdict(p.meta),
            "matches_gt_indices": is_,
            "matches_gt_guids": [gt[i].guid for i in is_],
            "pairwise_gt": pairwise,
            "iou_union_gt_vs_pred": float(iou_union),
            "local_compactness_pred_to_gt": float(local_compact)
        })

    # Dataset-level
    log_step("  Deriving dataset-level metrics")
    m3D_IoU_gt = float(np.mean(gt_ious) if gt_ious else 0.0)
    m3D_IoU_pred = float(np.mean(pr_ious) if pr_ious else 0.0)
    comp_gt_to_pred = float(np.mean(gt_local_compact) if gt_local_compact else 0.0)
    comp_pred_to_gt = float(np.mean(pr_local_compact) if pr_local_compact else 0.0)
    m3D_Compactness = 0.5 * (comp_gt_to_pred + comp_pred_to_gt)

    # Optional: correspondences table (sparse edges)
    edges = []
    for i in range(m):
        for j in range(n):
            if M[i, j] > 0.0:
                edges.append({
                    "gt_index": i, "gt_guid": gt[i].guid,
                    "pred_index": j, "pred_guid": pr[j].guid,
                    "iou": float(M[i, j])
                })

    log_step("Metric aggregation complete")
    return {
        "overall": {
            "m3D_IoU_mean_over_GT": m3D_IoU_gt,
            "m3D_IoU_mean_over_PRED": m3D_IoU_pred,
            "GT_to_PRED_Compactness": comp_gt_to_pred,
            "PRED_to_GT_Compactness": comp_pred_to_gt,
            "m3D_Compactness": m3D_Compactness
        },
        "per_gt": per_gt,
        "per_pred": per_pred,
        "correspondences": edges
    }

# ---------- CSV helpers (unchanged) ----------

def save_csvs(prefix: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    with open(f"{prefix}_per_gt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "gt_index","gt_guid","gt_ifc_type","iou_union_pred_vs_gt",
            "local_compactness_gt_to_pred","num_matches","match_pred_guids"
        ])
        for r in result["per_gt"]:
            w.writerow([
                r["gt_index"], r["gt_guid"], r["gt_ifc_type"],
                f'{r["iou_union_pred_vs_gt"]:.6f}',
                f'{r["local_compactness_gt_to_pred"]:.6f}',
                len(r["matches_pred_indices"]),
                ";".join(r["matches_pred_guids"])
            ])

    with open(f"{prefix}_per_pred.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pred_index","pred_guid","pred_ifc_type","iou_union_gt_vs_pred",
            "local_compactness_pred_to_gt","num_matches","match_gt_guids"
        ])
        for r in result["per_pred"]:
            w.writerow([
                r["pred_index"], r["pred_guid"], r["pred_ifc_type"],
                f'{r["iou_union_gt_vs_pred"]:.6f}',
                f'{r["local_compactness_pred_to_gt"]:.6f}',
                len(r["matches_gt_indices"]),
                ";".join(r["matches_gt_guids"])
            ])

    with open(f"{prefix}_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_index","gt_guid","pred_index","pred_guid","pairwise_iou"])
        for e in result["correspondences"]:
            w.writerow([e["gt_index"], e["gt_guid"], e["pred_index"], e["pred_guid"], f'{e["iou"]:.6f}'])

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Per-element 3D-IoU & 3D-Compactness between two IFCs, with metadata. (Open3D refactor)")
    ap.add_argument("--gt", required=True, help="Path to ground-truth IFC.")
    ap.add_argument("--pred", required=True, help="Path to predicted/reconstructed IFC.")
    ap.add_argument("--align", choices=["icp", "centroid", "none"], default="icp",
                    help="Alignment for PRED->GT. Uses ICP if available; otherwise centroid.")
    ap.add_argument("--backend", choices=["auto", "exact", "voxel"], default="auto",
                    help="CSG backend for IoU/union calculation.")
    ap.add_argument("--voxel-size", type=float, default=0.03, help="Voxel pitch (m) for fallback ops.")
    ap.add_argument("--epsilon", type=float, default=0.05, help="IoU threshold to consider a correspondence.")
    ap.add_argument("--save-json", type=str, default=None, help="Path to save a JSON report.")
    ap.add_argument("--save-csv-prefix", type=str, default=None, help="Prefix to save CSVs: <prefix>_per_gt.csv, _per_pred.csv, _edges.csv")
    ap.add_argument("--ifc-classes", nargs="+", default=None,
                    help="Optional list of IFC classes to include (e.g. IfcWall IfcDoor). You may also pass a single comma-separated string 'IfcWall,IfcDoor'.")

    # Default CLI arguments for convenience (used only when no CLI args are provided)
    DEFAULT_ARGS = [
        "--gt", ".\\input\\cenz-main-v1.ifc",
        "--pred", ".\\input\\cenz-main-v2-revit-geo.ifc",
        "--align", "icp",
        "--backend", "voxel",
        "--voxel-size", "0.05",
        "--epsilon", "0.8",
        "--save-json", "metrics.json",
        "--save-csv-prefix", "out/metrics",
        "--ifc-classes", "IfcWall", "IfcWallStandardCase"
    ]

    # If the script was invoked without extra CLI args, use DEFAULT_ARGS.
    # Otherwise respect actual command-line arguments.
    if len(sys.argv) == 1:
        log_step(f"No CLI arguments detected — using DEFAULT_ARGS: {' '.join(DEFAULT_ARGS)}")
        args = ap.parse_args(DEFAULT_ARGS)
    else:
        args = ap.parse_args()

    # Normalize --ifc-classes: allow single token comma-separated input
    if args.ifc_classes is not None and len(args.ifc_classes) == 1 and ',' in args.ifc_classes[0]:
        args.ifc_classes = [s.strip() for s in args.ifc_classes[0].split(',') if s.strip()]

    log_step(f"Loading GT IFC: {args.gt}")
    gt, _ = load_ifc_components(args.gt, include_types=args.ifc_classes)
    log_step(f"  GT elements loaded: {len(gt)}")

    log_step(f"Loading PRED IFC: {args.pred}")
    pr, _ = load_ifc_components(args.pred, include_types=args.ifc_classes)
    log_step(f"  PRED elements loaded: {len(pr)}")

    # # --- TESTING: Use only first 5 elements ---
    # gt = gt[:5]
    # pr = pr[:5]
    # log_step(f"  Using first 5 GT and PRED elements for testing")

    # Visualize pre-alignment (Open3D)
    gt_meshes = [_o3d_mesh_copy(c.mesh) for c in gt]
    pred_meshes = [_o3d_mesh_copy(c.mesh) for c in pr]
    for m in gt_meshes:
        m.paint_uniform_color([0, 1, 0])  # green
    for m in pred_meshes:
        m.paint_uniform_color([1, 0, 0])  # red
    try:
        o3d.visualization.draw_geometries(gt_meshes + pred_meshes)
    except Exception:
        pass

    # Align PRED -> GT
    if args.align != "none":
        log_step("Aligning PRED to GT ...")
        if args.align == "icp":
            if OPEN3D_OK:
                log_step("  Running Open3D ICP alignment")
                T = rigid_icp_align([c.mesh for c in pr], [c.mesh for c in gt])
            else:
                log_step("  Open3D unavailable; using centroid alignment instead")
                T = _centroid_align([c.mesh for c in pr], [c.mesh for c in gt])
        else:
            log_step("  Using centroid alignment")
            T = _centroid_align([c.mesh for c in pr], [c.mesh for c in gt])
        _o3d_transform_inplace(pr, T)
        log_step("  Alignment applied")

        # Visualize aligned meshes (Open3D)
        gt_meshes = [_o3d_mesh_copy(c.mesh) for c in gt]
        pred_meshes = [_o3d_mesh_copy(c.mesh) for c in pr]
        for m in gt_meshes:
            m.paint_uniform_color([0, 1, 0])
        for m in pred_meshes:
            m.paint_uniform_color([1, 0, 0])
        try:
            o3d.visualization.draw_geometries(gt_meshes + pred_meshes)
        except Exception:
            pass
    else:
        log_step("Skipping alignment step")

    # Metrics
    log_step("Computing metrics ...")
    res = compute_all_metrics(gt, pr, backend=args.backend, pitch=float(args.voxel_size), eps=float(args.epsilon))
    log_step("Metrics computation complete")

    # Console summary
    log_step("Rendering console summary")
    print("\n=== Overall ===")
    for k, v in res["overall"].items():
        print(f"{k}: {v:.6f}")

    print("\nPer-GT example rows (first 5):")
    for r in res["per_gt"][:5]:
        print(f'  [{r["gt_index"]:03d}] {r["gt_guid"]}  IoU={r["iou_union_pred_vs_gt"]:.4f}  matches={len(r["matches_pred_indices"])}')

    print("\nPer-PRED example rows (first 5):")
    for r in res["per_pred"][:5]:
        print(f'  [{r["pred_index"]:03d}] {r["pred_guid"]}  IoU={r["iou_union_gt_vs_pred"]:.4f}  matches={len(r["matches_gt_indices"])}')

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        log_step(f"Saved JSON to {args.save_json}")

    if args.save_csv_prefix:
        save_csvs(args.save_csv_prefix, res)
        log_step(f"Saved CSVs with prefix: {args.save_csv_prefix}")

    log_step("Processing complete")

if __name__ == "__main__":
    main()