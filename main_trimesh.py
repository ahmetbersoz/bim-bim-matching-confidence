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

def _get_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Robust volume helper: try Open3D's get_volume() first (fast),
    but catch exceptions (non-watertight meshes) and fall back to:
      1) a signed-volume triangle-based computation (_o3d_signed_volume),
      2) convex-hull -> signed-volume,
      3) finally return 0.0 on failure.
    Returns a positive float (absolute volume).
    """
    # Try to ensure consistent triangle orientation first (best-effort)
    try:
        if mesh.is_orientable():
            mesh.orient_triangles()
    except Exception:
        # ignore orientation failures; continue to robust volume paths
        pass

    # Primary fast attempt: Open3D's get_volume() (may throw for non-watertight meshes)
    try:
        vol = mesh.get_volume()
        return float(abs(vol))
    except Exception:
        # Fall back to a pure-Python signed-volume computation over triangles
        try:
            vol = _o3d_signed_volume(mesh)
            return float(abs(vol))
        except Exception:
            # As a last resort, try convex hull and compute signed volume on the hull
            try:
                hull, _ = mesh.compute_convex_hull()
                try:
                    if hull.is_orientable():
                        hull.orient_triangles()
                except Exception:
                    pass
                # compute signed volume on the hull (avoids relying on get_volume)
                hvol = _o3d_signed_volume(hull)
                return float(abs(hvol))
            except Exception:
                # give up and return zero
                return 0.0

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


def _o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> Optional['trimesh.Trimesh']:
    """Convert an Open3D triangle mesh into a trimesh.Trimesh."""
    if not TRIMESH_OK:
        return None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None
    try:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            return None
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
        try:
            tm.remove_duplicate_faces()
            tm.remove_degenerate_faces()
            tm.remove_unreferenced_vertices()
            tm.remove_infinite_values()
            tm.remove_nan()
        except Exception:
            pass
        return tm
    except Exception:
        return None


def _trimesh_to_o3d(mesh: 'trimesh.Trimesh') -> Optional[o3d.geometry.TriangleMesh]:
    """Convert a trimesh.Trimesh back into an Open3D TriangleMesh."""
    if mesh is None:
        return None
    try:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            return None
        out = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        try:
            out.compute_vertex_normals()
        except Exception:
            pass
        return out
    except Exception:
        return None


def _collapse_trimesh_result(result: Any) -> Optional['trimesh.Trimesh']:
    """Normalize trimesh boolean outputs into a single Trimesh instance."""
    if not TRIMESH_OK:
        return None
    if result is None:
        return None
    if isinstance(result, trimesh.Trimesh):
        return result
    if isinstance(result, (list, tuple)):
        meshes = [m for m in result if isinstance(m, trimesh.Trimesh) and len(getattr(m, 'faces', [])) > 0]
        if not meshes:
            return None
        if len(meshes) == 1:
            return meshes[0]
        try:
            return trimesh.util.concatenate(meshes)
        except Exception:
            return meshes[0]
    if isinstance(result, trimesh.Scene):
        try:
            collapsed = result.dump(concatenate=True)
            if isinstance(collapsed, trimesh.Trimesh):
                return collapsed
            return _collapse_trimesh_result(collapsed)
        except Exception:
            try:
                meshes = [g for g in result.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    return None
                if len(meshes) == 1:
                    return meshes[0]
                return trimesh.util.concatenate(meshes)
            except Exception:
                return None
    return None


def _o3d_boolean_intersection(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> Optional[o3d.geometry.TriangleMesh]:
    """Compute the boolean intersection using trimesh as the backend."""
    if not TRIMESH_OK:
        if not _LOG_STATE['boolean_intersection_fallback']:
            log_step('  Trimesh not available; falling back to voxel IoU where needed')
            _LOG_STATE['boolean_intersection_fallback'] = True
        return None

    a2 = _o3d_mesh_copy(a)
    b2 = _o3d_mesh_copy(b)
    for m in (a2, b2):
        _o3d_clean_mesh(m)

    tm_a = _o3d_to_trimesh(a2)
    tm_b = _o3d_to_trimesh(b2)
    if tm_a is None or tm_b is None:
        if not _LOG_STATE['boolean_intersection_fallback']:
            log_step('  Failed to convert meshes for intersection; will use voxel IoU')
            _LOG_STATE['boolean_intersection_fallback'] = True
        return None

    try:
        inter_raw = trimesh.boolean.intersection([tm_a, tm_b], engine=None)
        inter_tm = _collapse_trimesh_result(inter_raw)
    except Exception:
        inter_tm = None

    faces = getattr(inter_tm, 'faces', np.zeros((0, 3))) if inter_tm is not None else np.zeros((0, 3))
    if inter_tm is None or faces.size == 0:
        if not _LOG_STATE['boolean_intersection_fallback']:
            log_step('  Trimesh boolean_intersection failed; will fall back to voxel IoU where needed')
            _LOG_STATE['boolean_intersection_fallback'] = True
        return None

    inter_o3d = _trimesh_to_o3d(inter_tm)
    if inter_o3d is None or len(inter_o3d.triangles) == 0:
        if not _LOG_STATE['boolean_intersection_fallback']:
            log_step('  Intersection result empty after conversion; falling back to voxel IoU')
            _LOG_STATE['boolean_intersection_fallback'] = True
        return None
    return inter_o3d


def _o3d_boolean_union(meshes: List[o3d.geometry.TriangleMesh]) -> Optional[o3d.geometry.TriangleMesh]:
    """Compute the boolean union of meshes using trimesh."""
    if not meshes:
        return None

    if not TRIMESH_OK:
        if not _LOG_STATE['boolean_union_fallback']:
            log_step('  Trimesh not available; will fall back to voxel-based union approximation')
            _LOG_STATE['boolean_union_fallback'] = True
        return None

    converted: List['trimesh.Trimesh'] = []
    for mesh in meshes:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            continue
        mc = _o3d_mesh_copy(mesh)
        _o3d_clean_mesh(mc)
        tm = _o3d_to_trimesh(mc)
        faces = getattr(tm, 'faces', np.zeros((0, 3))) if tm is not None else np.zeros((0, 3))
        if tm is not None and faces.size > 0:
            converted.append(tm)
    if not converted:
        return None

    try:
        union_raw = trimesh.boolean.union(converted, engine=None)
        union_tm = _collapse_trimesh_result(union_raw)
    except Exception:
        union_tm = None

    faces = getattr(union_tm, 'faces', np.zeros((0, 3))) if union_tm is not None else np.zeros((0, 3))
    if union_tm is None or faces.size == 0:
        if not _LOG_STATE['boolean_union_fallback']:
            log_step('  Trimesh boolean_union failed; will fall back to voxel-based union approximation')
            _LOG_STATE['boolean_union_fallback'] = True
        return None

    union_o3d = _trimesh_to_o3d(union_tm)
    if union_o3d is None or len(union_o3d.triangles) == 0:
        if not _LOG_STATE['boolean_union_fallback']:
            log_step('  Union result empty after conversion; falling back to voxel approximation')
            _LOG_STATE['boolean_union_fallback'] = True
        return None
    return union_o3d

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

    # temporarily disable scale
    scale = 1

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


def rigid_icp_align_xy_only(pred_meshes: List[o3d.geometry.TriangleMesh], gt_meshes: List[o3d.geometry.TriangleMesh]) -> np.ndarray:
    """
    Align PRED -> GT using only an X/Y translation. No rotation is applied.
    Strategy:
      - Sample point clouds from pred and gt meshes.
      - Compute mean (centroid) of each point set.
      - Compute translation vector = (tgt_centroid - src_centroid) but only on X and Y.
      - Return 4x4 transform with identity rotation and translation [dx, dy, 0].
    If sampling fails (empty clouds), fall back to the bbox-centroid alignment but zero the Z translation.
    """
    # Sample point clouds
    src = _o3d_pcd_from_meshes(pred_meshes)
    tgt = _o3d_pcd_from_meshes(gt_meshes)

    # If either cloud is empty, fallback to centroid align but enforce no Z translation
    try:
        src_pts = np.asarray(src.points)
        tgt_pts = np.asarray(tgt.points)
    except Exception:
        src_pts = np.zeros((0, 3))
        tgt_pts = np.zeros((0, 3))

    if src_pts.size == 0 or tgt_pts.size == 0:
        T_cent = _centroid_align(pred_meshes, gt_meshes)
        T_cent[2, 3] = 0.0  # enforce no Z translation
        print("Alignment (fallback centroid, Z zeroed):")
        print(T_cent)
        return T_cent

    # Compute centroids
    src_c = src_pts.mean(axis=0)
    tgt_c = tgt_pts.mean(axis=0)

    # Build translation only in X and Y; keep Z unchanged
    dx = float(tgt_c[0] - src_c[0])
    dy = float(tgt_c[1] - src_c[1])
    T = np.eye(4, dtype=float)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = 0.0

    print("Alignment (XY-translation only):")
    print(T)
    return T

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


def visualize_voxel_occupancy(occ_a: np.ndarray, occ_b: np.ndarray, origin: np.ndarray, pitch: float, pts_mode: bool = True) -> None:
    """
    Visualize two boolean occupancy grids (occ_a, occ_b) in the same coordinate frame.
    occ_* : 3D boolean numpy arrays with shape (Nx, Ny, Nz)
    origin : 3-element array (min corner) for voxel index (0,0,0)
    pitch  : voxel size (float)
    pts_mode : True => draw voxel centers as colored points (fast). False => builds boxes (heavy).
    """
    if occ_a.shape != occ_b.shape:
        raise ValueError("occ_a and occ_b must have same shape")
    # voxel indices where occupied
    ia = np.argwhere(occ_a)
    ib = np.argwhere(occ_b)

    if ia.size == 0 and ib.size == 0:
        log_step("visualize_voxel_occupancy: no occupied voxels")
        return

    # compute centers
    centers_a = origin + (ia + 0.5) * pitch if ia.size else np.zeros((0, 3))
    centers_b = origin + (ib + 0.5) * pitch if ib.size else np.zeros((0, 3))

    # If a voxel is occupied in both, mark as overlap (blue)
    # Build a merged list with colors: A-only red, B-only green, overlap blue
    # Use sets of tuple indices for quick overlap test
    set_a = {tuple(x) for x in ia.tolist()}
    set_b = {tuple(x) for x in ib.tolist()}
    overlap_idx = np.array(sorted([x for x in set_a & set_b]))
    only_a_idx = np.array(sorted([x for x in set_a - set_b]))
    only_b_idx = np.array(sorted([x for x in set_b - set_a]))

    pts = []
    cols = []
    def append_from_indices(arr, color):
        if arr.size == 0:
            return
        arr = np.asarray(arr, dtype=float)
        centers = origin + (arr + 0.5) * pitch
        pts.append(centers)
        cols.append(np.tile(color, (centers.shape[0], 1)))

    append_from_indices(only_a_idx, [1.0, 0.0, 0.0])   # red
    append_from_indices(only_b_idx, [0.0, 1.0, 0.0])   # green
    append_from_indices(overlap_idx, [0.0, 0.0, 1.0])  # blue

    if not pts:
        log_step("visualize_voxel_occupancy: nothing to draw after classification")
        return

    P = np.vstack(pts)
    C = np.vstack(cols)

    if pts_mode:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P)
        pc.colors = o3d.utility.Vector3dVector(C)
        try:
            o3d.visualization.draw_geometries([pc], window_name="Voxel occupancy (A:red B:green overlap:blue)")
        except Exception:
            log_step("Open3D visualization failed (headless?)")
    else:
        # boxes mode: heavy - create box meshes per set (colored)
        def boxes_from_list(idx_arr, color):
            if idx_arr is None or idx_arr.size == 0:
                return o3d.geometry.TriangleMesh()
            boxes = []
            half = pitch / 2.0
            for (i, j, k) in np.asarray(idx_arr, dtype=int):
                center = origin + np.array([i + 0.5, j + 0.5, k + 0.5]) * pitch
                box = o3d.geometry.TriangleMesh.create_box(width=pitch, height=pitch, depth=pitch)
                box.translate(center - np.array([half, half, half]), relative=False)
                boxes.append(box)
            m = _o3d_concat_meshes(boxes) if boxes else o3d.geometry.TriangleMesh()
            if len(m.triangles) > 0:
                m.paint_uniform_color(color)
            return m
        mesh_a = boxes_from_list(only_a_idx, [1, 0, 0])
        mesh_b = boxes_from_list(only_b_idx, [0, 1, 0])
        mesh_o = boxes_from_list(overlap_idx, [0, 0, 1])
        try:
            o3d.visualization.draw_geometries([mesh_a, mesh_b, mesh_o], window_name="Voxel boxes")
        except Exception:
            log_step("Open3D visualization failed (headless?)")


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

    # visualize_voxel_occupancy(occ_a, occ_b, mn, pitch, pts_mode=True)

    inter = np.count_nonzero(occ_a & occ_b)
    union = np.count_nonzero(occ_a | occ_b)
    return float(inter / union) if union > 0 else 0.0

def visualize_meshes_with_volumes(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh, va: float, vb: float, show_hulls: bool = False) -> None:
    """
    Visualize two meshes together with simple coloring and print their computed volumes.
    - a : mesh A (will be painted red)
    - b : mesh B (will be painted green)
    - va, vb : their computed volumes (floats) — printed and shown in the window title
    - show_hulls : if True, also compute and show convex hulls (wireframe)
    This is a best-effort helper; visualization failures are caught and logged.
    """
    try:
        ma = _o3d_mesh_copy(a)
        mb = _o3d_mesh_copy(b)
        ma.paint_uniform_color([1.0, 0.0, 0.0])  # red
        mb.paint_uniform_color([0.0, 1.0, 0.0])  # green

        geoms = [ma, mb]

        if show_hulls:
            try:
                ha, _ = ma.compute_convex_hull()
                hb, _ = mb.compute_convex_hull()
                # paint hulls lightly and show wireframe
                ha.paint_uniform_color([1.0, 0.6, 0.6])
                hb.paint_uniform_color([0.6, 1.0, 0.6])
                geoms.extend([ha, hb])
            except Exception:
                log_step("  Convex hull computation for visualization failed; continuing without hulls")

        title = f"Mesh A (red) va={va:.6f}  |  Mesh B (green) vb={vb:.6f}"
        print(title)
        try:
            o3d.visualization.draw_geometries(geoms, window_name=title, mesh_show_wireframe=True)
        except Exception:
            log_step("Open3D visualization failed (headless or other issue)")
    except Exception as exc:
        log_step(f"Visualization helper failed: {exc}")


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

    # print(f"  Volumes: va={va:.6f} vb={vb:.6f} intersection={inter:.6f}")
    # print(f"  IoU = {inter:.6f} / {union:.6f} = {float(inter / union) if union > 0 else 0.0:.6f}")
    # visualize_meshes_with_volumes(a, b, va, vb, show_hulls=True)

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



def _union_many(
    meshes: List[o3d.geometry.TriangleMesh],
    backend: str,
    pitch: float,
    ref: Optional[o3d.geometry.TriangleMesh] = None
) -> Optional[o3d.geometry.TriangleMesh]:

    if not meshes:
        return o3d.geometry.TriangleMesh()
    if len(meshes) == 1:
        return _o3d_mesh_copy(meshes[0])

    pitch = float(pitch)
    if pitch <= 0:
        raise ValueError("voxel pitch must be > 0")

    # Try exact first
    if backend in ("exact", "auto"):
        u = _o3d_boolean_union(meshes)
        if u is not None and len(u.triangles) > 0:
            return u
        if backend == "exact":
            # respect user's explicit request
            log_step("  Exact union returned empty; falling back to voxel approximation")
            # (we deliberately do NOT raise here to keep the pipeline alive)

    # ---- voxel fallback ----
    bounds = [_o3d_bounds(m) for m in meshes]
    if ref is not None:
        bounds.append(_o3d_bounds(ref))
    mins = np.min(np.stack([b[0] for b in bounds], axis=0), axis=0)
    maxs = np.max(np.stack([b[1] for b in bounds], axis=0), axis=0)
    ext  = np.maximum(maxs - mins, 1e-9)
    dims = np.maximum(np.ceil(ext / pitch).astype(int) + 1, 1)

    if np.prod(dims) > 200_000_000:  # adapt to your RAM budget
        log_step(f"  Voxel grid too large ({tuple(int(d) for d in dims)}); using triangle concatenation")
        return _o3d_concat_meshes(meshes)

    if not _LOG_STATE["voxel_union_fallback"]:
        log_step("  Falling back to voxel-based union approximation (Open3D)")
        _LOG_STATE["voxel_union_fallback"] = True

    occ = np.zeros((dims[0], dims[1], dims[2]), dtype=bool)
    any_filled = False
    for m in meshes:
        occ_m = _o3d_voxelize_indices(m, pitch, mins, dims)
        any_filled |= occ_m.any()
        occ |= occ_m

    if not any_filled:
        # try a finer pitch once
        finer = pitch * 0.5
        log_step("  Voxelization empty; retrying with finer pitch")
        occ = np.zeros((dims[0]*2, dims[1]*2, dims[2]*2), dtype=bool)
        mins2 = mins
        dims2 = np.maximum(np.ceil(ext / finer).astype(int) + 1, 1)
        if np.prod(dims2) <= 200_000_000:
            for m in meshes:
                occ |= _o3d_voxelize_indices(m, finer, mins2, dims2)
            if occ.any():
                u_mesh = _o3d_boxes_from_indices(occ, finer, mins2)
                _o3d_clean_mesh(u_mesh)
                return u_mesh

        log_step("  Still empty after finer pitch; using triangle concatenation")
        return _o3d_concat_meshes(meshes)

    u_mesh = _o3d_boxes_from_indices(occ, pitch, mins)
    if u_mesh is None or len(u_mesh.triangles) == 0:
        log_step("  Box instancing produced empty mesh; using triangle concatenation")
        return _o3d_concat_meshes(meshes)

    _o3d_clean_mesh(u_mesh)
    return u_mesh


def _silence_vtk_output(log_to: str | None = None) -> None:
    """Prevent VTK from opening its GUI error window (Windows)."""
    try:
        import os
        # Try VTK 9 logger first (reduces stderr noise)
        try:
            from vtkmodules.vtkCommonCore import vtkLogger
            vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)
        except Exception:
            pass
        # Route OutputWindow to file or OS null device
        import vtk
        fn = ("NUL" if os.name == "nt" else "/dev/null") if log_to is None else log_to
        fow = vtk.vtkFileOutputWindow()
        fow.SetFileName(fn)
        vtk.vtkOutputWindow.SetInstance(fow)
        try:
            vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass
    except Exception:
        # If VTK is not importable here, nothing to do
        pass


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
    _silence_vtk_output() 

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
        "--gt", ".\\input\\WW_revit_model_v6.ifc",
        "--pred", ".\\input\\ww-v1-ifc4-geo.ifc",
        "--align", "icp",
        "--backend", "exact",
        "--voxel-size", "0.05",
        "--epsilon", "0.1",
        "--save-json", "metrics.json",
        "--save-csv-prefix", "out/metrics",
        "--ifc-classes", "IfcWall", "IfcWallStandardCase"
        # "--ifc-classes", "IfcSpace"
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
    # gt = gt[:20]
    # pr = pr[:20]
    # log_step(f"  Using first 5 GT and PRED elements for testing")

    # Visualize pre-alignment (Open3D)
    gt_meshes = [_o3d_mesh_copy(c.mesh) for c in gt]
    pred_meshes = [_o3d_mesh_copy(c.mesh) for c in pr]
    for m in gt_meshes:
        m.paint_uniform_color([0, 1, 0])  # green
    for m in pred_meshes:
        m.paint_uniform_color([1, 0, 0])  # red
    try:
        o3d.visualization.draw_geometries(gt_meshes, mesh_show_wireframe=True)
        o3d.visualization.draw_geometries(pred_meshes, mesh_show_wireframe=True)
        o3d.visualization.draw_geometries(gt_meshes + pred_meshes, mesh_show_wireframe=True)
    except Exception:
        pass

    # Align PRED -> GT
    if args.align != "none":
        log_step("Aligning PRED to GT ...")
        if args.align == "icp":
            if OPEN3D_OK:
                log_step("  Running Open3D ICP alignment")
                # T = rigid_icp_align([c.mesh for c in pr], [c.mesh for c in gt])
                T = rigid_icp_align_xy_only([c.mesh for c in pr], [c.mesh for c in gt])
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
            o3d.visualization.draw_geometries(gt_meshes + pred_meshes, mesh_show_wireframe=True)
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
