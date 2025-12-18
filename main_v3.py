#!/usr/bin/env python3
# ifc_metrics_with_metadata_mesh.py
# Compute per-element 3D-IoU & 3D-Compactness between two IFCs, preserving IFC metadata,
# using mesh boolean volumes on the actual triangle meshes (Open3D).
#
# - Loads IFCs with ifcopenshell
# - Aligns PRED -> GT (ICP if available, else centroid translation; optional XY-only ICP)
# - Builds pairwise IoU matrix using mesh boolean intersection/union volumes
# - Reports per-element metrics for *both* GT and PRED
# - Preserves metadata (GlobalId, Name, Type, attributes, and Psets)

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

# ---------- Dependencies ----------
try:
    import ifcopenshell
    import ifcopenshell.geom
    from ifcopenshell.util import unit as ifc_unit
    try:
        from ifcopenshell.util.element import get_psets as ifc_get_psets
    except Exception:
        ifc_get_psets = None
except Exception:
    print("This script needs ifcopenshell with geometry enabled. Try: pip install ifcopenshell", file=sys.stderr)
    raise

try:
    import open3d as o3d
except Exception:
    print("This script needs open3d. Try: pip install open3d", file=sys.stderr)
    raise

# Optional: trimesh backend for robust mesh booleans (recommended on Windows when Open3D/VTK booleans fail).
try:
    import trimesh  # type: ignore

    TRIMESH_OK = True
except Exception:
    trimesh = None  # type: ignore
    TRIMESH_OK = False

VERBOSE = True


def log_step(message: str) -> None:
    if VERBOSE:
        print(f"[step] {message}", flush=True)


def _sanitize_for_filename(value: Optional[str], fallback: str = "item") -> str:
    """
    Create a filesystem-friendly token derived from IFC identifiers or names.
    Ensures ASCII-only characters and avoids zero-length results.
    """
    if not value:
        value = fallback
    value = value.strip()
    if not value:
        value = fallback
    # collapse whitespace and remove characters that are problematic for filenames
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    # avoid leading dots which can create hidden files on some systems
    value = value.lstrip(".")
    if not value:
        value = fallback
    return value[:120]


def _derive_session_output_dir(base_dir: str, pred_path: str, gt_path: str) -> str:
    base_abs = os.path.abspath(base_dir or "out")
    os.makedirs(base_abs, exist_ok=True)

    pred_name = os.path.splitext(os.path.basename(pred_path or "pred"))[0]
    gt_name = os.path.splitext(os.path.basename(gt_path or "gt"))[0]

    pred_token = _sanitize_for_filename(pred_name or "pred", fallback="pred")
    gt_token = _sanitize_for_filename(gt_name or "gt", fallback="gt")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    session_dir_name = f"{timestamp}__pred_{pred_token}__gt_{gt_token}"
    session_dir = os.path.join(base_abs, session_dir_name)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def _prepare_output_paths(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    """
    Determine the session-scoped output directories and file paths.
    All generated artifacts are kept under a single session root directory.
    """
    session_root = _derive_session_output_dir(args.mesh_output_dir, args.pred, args.gt)
    mesh_dir = os.path.join(session_root, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    floor_area_name = os.path.basename(args.floor_area_csv) if args.floor_area_csv else "matched_space_floor_areas.csv"
    floor_area_base = os.path.join(session_root, floor_area_name)

    save_json_path = None
    if args.save_json:
        save_json_path = os.path.join(session_root, os.path.basename(args.save_json))

    save_csv_prefix = None
    if args.save_csv_prefix:
        save_csv_prefix = os.path.join(session_root, os.path.basename(args.save_csv_prefix))

    return {
        "session_root": session_root,
        "mesh_dir": mesh_dir,
        "floor_area_base": floor_area_base,
        "save_json": save_json_path,
        "save_csv_prefix": save_csv_prefix,
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
class OBB:
    center: np.ndarray        # (3,)
    R: np.ndarray             # (3,3) rotation, columns = axes
    extent: np.ndarray        # (3,) full side lengths along OBB axes
    half: np.ndarray          # (3,) = extent / 2
    planes: List[Tuple[np.ndarray, float]]  # list of (n, d) with n unit length, half-space n·x <= d
    corners: np.ndarray       # (8,3) 8 box corners in world coordinates

@dataclass
class Comp:
    idx: int
    guid: str
    etype: str
    meta: Meta
    mesh: o3d.geometry.TriangleMesh
    obb: OBB
    volume: float            # OBB volume
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    wall_thickness: Optional[float]


# ---------- Clone helpers ----------

def _copy_obb(obb: OBB) -> OBB:
    return OBB(
        center=np.array(obb.center, dtype=float, copy=True),
        R=np.array(obb.R, dtype=float, copy=True),
        extent=np.array(obb.extent, dtype=float, copy=True),
        half=np.array(obb.half, dtype=float, copy=True),
        planes=[(np.array(n, dtype=float, copy=True), float(d)) for n, d in obb.planes],
        corners=np.array(obb.corners, dtype=float, copy=True)
    )


def _clone_comp(comp: Comp) -> Comp:
    return Comp(
        idx=comp.idx,
        guid=comp.guid,
        etype=comp.etype,
        meta=copy.deepcopy(comp.meta),
        mesh=_o3d_mesh_copy(comp.mesh),
        obb=_copy_obb(comp.obb),
        volume=float(comp.volume),
        aabb_min=np.array(comp.aabb_min, dtype=float, copy=True),
        aabb_max=np.array(comp.aabb_max, dtype=float, copy=True),
        wall_thickness=comp.wall_thickness
    )


def _clone_comp_list(comps: List[Comp]) -> List[Comp]:
    return [_clone_comp(comp) for comp in comps]


# ---------- Open3D helpers ----------

def _o3d_mesh_copy(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(mesh)

def _o3d_clean_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
    except Exception:
        pass

def _o3d_bounds(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    aabb = mesh.get_axis_aligned_bounding_box()
    return np.asarray(aabb.min_bound), np.asarray(aabb.max_bound)

def _o3d_concat_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
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
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F)
    )

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


def _mesh_floor_area_xy(mesh: o3d.geometry.TriangleMesh) -> float:
    """Compute the projected floor area of a mesh onto the XY plane."""
    if mesh is None or len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        return 0.0
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if V.size == 0 or F.size == 0:
        return 0.0
    v1 = V[F[:, 0]]
    v2 = V[F[:, 1]]
    v3 = V[F[:, 2]]
    # Shoelace formula for each projected triangle
    area = 0.5 * np.abs(
        v1[:, 0] * (v2[:, 1] - v3[:, 1]) +
        v2[:, 0] * (v3[:, 1] - v1[:, 1]) +
        v3[:, 0] * (v1[:, 1] - v2[:, 1])
    )
    return 0.5 * float(np.sum(area))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _metric_to_rgb(value: Any) -> Tuple[float, float, float]:
    """
    Piecewise-linear color mapping:
      0.0 -> red, 0.5 -> yellow, 1.0 -> green.
    Values outside [0,1] are clamped.
    """
    v = _clamp01(_safe_float(value, default=0.0))
    if v <= 0.5:
        t = v / 0.5  # 0..1
        return 1.0, t, 0.0
    t = (v - 0.5) / 0.5  # 0..1
    return 1.0 - t, 1.0, 0.0


def _xyz_dimensions(comp: Comp) -> Tuple[float, float, float]:
    """Return axis-aligned (X, Y, Z) dimensions (AABB extents) in the current coordinate system."""
    try:
        mn = np.asarray(getattr(comp, "aabb_min", None), dtype=float)
        mx = np.asarray(getattr(comp, "aabb_max", None), dtype=float)
        if mn.shape == (3,) and mx.shape == (3,) and np.all(np.isfinite(mn)) and np.all(np.isfinite(mx)):
            ext = np.abs(mx - mn)
            return float(ext[0]), float(ext[1]), float(ext[2])
    except Exception:
        pass

    mesh = getattr(comp, "mesh", None)
    if mesh is None or len(mesh.vertices) == 0:
        return 0.0, 0.0, 0.0
    mn2, mx2 = _o3d_bounds(mesh)
    ext2 = np.abs(np.asarray(mx2, dtype=float) - np.asarray(mn2, dtype=float))
    return float(ext2[0]), float(ext2[1]), float(ext2[2])


def _space_xyz_dimensions(comp: Comp) -> Tuple[float, float, float]:
    return _xyz_dimensions(comp)


def _write_combined_mesh(meshes: List[o3d.geometry.TriangleMesh], path: str) -> None:
    """Export a combined mesh built from the provided list to a file path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined = _o3d_concat_meshes(meshes)
    if len(combined.vertices) == 0 or len(combined.triangles) == 0:
        log_step(f"  Warning: combined mesh at {path} is empty.")
    try:
        combined.compute_vertex_normals()
    except Exception:
        pass
    try:
        o3d.io.write_triangle_mesh(path, combined, write_ascii=True)
    except Exception as exc:
        log_step(f"  Failed to write mesh {path}: {exc}")


def _o3d_concat_meshes_with_vertex_colors(
    verts_list: List[np.ndarray],
    tris_list: List[np.ndarray],
    colors_list: List[np.ndarray]
) -> o3d.geometry.TriangleMesh:
    if not verts_list or not tris_list:
        return o3d.geometry.TriangleMesh()

    V = np.vstack(verts_list) if verts_list else np.zeros((0, 3), dtype=float)
    F = np.vstack(tris_list) if tris_list else np.zeros((0, 3), dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F.astype(np.int32, copy=False))
    )
    if colors_list:
        C = np.vstack(colors_list)
        if C.shape[0] == V.shape[0]:
            mesh.vertex_colors = o3d.utility.Vector3dVector(C.astype(np.float64, copy=False))
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    return mesh


def _write_triangle_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        log_step(f"  Warning: mesh at {path} is empty.")
    try:
        o3d.io.write_triangle_mesh(path, mesh, write_ascii=True)
    except Exception as exc:
        log_step(f"  Failed to write mesh {path}: {exc}")


def _export_metric_meshes_by_class(
    comps: List[Comp],
    metric_by_guid: Dict[str, float],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Export one colored mesh per IFC class, with per-element uniform coloring based on a metric value.
    Elements missing from metric_by_guid are treated as 0.0 (red).
    """
    os.makedirs(output_dir, exist_ok=True)

    comps_by_type: Dict[str, List[Comp]] = {}
    for comp in comps:
        comps_by_type.setdefault(comp.etype, []).append(comp)

    exports: List[Dict[str, Any]] = []
    for etype in sorted(comps_by_type.keys()):
        verts_list: List[np.ndarray] = []
        tris_list: List[np.ndarray] = []
        colors_list: List[np.ndarray] = []
        v_offset = 0

        for comp in comps_by_type[etype]:
            mesh = getattr(comp, "mesh", None)
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                continue
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
            if V.size == 0 or F.size == 0:
                continue

            metric_value = metric_by_guid.get(comp.guid, 0.0)
            color = np.asarray(_metric_to_rgb(metric_value), dtype=float).reshape(1, 3)

            verts_list.append(V)
            tris_list.append(F + v_offset)
            colors_list.append(np.tile(color, (V.shape[0], 1)))
            v_offset += V.shape[0]

        combined = _o3d_concat_meshes_with_vertex_colors(verts_list, tris_list, colors_list)
        if len(combined.vertices) == 0 or len(combined.triangles) == 0:
            log_step(f"  Warning: no geometry for metric export type {etype}; skipping.")
            continue
        filename = f"{_sanitize_for_filename(etype, fallback='IfcType')}.ply"
        path = os.path.join(output_dir, filename)
        _write_triangle_mesh(combined, path)
        exports.append({
            "ifc_type": etype,
            "count": int(len(comps_by_type[etype])),
            "path": path
        })

    return exports


def _metric_map_by_guid(
    records: List[Dict[str, Any]],
    guid_key: str,
    value_key: str
) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for row in records or []:
        guid = row.get(guid_key)
        if not guid:
            continue
        mapping[str(guid)] = _safe_float(row.get(value_key, 0.0), default=0.0)
    return mapping


def _export_space_meshes_for_round(
    round_label: str,
    space_matches: List[Dict[str, Any]],
    by_space: Dict[str, Any],
    spaces_gt: List[Comp],
    spaces_pr: List[Comp],
    elems_gt: List[Comp],
    elems_pr: List[Comp],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Export per-space mesh bundles for a given round.
    Each bundle contains a GT mesh (space + elements) and a PRED mesh.
    """
    exports: List[Dict[str, Any]] = []
    if not space_matches:
        return exports
    os.makedirs(output_dir, exist_ok=True)

    for rec in space_matches:
        match_idx = rec.get("match_index")
        if match_idx is None:
            match_idx = len(exports)
        gt_idx = rec.get("gt_index")
        pr_idx = rec.get("pred_index")
        gt_guid = rec.get("gt_guid") or f"gt-space-{gt_idx}"
        pr_guid = rec.get("pred_guid") or f"pred-space-{pr_idx}"

        if gt_idx is None or pr_idx is None:
            continue
        if gt_idx >= len(spaces_gt) or pr_idx >= len(spaces_pr):
            continue

        by_space_key = f"{rec.get('gt_guid') or f'<gt-space-{gt_idx}>'}::<->::{rec.get('pred_guid') or f'<pred-space-{pr_idx}>'}"
        pair_entry = by_space.get(by_space_key, {})
        elem_indices = pair_entry.get("element_indices", {})
        gt_elem_indices = elem_indices.get("gt", [])
        pr_elem_indices = elem_indices.get("pred", [])

        match_dir_name = f"{match_idx:03d}_{_sanitize_for_filename(gt_guid, fallback=f'gt_{gt_idx}')}"
        match_dir = os.path.join(output_dir, match_dir_name)
        os.makedirs(match_dir, exist_ok=True)
        gt_mesh_path = os.path.join(match_dir, "gt_space_with_elements.ply")
        pr_mesh_path = os.path.join(match_dir, "pred_space_with_elements.ply")

        gt_space = spaces_gt[gt_idx]
        pr_space = spaces_pr[pr_idx]

        gt_meshes = [gt_space.mesh] + [elems_gt[i].mesh for i in gt_elem_indices if 0 <= i < len(elems_gt)]
        pr_meshes = [pr_space.mesh] + [elems_pr[i].mesh for i in pr_elem_indices if 0 <= i < len(elems_pr)]

        log_step(
            f"  [{round_label}] Exporting space match {match_idx} -> {match_dir_name} "
            f"(GT {gt_guid} / PRED {pr_guid})"
        )

        _write_combined_mesh(gt_meshes, gt_mesh_path)
        _write_combined_mesh(pr_meshes, pr_mesh_path)

        exports.append({
            "match_index": match_idx,
            "gt_index": gt_idx,
            "pred_index": pr_idx,
            "gt_guid": gt_guid,
            "pred_guid": pr_guid,
            "directory": match_dir,
            "gt_mesh": gt_mesh_path,
            "pred_mesh": pr_mesh_path
        })

    return exports


# ---------- OBB math ----------

def _transform_obb(obb: OBB, T: np.ndarray) -> OBB:
    if obb is None:
        raise ValueError("Cannot transform a null OBB")
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")

    R_T = T[:3, :3]
    t_T = T[:3, 3]

    old_center = np.asarray(obb.center, dtype=float)
    center_transformed = R_T @ old_center + t_T

    old_axes = np.asarray(obb.R, dtype=float)
    axis_list: List[np.ndarray] = []
    for i in range(3):
        axis = R_T @ old_axes[:, i]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-18:
            axis = np.zeros(3, dtype=float)
            axis[i] = 1.0
            norm = 1.0
        axis_list.append(axis / norm)
    R_new = np.column_stack(axis_list)

    old_corners = np.asarray(obb.corners, dtype=float)
    corners_h = np.concatenate([old_corners, np.ones((old_corners.shape[0], 1), dtype=float)], axis=1)
    transformed_corners = (corners_h @ T.T)[:, :3]

    offsets = transformed_corners - center_transformed
    half = np.zeros(3, dtype=float)
    for i in range(3):
        axis = R_new[:, i]
        half[i] = float(np.max(np.abs(offsets @ axis)))
    extent = half * 2.0

    corners: List[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = center_transformed.copy()
                corner += sx * half[0] * R_new[:, 0]
                corner += sy * half[1] * R_new[:, 1]
                corner += sz * half[2] * R_new[:, 2]
                corners.append(corner)
    corners_arr = np.asarray(corners, dtype=float)

    planes: List[Tuple[np.ndarray, float]] = []
    for i in range(3):
        axis = R_new[:, i]
        pos = center_transformed + half[i] * axis
        neg = center_transformed - half[i] * axis
        planes.append((axis.copy(), float(np.dot(axis, pos))))
        neg_axis = -axis
        planes.append((neg_axis.copy(), float(np.dot(neg_axis, neg))))

    return OBB(
        center=center_transformed,
        R=R_new,
        extent=extent,
        half=half,
        planes=planes,
        corners=corners_arr
    )


def _obb_volume(obb: OBB) -> float:
    return float(np.prod(obb.extent))


def _plane_triple_intersection(p1, p2, p3, tol: float = 1e-12) -> Optional[np.ndarray]:
    n1, d1 = p1
    n2, d2 = p2
    n3, d3 = p3
    N = np.vstack([n1, n2, n3])
    det = np.linalg.det(N)
    if abs(det) < tol:
        return None
    try:
        x = np.linalg.solve(N, np.array([d1, d2, d3], dtype=float))
        return x
    except Exception:
        return None


def _point_in_halfspaces(x: np.ndarray, planes: List[Tuple[np.ndarray, float]], eps: float) -> bool:
    for (n, d) in planes:
        if np.dot(n, x) - d > eps:
            return False
    return True


def _dedup_points(pts: List[np.ndarray], eps: float) -> np.ndarray:
    if not pts:
        return np.zeros((0, 3), dtype=float)
    scale = max(1.0, 1.0 / max(eps, 1e-12))
    key = np.round(np.asarray(pts) * scale).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.asarray(pts, dtype=float)[np.sort(idx)]


def _orthonormal_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        n = np.array([0, 0, 1.0], dtype=float)
        n_norm = 1.0
    n = n / n_norm
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        a = np.array([0.0, 0.0, 1.0])
        u = np.cross(n, a)
        u_norm = np.linalg.norm(u)
    u = u / (u_norm + 1e-18)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-18)
    return u, v, n


def _polyhedron_volume_from_planes_and_vertices(
    planes: List[Tuple[np.ndarray, float]],
    verts: np.ndarray,
    eps_on_plane: float = 1e-7
) -> float:
    """
    Given a convex polyhedron defined by halfspaces {n·x <= d}, and the set of its vertices,
    compute volume WITHOUT convex hull by reconstructing faces on each active plane
    and summing signed tetra volumes w.r.t. origin.
    """
    if verts.shape[0] < 4:
        return 0.0

    vol = 0.0
    for (n, d) in planes:
        mask = np.abs(verts @ n - d) <= eps_on_plane
        pts = verts[mask]
        if pts.shape[0] < 3:
            continue

        c = np.mean(pts, axis=0)
        u, v, n_hat = _orthonormal_basis_from_normal(n)
        rel = pts - c
        ang = np.arctan2(rel @ v, rel @ u)
        order = np.argsort(ang)
        poly = pts[order]

        for i in range(1, poly.shape[0] - 1):
            a = poly[0]
            b = poly[i]
            ctri = poly[i + 1]
            tri_n = np.cross(b - a, ctri - a)
            if np.dot(tri_n, n_hat) < 0:
                b, ctri = ctri, b
                tri_n = -tri_n
            vol += np.dot(a, np.cross(b, ctri)) / 6.0

    return float(abs(vol))


def _intersection_vertices_from_planes(planes: List[Tuple[np.ndarray, float]], eps_inside: float = 1e-7) -> np.ndarray:
    pts: List[np.ndarray] = []
    L = len(planes)
    for i in range(L):
        for j in range(i + 1, L):
            for k in range(j + 1, L):
                x = _plane_triple_intersection(planes[i], planes[j], planes[k])
                if x is None:
                    continue
                if _point_in_halfspaces(x, planes, eps_inside):
                    pts.append(x)
    return _dedup_points(pts, eps_inside)


_MESH_BOOL_LOG_STATE = {
    "trimesh_missing": False,
    "trimesh_boolean_failed": False,
    "raycast_missing": False,
    "raycast_failed": False,
}

# IoU backend settings (populated from CLI in main()).
# - auto: trimesh booleans if available, else raycast occupancy integration
# - trimesh: force trimesh booleans (fails -> 0.0)
# - raycast: force raycast occupancy integration (no booleans)
IOU_BACKEND: str = "auto"

# Raycast backend parameters (meters).
IOU_RAYCAST_PITCH: float = 0.05
IOU_RAYCAST_MAX_VOXELS: int = 2_000_000
IOU_RAYCAST_CHUNK: int = 250_000

# Internal caches (invalidated whenever meshes are transformed).
_TRIMESH_CACHE: Dict[int, Any] = {}
_RAYCAST_SCENE_CACHE: Dict[int, Any] = {}


def _clear_mesh_iou_caches() -> None:
    _TRIMESH_CACHE.clear()
    _RAYCAST_SCENE_CACHE.clear()


def legacy_to_tensor_minimal(legacy_mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    """
    Convert an Open3D legacy TriangleMesh to a minimal tensor TriangleMesh containing only:
      - vertex.positions
      - triangle.indices
    This avoids attribute-related boolean issues across Open3D versions.
    """
    vp = o3d.core.Tensor(np.asarray(legacy_mesh.vertices), o3d.core.Dtype.Float32)
    ti = o3d.core.Tensor(np.asarray(legacy_mesh.triangles), o3d.core.Dtype.Int64)
    return o3d.t.geometry.TriangleMesh(vp, ti)


def drop_all_but_positions_indices(tm: Any) -> None:
    """
    Remove all vertex attributes except 'positions' and all triangle attributes except 'indices'.
    Defensive across Open3D versions (TensorMap APIs can differ).
    """

    def _list_attr_map(attr_map: Any) -> List[str]:
        try:
            if hasattr(attr_map, "get_attribute_names") and callable(attr_map.get_attribute_names):
                return list(attr_map.get_attribute_names())
            if hasattr(attr_map, "attribute_names"):
                try:
                    return list(attr_map.attribute_names)
                except Exception:
                    pass
            if isinstance(attr_map, dict):
                return list(attr_map.keys())
            keys_attr = getattr(attr_map, "keys", None)
            if callable(keys_attr):
                try:
                    return list(keys_attr())
                except Exception:
                    pass
            return []
        except Exception:
            return []

    def _try_remove_attr(attr_map: Any, name: str) -> None:
        for remover in (
            getattr(attr_map, "remove_attribute", None),
            getattr(attr_map, "pop", None),
        ):
            try:
                if callable(remover):
                    if remover.__name__ == "pop":
                        remover(name, None)
                    else:
                        remover(name)
                    return
            except Exception:
                pass
        try:
            del attr_map[name]
            return
        except Exception:
            pass

    try:
        vmap = tm.vertex
        for k in _list_attr_map(vmap):
            if k != "positions":
                _try_remove_attr(vmap, k)
    except Exception:
        pass

    try:
        tmap = tm.triangle
        for k in _list_attr_map(tmap):
            if k != "indices":
                _try_remove_attr(tmap, k)
    except Exception:
        pass


def _o3d_signed_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """Signed volume via divergence theorem (assumes a closed, consistently-oriented surface)."""
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    if V.size == 0 or F.size == 0:
        return 0.0
    # Recenter to improve numerical stability for large-coordinate BIM models.
    try:
        ref = V.mean(axis=0)
        if ref.shape == (3,) and np.all(np.isfinite(ref)):
            V = V - ref
    except Exception:
        pass

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    cross = np.cross(v1, v2)
    vol = np.einsum("ij,ij->i", v0, cross).sum() / 6.0
    return float(vol)


def _o3d_clean_mesh_for_volume(mesh: o3d.geometry.TriangleMesh) -> None:
    # Best-effort cleanup before volume computation; avoid aggressive topology edits.
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
    except Exception:
        pass


def mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Compute the (positive) enclosed volume of a triangle mesh using the actual mesh geometry.
    Requires a (mostly) watertight surface for meaningful results.
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0
    m = _o3d_mesh_copy(mesh)
    _o3d_clean_mesh_for_volume(m)
    try:
        if m.is_orientable():
            m.orient_triangles()
    except Exception:
        pass

    # Avoid calling Open3D's TriangleMesh.get_volume() because it emits a hard error log
    # for meshes it considers non-watertight (even if downstream volume is still computable).
    try:
        vol = float(abs(_o3d_signed_volume(m)))
        return vol if math.isfinite(vol) else 0.0
    except Exception:
        pass

    # Optional fallback: trimesh (if installed) can sometimes compute volumes more robustly.
    try:
        import trimesh  # type: ignore

        V = np.asarray(m.vertices, dtype=np.float64)
        F = np.asarray(m.triangles, dtype=np.int64)
        tm = trimesh.Trimesh(vertices=V, faces=F, process=True)
        vol = float(abs(tm.volume))
        return vol if math.isfinite(vol) else 0.0
    except Exception:
        return 0.0


def _o3d_to_trimesh_cached(mesh: o3d.geometry.TriangleMesh) -> Optional[Any]:
    if not TRIMESH_OK:
        if not _MESH_BOOL_LOG_STATE["trimesh_missing"]:
            log_step("  Trimesh not available; falling back to raycast IoU where needed.")
            _MESH_BOOL_LOG_STATE["trimesh_missing"] = True
        return None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    key = id(mesh)
    cached = _TRIMESH_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            return None
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)  # type: ignore[attr-defined]
        try:
            tm.remove_duplicate_faces()
            tm.remove_degenerate_faces()
            tm.remove_unreferenced_vertices()
            tm.remove_infinite_values()
            tm.remove_nan()
        except Exception:
            pass
        _TRIMESH_CACHE[key] = tm
        return tm
    except Exception:
        return None


def _collapse_trimesh_result(result: Any) -> Optional[Any]:
    """Normalize trimesh boolean outputs into a single Trimesh instance."""
    if not TRIMESH_OK or result is None:
        return None
    try:
        if isinstance(result, trimesh.Trimesh):  # type: ignore[attr-defined]
            return result
        if isinstance(result, (list, tuple)):
            meshes = [m for m in result if isinstance(m, trimesh.Trimesh) and len(getattr(m, "faces", [])) > 0]  # type: ignore[attr-defined]
            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            try:
                return trimesh.util.concatenate(meshes)  # type: ignore[attr-defined]
            except Exception:
                return meshes[0]
        if isinstance(result, trimesh.Scene):  # type: ignore[attr-defined]
            try:
                collapsed = result.dump(concatenate=True)
                return _collapse_trimesh_result(collapsed)
            except Exception:
                try:
                    meshes = [g for g in result.geometry.values() if isinstance(g, trimesh.Trimesh)]  # type: ignore[attr-defined]
                    if not meshes:
                        return None
                    if len(meshes) == 1:
                        return meshes[0]
                    return trimesh.util.concatenate(meshes)  # type: ignore[attr-defined]
                except Exception:
                    return None
    except Exception:
        return None
    return None


def _trimesh_volume(mesh: Any) -> float:
    if not TRIMESH_OK or mesh is None:
        return 0.0
    try:
        v = float(abs(mesh.volume))
        return v if math.isfinite(v) else 0.0
    except Exception:
        pass
    try:
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)
        if V.size == 0 or F.size == 0:
            return 0.0
        ref = V.mean(axis=0)
        if ref.shape == (3,) and np.all(np.isfinite(ref)):
            V = V - ref
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
        vol = float(abs(vol))
        return vol if math.isfinite(vol) else 0.0
    except Exception:
        return 0.0


def _trimesh_boolean_intersection_tms(tm_a: Any, tm_b: Any) -> Tuple[bool, Optional[Any]]:
    if not TRIMESH_OK or tm_a is None or tm_b is None:
        return False, None
    succeeded = False
    for engine in ("manifold", None):
        try:
            inter_raw = trimesh.boolean.intersection([tm_a, tm_b], engine=engine)  # type: ignore[attr-defined]
            succeeded = True
            inter_tm = _collapse_trimesh_result(inter_raw)
            if inter_tm is not None and len(getattr(inter_tm, "faces", [])) > 0:
                return True, inter_tm
            # empty intersection
            return True, trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)  # type: ignore[attr-defined]
        except Exception:
            continue
    return (True, trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)) if succeeded else (False, None)  # type: ignore[attr-defined]


def _trimesh_boolean_intersection(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> Optional[Any]:
    if not TRIMESH_OK:
        return None
    tm_a = _o3d_to_trimesh_cached(a)
    tm_b = _o3d_to_trimesh_cached(b)
    if tm_a is None or tm_b is None:
        return None

    ok, inter_tm = _trimesh_boolean_intersection_tms(tm_a, tm_b)
    if ok:
        return inter_tm
    if not _MESH_BOOL_LOG_STATE["trimesh_boolean_failed"]:
        log_step("  Trimesh boolean intersection failed; falling back to raycast IoU where needed.")
        _MESH_BOOL_LOG_STATE["trimesh_boolean_failed"] = True
    return None


def _trimesh_boolean_union(meshes: List[o3d.geometry.TriangleMesh]) -> Optional[Any]:
    if not TRIMESH_OK:
        return None
    converted = []
    for m in meshes or []:
        tm = _o3d_to_trimesh_cached(m)
        if tm is not None and len(getattr(tm, "faces", [])) > 0:
            converted.append(tm)
    if not converted:
        return None
    if len(converted) == 1:
        return converted[0]

    for engine in ("manifold", None):
        try:
            union_raw = trimesh.boolean.union(converted, engine=engine)  # type: ignore[attr-defined]
            union_tm = _collapse_trimesh_result(union_raw)
            if union_tm is not None and len(getattr(union_tm, "faces", [])) > 0:
                return union_tm
        except Exception:
            continue

    if not _MESH_BOOL_LOG_STATE["trimesh_boolean_failed"]:
        log_step("  Trimesh boolean union failed; falling back to raycast IoU where needed.")
        _MESH_BOOL_LOG_STATE["trimesh_boolean_failed"] = True
    return None


def _has_raycasting_scene() -> bool:
    try:
        return bool(getattr(o3d, "t", None) and getattr(o3d.t, "geometry", None) and hasattr(o3d.t.geometry, "RaycastingScene"))
    except Exception:
        return False


def _raycast_scene_for_mesh(mesh: o3d.geometry.TriangleMesh) -> Optional[Any]:
    if not _has_raycasting_scene():
        if not _MESH_BOOL_LOG_STATE["raycast_missing"]:
            log_step("  Open3D RaycastingScene unavailable; raycast IoU cannot be used.")
            _MESH_BOOL_LOG_STATE["raycast_missing"] = True
        return None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    key = id(mesh)
    cached = _RAYCAST_SCENE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        m = _o3d_mesh_copy(mesh)
        _o3d_clean_mesh_for_volume(m)
        try:
            if m.is_orientable():
                m.orient_triangles()
        except Exception:
            pass

        try:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(m)
        except Exception:
            mesh_t = legacy_to_tensor_minimal(m)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)
        _RAYCAST_SCENE_CACHE[key] = scene
        return scene
    except Exception:
        if not _MESH_BOOL_LOG_STATE["raycast_failed"]:
            log_step("  RaycastingScene setup failed; raycast IoU will return 0 for affected pairs.")
            _MESH_BOOL_LOG_STATE["raycast_failed"] = True
        return None


def _raycast_scene_for_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> Optional[Any]:
    """Build a temporary RaycastingScene containing all meshes (used for union-of-set)."""
    if not _has_raycasting_scene():
        if not _MESH_BOOL_LOG_STATE["raycast_missing"]:
            log_step("  Open3D RaycastingScene unavailable; raycast IoU cannot be used.")
            _MESH_BOOL_LOG_STATE["raycast_missing"] = True
        return None
    meshes = [m for m in (meshes or []) if m is not None and len(m.vertices) > 0 and len(m.triangles) > 0]
    if not meshes:
        return None
    try:
        scene = o3d.t.geometry.RaycastingScene()
        for mesh in meshes:
            m = _o3d_mesh_copy(mesh)
            _o3d_clean_mesh_for_volume(m)
            try:
                if m.is_orientable():
                    m.orient_triangles()
            except Exception:
                pass
            try:
                mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(m)
            except Exception:
                mesh_t = legacy_to_tensor_minimal(m)
            _ = scene.add_triangles(mesh_t)
        return scene
    except Exception:
        if not _MESH_BOOL_LOG_STATE["raycast_failed"]:
            log_step("  RaycastingScene setup failed; raycast IoU will return 0 for affected pairs.")
            _MESH_BOOL_LOG_STATE["raycast_failed"] = True
        return None


def _raycast_occupancy(scene: Any, points: np.ndarray) -> Optional[np.ndarray]:
    if scene is None or points is None or points.size == 0:
        return None
    try:
        pts = np.asarray(points, dtype=np.float32)
        pts_t = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
        occ_t = scene.compute_occupancy(pts_t)
        try:
            occ = occ_t.numpy()
        except Exception:
            occ = np.asarray(occ_t)
        return np.asarray(occ).reshape(-1) > 0.5
    except Exception:
        return None


def _adaptive_raycast_pitch(mins: np.ndarray, maxs: np.ndarray, pitch: float, max_voxels: int) -> float:
    pitch = float(pitch)
    if pitch <= 0:
        pitch = 0.05
    max_voxels = int(max_voxels) if max_voxels is not None else 0
    if max_voxels <= 0:
        return pitch
    ext = np.maximum(np.asarray(maxs, dtype=float) - np.asarray(mins, dtype=float), 1e-9)
    try:
        # target ~max_voxels samples in the AABB
        pitch_needed = float((float(ext[0] * ext[1] * ext[2]) / float(max_voxels)) ** (1.0 / 3.0))
        if math.isfinite(pitch_needed) and pitch_needed > pitch:
            return pitch_needed
    except Exception:
        pass
    return pitch


def _raycast_iou_between_scenes(scene_a: Any, scene_b: Any, aabb_min: np.ndarray, aabb_max: np.ndarray, eps: float) -> float:
    if scene_a is None or scene_b is None:
        return 0.0

    pitch = _adaptive_raycast_pitch(aabb_min, aabb_max, IOU_RAYCAST_PITCH, IOU_RAYCAST_MAX_VOXELS)
    pitch = float(pitch)
    if pitch <= 0:
        return 0.0

    mins = np.asarray(aabb_min, dtype=float) - 0.5 * pitch
    maxs = np.asarray(aabb_max, dtype=float) + 0.5 * pitch
    ext = np.maximum(maxs - mins, 1e-9)
    dims = np.maximum(np.ceil(ext / pitch).astype(np.int64), 1)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    total = nx * ny * nz
    if total <= 0:
        return 0.0

    # voxel center coordinates per axis
    xs = mins[0] + (np.arange(nx, dtype=np.float64) + 0.5) * pitch
    ys = mins[1] + (np.arange(ny, dtype=np.float64) + 0.5) * pitch
    zs = mins[2] + (np.arange(nz, dtype=np.float64) + 0.5) * pitch

    inter_cnt = 0
    union_cnt = 0
    chunk = max(10_000, int(IOU_RAYCAST_CHUNK))
    plane = nx * ny
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        idx = np.arange(start, end, dtype=np.int64)
        ix = idx % nx
        iy = (idx // nx) % ny
        iz = idx // plane
        pts = np.column_stack([xs[ix], ys[iy], zs[iz]])

        occ_a = _raycast_occupancy(scene_a, pts)
        occ_b = _raycast_occupancy(scene_b, pts)
        if occ_a is None or occ_b is None:
            return 0.0
        inter = occ_a & occ_b
        union = occ_a | occ_b
        inter_cnt += int(inter.sum())
        union_cnt += int(union.sum())

    if union_cnt <= 0:
        return 0.0
    return float(inter_cnt / union_cnt)


def _o3d_boolean_intersection(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Boolean intersection using trimesh (preferred). Returns None on failure.
    """
    if a is None or b is None or len(a.triangles) == 0 or len(b.triangles) == 0:
        return None
    if not TRIMESH_OK:
        return None
    inter_tm = _trimesh_boolean_intersection(a, b)
    if inter_tm is None:
        return None
    try:
        vertices = np.asarray(inter_tm.vertices, dtype=np.float64)
        faces = np.asarray(inter_tm.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            return None
        out = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False))
        )
        return out if len(out.triangles) > 0 else None
    except Exception:
        return None


def _o3d_boolean_union(meshes: List[o3d.geometry.TriangleMesh]) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Boolean union using trimesh (preferred). Returns None on failure.
    """
    meshes = [m for m in (meshes or []) if m is not None and len(m.triangles) > 0 and len(m.vertices) > 0]
    if not meshes:
        return None
    if len(meshes) == 1:
        return _o3d_mesh_copy(meshes[0])
    if not TRIMESH_OK:
        return None

    union_tm = _trimesh_boolean_union(meshes)
    if union_tm is None:
        return None
    try:
        vertices = np.asarray(union_tm.vertices, dtype=np.float64)
        faces = np.asarray(union_tm.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            return None
        out = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False))
        )
        return out if len(out.triangles) > 0 else None
    except Exception:
        return None


def intersection_volume_meshes(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> float:
    inter = _o3d_boolean_intersection(a, b)
    if inter is None:
        return 0.0
    return mesh_volume(inter)


def union_volume_meshes(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh) -> float:
    u = _o3d_boolean_union([a, b])
    if u is not None:
        return mesh_volume(u)
    # Fallback: volume(A) + volume(B) - volume(intersection)
    va = mesh_volume(a)
    vb = mesh_volume(b)
    inter = intersection_volume_meshes(a, b)
    return max(0.0, float(va + vb - inter))


def iou_between_two_meshes(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh, eps: float = 1e-7) -> float:
    """
    Mesh IoU using a robust backend:
      - auto/trimesh: boolean intersection via trimesh (prefer manifold), union via volume identity
      - raycast: occupancy integration via Open3D RaycastingScene (no booleans)
    """
    if a is None or b is None or len(a.triangles) == 0 or len(b.triangles) == 0:
        return 0.0
    try:
        a_min, a_max = _o3d_bounds(a)
        b_min, b_max = _o3d_bounds(b)
        if not _aabb_overlap(a_min, a_max, b_min, b_max):
            return 0.0
    except Exception:
        pass

    backend = (IOU_BACKEND or "auto").strip().lower()
    if backend not in {"auto", "trimesh", "raycast"}:
        backend = "auto"

    # ---- trimesh boolean backend ----
    if backend in {"auto", "trimesh"} and TRIMESH_OK:
        inter_tm = _trimesh_boolean_intersection(a, b)
        if inter_tm is not None:
            faces = getattr(inter_tm, "faces", None)
            if faces is None or len(faces) == 0:
                return 0.0

            inter = _trimesh_volume(inter_tm)
            if inter <= eps:
                return 0.0

            va = mesh_volume(a)
            vb = mesh_volume(b)
            if va <= eps or vb <= eps:
                return 0.0
            union = max(0.0, float(va + vb - inter))
            return float(inter / union) if union > eps else 0.0

        if backend == "trimesh":
            return 0.0

    # ---- raycast fallback ----
    scene_a = _raycast_scene_for_mesh(a)
    scene_b = _raycast_scene_for_mesh(b)
    try:
        a_min, a_max = _o3d_bounds(a)
        b_min, b_max = _o3d_bounds(b)
        mins = np.minimum(a_min, b_min)
        maxs = np.maximum(a_max, b_max)
    except Exception:
        mins = np.zeros(3, dtype=float)
        maxs = np.ones(3, dtype=float)
    return _raycast_iou_between_scenes(scene_a, scene_b, mins, maxs, eps=eps)


def iou_union_of_set_vs_single(
    set_meshes: List[o3d.geometry.TriangleMesh],
    ref_mesh: o3d.geometry.TriangleMesh,
    eps: float = 1e-7,
    max_k_for_ie: int = 8
) -> float:
    """
    IoU( union(set_meshes), ref_mesh ).

    For performance, max_k_for_ie is treated as a cap on how many meshes to union.
    Backend selection matches iou_between_two_meshes().
    """
    meshes = [m for m in (set_meshes or []) if m is not None and len(m.triangles) > 0 and len(m.vertices) > 0]
    if not meshes or ref_mesh is None or len(ref_mesh.triangles) == 0:
        return 0.0
    if max_k_for_ie is not None and max_k_for_ie > 0 and len(meshes) > max_k_for_ie:
        meshes = meshes[:max_k_for_ie]

    backend = (IOU_BACKEND or "auto").strip().lower()
    if backend not in {"auto", "trimesh", "raycast"}:
        backend = "auto"

    # ---- trimesh boolean backend ----
    if backend in {"auto", "trimesh"} and TRIMESH_OK:
        union_tm = _trimesh_boolean_union(meshes)
        if union_tm is not None and len(getattr(union_tm, "faces", [])) > 0:
            tm_ref = _o3d_to_trimesh_cached(ref_mesh)
            if tm_ref is not None and len(getattr(tm_ref, "faces", [])) > 0:
                ok, inter_tm = _trimesh_boolean_intersection_tms(union_tm, tm_ref)
                if ok and inter_tm is not None:
                    if len(getattr(inter_tm, "faces", [])) == 0:
                        return 0.0
                    inter = _trimesh_volume(inter_tm)
                    if inter <= eps:
                        return 0.0
                    v_union = _trimesh_volume(union_tm)
                    v_ref = mesh_volume(ref_mesh)
                    den = max(0.0, float(v_union + v_ref - inter))
                    return float(inter / den) if den > eps else 0.0

        if backend == "trimesh":
            return 0.0

    # ---- raycast fallback ----
    scene_union = _raycast_scene_for_meshes(meshes)
    scene_ref = _raycast_scene_for_mesh(ref_mesh)
    try:
        bounds = [_o3d_bounds(ref_mesh)] + [_o3d_bounds(m) for m in meshes]
        mins = np.min(np.stack([b[0] for b in bounds], axis=0), axis=0)
        maxs = np.max(np.stack([b[1] for b in bounds], axis=0), axis=0)
    except Exception:
        mins = np.zeros(3, dtype=float)
        maxs = np.ones(3, dtype=float)
    return _raycast_iou_between_scenes(scene_union, scene_ref, mins, maxs, eps=eps)


# ---------- IFC -> mesh & metadata ----------

def _ifc_length_scale_m(ifc) -> float:
    try:
        return float(ifc_unit.calculate_unit_scale(ifc))
    except Exception:
        return 1.0

def _get_product_psets(p) -> Optional[Dict[str, Any]]:
    if ifc_get_psets is None:
        return None
    try:
        return ifc_get_psets(p, include_inherited=True, recursive=True)
    except Exception:
        return None

def _product_meta(p, psets: Optional[Dict[str, Any]] = None) -> Meta:
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
    if psets is None:
        psets = _get_product_psets(p)
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

def _numeric_candidates_from_value(value: Any) -> List[float]:
    nums: List[float] = []

    def _collect(val: Any) -> None:
        if val is None:
            return
        if isinstance(val, bool):
            return
        if isinstance(val, (int, float, np.integer, np.floating)):
            fv = float(val)
            if math.isfinite(fv):
                nums.append(fv)
            return
        if isinstance(val, str):
            tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val.replace(",", "."))
            for tok in tokens:
                try:
                    fv = float(tok)
                except ValueError:
                    continue
                if math.isfinite(fv):
                    nums.append(fv)
            return
        if isinstance(val, (list, tuple, set)):
            for item in val:
                _collect(item)
            return
        if isinstance(val, dict):
            for key in ("value", "Value", "NominalValue", "UpperBoundValue", "LowerBoundValue"):
                if key in val:
                    _collect(val[key])
            for item in val.values():
                if isinstance(item, (dict, list, tuple, set)):
                    _collect(item)
            return
        attr = getattr(val, "wrappedValue", None)
        if attr is not None and attr is not val:
            _collect(attr)
            return
        attr = getattr(val, "NominalValue", None)
        if attr is not None and attr is not val:
            _collect(attr)
            return
        attr = getattr(val, "Value", None)
        if attr is not None and attr is not val:
            _collect(attr)
            return
        try:
            fv = float(val)
        except (TypeError, ValueError):
            return
        if math.isfinite(fv):
            nums.append(fv)

    _collect(value)
    return nums

def _collect_wall_thickness_candidates_from_psets(psets: Optional[Dict[str, Any]]) -> List[Tuple[float, bool]]:
    candidates: List[Tuple[float, bool]] = []
    if not isinstance(psets, dict):
        return candidates
    for props in psets.values():
        if not isinstance(props, dict):
            continue
        for prop_name, prop_value in props.items():
            if not isinstance(prop_name, str):
                continue
            key = prop_name.lower()
            if not any(term in key for term in ("thick", "width", "depth")):
                continue
            values = _numeric_candidates_from_value(prop_value)
            if not values:
                continue
            is_strong = "thick" in key
            for v in values:
                candidates.append((v, is_strong))
    return candidates

def _collect_candidates_from_layer_set(layer_set: Any) -> List[Tuple[float, bool]]:
    candidates: List[Tuple[float, bool]] = []
    if layer_set is None:
        return candidates
    total_attr = getattr(layer_set, "TotalThickness", None)
    for v in _numeric_candidates_from_value(total_attr):
        candidates.append((v, True))
    try:
        layers = getattr(layer_set, "MaterialLayers", None)
    except Exception:
        layers = None
    if layers:
        layer_values: List[float] = []
        try:
            iterator = list(layers)
        except TypeError:
            iterator = [layers]
        except Exception:
            iterator = []
        for layer in iterator:
            layer_thickness = getattr(layer, "LayerThickness", None)
            layer_values.extend(_numeric_candidates_from_value(layer_thickness))
        if layer_values:
            candidates.append((sum(layer_values), True))
    return candidates

def _collect_wall_thickness_candidates_from_material(material: Any) -> List[Tuple[float, bool]]:
    candidates: List[Tuple[float, bool]] = []
    if material is None:
        return candidates
    type_name = ""
    try:
        type_name = material.is_a()
    except Exception:
        type_name = ""
    if type_name == "IfcMaterialLayerSetUsage":
        candidates.extend(_collect_candidates_from_layer_set(getattr(material, "ForLayerSet", None)))
    elif type_name == "IfcMaterialLayerSet":
        candidates.extend(_collect_candidates_from_layer_set(material))
    else:
        try:
            if getattr(material, "MaterialLayers", None):
                candidates.extend(_collect_candidates_from_layer_set(material))
        except Exception:
            pass
    total_attr = getattr(material, "TotalThickness", None)
    for v in _numeric_candidates_from_value(total_attr):
        candidates.append((v, True))
    single_layer = getattr(material, "MaterialLayer", None)
    if single_layer is not None:
        layer_values = _numeric_candidates_from_value(getattr(single_layer, "LayerThickness", None))
        if layer_values:
            candidates.append((sum(layer_values), True))
    return candidates

def _collect_wall_thickness_candidates_from_product_materials(p: Any) -> List[Tuple[float, bool]]:
    candidates: List[Tuple[float, bool]] = []
    associations = getattr(p, "HasAssociations", None)
    if not associations:
        return candidates
    try:
        iterator = list(associations)
    except TypeError:
        iterator = [associations]
    except Exception:
        iterator = []
    for rel in iterator:
        try:
            if not rel.is_a("IfcRelAssociatesMaterial"):
                continue
        except Exception:
            continue
        material = getattr(rel, "RelatingMaterial", None)
        candidates.extend(_collect_wall_thickness_candidates_from_material(material))
    return candidates

def _extract_wall_thickness(p: Any, psets: Optional[Dict[str, Any]], obb: Optional[OBB], scale_to_m: float) -> Optional[float]:
    etype = ""
    try:
        etype = p.is_a()
    except Exception:
        etype = ""
    if not isinstance(etype, str) or not etype.lower().startswith("ifcwall"):
        return None

    strong: List[float] = []
    weak: List[float] = []

    if psets is None:
        psets = _get_product_psets(p)
    for value, is_strong in _collect_wall_thickness_candidates_from_psets(psets):
        scaled = float(value) * float(scale_to_m)
        if not math.isfinite(scaled) or scaled <= 0.0:
            continue
        if is_strong:
            strong.append(scaled)
        else:
            weak.append(scaled)

    for value, _ in _collect_wall_thickness_candidates_from_product_materials(p):
        scaled = float(value) * float(scale_to_m)
        if math.isfinite(scaled) and scaled > 0.0:
            strong.append(scaled)

    if strong:
        return min(strong)
    if weak:
        return min(weak)
    if obb is not None:
        try:
            extent = np.asarray(obb.extent, dtype=float)
            if extent.size:
                min_edge = float(np.min(np.abs(extent)))
                if math.isfinite(min_edge) and min_edge > 0.0:
                    return min_edge
        except Exception:
            pass
    return None

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


def _object_obb_from_shape(shape, scale_to_m: float) -> Optional[OBB]:
    g = shape.geometry
    verts = getattr(g, "verts", None)
    if verts is None:
        verts = getattr(g, "vertices", None)
    if verts is None:
        return None
    V = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if V.shape[0] == 0:
        return None

    local_min = V.min(axis=0)
    local_max = V.max(axis=0)
    extent_local = local_max - local_min
    center_local = (local_min + local_max) * 0.5
    half_local = extent_local * 0.5

    M = np.array(shape.transformation.matrix, dtype=np.float64).reshape(4, 4).T
    R_raw = M[:3, :3]
    t_raw = M[:3, 3]

    axes: List[np.ndarray] = []
    axis_scales = np.zeros(3, dtype=float)
    for i in range(3):
        col = R_raw[:, i]
        norm = float(np.linalg.norm(col))
        if norm < 1e-18:
            col = np.zeros(3, dtype=float)
            col[i] = 1.0
            norm = 1.0
        axes.append(col / norm)
        axis_scales[i] = norm
    R_world = np.column_stack(axes)

    extent = extent_local * axis_scales * scale_to_m
    half = half_local * axis_scales * scale_to_m
    center_world = (R_raw @ center_local + t_raw) * scale_to_m

    planes: List[Tuple[np.ndarray, float]] = []
    corners: List[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = center_world.copy()
                corner += sx * half[0] * R_world[:, 0]
                corner += sy * half[1] * R_world[:, 1]
                corner += sz * half[2] * R_world[:, 2]
                corners.append(corner)
    corners_arr = np.asarray(corners, dtype=float)

    for i in range(3):
        axis = R_world[:, i]
        pos = center_world + half[i] * axis
        neg = center_world - half[i] * axis
        planes.append((axis.copy(), float(np.dot(axis, pos))))
        neg_axis = -axis
        planes.append((neg_axis.copy(), float(np.dot(neg_axis, neg))))

    return OBB(
        center=center_world,
        R=R_world,
        extent=extent,
        half=half,
        planes=planes,
        corners=corners_arr
    )

def _build_geom_settings(use_python_occ: bool) -> Optional["ifcopenshell.geom.settings"]:
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
        if tl == etl.replace("ifc", ""):
            return True
    return False

def load_ifc_components(path: str, include_types: Optional[List[str]] = None) -> Tuple[List[Comp], float]:
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

    # Typically keep at 1.0 unless you know otherwise
    scale = 1.0

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
    progress_step = 50
    for p in ifc.by_type("IfcProduct"):
        if not getattr(p, "Representation", None):
            continue

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

        obb = _object_obb_from_shape(shape, scale_to_m=scale)
        if obb is None:
            log_step(f"  Could not derive object-oriented OBB for {guid} ({etype}); skipping")
            continue
        vol = _obb_volume(obb)
        if vol <= 0.0:
            log_step(f"  Non-positive OBB volume for {guid} ({etype}); skipping")
            continue

        aabb_min, aabb_max = _o3d_bounds(mesh)
        psets = _get_product_psets(p)
        meta = _product_meta(p, psets=psets)
        wall_thickness = _extract_wall_thickness(p, psets, obb, scale)
        comps.append(Comp(
            idx=len(comps),
            guid=meta.GlobalId,
            etype=meta.IfcType,
            meta=meta,
            mesh=mesh,
            obb=obb,
            volume=vol,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            wall_thickness=wall_thickness
        ))
        if len(comps) % progress_step == 0:
            log_step(f"  Meshed {len(comps)} components so far")
    if not comps:
        raise RuntimeError(f"No meshable IfcProducts with positive OBB volume found in {os.path.basename(path)}")
    log_step(f"Finished meshing {len(comps)} components (processed {total_products} candidates)")
    return comps, scale


# ---------- Space membership + matching ----------

def _guid_or_id(entity) -> str:
    guid = getattr(entity, "GlobalId", None)
    if isinstance(guid, str) and guid:
        return guid
    try:
        return str(int(entity.id()))
    except Exception:
        try:
            return str(entity.id())
        except Exception:
            return repr(entity)


def _clean_include_types_for_elements(include_types: Optional[List[str]]) -> Optional[List[str]]:
    if not include_types:
        return None
    cleaned: List[str] = []
    for t in include_types:
        if not t:
            continue
        parts = [p.strip() for p in str(t).split(",")]
        for p in parts:
            if not p:
                continue
            if p.lower() == "ifcspace":
                continue
            if p not in cleaned:
                cleaned.append(p)
    return cleaned or None


def build_space_membership(
    ifc_path: str,
    include_types_for_elements: Optional[List[str]]
) -> Tuple[List[Comp], List[Comp], Dict[str, List[int]], Dict[str, List[str]]]:
    """
    Load spaces and elements (excluding IfcSpace) while building membership maps.
    Returns (spaces, elements, space_guid_to_elem_indices, elem_guid_to_space_guids).
    """
    log_step(f"Building space membership from {ifc_path}")
    element_types = _clean_include_types_for_elements(include_types_for_elements)

    spaces, _ = load_ifc_components(ifc_path, include_types=["IfcSpace"])
    log_step(f"  Loaded {len(spaces)} spaces with geometry")

    elements, _ = load_ifc_components(ifc_path, include_types=element_types)
    if element_types is None:
        elements = [c for c in elements if c.etype.lower() != "ifcspace"]
    else:
        elements = [c for c in elements if c.etype.lower() != "ifcspace"]
    log_step(f"  Loaded {len(elements)} elements (non-space)")

    space_guid_to_elem_indices: Dict[str, List[int]] = {s.guid: [] for s in spaces}
    elem_guid_to_space_guids: Dict[str, List[str]] = {}
    elem_guid_to_index: Dict[str, int] = {}
    for idx, comp in enumerate(elements):
        elem_guid_to_index[comp.guid] = idx

    try:
        ifc = ifcopenshell.open(ifc_path)
    except Exception as exc:
        log_step(f"  Failed to reopen IFC for membership lookup: {exc}")
        return spaces, elements, space_guid_to_elem_indices, elem_guid_to_space_guids

    def record_membership(space_guid: str, elem_guid: str) -> None:
        idx = elem_guid_to_index.get(elem_guid)
        if idx is None:
            return
        lst = space_guid_to_elem_indices.setdefault(space_guid, [])
        if idx not in lst:
            lst.append(idx)
        elem_lst = elem_guid_to_space_guids.setdefault(elem_guid, [])
        if space_guid not in elem_lst:
            elem_lst.append(space_guid)

    space_guids = set(space_guid_to_elem_indices.keys())
    try:
        for rel in ifc.by_type("IfcRelContainedInSpatialStructure"):
            space = getattr(rel, "RelatingStructure", None)
            if space is None or not space.is_a("IfcSpace"):
                continue
            space_guid = _guid_or_id(space)
            if space_guid not in space_guids:
                continue
            for obj in getattr(rel, "RelatedElements", []) or []:
                record_membership(space_guid, _guid_or_id(obj))
    except Exception as exc:
        log_step(f"  Membership via IfcRelContainedInSpatialStructure failed: {exc}")

    # Defensive: also inspect inverse ContainsElements on spaces.
    try:
        for space in ifc.by_type("IfcSpace"):
            space_guid = _guid_or_id(space)
            if space_guid not in space_guids:
                continue
            rels = getattr(space, "ContainsElements", None)
            if not rels:
                continue
            for rel in rels:
                for obj in getattr(rel, "RelatedElements", []) or []:
                    record_membership(space_guid, _guid_or_id(obj))
    except Exception:
        pass

    BOUNDARY_TYPES = (
        "IfcRelSpaceBoundary2ndLevel",
        "IfcRelSpaceBoundary1stLevel",
        "IfcRelSpaceBoundary",
    )

    added_any_boundary_memberships = False
    for t in BOUNDARY_TYPES:
        try:
            rsbs = ifc.by_type(t)
        except Exception as exc:
            # IFC2X3 does not define the 1st/2nd level boundary entities; keep going so that
            # IfcRelSpaceBoundary can still be processed.
            log_step(f"  Skipping {t} membership lookup: {exc}")
            continue

        for rsb in rsbs:
            sp = getattr(rsb, "RelatingSpace", None)
            el = getattr(rsb, "RelatedBuildingElement", None)
            if sp is None or el is None:
                # space-to-space or virtual boundaries; nothing to add
                continue

            space_guid = _guid_or_id(sp)
            if space_guid not in space_guid_to_elem_indices:
                continue

            elem_guid = _guid_or_id(el)

            # If the exact element GUID isn’t in our elements list (because of include_types
            # filtering or decomposition), try bubbling up to a parent that *is* in the list.
            idx = elem_guid_to_index.get(elem_guid)
            if idx is None:
                try:
                    # el.Decomposes -> parents (assemblies)
                    parents = getattr(el, "Decomposes", None) or []
                    for rel in parents:
                        par = getattr(rel, "RelatingObject", None)
                        if par:
                            par_guid = _guid_or_id(par)
                            if par_guid in elem_guid_to_index:
                                elem_guid = par_guid
                                break
                except Exception:
                    pass

            before = len(space_guid_to_elem_indices.get(space_guid, []))
            record_membership(space_guid, elem_guid)
            after = len(space_guid_to_elem_indices.get(space_guid, []))
            if after > before:
                added_any_boundary_memberships = True

    if added_any_boundary_memberships:
        log_step("  Added memberships from IfcRelSpaceBoundary*")

    # Ensure deterministic ordering of element indices in each space.
    for guid, lst in space_guid_to_elem_indices.items():
        if not lst:
            continue
        lst.sort()

    log_step("  Space membership mapping complete")
    return spaces, elements, space_guid_to_elem_indices, elem_guid_to_space_guids


def compute_space_iou_matrix(
    spaces_gt: List[Comp],
    spaces_pr: List[Comp],
    thresh: float,
    inside_eps: float = 1e-7
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    m, n = len(spaces_gt), len(spaces_pr)
    log_step(f"Computing space IoU matrix ({m}x{n})")
    S = np.zeros((m, n), dtype=float)
    above_thresh: List[Tuple[int, int, float]] = []
    for i, g in enumerate(spaces_gt):
        for j, p in enumerate(spaces_pr):
            v = iou_between_two_meshes(g.mesh, p.mesh, eps=inside_eps)
            S[i, j] = v
            if v >= thresh:
                above_thresh.append((i, j, float(v)))
    log_step("Space IoU matrix ready")
    return S, above_thresh


def match_spaces_one_to_one(S: np.ndarray, thresh: float) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = float(S[i, j])
            if v >= thresh:
                candidates.append((v, i, j))
    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pr: set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for v, i, j in candidates:
        if i in matched_gt or j in matched_pr:
            continue
        matched_gt.add(i)
        matched_pr.add(j)
        matches.append((i, j, v))
    log_step(f"Matched {len(matches)} space pairs (threshold={thresh})")
    return matches


def align_space_pair(gt_space: Comp, pr_space: Comp, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "none":
        return np.eye(4, dtype=float)
    if mode == "icp":
        return rigid_icp_align([pr_space.mesh], [gt_space.mesh])
    if mode == "icp_xy":
        return rigid_icp_align_xy_only([pr_space.mesh], [gt_space.mesh])
    if mode == "centroid":
        return _centroid_align([pr_space.mesh], [gt_space.mesh])
    raise ValueError(f"Unsupported space alignment mode: {mode}")


def apply_transform_to_indices(comps: List[Comp], indices: List[int], T: np.ndarray) -> None:
    if not indices:
        return
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")
    for idx in indices:
        if idx < 0 or idx >= len(comps):
            continue
        comp = comps[idx]
        comp.mesh.transform(T)
        comp.aabb_min, comp.aabb_max = _o3d_bounds(comp.mesh)
        comp.obb = _transform_obb(comp.obb, T)
        comp.volume = _obb_volume(comp.obb)
    _clear_mesh_iou_caches()


def _matrix_to_nested_list(T: np.ndarray) -> List[List[float]]:
    T_arr = np.asarray(T, dtype=np.float64)
    if T_arr.shape != (4, 4):
        raise ValueError("Expected a 4x4 matrix")
    return [[float(x) for x in row] for row in T_arr.tolist()]


def compute_metrics_for_space(
    gt_elems: List[Comp],
    pr_elems: List[Comp],
    eps: float,
    inside_eps: float,
    max_k_for_ie: int,
    gt_space: Optional[Comp] = None,
    pr_space: Optional[Comp] = None
) -> Dict[str, Any]:
    metrics = compute_all_metrics(
        gt_elems,
        pr_elems,
        eps=eps,
        inside_eps=inside_eps,
        max_k_for_ie=max_k_for_ie
    )
    if gt_space is not None:
        for row in metrics["per_gt"]:
            row["space_guid"] = gt_space.guid
            row["space_name"] = gt_space.meta.Name
            row["space_ifc_type"] = gt_space.etype
    if pr_space is not None:
        for row in metrics["per_pred"]:
            row["space_guid"] = pr_space.guid
            row["space_name"] = pr_space.meta.Name
            row["space_ifc_type"] = pr_space.etype
    return metrics


def _run_round(
    round_label: str,
    spaces_gt: List[Comp],
    elems_gt: List[Comp],
    spaces_pr: List[Comp],
    elems_pr: List[Comp],
    matches: List[Tuple[int, int, float]],
    space_guid_to_elem_indices_gt: Dict[str, List[int]],
    space_guid_to_elem_indices_pr: Dict[str, List[int]],
    eps: float,
    inside_eps: float,
    max_k_for_ie: int,
    space_align_mode: str,
    apply_local_alignment: bool,
    target_guid_upper: str,
    visualize_per_space: bool = False
) -> Tuple[Dict[str, Any], List[Tuple[Comp, Dict[str, Any]]]]:
    log_step(f"Starting '{round_label}' round ({'local' if apply_local_alignment else 'global-only'} evaluation)")
    by_space: Dict[str, Any] = {}
    per_space_csv_payload: List[Tuple[Comp, Dict[str, Any]]] = []
    space_match_records: List[Dict[str, Any]] = []
    floor_area_records: List[Dict[str, Any]] = []

    matched_gt_space_indices: Set[int] = set()
    matched_pr_space_indices: Set[int] = set()
    matched_gt_elem_indices: Set[int] = set()
    matched_pr_elem_indices: Set[int] = set()
    transformed_pred_elem_indices: Set[int] = set()

    for match_idx, (i_gt, j_pr, _) in enumerate(matches):
        gt_space = spaces_gt[i_gt]
        pr_space = spaces_pr[j_pr]
        gt_guid = gt_space.guid or f"<gt-space-{i_gt}>"
        pr_guid = pr_space.guid or f"<pred-space-{j_pr}>"
        gt_space_guid_key = gt_space.guid
        pr_space_guid_key = pr_space.guid

        is_target_space = ((gt_space.guid or "").upper() == target_guid_upper)

        matched_gt_space_indices.add(i_gt)
        matched_pr_space_indices.add(j_pr)

        pre_iou = iou_between_two_meshes(gt_space.mesh, pr_space.mesh, eps=inside_eps)

        log_step(f"[{round_label}] Pair {match_idx + 1}/{len(matches)} GT {gt_guid} <-> PRED {pr_guid} (IoU={pre_iou:.3f})")
        if apply_local_alignment:
            _print_spaces_summary(f"{round_label.upper()} | GT space {match_idx+1}", [gt_space], max_print=1)
            _print_spaces_summary(f"{round_label.upper()} | PRED space {match_idx+1} (before)", [pr_space], max_print=1)

            # try:
            #     _visualize_spaces([gt_space], [pr_space], title=f"{round_label.title()} match {match_idx+1}: BEFORE")
            # except Exception as exc:
            #     log_step(f"Per-space visualization (before) failed (match {match_idx + 1}): {exc}")

        T_space = np.eye(4, dtype=float)
        alignment_mode = "none"
        transform_candidates: List[int] = []
        skipped: List[int] = []

        pred_elem_indices_all = sorted(set(space_guid_to_elem_indices_pr.get(pr_space_guid_key, [])))

        if apply_local_alignment and space_align_mode != "none":
            T_space = align_space_pair(gt_space, pr_space, space_align_mode)
            alignment_mode = space_align_mode
            apply_transform_to_indices(spaces_pr, [j_pr], T_space)
            transform_candidates = [idx for idx in pred_elem_indices_all if idx not in transformed_pred_elem_indices]
            skipped = [idx for idx in pred_elem_indices_all if idx in transformed_pred_elem_indices]
            if transform_candidates:
                apply_transform_to_indices(elems_pr, transform_candidates, T_space)
                transformed_pred_elem_indices.update(transform_candidates)

        pred_elem_indices = pred_elem_indices_all
        gt_elem_indices = sorted(set(space_guid_to_elem_indices_gt.get(gt_space_guid_key, [])))

        matched_gt_elem_indices.update(gt_elem_indices)
        matched_pr_elem_indices.update(pred_elem_indices)

        gt_elems = [elems_gt[idx] for idx in gt_elem_indices]
        pr_elems = [elems_pr[idx] for idx in pred_elem_indices]

        space_metrics = compute_metrics_for_space(
            gt_elems,
            pr_elems,
            eps=eps,
            inside_eps=inside_eps,
            max_k_for_ie=max_k_for_ie,
            gt_space=gt_space,
            pr_space=spaces_pr[j_pr]
        )

        post_iou = iou_between_two_meshes(gt_space.mesh, spaces_pr[j_pr].mesh, eps=inside_eps)
        gt_floor_area = _mesh_floor_area_xy(gt_space.mesh)
        pred_floor_area = _mesh_floor_area_xy(spaces_pr[j_pr].mesh)
        gt_dim_x, gt_dim_y, gt_dim_z = _space_xyz_dimensions(gt_space)
        pred_dim_x, pred_dim_y, pred_dim_z = _space_xyz_dimensions(spaces_pr[j_pr])

        floor_area_records.append({
            "gt_guid": gt_guid,
            "gt_name": gt_space.meta.Name,
            "pred_guid": pr_guid,
            "pred_name": spaces_pr[j_pr].meta.Name,
            "space_iou": float(post_iou),
            "gt_floor_area": gt_floor_area,
            "pred_floor_area": pred_floor_area,
            "floor_area_diff": pred_floor_area - gt_floor_area,
            "floor_area_abs_diff": abs(pred_floor_area - gt_floor_area),
            "gt_dim_x": gt_dim_x,
            "gt_dim_y": gt_dim_y,
            "gt_dim_z": gt_dim_z,
            "pred_dim_x": pred_dim_x,
            "pred_dim_y": pred_dim_y,
            "pred_dim_z": pred_dim_z,
            "is_target": is_target_space
        })

        alignment_info: Dict[str, Any] = {
            "round": round_label,
            "mode": alignment_mode,
            "T": _matrix_to_nested_list(T_space),
            "space_iou_before": float(pre_iou),
            "space_iou_after": float(post_iou),
            "is_target": is_target_space
        }
        if transform_candidates:
            alignment_info["transformed_pred_element_indices"] = transform_candidates
        if skipped:
            alignment_info["skipped_pred_element_indices"] = skipped

        by_space_key = f"{gt_guid}::<->::{pr_guid}"
        by_space[by_space_key] = {
            "round": round_label,
            "gt_space": {"guid": gt_guid, "name": gt_space.meta.Name, "meta": asdict(gt_space.meta)},
            "pred_space": {"guid": pr_guid, "name": spaces_pr[j_pr].meta.Name, "meta": asdict(spaces_pr[j_pr].meta)},
            "alignment": alignment_info,
            "metrics": space_metrics,
            "counts": {
                "gt_elems": len(gt_elem_indices),
                "pred_elems": len(pred_elem_indices)
            },
            "element_indices": {
                "gt": gt_elem_indices,
                "pred": pred_elem_indices
            }
        }
        per_space_csv_payload.append((gt_space, space_metrics))

        space_match_records.append({
            "round": round_label,
            "match_index": match_idx,
            "gt_index": i_gt,
            "pred_index": j_pr,
            "gt_guid": gt_guid,
            "pred_guid": pr_guid,
            "space_iou_before": float(pre_iou),
            "space_iou": float(post_iou),
            "is_target": is_target_space
        })

        if apply_local_alignment:
            _print_spaces_summary(f"{round_label.upper()} | PRED space {match_idx+1} (after)", [spaces_pr[j_pr]], max_print=1)
            # try:
            #     _visualize_spaces([gt_space], [spaces_pr[j_pr]], title=f"{round_label.title()} match {match_idx+1}: AFTER")
            # except Exception as exc:
            #     log_step(f"Per-space visualization failed (match {match_idx + 1}): {exc}")

    return (
        {
            "round": round_label,
            "by_space": by_space,
            "space_match_records": space_match_records,
            "floor_area_records": floor_area_records,
            "matched_gt_space_indices": matched_gt_space_indices,
            "matched_pr_space_indices": matched_pr_space_indices,
            "matched_gt_elem_indices": matched_gt_elem_indices,
            "matched_pr_elem_indices": matched_pr_elem_indices,
            "transformed_pred_elem_indices": transformed_pred_elem_indices
        },
        per_space_csv_payload
    )


def _compute_round_element_metrics(
    round_eval: Dict[str, Any],
    spaces_gt: List[Comp],
    spaces_pr: List[Comp],
    elems_gt: List[Comp],
    elems_pr: List[Comp],
    space_guid_to_elem_indices_gt: Dict[str, List[int]],
    space_guid_to_elem_indices_pr: Dict[str, List[int]],
    elem_guid_to_space_gt: Dict[str, List[str]],
    elem_guid_to_space_pr: Dict[str, List[str]],
    include_unmatched_mode: str,
    eps: float,
    inside_eps: float,
    max_k_for_ie: int,
    allow_unmatched_alignment: bool,
    align_mode: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    matched_gt_space_indices = set(round_eval.get("matched_gt_space_indices", set()))
    matched_pr_space_indices = set(round_eval.get("matched_pr_space_indices", set()))
    matched_gt_elem_indices = set(round_eval.get("matched_gt_elem_indices", set()))
    matched_pr_elem_indices = set(round_eval.get("matched_pr_elem_indices", set()))

    all_gt_space_indices = set(range(len(spaces_gt)))
    all_pr_space_indices = set(range(len(spaces_pr)))
    unmatched_gt_space_indices = sorted(all_gt_space_indices - matched_gt_space_indices)
    unmatched_pr_space_indices = sorted(all_pr_space_indices - matched_pr_space_indices)

    gt_elements_without_space = [
        idx for idx, comp in enumerate(elems_gt)
        if not elem_guid_to_space_gt.get(comp.guid)
    ]
    pred_elements_without_space = [
        idx for idx, comp in enumerate(elems_pr)
        if not elem_guid_to_space_pr.get(comp.guid)
    ]

    gt_unmatched_elem_indices: Set[int] = set()
    for idx in unmatched_gt_space_indices:
        space = spaces_gt[idx]
        gt_unmatched_elem_indices.update(space_guid_to_elem_indices_gt.get(space.guid, []))
    gt_unmatched_elem_indices.update(gt_elements_without_space)
    gt_unmatched_elem_indices -= matched_gt_elem_indices

    pred_unmatched_elem_indices: Set[int] = set()
    for idx in unmatched_pr_space_indices:
        space = spaces_pr[idx]
        pred_unmatched_elem_indices.update(space_guid_to_elem_indices_pr.get(space.guid, []))
    pred_unmatched_elem_indices.update(pred_elements_without_space)
    pred_unmatched_elem_indices -= matched_pr_elem_indices

    unmatched_bucket: Optional[Dict[str, Any]] = None
    if include_unmatched_mode == "global" and gt_unmatched_elem_indices and pred_unmatched_elem_indices:
        unmatched_gt_indices_sorted = sorted(gt_unmatched_elem_indices)
        unmatched_pr_indices_sorted = sorted(pred_unmatched_elem_indices)
        unmatched_gt_elems = [elems_gt[idx] for idx in unmatched_gt_indices_sorted]
        unmatched_pr_elems = [elems_pr[idx] for idx in unmatched_pr_indices_sorted]
        unmatched_alignment = np.eye(4, dtype=float)
        alignment_mode = "none"
        if allow_unmatched_alignment and align_mode != "none":
            alignment_mode = align_mode
            if align_mode == "icp":
                unmatched_alignment = rigid_icp_align([comp.mesh for comp in unmatched_pr_elems], [comp.mesh for comp in unmatched_gt_elems])
            elif align_mode == "icp_xy":
                unmatched_alignment = rigid_icp_align_xy_only([comp.mesh for comp in unmatched_pr_elems], [comp.mesh for comp in unmatched_gt_elems])
            else:
                unmatched_alignment = _centroid_align([comp.mesh for comp in unmatched_pr_elems], [comp.mesh for comp in unmatched_gt_elems])
            if unmatched_pr_indices_sorted:
                apply_transform_to_indices(elems_pr, unmatched_pr_indices_sorted, unmatched_alignment)
        unmatched_metrics = compute_all_metrics(
            unmatched_gt_elems,
            unmatched_pr_elems,
            eps=eps,
            inside_eps=inside_eps,
            max_k_for_ie=max_k_for_ie
        )
        unmatched_bucket = {
            "alignment": {
                "mode": alignment_mode,
                "T": _matrix_to_nested_list(unmatched_alignment)
            },
            "metrics": unmatched_metrics,
            "counts": {
                "gt_elems": len(unmatched_gt_indices_sorted),
                "pred_elems": len(unmatched_pr_indices_sorted)
            }
        }

    included_gt_indices = set(matched_gt_elem_indices)
    included_pr_indices = set(matched_pr_elem_indices)
    if include_unmatched_mode == "global":
        included_gt_indices.update(gt_unmatched_elem_indices)
        included_pr_indices.update(pred_unmatched_elem_indices)

    included_gt_indices_sorted = sorted(included_gt_indices)
    included_pr_indices_sorted = sorted(included_pr_indices)

    gt_for_report = [elems_gt[idx] for idx in included_gt_indices_sorted]
    pr_for_report = [elems_pr[idx] for idx in included_pr_indices_sorted]

    if not gt_for_report and elems_gt and include_unmatched_mode == "ignore":
        log_step(
            "WARNING: No GT elements were included in per-element metrics for this round. "
            "This commonly happens when GT elements are not linked to spaces and --include-unmatched is 'ignore'. "
            "Try --include-unmatched global."
        )
    if not pr_for_report and elems_pr and include_unmatched_mode == "ignore":
        log_step(
            "WARNING: No PRED elements were included in per-element metrics for this round. "
            "This commonly happens when PRED elements are not linked to spaces and --include-unmatched is 'ignore'. "
            "Try --include-unmatched global."
        )

    global_metrics = compute_all_metrics(
        gt_for_report,
        pr_for_report,
        eps=eps,
        inside_eps=inside_eps,
        max_k_for_ie=max_k_for_ie
    )

    gt_space_guid_to_name = {space.guid: space.meta.Name for space in spaces_gt}
    pred_space_guid_to_name = {space.guid: space.meta.Name for space in spaces_pr}

    for row in global_metrics["per_gt"]:
        idx = row.get("gt_global_index")
        if idx is None or not (0 <= idx < len(elems_gt)):
            continue
        comp = elems_gt[idx]
        guids = elem_guid_to_space_gt.get(comp.guid, [])
        row["space_guids"] = guids
        row["spaces"] = [{"guid": guid, "name": gt_space_guid_to_name.get(guid)} for guid in guids]

    for row in global_metrics["per_pred"]:
        idx = row.get("pred_global_index")
        if idx is None or not (0 <= idx < len(elems_pr)):
            continue
        comp = elems_pr[idx]
        guids = elem_guid_to_space_pr.get(comp.guid, [])
        row["space_guids"] = guids
        row["spaces"] = [{"guid": guid, "name": pred_space_guid_to_name.get(guid)} for guid in guids]

    unmatched_summary = {
        "mode": include_unmatched_mode,
        "gt_spaces_unmatched": len(unmatched_gt_space_indices),
        "pred_spaces_unmatched": len(unmatched_pr_space_indices),
        "gt_elements_unmatched": len(gt_unmatched_elem_indices),
        "pred_elements_unmatched": len(pred_unmatched_elem_indices),
        "gt_spaces_ignored": len(unmatched_gt_space_indices) if include_unmatched_mode == "ignore" else 0,
        "pred_spaces_ignored": len(unmatched_pr_space_indices) if include_unmatched_mode == "ignore" else 0,
        "gt_elements_ignored": len(gt_unmatched_elem_indices) if include_unmatched_mode == "ignore" else 0,
        "pred_elements_ignored": len(pred_unmatched_elem_indices) if include_unmatched_mode == "ignore" else 0,
        "gt_elements_in_unmatched_bucket": len(gt_unmatched_elem_indices) if include_unmatched_mode == "global" else 0,
        "pred_elements_in_unmatched_bucket": len(pred_unmatched_elem_indices) if include_unmatched_mode == "global" else 0,
        "gt_elements_without_space": len(gt_elements_without_space),
        "pred_elements_without_space": len(pred_elements_without_space)
    }

    return global_metrics, unmatched_summary, unmatched_bucket


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
    src = _o3d_pcd_from_meshes(pred_meshes)
    tgt = _o3d_pcd_from_meshes(gt_meshes)
    try:
        src_pts = np.asarray(src.points)
        tgt_pts = np.asarray(tgt.points)
    except Exception:
        src_pts = np.zeros((0, 3))
        tgt_pts = np.zeros((0, 3))
    if src_pts.size == 0 or tgt_pts.size == 0:
        T_cent = _centroid_align(pred_meshes, gt_meshes)
        T_cent[2, 3] = 0.0
        print("Alignment (fallback centroid, Z zeroed):")
        print(T_cent)
        return T_cent

    src_c = src_pts.mean(axis=0)
    tgt_c = tgt_pts.mean(axis=0)
    dx = float(tgt_c[0] - src_c[0])
    dy = float(tgt_c[1] - src_c[1])
    T = np.eye(4, dtype=float)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = 0.0

    print("Alignment (XY-translation only):")
    print(T)
    return T


def _o3d_transform_inplace(comps: List[Comp], T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix T must be 4x4.")
    for c in comps:
        c.mesh.transform(T)
        c.aabb_min, c.aabb_max = _o3d_bounds(c.mesh)
        c.obb = _transform_obb(c.obb, T)
        c.volume = _obb_volume(c.obb)
    _clear_mesh_iou_caches()


# ---------- IoU + Compactness (mesh-boolean) ----------

def _aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    return np.all(a_min <= b_max) and np.all(b_min <= a_max) and np.all(a_max >= b_min) and np.all(b_max >= a_min)

_WALL_FAMILY_TYPES: Set[str] = {"IfcWall", "IfcWallStandardCase"}


def _ifc_types_compatible_for_matching(gt_ifc_type: str, pred_ifc_type: str) -> bool:
    """
    Element-to-element comparisons for IoU/compactness must be class-consistent.

    Rules:
    - Default: only compare same IFC entity type (e.g., IfcDoor <-> IfcDoor).
    - Exception: IfcWall and IfcWallStandardCase are treated as compatible both ways.
    """
    if not gt_ifc_type or not pred_ifc_type:
        return False
    if gt_ifc_type == pred_ifc_type:
        return True
    return (gt_ifc_type in _WALL_FAMILY_TYPES) and (pred_ifc_type in _WALL_FAMILY_TYPES)


def _pairwise_iou(gt: List[Comp], pr: List[Comp], eps: float, inside_eps: float = 1e-7) -> np.ndarray:
    m, n = len(gt), len(pr)
    log_step(f"Computing pairwise IoU matrix ({m}x{n}) using mesh boolean volumes")
    M = np.zeros((m, n), dtype=float)
    progress_stride = max(1, min(50, (m // 10) or 5))
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            if not _ifc_types_compatible_for_matching(g.etype, p.etype):
                continue
            if not _aabb_overlap(g.aabb_min, g.aabb_max, p.aabb_min, p.aabb_max):
                continue
            v = iou_between_two_meshes(g.mesh, p.mesh, eps=inside_eps)
            M[i, j] = v if v >= eps else 0.0
        if VERBOSE and (m > 0) and (((i + 1) % progress_stride == 0) or (i == m - 1)):
            log_step(f"  IoU progress: processed {i + 1}/{m} GT elements")
    log_step("Pairwise IoU matrix ready")
    return M


# ---------- Metrics aggregation ----------

def compute_all_metrics(
    gt: List[Comp],
    pr: List[Comp],
    eps: float,
    inside_eps: float = 1e-7,
    max_k_for_ie: int = 8
) -> Dict[str, Any]:
    log_step("Aggregating IoU and compactness metrics (mesh-boolean)")
    M = _pairwise_iou(gt, pr, eps=eps, inside_eps=inside_eps)
    m, n = M.shape

    # Per-GT
    per_gt = []
    gt_local_compact = []
    gt_ious = []
    log_step("  Computing per-GT aggregates")
    for i, g in enumerate(gt):
        gt_dim_x, gt_dim_y, gt_dim_z = _xyz_dimensions(g)
        js = [j for j in range(n) if M[i, j] > 0.0]
        local_compact = (1.0 / len(js)) if js else 0.0
        gt_local_compact.append(local_compact)

        pairwise = [{"pred_index": j, "pred_guid": pr[j].guid, "iou": float(M[i, j])} for j in js]
        for entry, j in zip(pairwise, js):
            entry["pred_global_index"] = pr[j].idx
        if js:
            js_for_union = sorted(js, key=lambda j: float(M[i, j]), reverse=True)
            iou_union = iou_union_of_set_vs_single(
                [pr[j].mesh for j in js_for_union],
                g.mesh,
                eps=inside_eps,
                max_k_for_ie=max_k_for_ie
            )
        else:
            iou_union = 0.0
        gt_ious.append(iou_union)
        matched_pred_thickness = [
            float(pr[j].wall_thickness)
            for j in js
            if pr[j].wall_thickness is not None and math.isfinite(pr[j].wall_thickness)
        ]
        avg_match_pred_wall = float(np.mean(matched_pred_thickness)) if matched_pred_thickness else None

        per_gt.append({
            "gt_index": i,
            "gt_guid": g.guid,
            "gt_ifc_type": g.etype,
            "gt_dim_x": gt_dim_x,
            "gt_dim_y": gt_dim_y,
            "gt_dim_z": gt_dim_z,
            "gt_meta": asdict(g.meta),
            "gt_wall_thickness": float(g.wall_thickness) if g.wall_thickness is not None else None,
            "avg_matched_pred_wall_thickness": avg_match_pred_wall,
            "matches_pred_indices": js,
            "matches_pred_guids": [pr[j].guid for j in js],
            "pairwise_pred": pairwise,
            "iou_union_pred_vs_gt": float(iou_union),
            "local_compactness_gt_to_pred": float(local_compact),
            "gt_global_index": g.idx
        })

    # Per-PRED
    per_pred = []
    pr_local_compact = []
    pr_ious = []
    log_step("  Computing per-PRED aggregates")
    for j, p in enumerate(pr):
        pred_dim_x, pred_dim_y, pred_dim_z = _xyz_dimensions(p)
        is_ = [i for i in range(m) if M[i, j] > 0.0]
        local_compact = (1.0 / len(is_)) if is_ else 0.0
        pr_local_compact.append(local_compact)

        pairwise = [{"gt_index": i, "gt_guid": gt[i].guid, "iou": float(M[i, j])} for i in is_]
        for entry, i in zip(pairwise, is_):
            entry["gt_global_index"] = gt[i].idx
        if is_:
            is_for_union = sorted(is_, key=lambda i: float(M[i, j]), reverse=True)
            iou_union = iou_union_of_set_vs_single(
                [gt[i].mesh for i in is_for_union],
                p.mesh,
                eps=inside_eps,
                max_k_for_ie=max_k_for_ie
            )
        else:
            iou_union = 0.0
        pr_ious.append(iou_union)

        per_pred.append({
            "pred_index": j,
            "pred_guid": p.guid,
            "pred_ifc_type": p.etype,
            "pred_dim_x": pred_dim_x,
            "pred_dim_y": pred_dim_y,
            "pred_dim_z": pred_dim_z,
            "pred_meta": asdict(p.meta),
            "pred_wall_thickness": float(p.wall_thickness) if p.wall_thickness is not None else None,
            "matches_gt_indices": is_,
            "matches_gt_guids": [gt[i].guid for i in is_],
            "pairwise_gt": pairwise,
            "iou_union_gt_vs_pred": float(iou_union),
            "local_compactness_pred_to_gt": float(local_compact),
            "pred_global_index": p.idx
        })

    # Dataset-level
    log_step("  Deriving dataset-level metrics")
    m3D_IoU_gt = float(np.mean(gt_ious) if gt_ious else 0.0)
    m3D_IoU_pred = float(np.mean(pr_ious) if pr_ious else 0.0)
    comp_gt_to_pred = float(np.mean(gt_local_compact) if gt_local_compact else 0.0)
    comp_pred_to_gt = float(np.mean(pr_local_compact) if pr_local_compact else 0.0)
    m3D_Compactness = 0.5 * (comp_gt_to_pred + comp_pred_to_gt)

    edges = []
    for i in range(m):
        for j in range(n):
            if M[i, j] > 0.0:
                edges.append({
                    "gt_index": i, "gt_guid": gt[i].guid,
                    "pred_index": j, "pred_guid": pr[j].guid,
                    "iou": float(M[i, j]),
                    "gt_global_index": gt[i].idx,
                    "pred_global_index": pr[j].idx
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


# ---------- CSV helpers ----------

def _write_csv_bundle(prefix: str, per_gt: List[Dict[str, Any]], per_pred: List[Dict[str, Any]], correspondences: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    with open(f"{prefix}_per_gt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "gt_index","gt_guid","gt_ifc_type","iou_union_pred_vs_gt",
            "local_compactness_gt_to_pred","gt_wall_thickness","avg_matched_pred_wall_thickness","num_matches","match_pred_guids",
            "gt_dim_x","gt_dim_y","gt_dim_z"
        ])
        for r in per_gt:
            thickness = r.get("gt_wall_thickness")
            matched_avg = r.get("avg_matched_pred_wall_thickness")
            dim_x = r.get("gt_dim_x")
            dim_y = r.get("gt_dim_y")
            dim_z = r.get("gt_dim_z")
            w.writerow([
                r.get("gt_index"),
                r.get("gt_guid"),
                r.get("gt_ifc_type"),
                f'{float(r.get("iou_union_pred_vs_gt", 0.0)):.6f}',
                f'{float(r.get("local_compactness_gt_to_pred", 0.0)):.6f}',
                "" if thickness in (None, "") else f'{float(thickness):.6f}',
                "" if matched_avg in (None, "") else f'{float(matched_avg):.6f}',
                len(r.get("matches_pred_indices", [])),
                ";".join(r.get("matches_pred_guids", [])),
                "" if dim_x in (None, "") else f'{float(dim_x):.6f}',
                "" if dim_y in (None, "") else f'{float(dim_y):.6f}',
                "" if dim_z in (None, "") else f'{float(dim_z):.6f}'
            ])

    with open(f"{prefix}_per_pred.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pred_index","pred_guid","pred_ifc_type","iou_union_gt_vs_pred",
            "local_compactness_pred_to_gt","pred_wall_thickness","num_matches","match_gt_guids",
            "pred_dim_x","pred_dim_y","pred_dim_z"
        ])
        for r in per_pred:
            thickness = r.get("pred_wall_thickness")
            dim_x = r.get("pred_dim_x")
            dim_y = r.get("pred_dim_y")
            dim_z = r.get("pred_dim_z")
            w.writerow([
                r.get("pred_index"),
                r.get("pred_guid"),
                r.get("pred_ifc_type"),
                f'{float(r.get("iou_union_gt_vs_pred", 0.0)):.6f}',
                f'{float(r.get("local_compactness_pred_to_gt", 0.0)):.6f}',
                "" if thickness in (None, "") else f'{float(thickness):.6f}',
                len(r.get("matches_gt_indices", [])),
                ";".join(r.get("matches_gt_guids", [])),
                "" if dim_x in (None, "") else f'{float(dim_x):.6f}',
                "" if dim_y in (None, "") else f'{float(dim_y):.6f}',
                "" if dim_z in (None, "") else f'{float(dim_z):.6f}'
            ])

    with open(f"{prefix}_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_index","gt_guid","pred_index","pred_guid","pairwise_iou"])
        for e in correspondences:
            w.writerow([
                e.get("gt_index"),
                e.get("gt_guid"),
                e.get("pred_index"),
                e.get("pred_guid"),
                f'{float(e.get("iou", 0.0)):.6f}'
            ])


def save_csvs(prefix: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    _write_csv_bundle(
        prefix,
        result.get("per_gt", []),
        result.get("per_pred", []),
        result.get("correspondences", [])
    )


def _space_label_for_filename(gt_space: Comp) -> str:
    parts: List[str] = []
    if gt_space.meta.Name and gt_space.meta.Name.strip():
        parts.append(gt_space.meta.Name.strip())
    if gt_space.guid:
        parts.append(gt_space.guid)
    if not parts:
        parts.append(f"space_{gt_space.idx}")
    label = "_".join(parts)
    label = re.sub(r"[^0-9A-Za-z_-]+", "_", label)
    label = label.strip("_")
    if not label:
        label = f"space_{gt_space.idx}"
    # constrain length to keep filenames manageable
    return (label[:80] if len(label) > 80 else label)


def save_space_csvs(prefix: str, gt_space: Comp, metrics: Dict[str, Any]) -> None:
    label = _space_label_for_filename(gt_space)
    space_dir = os.path.join(f"{prefix}_spaces", f"{label}_{gt_space.idx}")
    os.makedirs(os.path.abspath(space_dir), exist_ok=True)
    bundle_prefix = os.path.join(space_dir, "metrics")
    _write_csv_bundle(
        bundle_prefix,
        metrics.get("per_gt", []),
        metrics.get("per_pred", []),
        metrics.get("correspondences", [])
    )


def _path_with_round_suffix(path: str, round_label: str, default_ext: str = ".csv") -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        ext = default_ext
    return f"{base}_{round_label}{ext}"


def write_floor_area_csv(path: str, round_label: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "round",
            "gt_guid", "gt_name", "pred_guid", "pred_name",
            "space_iou",
            "gt_floor_area", "pred_floor_area",
            "floor_area_diff", "floor_area_abs_diff",
            "gt_dim_x", "gt_dim_y", "gt_dim_z",
            "pred_dim_x", "pred_dim_y", "pred_dim_z",
            "is_target"
        ])
        for rec in records:
            w.writerow([
                round_label,
                rec.get("gt_guid"),
                rec.get("gt_name"),
                rec.get("pred_guid"),
                rec.get("pred_name"),
                f'{float(rec.get("space_iou", 0.0)):.6f}',
                f'{float(rec.get("gt_floor_area", 0.0)):.6f}',
                f'{float(rec.get("pred_floor_area", 0.0)):.6f}',
                f'{float(rec.get("floor_area_diff", 0.0)):.6f}',
                f'{float(rec.get("floor_area_abs_diff", 0.0)):.6f}',
                f'{float(rec.get("gt_dim_x", 0.0)):.6f}',
                f'{float(rec.get("gt_dim_y", 0.0)):.6f}',
                f'{float(rec.get("gt_dim_z", 0.0)):.6f}',
                f'{float(rec.get("pred_dim_x", 0.0)):.6f}',
                f'{float(rec.get("pred_dim_y", 0.0)):.6f}',
                f'{float(rec.get("pred_dim_z", 0.0)):.6f}',
                int(bool(rec.get("is_target")))
            ])
    if records:
        log_step(f"Wrote {len(records)} floor area comparison rows to {path}")
    else:
        log_step(f"Wrote floor area CSV header (no rows) to {path}")


# ---------- CLI ----------

def _silence_vtk_output(log_to: str | None = None) -> None:
    """Prevent VTK from opening its GUI error window (Windows)."""
    try:
        import os
        try:
            from vtkmodules.vtkCommonCore import vtkLogger
            vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)
        except Exception:
            pass
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
        pass


def _print_spaces_summary(prefix: str, spaces: List[Comp], max_print: int = 10) -> None:
    """Print a concise summary of spaces: index, guid, name, OBB center and extent."""
    try:
        print(f"\n=== {prefix} spaces (count={len(spaces)}) ===")
        for i, s in enumerate(spaces[:max_print]):
            c = s.obb.center if getattr(s, 'obb', None) is not None else None
            e = s.obb.extent if getattr(s, 'obb', None) is not None else None
            name = s.meta.Name if getattr(s, 'meta', None) is not None else ''
            print(f"[{i}] guid={s.guid} name='{name}' center={np.array2string(np.asarray(c), precision=3) if c is not None else 'N/A'} extent={np.array2string(np.asarray(e), precision=3) if e is not None else 'N/A'}")
        if len(spaces) > max_print:
            print(f"  ... and {len(spaces)-max_print} more spaces")
    except Exception as exc:
        print(f"Failed to print spaces summary: {exc}")


def _obb_to_lineset(obb: OBB, color=(1.0, 0.0, 0.0)) -> o3d.geometry.LineSet:
    """Convert an OBB to an Open3D LineSet for visualization.

    Color is RGB tuple in 0..1 (default red).
    """
    # Recompute corners directly from OBB properties for robustness.
    # This ensures the visualization perfectly matches the OBB definition.
    center = np.asarray(obb.center)
    R = np.asarray(obb.R)
    half_extent = np.asarray(obb.extent) * 0.5

    # 8 corners relative to the center
    relative_corners = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ]) * half_extent

    # Transform corners to world coordinates
    corners = center + (R @ relative_corners.T).T

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    )
    colors = np.tile(np.array(color, dtype=float).reshape(1, 3), (len(edges), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _visualize_spaces(spaces_gt: List[Comp], spaces_pr: List[Comp], title: str = "GT vs PRED") -> None:
    """Open a simple Open3D visualization overlaying GT (green) and PRED (red) space OBBs.

    This creates line sets for each OBB and shows them together. It's lightweight and
    intended for quick inspection only.
    """
    def _obb_to_mesh(obb: OBB, color=(1.0, 0.0, 0.0)) -> o3d.geometry.TriangleMesh:
        """Build a TriangleMesh representing the OBB as a closed box.

        The OBB.corners are used as vertices and a fixed face index layout is applied.
        """
        try:
            corners = np.asarray(obb.corners, dtype=float)
        except Exception:
            corners = np.zeros((8, 3), dtype=float)

        # Triangles for a box (using corner ordering consistent with get_box_points())
        tris = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 4, 5], [0, 5, 1],  # side
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0]
        ], dtype=np.int32)

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(corners),
            triangles=o3d.utility.Vector3iVector(tris)
        )
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass
        try:
            mesh.paint_uniform_color(list(color))
        except Exception:
            pass
        return mesh

    vis_objs = []
    for s in spaces_gt:
        try:
            # m = _obb_to_mesh(s.obb, color=(0.0, 0.8, 0.0))
            # Green for GT
            ls = _obb_to_lineset(s.obb, color=(0.0, 0.8, 0.0))
            vis_objs.append(ls)
        except Exception:
            pass
    for s in spaces_pr:
        try:
            # m = _obb_to_mesh(s.obb, color=(0.8, 0.0, 0.0))
            # Red for PRED
            ls = _obb_to_lineset(s.obb, color=(0.8, 0.0, 0.0))
            vis_objs.append(ls)
        except Exception:
            pass
    if not vis_objs:
        raise RuntimeError("No space OBBs available for visualization")

    try:
        o3d.visualization.draw(vis_objs, window_name=title)
    except Exception:
        # Fallback to draw_geometries if draw() not available (older Open3D)
        o3d.visualization.draw_geometries(vis_objs, window_name=title)


def main():
    _silence_vtk_output()

    ap = argparse.ArgumentParser(description="Per-element 3D-IoU & 3D-Compactness between two IFCs, with metadata. (mesh booleans)")
    ap.add_argument("--gt", required=True, help="Path to ground-truth IFC.")
    ap.add_argument("--pred", required=True, help="Path to predicted/reconstructed IFC.")
    ap.add_argument("--target-space-guid", "--align-space-guid", required=False, default=None,
                    help="(Optional) GlobalId of a GT IfcSpace used to derive a global (model-wide) ICP refinement. "
                         "When provided, per-space/local alignment (--space-align) is disabled.")
    ap.add_argument("--mesh-output-dir", type=str, default="out",
                    help="Directory to write combined GT and aligned PRED meshes.")
    ap.add_argument("--floor-area-csv", type=str, default="out/matched_space_floor_areas.csv",
                    help="CSV file path to record floor area comparisons for matched spaces.")
    ap.add_argument("--align", choices=["icp", "centroid", "none", "icp_xy"], default="centroid",
                    help="Global alignment for PRED->GT before space matching.")
    ap.add_argument("--space-align", choices=["icp", "icp_xy", "centroid", "none"], default="icp",
                    help="Per-space alignment mode for matched PRED spaces.")
    ap.add_argument("--visualize", action="store_true", help="Show an Open3D visualization of GT (green) and PRED (red) spaces before/after global alignment.")
    ap.add_argument("--visualize-per-space", action="store_true", help="Open a small Open3D visualization for each matched GT/PRED space pair before/after per-space alignment.")
    ap.add_argument("--space-match-thresh", type=float, default=0.5,
                    help="IoU threshold to accept a GT/PRED space match.")
    ap.add_argument("--epsilon", type=float, default=0.05, help="IoU threshold to consider a correspondence.")
    ap.add_argument("--save-json", type=str, default=None, help="Path to save a JSON report.")
    ap.add_argument("--save-csv-prefix", type=str, default=None, help="Prefix to save CSVs: <prefix>_per_gt.csv, _per_pred.csv, _edges.csv")
    ap.add_argument("--ifc-classes", nargs="+", default=None,
                    help="Optional list of IFC classes to include (e.g. IfcWall IfcDoor). You may also pass a single comma-separated string 'IfcWall,IfcDoor'.")
    ap.add_argument("--include-unmatched", choices=["ignore", "global"], default="ignore",
                    help="Handling for elements in unmatched spaces. 'ignore' skips them; 'global' aggregates them in a single unmatched bucket.")
    ap.add_argument("--inside-eps", type=float, default=1e-7,
                    help="Numerical epsilon for mesh IoU volumes (treat volumes <= eps as empty).")
    ap.add_argument("--ie-cap", type=int, default=8,
                    help="Max number of meshes to union for many-to-one IoU aggregation (runtime cap).")
    ap.add_argument("--iou-backend", choices=["auto", "trimesh", "raycast"], default="auto",
                    help="IoU backend: 'trimesh' uses robust mesh booleans (recommended), "
                         "'raycast' uses occupancy integration (no booleans), "
                         "'auto' tries trimesh then falls back to raycast.")
    ap.add_argument("--raycast-pitch", type=float, default=0.05,
                    help="Base voxel pitch (meters) for the raycast IoU backend (may be increased to respect --raycast-max-voxels).")
    ap.add_argument("--raycast-max-voxels", type=int, default=2_000_000,
                    help="Maximum occupancy samples per IoU evaluation for the raycast backend.")
    ap.add_argument("--raycast-chunk", type=int, default=250_000,
                    help="Chunk size for occupancy queries in the raycast backend.")

    # PAPER MODELS

    # # Default CLI arguments for convenience (used only when no CLI args are provided)
    # DEFAULT_ARGS = [
    #     "--gt", "./input/JM_2nd_floor_w_spaces_updated.ifc",
    #     "--pred", "./input/john-muir-2nd-v2-ifc4-geo-clean-autorotated.ifc",
    #     "--target-space-guid", "1663O7_YHCi8qg8Qi5si4T", #gt-space
    #     "--mesh-output-dir", "out",
    #     "--floor-area-csv", "out/matched_space_floor_areas.csv",
    #     "--align", "centroid",
    #     "--space-align", "none",
    #     "--space-match-thresh", "0.25",
    #     "--epsilon", "0.10",
    #     "--save-json", "metrics_obb.json",
    #     "--save-csv-prefix", "out/metrics_obb",
    #     "--ifc-classes", "IfcWall", "IfcWallStandardCase", "IfcSpace", "IfcDoor", "IfcWindow",
    #     "--include-unmatched", "ignore",
    # ]

    # # Default CLI arguments for convenience (used only when no CLI args are provided)
    # DEFAULT_ARGS = [
    #     "--gt", "./input/JM_1st_floor_w_spaces.ifc",
    #     "--pred", "./input/john-muir-1st-v2-ifc4-geo-autorotated.ifc",
    #     "--target-space-guid", "1663O7_YHCi8qg8Qi5si4v", #gt-space
    #     "--mesh-output-dir", "out",
    #     "--floor-area-csv", "out/matched_space_floor_areas.csv",
    #     "--align", "centroid",
    #     "--space-align", "none",
    #     "--space-match-thresh", "0.25",
    #     "--epsilon", "0.10",
    #     "--save-json", "metrics_obb.json",
    #     "--save-csv-prefix", "out/metrics_obb",
    #     "--ifc-classes", "IfcWall", "IfcWallStandardCase", "IfcSpace", "IfcDoor", "IfcWindow",
    #     "--include-unmatched", "ignore",
    # ]


    # Default CLI arguments for convenience (used only when no CLI args are provided)
    DEFAULT_ARGS = [
        "--gt", ".\\input\\WW_revit_model_v6_w_spaces.ifc",
        "--pred", ".\\input\\ww-v1-ifc4-geo.ifc",
        "--target-space-guid", "1UABPD7uD69BRTD38UZ2$k",
        "--mesh-output-dir", "out",
        "--floor-area-csv", "out/matched_space_floor_areas.csv",
        "--align", "centroid",
        "--space-align", "none",
        "--space-match-thresh", "0.25",
        "--epsilon", "0.10",
        "--save-json", "metrics_mesh.json",
        "--save-csv-prefix", "out/metrics_mesh",
        "--ifc-classes", "IfcWall", "IfcWallStandardCase", "IfcSpace", "IfcDoor", "IfcWindow",
        "--include-unmatched", "ignore",
    ]

    

    # # Default CLI arguments for convenience (used only when no CLI args are provided)
    # DEFAULT_ARGS = [
    #     "--gt", "./input/JM_1st_floor_w_spaces.ifc",
    #     "--pred", "./input/john-muir-1st-v2-ifc4-geo-rotated.ifc",
    #     # "--target-space-guid", "1663O7_YHCi8qg8Qi5si4r", #gt-space
    #     # "--target-space-guid", "0UhoA17yvDzgPS6x1bHitJ", #pred-space
    #     "--mesh-output-dir", "out",
    #     "--floor-area-csv", "out/matched_space_floor_areas.csv",
    #     "--align", "centroid",
    #     "--space-align", "none",
    #     "--space-match-thresh", "0.25",
    #     "--epsilon", "0.10",
    #     "--save-json", "metrics_obb.json",
    #     "--save-csv-prefix", "out/metrics_obb",
    #     "--ifc-classes", "IfcWall", "IfcWallStandardCase", "IfcSpace", "IfcDoor", "IfcWindow",
    #     "--include-unmatched", "ignore",
    # ]

    # # Default CLI arguments for convenience (used only when no CLI args are provided)
    # DEFAULT_ARGS = [
    #     "--gt", ".\\input\\JohnMuir_revit_rotated_spaces_1st.ifc",
    #     "--pred", ".\\input\\john-muir-1st-v1-ifc4-geo-rotated.ifc",
    #     "--target-space-guid", "1663O7_YHCi8qg8Qi5si4a",
    #     # "--target-space-guid", "1663O7_YHCi8qg8Qi5si4R",
    #     "--mesh-output-dir", "out",
    #     "--floor-area-csv", "out/matched_space_floor_areas.csv",
    #     "--align", "icp",
    #     "--space-align", "none",
    #     "--space-match-thresh", "0.25",
    #     "--epsilon", "0.10",
    #     "--save-json", "metrics_obb.json",
    #     "--save-csv-prefix", "out/metrics_obb",
    #     "--ifc-classes", "IfcWall", "IfcWallStandardCase", "IfcSpace", "IfcDoor", "IfcWindow",
    #     "--include-unmatched", "ignore",
    # ]

    # # Default CLI arguments for convenience (used only when no CLI args are provided)
    # DEFAULT_ARGS = [
    #     "--gt", ".\\input\\WW_revit_model_v6_w_spaces.ifc",
    #     "--pred", ".\\input\\ww-v1-ifc4-geo.ifc",
    #     # "--target-space-guid", "1UABPD7uD69BRTD38UZ2$I",
    #     "--mesh-output-dir", "out",
    #     "--floor-area-csv", "out/matched_space_floor_areas.csv",
    #     "--align", "centroid",
    #     "--space-align", "none",
    #     "--space-match-thresh", "0.25",
    #     "--epsilon", "0.10",
    #     "--save-json", "metrics_obb.json",
    #     "--save-csv-prefix", "out/metrics_obb",
    #     "--ifc-classes", "IfcWall", "IfcWallStandardCase", "IfcSpace", "IfcDoor", "IfcWindow",
    #     "--include-unmatched", "ignore",
    # ]

    if len(sys.argv) == 1:
        log_step(f"No CLI arguments detected - using DEFAULT_ARGS: {' '.join(DEFAULT_ARGS)}")
        args = ap.parse_args(DEFAULT_ARGS)
    else:
        args = ap.parse_args()

    if args.ifc_classes is not None and len(args.ifc_classes) == 1 and ',' in args.ifc_classes[0]:
        args.ifc_classes = [s.strip() for s in args.ifc_classes[0].split(',') if s.strip()]

    # Configure IoU backend globals (used by iou_between_two_meshes / iou_union_of_set_vs_single).
    global IOU_BACKEND, IOU_RAYCAST_PITCH, IOU_RAYCAST_MAX_VOXELS, IOU_RAYCAST_CHUNK
    IOU_BACKEND = str(getattr(args, "iou_backend", "auto")).strip().lower() or "auto"
    IOU_RAYCAST_PITCH = float(getattr(args, "raycast_pitch", IOU_RAYCAST_PITCH))
    IOU_RAYCAST_MAX_VOXELS = int(getattr(args, "raycast_max_voxels", IOU_RAYCAST_MAX_VOXELS))
    IOU_RAYCAST_CHUNK = int(getattr(args, "raycast_chunk", IOU_RAYCAST_CHUNK))
    if IOU_BACKEND == "trimesh" and not TRIMESH_OK:
        log_step("IoU backend 'trimesh' requested but trimesh is not installed; switching to 'raycast'.")
        IOU_BACKEND = "raycast"
    elif IOU_BACKEND in {"auto", "trimesh"} and not TRIMESH_OK:
        log_step("Trimesh not installed; IoU backend will use raycast fallback.")
    _clear_mesh_iou_caches()

    args.target_space_guid = (args.target_space_guid or "").strip() or None
    target_guid_provided = args.target_space_guid is not None
    align_lower = args.align.lower()

    if target_guid_provided:
        # Enforce the workflow: centroid global alignment followed by targeted ICP refinement.
        if align_lower != "centroid":
            log_step("Overriding --align to 'centroid' when a target space GUID is provided.")
            args.align = "centroid"
        if args.space_align.lower() != "none":
            log_step("Overriding --space-align to 'none' when a target space GUID is provided (global-only alignment).")
            args.space_align = "none"
    else:
        if align_lower not in {"icp", "centroid"}:
            raise ValueError("When --target-space-guid is omitted, --align must be either 'icp' or 'centroid'.")
        log_step(f"No --target-space-guid provided; performing global alignment via '{args.align}'.")

    output_paths = _prepare_output_paths(args)
    session_output_dir = output_paths["session_root"]
    mesh_output_dir_abs = output_paths["mesh_dir"]
    floor_area_csv_base = output_paths["floor_area_base"]
    save_json_path = output_paths["save_json"]
    save_csv_prefix = output_paths["save_csv_prefix"]
    log_step(f"Session output directory: {session_output_dir}")

    log_step(f"Loading GT IFC: {args.gt}")
    spaces_gt, elems_gt, space_guid_to_elem_indices_gt, elem_guid_to_space_gt = build_space_membership(
        args.gt,
        include_types_for_elements=args.ifc_classes
    )
    log_step(f"  GT spaces loaded: {len(spaces_gt)} | elements: {len(elems_gt)}")
    gt_elements_without_space = sum(1 for comp in elems_gt if not elem_guid_to_space_gt.get(comp.guid))
    if elems_gt and gt_elements_without_space == len(elems_gt) and args.include_unmatched == "ignore":
        log_step(
            "WARNING: 100% of GT elements have no associated IfcSpace; with --include-unmatched ignore, "
            "`metrics_*_per_gt.csv` will be empty. Use --include-unmatched global or provide an IFC "
            "with element-to-space relationships."
        )

    # Display GT spaces before any alignment
    _print_spaces_summary("GT (before alignment)", spaces_gt)
    if args.visualize:
        try:
            _visualize_spaces(spaces_gt, [], title="GT (before)")
        except Exception as exc:
            log_step(f"Visualization (GT before) failed: {exc}")

    log_step(f"Loading PRED IFC: {args.pred}")
    spaces_pr, elems_pr, space_guid_to_elem_indices_pr, elem_guid_to_space_pr = build_space_membership(
        args.pred,
        include_types_for_elements=args.ifc_classes
    )
    log_step(f"  PRED spaces loaded: {len(spaces_pr)} | elements: {len(elems_pr)}")
    pred_elements_without_space = sum(1 for comp in elems_pr if not elem_guid_to_space_pr.get(comp.guid))
    if elems_pr and pred_elements_without_space == len(elems_pr) and args.include_unmatched == "ignore":
        log_step(
            "WARNING: 100% of PRED elements have no associated IfcSpace; with --include-unmatched ignore, "
            "per-element metrics will be empty. Use --include-unmatched global or provide an IFC with "
            "element-to-space relationships."
        )

    # Display PRED spaces before any alignment
    _print_spaces_summary("PRED (before alignment)", spaces_pr)
    if args.visualize:
        try:
            _visualize_spaces([], spaces_pr, title="PRED (before)")
        except Exception as exc:
            log_step(f"Visualization (PRED before) failed: {exc}")

    eps = float(args.epsilon)
    inside_eps = float(args.inside_eps)
    max_k_for_ie = int(args.ie_cap)
    space_match_thresh = float(args.space_match_thresh)

    global_alignment = np.eye(4, dtype=float)
    global_alignment_mode = "identity"
    if args.align != "none":
        log_step("Running global alignment on full dataset")
        pred_meshes = [comp.mesh for comp in elems_pr] + [space.mesh for space in spaces_pr]
        gt_meshes = [comp.mesh for comp in elems_gt] + [space.mesh for space in spaces_gt]
        if pred_meshes and gt_meshes:
            if args.align == "icp":
                log_step("  Using ICP (full)")
                global_alignment = rigid_icp_align(pred_meshes, gt_meshes)
            elif args.align == "icp_xy":
                log_step("  Using ICP (XY translation)")
                global_alignment = rigid_icp_align_xy_only(pred_meshes, gt_meshes)
            else:
                log_step("  Using centroid alignment")
                global_alignment = _centroid_align(pred_meshes, gt_meshes)
            global_alignment_mode = args.align
            _o3d_transform_inplace(elems_pr, global_alignment)
            _o3d_transform_inplace(spaces_pr, global_alignment)
            log_step("  Global alignment applied")
            # Display PRED spaces after global alignment
            _print_spaces_summary("PRED (after global alignment)", spaces_pr)
            if args.visualize:
                try:
                    _visualize_spaces(spaces_gt, spaces_pr, title="GT vs PRED (after global alignment)")
                except Exception as exc:
                    log_step(f"Visualization (after global) failed: {exc}")
        else:
            log_step("  Skipping global alignment - insufficient geometry")
    else:
        log_step("Skipping global alignment step")

    space_iou_matrix, _ = compute_space_iou_matrix(spaces_gt, spaces_pr, thresh=space_match_thresh, inside_eps=inside_eps)
    matches = match_spaces_one_to_one(space_iou_matrix, space_match_thresh)

    target_guid_upper = args.target_space_guid.upper() if args.target_space_guid else None
    target_refinement_info: Optional[Dict[str, Any]] = None
    if target_guid_upper:
        target_match_entry: Optional[Tuple[int, int, float]] = None
        for entry in matches:
            i_gt, j_pr, score = entry
            if (spaces_gt[i_gt].guid or "").upper() == target_guid_upper:
                target_match_entry = entry
                break
        if target_match_entry is None:
            raise RuntimeError(f"Target space GUID {args.target_space_guid} was not matched. "
                               "Try adjusting --space-match-thresh or verify the GUID.")

        target_gt_idx, target_pr_idx, target_pre_iou = target_match_entry
        target_gt_space = spaces_gt[target_gt_idx]
        target_pr_space = spaces_pr[target_pr_idx]

        log_step(f"Running targeted ICP refinement for GT space {target_gt_space.guid} "
                 f"against PRED space {target_pr_space.guid}")
        target_icp_alignment = align_space_pair(target_gt_space, target_pr_space, "icp")
        _o3d_transform_inplace(spaces_pr, target_icp_alignment)
        _o3d_transform_inplace(elems_pr, target_icp_alignment)
        global_alignment = target_icp_alignment @ global_alignment
        global_alignment_mode = "centroid_then_target_icp"
        target_post_iou = iou_between_two_meshes(
            target_gt_space.mesh,
            spaces_pr[target_pr_idx].mesh,
            eps=inside_eps
        )
        log_step(f"  Target space IoU (before ICP): {float(target_pre_iou):.6f} | after ICP: {float(target_post_iou):.6f}")
        target_refinement_info = {
            "gt_guid": target_gt_space.guid,
            "gt_name": target_gt_space.meta.Name,
            "gt_index": target_gt_idx,
            "pred_guid": spaces_pr[target_pr_idx].guid,
            "pred_name": spaces_pr[target_pr_idx].meta.Name,
            "pred_index": target_pr_idx,
            "pre_icp_iou": float(target_pre_iou),
            "post_icp_iou": float(target_post_iou),
            "T": _matrix_to_nested_list(target_icp_alignment)
        }
    else:
        log_step("Skipping target-space ICP refinement; no --target-space-guid provided.")

    spaces_pr_global_aligned = _clone_comp_list(spaces_pr)
    elems_pr_global_aligned = _clone_comp_list(elems_pr)

    rounds: Dict[str, Any] = {}
    per_space_payloads: Dict[str, List[Tuple[Comp, Dict[str, Any]]]] = {}

    global_round_eval, global_per_space_payload = _run_round(
        "global",
        spaces_gt,
        elems_gt,
        spaces_pr_global_aligned,
        elems_pr_global_aligned,
        matches,
        space_guid_to_elem_indices_gt,
        space_guid_to_elem_indices_pr,
        eps,
        inside_eps,
        max_k_for_ie,
        "none",
        False,
        target_guid_upper,
        False
    )
    per_space_payloads["global"] = global_per_space_payload

    global_metrics_round, global_unmatched_summary, global_unmatched_bucket = _compute_round_element_metrics(
        global_round_eval,
        spaces_gt,
        spaces_pr_global_aligned,
        elems_gt,
        elems_pr_global_aligned,
        space_guid_to_elem_indices_gt,
        space_guid_to_elem_indices_pr,
        elem_guid_to_space_gt,
        elem_guid_to_space_pr,
        args.include_unmatched,
        eps,
        inside_eps,
        max_k_for_ie,
        False,
        args.align
    )

    rounds["global"] = {
        "overall": global_metrics_round["overall"],
        "per_gt": global_metrics_round["per_gt"],
        "per_pred": global_metrics_round["per_pred"],
        "correspondences": global_metrics_round["correspondences"],
        "by_space": global_round_eval["by_space"],
        "space_matches": global_round_eval["space_match_records"],
        "unmatched_summary": global_unmatched_summary,
        "floor_area_records": global_round_eval["floor_area_records"]
    }
    if global_unmatched_bucket is not None:
        rounds["global"]["unmatched_bucket"] = global_unmatched_bucket

    local_apply_alignment = args.space_align.lower() != "none"
    local_round_eval, local_per_space_payload = _run_round(
        "local",
        spaces_gt,
        elems_gt,
        spaces_pr,
        elems_pr,
        matches,
        space_guid_to_elem_indices_gt,
        space_guid_to_elem_indices_pr,
        eps,
        inside_eps,
        max_k_for_ie,
        args.space_align,
        local_apply_alignment,
        target_guid_upper,
        args.visualize_per_space and local_apply_alignment
    )
    per_space_payloads["local"] = local_per_space_payload

    local_metrics_round, local_unmatched_summary, local_unmatched_bucket = _compute_round_element_metrics(
        local_round_eval,
        spaces_gt,
        spaces_pr,
        elems_gt,
        elems_pr,
        space_guid_to_elem_indices_gt,
        space_guid_to_elem_indices_pr,
        elem_guid_to_space_gt,
        elem_guid_to_space_pr,
        args.include_unmatched,
        eps,
        inside_eps,
        max_k_for_ie,
        local_apply_alignment,
        args.align
    )

    rounds["local"] = {
        "overall": local_metrics_round["overall"],
        "per_gt": local_metrics_round["per_gt"],
        "per_pred": local_metrics_round["per_pred"],
        "correspondences": local_metrics_round["correspondences"],
        "by_space": local_round_eval["by_space"],
        "space_matches": local_round_eval["space_match_records"],
        "unmatched_summary": local_unmatched_summary,
        "floor_area_records": local_round_eval["floor_area_records"]
    }
    if local_unmatched_bucket is not None:
        rounds["local"]["unmatched_bucket"] = local_unmatched_bucket

    combined_space_matches: List[Dict[str, Any]] = []
    for rec_local in rounds["local"]["space_matches"]:
        space_iou_global = float(rec_local.get("space_iou_before", rec_local.get("space_iou", 0.0)))
        space_iou_local = float(rec_local.get("space_iou", 0.0))
        combined_space_matches.append({
            "match_index": rec_local["match_index"],
            "gt_index": rec_local["gt_index"],
            "pred_index": rec_local["pred_index"],
            "gt_guid": rec_local["gt_guid"],
            "pred_guid": rec_local["pred_guid"],
            "is_target": rec_local["is_target"],
            "space_iou_global": space_iou_global,
            "space_iou_local": space_iou_local,
            "space_iou_delta": space_iou_local - space_iou_global
        })

    gt_mesh_path = os.path.join(mesh_output_dir_abs, "gt_mesh.ply")
    pred_mesh_path = os.path.join(mesh_output_dir_abs, "pred_aligned_mesh.ply")
    pred_mesh_global_round_path = os.path.join(mesh_output_dir_abs, "pred_global_round_mesh.ply")

    floor_area_csv_paths: Dict[str, str] = {}
    for round_label, round_data in rounds.items():
        round_path = _path_with_round_suffix(floor_area_csv_base, round_label)
        write_floor_area_csv(round_path, round_label, round_data.get("floor_area_records", []))
        round_data["floor_area_csv_path"] = round_path
        floor_area_csv_paths[round_label] = round_path

    report_config = {
        "gt_path": args.gt,
        "pred_path": args.pred,
        "align": args.align,
        "space_align": args.space_align,
        "space_match_thresh": space_match_thresh,
        "epsilon": eps,
        "include_unmatched": args.include_unmatched,
        "ifc_classes": args.ifc_classes,
        "inside_eps": inside_eps,
        "ie_cap": max_k_for_ie,
        "target_space_guid": args.target_space_guid,
        "session_output_dir": session_output_dir,
        "mesh_output_dir": mesh_output_dir_abs,
        "floor_area_csv_base": floor_area_csv_base,
        "floor_area_csv_paths": floor_area_csv_paths,
        "save_json_path": save_json_path,
        "save_csv_prefix": save_csv_prefix
    }

    report_mesh_exports: Dict[str, Any] = {
        "session_root": session_output_dir,
        "directory": mesh_output_dir_abs,
        "gt_mesh": gt_mesh_path,
        "pred_mesh": pred_mesh_path,
        "pred_mesh_global_round": pred_mesh_global_round_path
    }

    try:
        metric_mesh_root_dir = os.path.join(mesh_output_dir_abs, "metrics")
        os.makedirs(metric_mesh_root_dir, exist_ok=True)

        def _export_metric_meshes_for_round(round_label: str, pred_elems: List[Comp], pred_spaces: List[Comp]) -> Dict[str, Any]:
            round_data = rounds.get(round_label) or {}
            per_gt = round_data.get("per_gt") or []
            per_pred = round_data.get("per_pred") or []

            iou_gt_map = _metric_map_by_guid(per_gt, "gt_guid", "iou_union_pred_vs_gt")
            iou_pred_map = _metric_map_by_guid(per_pred, "pred_guid", "iou_union_gt_vs_pred")
            comp_gt_map = _metric_map_by_guid(per_gt, "gt_guid", "local_compactness_gt_to_pred")
            comp_pred_map = _metric_map_by_guid(per_pred, "pred_guid", "local_compactness_pred_to_gt")

            space_matches = round_data.get("space_matches") or []
            space_iou_gt_map: Dict[str, float] = {}
            space_iou_pred_map: Dict[str, float] = {}
            space_compact_gt_map: Dict[str, float] = {}
            space_compact_pred_map: Dict[str, float] = {}
            for rec in space_matches:
                gt_guid = rec.get("gt_guid")
                pred_guid = rec.get("pred_guid")
                iou_value = _safe_float(rec.get("space_iou", 0.0), default=0.0)
                if gt_guid:
                    space_iou_gt_map[str(gt_guid)] = iou_value
                    space_compact_gt_map[str(gt_guid)] = 1.0
                if pred_guid:
                    space_iou_pred_map[str(pred_guid)] = iou_value
                    space_compact_pred_map[str(pred_guid)] = 1.0

            iou_gt_map.update(space_iou_gt_map)
            iou_pred_map.update(space_iou_pred_map)
            comp_gt_map.update(space_compact_gt_map)
            comp_pred_map.update(space_compact_pred_map)

            round_mesh_dir = os.path.join(metric_mesh_root_dir, f"{round_label}_round")
            iou_root = os.path.join(round_mesh_dir, "3DIou")
            iou_gt_dir = os.path.join(iou_root, "gt")
            iou_pred_dir = os.path.join(iou_root, "pred")
            comp_gt_dir = os.path.join(round_mesh_dir, "3Dcompactness_gt")
            comp_pred_dir = os.path.join(round_mesh_dir, "3Dcompactness_pred")

            log_step(f"Exporting metric meshes for '{round_label}' round to {round_mesh_dir}")
            return {
                "directory": round_mesh_dir,
                "3DIou": {
                    "gt": {
                        "directory": iou_gt_dir,
                        "by_class": _export_metric_meshes_by_class(elems_gt + spaces_gt, iou_gt_map, iou_gt_dir)
                    },
                    "pred": {
                        "directory": iou_pred_dir,
                        "by_class": _export_metric_meshes_by_class(pred_elems + pred_spaces, iou_pred_map, iou_pred_dir)
                    }
                },
                "3Dcompactness_gt": {
                    "directory": comp_gt_dir,
                    "by_class": _export_metric_meshes_by_class(elems_gt + spaces_gt, comp_gt_map, comp_gt_dir)
                },
                "3Dcompactness_pred": {
                    "directory": comp_pred_dir,
                    "by_class": _export_metric_meshes_by_class(pred_elems + pred_spaces, comp_pred_map, comp_pred_dir)
                }
            }

        report_mesh_exports["metrics"] = {"directory": metric_mesh_root_dir, "rounds": {}}
        report_mesh_exports["metrics"]["rounds"]["global"] = _export_metric_meshes_for_round(
            "global",
            elems_pr_global_aligned,
            spaces_pr_global_aligned
        )
        report_mesh_exports["metrics"]["rounds"]["local"] = _export_metric_meshes_for_round(
            "local",
            elems_pr,
            spaces_pr
        )
    except Exception as exc:
        log_step(f"Metric mesh export failed: {exc}")

    report: Dict[str, Any] = {
        "config": report_config,
        "global_alignment": {"mode": global_alignment_mode, "T": _matrix_to_nested_list(global_alignment)},
        "target_refinement": target_refinement_info,
        "rounds": rounds,
        "space_matches_combined": combined_space_matches,
        "mesh_exports": report_mesh_exports
    }

    log_step(f"Exporting combined meshes to {mesh_output_dir_abs}")
    gt_meshes_for_export = [comp.mesh for comp in elems_gt] + [space.mesh for space in spaces_gt]
    pred_meshes_for_global_round = [comp.mesh for comp in elems_pr_global_aligned] + [space.mesh for space in spaces_pr_global_aligned]
    pred_meshes_for_export = [comp.mesh for comp in elems_pr] + [space.mesh for space in spaces_pr]
    _write_combined_mesh(gt_meshes_for_export, gt_mesh_path)
    _write_combined_mesh(pred_meshes_for_global_round, pred_mesh_global_round_path)
    _write_combined_mesh(pred_meshes_for_export, pred_mesh_path)
    log_step("  Saved combined meshes (GT, global-round PRED snapshot, final PRED).")

    global_space_mesh_exports: List[Dict[str, Any]] = []
    if "global" in rounds and rounds["global"].get("space_matches"):
        global_space_mesh_dir = os.path.join(mesh_output_dir_abs, "global_round_spaces")
        global_space_mesh_exports = _export_space_meshes_for_round(
            "global",
            rounds["global"]["space_matches"],
            rounds["global"].get("by_space", {}),
            spaces_gt,
            spaces_pr_global_aligned,
            elems_gt,
            elems_pr_global_aligned,
            global_space_mesh_dir
        )
        report_mesh_exports["global_round"] = {
            "directory": global_space_mesh_dir,
            "spaces": global_space_mesh_exports,
            "pred_mesh": pred_mesh_global_round_path
        }
        log_step(f"Exported {len(global_space_mesh_exports)} global round space mesh bundles to {global_space_mesh_dir}")

    local_space_mesh_exports: List[Dict[str, Any]] = []
    if "local" in rounds and rounds["local"].get("space_matches"):
        local_space_mesh_dir = os.path.join(mesh_output_dir_abs, "local_round_spaces")
        local_space_mesh_exports = _export_space_meshes_for_round(
            "local",
            rounds["local"]["space_matches"],
            rounds["local"].get("by_space", {}),
            spaces_gt,
            spaces_pr,
            elems_gt,
            elems_pr,
            local_space_mesh_dir
        )
        report_mesh_exports["local_round"] = {
            "directory": local_space_mesh_dir,
            "spaces": local_space_mesh_exports
        }
        log_step(f"Exported {len(local_space_mesh_exports)} local round space mesh bundles to {local_space_mesh_dir}")

    log_step("Rendering console summary")
    print("\n=== Space Matching (global -> local) ===")
    print(f"GT spaces: {len(spaces_gt)} | PRED spaces: {len(spaces_pr)} | Matched: {len(matches)} | Threshold: {space_match_thresh:.2f}")
    if combined_space_matches:
        print("Top space matches (first 5):")
        for rec in combined_space_matches[:5]:
            flag = " (target)" if rec.get("is_target") else ""
            print(
                f"  GT {rec['gt_guid']} -> PRED {rec['pred_guid']} | "
                f"IoU_global={rec['space_iou_global']:.3f} -> IoU_local={rec['space_iou_local']:.3f}{flag}"
            )
    else:
        print("  No matched spaces.")

    for round_label in ("global", "local"):
        if round_label not in rounds:
            continue
        round_title = round_label.title()
        overall = rounds[round_label]["overall"]
        summary = rounds[round_label]["unmatched_summary"]
        print(f"\n=== {round_title} Round Overall (aggregated elements) ===")
        for k, v in overall.items():
            print(f"{k}: {v:.6f}")
        print(f"\n{round_title} Round Unmatched summary:")
        print(f"  Mode: {summary['mode']}")
        print(f"  GT spaces unmatched: {summary['gt_spaces_unmatched']} (ignored: {summary['gt_spaces_ignored']})")
        print(f"  PRED spaces unmatched: {summary['pred_spaces_unmatched']} (ignored: {summary['pred_spaces_ignored']})")
        print(f"  GT elements unmatched: {summary['gt_elements_unmatched']} (ignored: {summary['gt_elements_ignored']})")
        print(f"  PRED elements unmatched: {summary['pred_elements_unmatched']} (ignored: {summary['pred_elements_ignored']})")

    if floor_area_csv_paths:
        print("\nFloor area CSV files:")
        for round_label, path in floor_area_csv_paths.items():
            print(f"  {round_label.title()}: {path}")

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log_step(f"Saved JSON to {save_json_path}")

    if save_csv_prefix:
        for round_label, round_data in rounds.items():
            round_prefix = f"{save_csv_prefix}_{round_label}"
            save_csvs(round_prefix, round_data)
            log_step(f"Saved CSV bundle for round '{round_label}' with prefix: {round_prefix}")
            payload = per_space_payloads.get(round_label, [])
            for gt_space, metrics in payload:
                save_space_csvs(round_prefix, gt_space, metrics)
            if payload:
                log_step(f"Saved per-space CSV bundles for round '{round_label}' ({len(payload)} spaces)")
            unmatched_bucket = round_data.get("unmatched_bucket")
            if unmatched_bucket:
                unmatched_prefix = f"{round_prefix}_unmatched"
                _write_csv_bundle(
                    unmatched_prefix,
                    unmatched_bucket["metrics"].get("per_gt", []),
                    unmatched_bucket["metrics"].get("per_pred", []),
                    unmatched_bucket["metrics"].get("correspondences", [])
                )
                log_step(f"Saved unmatched bucket CSV bundle for round '{round_label}'")

    log_step("Processing complete")


if __name__ == "__main__":
    main()
