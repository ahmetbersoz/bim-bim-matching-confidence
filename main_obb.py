#!/usr/bin/env python3
# ifc_metrics_with_metadata_obb.py
# Compute per-element 3D-IoU & 3D-Compactness between two IFCs, preserving IFC metadata,
# using ONLY Oriented Bounding Boxes (OBBs). No mesh booleans, no convex hulls.
#
# - Loads IFCs with ifcopenshell
# - Aligns PRED -> GT (ICP if available, else centroid translation; optional XY-only ICP)
# - Builds pairwise IoU matrix using exact OBB intersection/union math
# - Reports per-element metrics for *both* GT and PRED
# - Preserves metadata (GlobalId, Name, Type, attributes, and Psets)

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

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

VERBOSE = True


def log_step(message: str) -> None:
    if VERBOSE:
        print(f"[step] {message}", flush=True)


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


# ---------- OBB math ----------

def _obb_from_mesh(mesh: o3d.geometry.TriangleMesh) -> OBB:
    ob = mesh.get_oriented_bounding_box()
    center = np.array(ob.center, dtype=float, copy=True)
    R = np.array(ob.R, dtype=float, copy=True)          # ensure writable
    extent = np.array(ob.extent, dtype=float, copy=True)
    half = extent * 0.5
    planes: List[Tuple[np.ndarray, float]] = []
    for i in range(3):
        # copy to avoid modifying a read-only view
        axis = R[:, i].copy()
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-18:
            # fallback axis if something degenerate happens
            axis = np.zeros(3, dtype=float)
            axis[i] = 1.0
            axis_norm = 1.0
        axis = axis / axis_norm
        # + side
        n1 = axis
        p1 = center + half[i] * axis
        d1 = float(np.dot(n1, p1))
        planes.append((n1, d1))
        # - side
        n2 = -axis
        p2 = center - half[i] * axis
        d2 = float(np.dot(n2, p2))
        planes.append((n2, d2))
    corners = np.array(ob.get_box_points(), dtype=float, copy=True)
    return OBB(center=center, R=R, extent=extent, half=half, planes=planes, corners=corners)


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


def intersection_volume_obbs(obbs: List[OBB], eps: float = 1e-7) -> float:
    if not obbs:
        return 0.0
    if len(obbs) == 1:
        return _obb_volume(obbs[0])
    planes: List[Tuple[np.ndarray, float]] = []
    for obb in obbs:
        planes.extend(obb.planes)
    verts = _intersection_vertices_from_planes(planes, eps_inside=eps)
    if verts.shape[0] < 4:
        return 0.0
    return _polyhedron_volume_from_planes_and_vertices(planes, verts, eps_on_plane=10 * eps)


def union_volume_obbs(obbs: List[OBB], eps: float = 1e-7, max_k_for_ie: int = 8) -> float:
    k = len(obbs)
    if k == 0:
        return 0.0
    if k == 1:
        return _obb_volume(obbs[0])

    if k > max_k_for_ie:
        # Approximate: sum vols minus pairwise intersections only.
        vol = sum(_obb_volume(o) for o in obbs)
        pair = 0.0
        for a, b in combinations(obbs, 2):
            pair += intersection_volume_obbs([a, b], eps)
        return max(0.0, vol - pair)

    total = 0.0
    for r in range(1, k + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for combo in combinations(obbs, r):
            total += sign * intersection_volume_obbs(list(combo), eps)
    return max(0.0, float(total))


def iou_between_two_obbs(a: OBB, b: OBB, eps: float = 1e-7) -> float:
    inter = intersection_volume_obbs([a, b], eps)
    if inter <= 0.0:
        return 0.0
    va = _obb_volume(a)
    vb = _obb_volume(b)
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


def iou_union_of_set_vs_single(set_obbs: List[OBB], ref_obb: OBB, eps: float = 1e-7, max_k_for_ie: int = 8) -> float:
    if not set_obbs:
        return 0.0

    m = len(set_obbs)
    cap = max_k_for_ie
    if m > cap:
        set_obbs = set_obbs[:cap]
        m = cap

    # numerator: union over intersections with ref
    num = 0.0
    for r in range(1, m + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for combo in combinations(set_obbs, r):
            num += sign * intersection_volume_obbs([ref_obb, *combo], eps)

    # denominator: union over ref + all in set
    den = union_volume_obbs([ref_obb] + set_obbs, eps=eps, max_k_for_ie=max_k_for_ie)
    if den <= 0.0:
        return 0.0
    return float(max(0.0, num) / den)


# ---------- IFC -> mesh & metadata ----------

def _ifc_length_scale_m(ifc) -> float:
    try:
        return float(ifc_unit.calculate_unit_scale(ifc))
    except Exception:
        return 1.0

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

        obb = _obb_from_mesh(mesh)
        vol = _obb_volume(obb)
        if vol <= 0.0:
            log_step(f"  Non-positive OBB volume for {guid} ({etype}); skipping")
            continue

        aabb_min, aabb_max = _o3d_bounds(mesh)
        meta = _product_meta(p)
        comps.append(Comp(
            idx=len(comps),
            guid=meta.GlobalId,
            etype=meta.IfcType,
            meta=meta,
            mesh=mesh,
            obb=obb,
            volume=vol,
            aabb_min=aabb_min,
            aabb_max=aabb_max
        ))
        if len(comps) % progress_step == 0:
            log_step(f"  Meshed {len(comps)} components so far")
    if not comps:
        raise RuntimeError(f"No meshable IfcProducts with positive OBB volume found in {os.path.basename(path)}")
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
        # Recompute OBB & volume from the transformed mesh to avoid mutating internals.
        c.obb = _obb_from_mesh(c.mesh)
        c.volume = _obb_volume(c.obb)


# ---------- IoU + Compactness (OBB-based) ----------

def _aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    return np.all(a_min <= b_max) and np.all(b_min <= a_max) and np.all(a_max >= b_min) and np.all(b_max >= a_min)

def _pairwise_iou(gt: List[Comp], pr: List[Comp], eps: float, inside_eps: float = 1e-7) -> np.ndarray:
    m, n = len(gt), len(pr)
    log_step(f"Computing pairwise IoU matrix ({m}x{n}) using OBB intersection/union")
    M = np.zeros((m, n), dtype=float)
    progress_stride = max(1, min(50, (m // 10) or 5))
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            if not _aabb_overlap(g.aabb_min, g.aabb_max, p.aabb_min, p.aabb_max):
                continue
            v = iou_between_two_obbs(g.obb, p.obb, eps=inside_eps)
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
    log_step("Aggregating IoU and compactness metrics (OBB-based)")
    M = _pairwise_iou(gt, pr, eps=eps, inside_eps=inside_eps)
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
            iou_union = iou_union_of_set_vs_single([pr[j].obb for j in js], g.obb, eps=inside_eps, max_k_for_ie=max_k_for_ie)
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
            iou_union = iou_union_of_set_vs_single([gt[i].obb for i in is_], p.obb, eps=inside_eps, max_k_for_ie=max_k_for_ie)
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


# ---------- CSV helpers ----------

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


def main():
    _silence_vtk_output()

    ap = argparse.ArgumentParser(description="Per-element 3D-IoU & 3D-Compactness between two IFCs, with metadata. (OBB-only)")
    ap.add_argument("--gt", required=True, help="Path to ground-truth IFC.")
    ap.add_argument("--pred", required=True, help="Path to predicted/reconstructed IFC.")
    ap.add_argument("--align", choices=["icp", "centroid", "none", "icp_xy"], default="icp",
                    help="Alignment for PRED->GT. 'icp_xy' uses only XY translation.")
    ap.add_argument("--epsilon", type=float, default=0.05, help="IoU threshold to consider a correspondence.")
    ap.add_argument("--save-json", type=str, default=None, help="Path to save a JSON report.")
    ap.add_argument("--save-csv-prefix", type=str, default=None, help="Prefix to save CSVs: <prefix>_per_gt.csv, _per_pred.csv, _edges.csv")
    ap.add_argument("--ifc-classes", nargs="+", default=None,
                    help="Optional list of IFC classes to include (e.g. IfcWall IfcDoor). You may also pass a single comma-separated string 'IfcWall,IfcDoor'.")
    ap.add_argument("--inside-eps", type=float, default=1e-7, help="Tolerance for half-space tests / plane membership.")
    ap.add_argument("--ie-cap", type=int, default=8, help="Max K for exact inclusion–exclusion before pairwise approximation.")

    # Default CLI arguments for convenience (used only when no CLI args are provided)
    DEFAULT_ARGS = [
        "--gt", ".\\input\\WW_revit_model_v6.ifc",
        "--pred", ".\\input\\ww-v1-ifc4-geo.ifc",
        "--align", "icp",
        "--epsilon", "0.10",
        "--save-json", "metrics_obb.json",
        "--save-csv-prefix", "out/metrics_obb",
        "--ifc-classes", "IfcSpace"
        # "--ifc-classes", "IfcWall", "IfcWallStandardCase"
    ]

    if len(sys.argv) == 1:
        log_step(f"No CLI arguments detected — using DEFAULT_ARGS: {' '.join(DEFAULT_ARGS)}")
        args = ap.parse_args(DEFAULT_ARGS)
    else:
        args = ap.parse_args()

    if args.ifc_classes is not None and len(args.ifc_classes) == 1 and ',' in args.ifc_classes[0]:
        args.ifc_classes = [s.strip() for s in args.ifc_classes[0].split(',') if s.strip()]

    log_step(f"Loading GT IFC: {args.gt}")
    gt, _ = load_ifc_components(args.gt, include_types=args.ifc_classes)
    log_step(f"  GT elements loaded: {len(gt)}")

    log_step(f"Loading PRED IFC: {args.pred}")
    pr, _ = load_ifc_components(args.pred, include_types=args.ifc_classes)
    log_step(f"  PRED elements loaded: {len(pr)}")

    # Visualize pre-alignment (optional)
    try:
        gt_meshes = [_o3d_mesh_copy(c.mesh) for c in gt]
        pred_meshes = [_o3d_mesh_copy(c.mesh) for c in pr]
        for m in gt_meshes:
            m.paint_uniform_color([0, 1, 0])  # green
        for m in pred_meshes:
            m.paint_uniform_color([1, 0, 0])  # red
        o3d.visualization.draw_geometries(gt_meshes, mesh_show_wireframe=True)
        o3d.visualization.draw_geometries(pred_meshes, mesh_show_wireframe=True)
        o3d.visualization.draw_geometries(gt_meshes + pred_meshes, mesh_show_wireframe=True)
    except Exception:
        pass

    # Align PRED -> GTcc
    if args.align != "none":
        log_step("Aligning PRED to GT ...")
        if args.align == "icp":
            log_step("  Running Open3D ICP alignment (full)")
            T = rigid_icp_align([c.mesh for c in pr], [c.mesh for c in gt])
        elif args.align == "icp_xy":
            log_step("  Running Open3D ICP XY-translation alignment")
            T = rigid_icp_align_xy_only([c.mesh for c in pr], [c.mesh for c in gt])
        else:
            log_step("  Using centroid alignment")
            T = _centroid_align([c.mesh for c in pr], [c.mesh for c in gt])

        _o3d_transform_inplace(pr, T)
        log_step("  Alignment applied")

        # Visualize aligned meshes
        try:
            gt_meshes = [_o3d_mesh_copy(c.mesh) for c in gt]
            pred_meshes = [_o3d_mesh_copy(c.mesh) for c in pr]
            for m in gt_meshes:
                m.paint_uniform_color([0, 1, 0])
            for m in pred_meshes:
                m.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries(gt_meshes + pred_meshes, mesh_show_wireframe=True)
        except Exception:
            pass
    else:
        log_step("Skipping alignment step")

    # Metrics (OBB-only)
    log_step("Computing OBB-based metrics ...")
    res = compute_all_metrics(gt, pr, eps=float(args.epsilon), inside_eps=float(args.inside_eps), max_k_for_ie=int(args.ie_cap))
    log_step("Metrics computation complete")

    # Console summary
    log_step("Rendering console summary")
    print("\n=== Overall (OBB) ===")
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
