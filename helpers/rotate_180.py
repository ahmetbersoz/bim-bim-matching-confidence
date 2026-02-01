#!/usr/bin/env python3
"""
Rotate all IfcBuildingStorey objects and all elements contained in each storey
by 180 degrees around the global Z axis, then save as a new IFC.

Usage:
  python rotate_storeys_180.py input.ifc output.ifc
"""

import sys
import math
import numpy as np
import ifcopenshell
import ifcopenshell.util.placement


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.eye(4, dtype=float)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def ensure_dir(file, existing, ratios3):
    ratios3 = [float(r) for r in ratios3]
    if existing is None:
        return file.create_entity("IfcDirection", ratios3)
    existing.DirectionRatios = ratios3
    return existing


def ensure_pt(file, existing, coords):
    coords = [float(c) for c in coords]
    if existing is None:
        return file.create_entity("IfcCartesianPoint", coords)
    existing.Coordinates = coords
    return existing


def set_relative_placement_from_matrix(file, local_placement, rel_matrix: np.ndarray):
    """
    Write rel_matrix into local_placement.RelativePlacement (Axis2Placement3D/2D).
    Only supports IfcLocalPlacement with RelativePlacement of Axis2Placement3D or Axis2Placement2D.
    """
    rp = local_placement.RelativePlacement
    if rp is None:
        raise ValueError("IfcLocalPlacement has no RelativePlacement")

    if rp.is_a("IfcAxis2Placement3D"):
        loc = rel_matrix[:3, 3].astype(float)

        # Extract axes from matrix
        x = rel_matrix[:3, 0]
        z = rel_matrix[:3, 2]

        z = unit(z)
        x = x - z * float(np.dot(x, z))  # make x orthogonal to z
        x = unit(x)

        # If x collapsed, pick a fallback perpendicular to z
        if np.linalg.norm(x) < 1e-12:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(fallback, z))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            x = unit(fallback - z * float(np.dot(fallback, z)))

        rp.Location = ensure_pt(file, rp.Location, loc.tolist())
        rp.Axis = ensure_dir(file, rp.Axis, z.tolist())
        rp.RefDirection = ensure_dir(file, rp.RefDirection, x.tolist())
        return

    if rp.is_a("IfcAxis2Placement2D"):
        loc = rel_matrix[:2, 3].astype(float)

        # 2D x-axis is first column (x,y); rotation is in XY plane
        x = rel_matrix[:2, 0]
        x = unit(x)

        rp.Location = ensure_pt(file, rp.Location, loc.tolist())
        rp.RefDirection = ensure_dir(file, rp.RefDirection, x.tolist())
        return

    raise ValueError(f"Unsupported RelativePlacement type: {rp.is_a()}")


def get_world_matrix(local_placement):
    # IfcOpenShell util returns a 4x4 numpy matrix in world coordinates
    return np.array(ifcopenshell.util.placement.get_local_placement(local_placement), dtype=float)


def get_parent_world_matrix(local_placement, placement_new_world_map):
    parent = getattr(local_placement, "PlacementRelTo", None)
    if parent is None:
        return np.eye(4, dtype=float)
    if parent in placement_new_world_map:
        return placement_new_world_map[parent]
    return get_world_matrix(parent)


def collect_storey_elements(storey):
    elems = []
    # Inverse: storey.ContainsElements -> list[IfcRelContainedInSpatialStructure]
    rels = getattr(storey, "ContainsElements", None) or []
    for rel in rels:
        for e in (getattr(rel, "RelatedElements", None) or []):
            elems.append(e)
    return elems


def main(inp, outp):
    f = ifcopenshell.open(inp)

    angle = math.radians(180.0)
    R = rot_z(angle)

    storeys = f.by_type("IfcBuildingStorey")
    if not storeys:
        print("No IfcBuildingStorey found. Writing file unchanged.")
        f.write(outp)
        return

    # Build a list of products to rotate: all storeys + their contained elements
    products_to_rotate = []
    seen = set()

    for s in storeys:
        if s not in seen:
            products_to_rotate.append(s)
            seen.add(s)

        for e in collect_storey_elements(s):
            if e not in seen:
                products_to_rotate.append(e)
                seen.add(e)

    # Precompute original world matrices for placements and also compute rotated world matrices.
    # We rotate about the GLOBAL origin (0,0,0). (Common for “flip 180°” requests.)
    placement_new_world_map = {}  # IfcLocalPlacement -> new world matrix

    for p in products_to_rotate:
        plc = getattr(p, "ObjectPlacement", None)
        if plc is None or not plc.is_a("IfcLocalPlacement"):
            continue
        old_world = get_world_matrix(plc)
        new_world = R @ old_world
        placement_new_world_map[plc] = new_world

    # Now write back each placement as a new relative transform vs (possibly rotated) parent
    updated = 0
    skipped = 0

    for plc, new_world in placement_new_world_map.items():
        parent_world = get_parent_world_matrix(plc, placement_new_world_map)

        # rel = inv(parent_world) * new_world
        try:
            rel = np.linalg.inv(parent_world) @ new_world
        except np.linalg.LinAlgError:
            skipped += 1
            continue

        try:
            set_relative_placement_from_matrix(f, plc, rel)
            updated += 1
        except Exception:
            skipped += 1

    print(f"Rotated placements updated: {updated}, skipped: {skipped}")
    f.write(outp)
    print(f"Wrote: {outp}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rotate_storeys_180.py input.ifc output.ifc")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
