#!/usr/bin/env python3
"""
IFC storey editor:
- Remove storey named "L0"
- Rename "Level 0" -> "Storey 0"
- Rename "L4" -> "Storey 1"
- Remove storey named "L3"
Preserves elements by reassigning containment/decomposition from removed storeys.

Usage:
  python edit_storeys.py input.ifc output.ifc
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List, Optional, Set

import ifcopenshell


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def find_storeys_by_name(model: ifcopenshell.file, name: str) -> List:
    """Match against Name or LongName (case-insensitive, trimmed)."""
    target = _norm(name)
    out = []
    seen_ids: Set[int] = set()
    for st in model.by_type("IfcBuildingStorey"):
        cand = _norm(getattr(st, "Name", None))
        cand2 = _norm(getattr(st, "LongName", None))
        if cand == target or cand2 == target:
            # avoid duplicates if both fields match
            sid = st.id()
            if sid not in seen_ids:
                out.append(st)
                seen_ids.add(sid)
    return out


def rename_storey(storey, new_name: str) -> None:
    # Most IFCs use Name; some use LongName too.
    if hasattr(storey, "Name"):
        storey.Name = new_name
    if hasattr(storey, "LongName") and storey.LongName:
        storey.LongName = new_name


def iter_rels_contained(model: ifcopenshell.file):
    # Elements contained in spatial structure
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        yield rel


def iter_rels_referenced(model: ifcopenshell.file):
    # Elements referenced in spatial structure (less common but exists)
    for rel in model.by_type("IfcRelReferencedInSpatialStructure"):
        yield rel


def reassign_spatial_rels(model: ifcopenshell.file, old_storey, new_storey) -> int:
    """Move all containment/reference relationships from old_storey to new_storey."""
    changed = 0
    for rel in iter_rels_contained(model):
        if rel.RelatingStructure == old_storey:
            rel.RelatingStructure = new_storey
            changed += 1
    for rel in iter_rels_referenced(model):
        if rel.RelatingStructure == old_storey:
            rel.RelatingStructure = new_storey
            changed += 1
    return changed


def remove_from_parent_aggregations(model: ifcopenshell.file, storey) -> int:
    """
    Remove storey from any IfcRelAggregates.RelatedObjects lists (typically Building->Storeys).
    """
    changed = 0
    for rel in list(model.by_type("IfcRelAggregates")):
        related = list(rel.RelatedObjects or [])
        if storey in related:
            related = [x for x in related if x != storey]
            if related:
                rel.RelatedObjects = related
            else:
                safe_remove_entity(model, rel)
            changed += 1
    return changed


def reassign_child_decomposition(model: ifcopenshell.file, old_storey, new_storey) -> int:
    """
    If the old storey decomposes things (e.g., IfcSpace) via IfcRelAggregates,
    reattach them to the new storey.
    """
    changed = 0
    for rel in list(model.by_type("IfcRelAggregates")):
        if rel.RelatingObject == old_storey:
            children = list(rel.RelatedObjects or [])
            if not children:
                # nothing to move, just remove rel
                safe_remove_entity(model, rel)
                changed += 1
                continue

            # Try to merge into an existing aggregates rel on the new storey if one exists
            target_rel = None
            for r2 in model.by_type("IfcRelAggregates"):
                if r2 is rel:
                    continue
                if r2.RelatingObject == new_storey:
                    target_rel = r2
                    break

            if target_rel is not None:
                existing = list(target_rel.RelatedObjects or [])
                # preserve order, avoid duplicates
                seen = set(x.id() for x in existing)
                merged = existing[:]
                for c in children:
                    if c.id() not in seen:
                        merged.append(c)
                        seen.add(c.id())
                target_rel.RelatedObjects = merged
                safe_remove_entity(model, rel)
            else:
                # Reuse the same relationship object
                rel.RelatingObject = new_storey

            changed += 1
    return changed


def safe_remove_entity(model: ifcopenshell.file, ent) -> None:
    """
    Remove an entity from the model, compatible with different ifcopenshell versions.
    """
    try:
        model.remove(ent)
    except Exception:
        try:
            model.remove(ent.id())
        except Exception as e:
            raise RuntimeError(f"Failed to remove entity #{ent.id()} ({ent.is_a()}): {e}") from e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_ifc", help="Path to input IFC")
    ap.add_argument("output_ifc", help="Path to output IFC")
    args = ap.parse_args()

    model = ifcopenshell.open(args.input_ifc)

    # Locate key storeys (may be multiple with same name; we handle all matches)
    level0_storeys = find_storeys_by_name(model, "Level 0")
    l4_storeys = find_storeys_by_name(model, "L4")
    l0_storeys = find_storeys_by_name(model, "L0")
    l3_storeys = find_storeys_by_name(model, "L3")

    if not level0_storeys:
        print("ERROR: Could not find any storey named 'Level 0'.", file=sys.stderr)
        return 2
    if not l4_storeys:
        print("ERROR: Could not find any storey named 'L4'.", file=sys.stderr)
        return 2

    # Pick the first as the reassignment target (common case: exactly one)
    level0 = level0_storeys[0]
    l4 = l4_storeys[0]

    # Rename targets
    rename_storey(level0, "Storey 0")
    rename_storey(l4, "Storey 1")

    # Mapping for where to move contents of removed storeys:
    # - L0's contents -> Storey 0 (formerly Level 0)
    # - L3's contents -> Storey 1 (formerly L4)
    # If you need different behavior, change these two lines.
    reassign_map = {
        "l0": level0,
        "l3": l4,
    }

    def process_removal(storeys_to_remove: Iterable, key: str) -> None:
        for st in list(storeys_to_remove):
            target = reassign_map[_norm(key)]
            # 1) Move contained/referenced elements to target storey
            reassign_spatial_rels(model, st, target)
            # 2) Move decomposed children (e.g., IfcSpace) to target storey
            reassign_child_decomposition(model, st, target)
            # 3) Remove from parent aggregations (e.g., Building->Storeys)
            remove_from_parent_aggregations(model, st)
            # 4) Finally remove the storey entity itself
            safe_remove_entity(model, st)

    # Remove L0 and L3 (while preserving all elements via reassignment)
    process_removal(l0_storeys, "L0")
    process_removal(l3_storeys, "L3")

    model.write(args.output_ifc)
    print(f"Done. Wrote: {args.output_ifc}")
    return 0


if __name__ == "__main__":
    main()

