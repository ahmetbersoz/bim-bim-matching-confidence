#!/usr/bin/env python3
"""
Split an IFC file by IfcBuildingStorey, saving each storey and its
contained elements into a separate IFC file.

Usage:
  python split_by_storey.py input.ifc [output_dir]

Each output file is named  <input_basename>__<storey_name>.ifc  and contains:
  - The full spatial hierarchy (IfcProject / IfcSite / IfcBuilding)
  - One IfcBuildingStorey and all elements directly contained in it
    (via IfcRelContainedInSpatialStructure)
  - Spaces / sub-elements decomposed under the storey
    (via IfcRelAggregates where RelatingObject == storey)
  - All geometry, property sets, and type objects that belong to those elements
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Set

import ifcopenshell
import ifcopenshell.util.element


def _safe_name(name: str) -> str:
    """Convert a storey name to a safe filesystem component."""
    clean = re.sub(r"[^\w\-]", "_", name).strip("_")
    return clean or "unnamed"


def _collect_direct_elements(model: ifcopenshell.file, storey) -> Set:
    """
    Return all IFC product entities that directly belong to this storey:
      - contained via IfcRelContainedInSpatialStructure
      - decomposed via IfcRelAggregates (e.g. IfcSpace children)
    Referenced elements (IfcRelReferencedInSpatialStructure) are intentionally
    excluded because their canonical home is another storey.
    """
    elements: Set = set()

    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure == storey:
            for elem in (rel.RelatedElements or []):
                elements.add(elem)

    for rel in model.by_type("IfcRelAggregates"):
        if rel.RelatingObject == storey:
            for obj in (rel.RelatedObjects or []):
                elements.add(obj)

    return elements


def _build_indexes(model: ifcopenshell.file):
    """
    Pre-build lookup dicts so element collection doesn't scan all rels repeatedly.
    Returns (agg_children, void_openings) dicts keyed by entity.
    """
    agg_children: dict = {}
    for rel in model.by_type("IfcRelAggregates"):
        parent = rel.RelatingObject
        agg_children.setdefault(parent, []).extend(rel.RelatedObjects or [])

    void_openings: dict = {}
    for rel in model.by_type("IfcRelVoidsElement"):
        host = rel.RelatingBuildingElement
        if rel.RelatedOpeningElement:
            void_openings.setdefault(host, []).append(rel.RelatedOpeningElement)

    return agg_children, void_openings


def _collect_all_elements(model: ifcopenshell.file, storey,
                          agg_children: dict, void_openings: dict) -> Set:
    """
    Recursively collect all elements and their sub-elements under a storey
    using pre-built indexes for O(1) child lookups.
    """
    elements: Set = set()
    queue = list(_collect_direct_elements(model, storey))
    while queue:
        elem = queue.pop()
        if elem in elements:
            continue
        elements.add(elem)
        for sub in agg_children.get(elem, []):
            if sub not in elements:
                queue.append(sub)
        for opening in void_openings.get(elem, []):
            if opening not in elements:
                queue.append(opening)
    return elements


def _remove_storey_with_elements(model: ifcopenshell.file, storey,
                                  agg_children: dict, void_openings: dict) -> None:
    """
    Remove a storey together with all elements it contains/decomposes and
    all their exclusively-referenced geometry and properties.

    Relationships are severed explicitly first so that remove_deep2 (called
    without also_consider) can quickly determine which geometry entities are
    now exclusively referenced by the element being removed and can be purged.
    """
    elements = _collect_all_elements(model, storey, agg_children, void_openings)

    # --- Sever all relationships that reference this storey or its elements ---

    for rel in list(model.by_type("IfcRelContainedInSpatialStructure")):
        if rel.RelatingStructure == storey:
            model.remove(rel)

    for rel in list(model.by_type("IfcRelReferencedInSpatialStructure")):
        if rel.RelatingStructure == storey:
            model.remove(rel)

    for rel in list(model.by_type("IfcRelAggregates")):
        if rel.RelatingObject == storey or rel.RelatingObject in elements:
            model.remove(rel)

    for rel in list(model.by_type("IfcRelVoidsElement")):
        if rel.RelatingBuildingElement in elements or rel.RelatedOpeningElement in elements:
            model.remove(rel)

    for rel in list(model.by_type("IfcRelFillsElement")):
        if rel.RelatedBuildingElement in elements or rel.RelatingOpeningElement in elements:
            model.remove(rel)

    # Space boundaries can cross storeys (Level-0 wall bounding a Level-1 space).
    # Remove any boundary whose space OR bounding element is being deleted.
    for rel in list(model.by_type("IfcRelSpaceBoundary")):
        relating = getattr(rel, "RelatingSpace", None)
        related  = getattr(rel, "RelatedBuildingElement", None)
        if relating in elements or related in elements:
            model.remove(rel)

    # Relationships that list multiple objects: keep only those not being removed.
    for rel_type in ("IfcRelDefinesByProperties", "IfcRelDefinesByType",
                     "IfcRelAssociatesMaterial", "IfcRelAssignsToGroup",
                     "IfcRelAssociatesConstraint", "IfcRelAssociatesDocument",
                     "IfcRelAssociatesClassification"):
        for rel in list(model.by_type(rel_type)):
            attr = "RelatedObjects" if hasattr(rel, "RelatedObjects") else None
            if attr is None:
                continue
            kept = [e for e in (getattr(rel, attr) or []) if e not in elements]
            if not kept:
                model.remove(rel)
            elif len(kept) < len(list(getattr(rel, attr))):
                setattr(rel, attr, kept)

    # --- Sweep any remaining inverses before deep-removal ---
    # After the explicit passes above, only edge-case rels (e.g. IfcRelServicesBuildings,
    # IfcRelInterferesElements, custom relationships) might still reference the elements.
    # Removing those inverses lets remove_deep2 cleanly purge geometry.
    for elem in list(elements):
        for inv in list(model.get_inverse(elem)):
            try:
                model.remove(inv)
            except Exception:
                pass

    # --- Deep-remove elements (geometry purged because all inverses are gone) ---
    for elem in list(elements):
        try:
            ifcopenshell.util.element.remove_deep2(model, elem)
        except Exception:
            try:
                model.remove(elem)
            except Exception:
                pass

    # Remove storey from the parent building's aggregation list
    for rel in list(model.by_type("IfcRelAggregates")):
        related = list(rel.RelatedObjects or [])
        if storey in related:
            related = [x for x in related if x != storey]
            if related:
                rel.RelatedObjects = related
            else:
                model.remove(rel)

    # Sweep remaining inverses of the storey itself before removing it
    for inv in list(model.get_inverse(storey)):
        try:
            model.remove(inv)
        except Exception:
            pass

    try:
        model.remove(storey)
    except Exception:
        pass


def extract_storey(source_path: str, target_guid: str, output_path: str) -> None:
    """
    Open *source_path* fresh, strip every storey except the one with
    *target_guid*, and write the result to *output_path*.
    """
    model = ifcopenshell.open(source_path)

    all_storeys = model.by_type("IfcBuildingStorey")
    target_storey = next(
        (st for st in all_storeys if st.GlobalId == target_guid), None
    )
    if target_storey is None:
        raise RuntimeError(
            f"Storey with GlobalId '{target_guid}' not found in {source_path}"
        )

    agg_children, void_openings = _build_indexes(model)

    for st in list(all_storeys):
        if st.GlobalId != target_guid:
            _remove_storey_with_elements(model, st, agg_children, void_openings)

    model.write(output_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split an IFC file into one file per IfcBuildingStorey."
    )
    ap.add_argument("input_ifc", help="Path to the input IFC file")
    ap.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory for output files (default: same directory as input)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input_ifc):
        print(f"ERROR: Input file not found: {args.input_ifc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_ifc))
    os.makedirs(output_dir, exist_ok=True)

    model = ifcopenshell.open(args.input_ifc)
    storeys = model.by_type("IfcBuildingStorey")

    if not storeys:
        print("No IfcBuildingStorey entities found in the model.", file=sys.stderr)
        return 1

    base_name = os.path.splitext(os.path.basename(args.input_ifc))[0]

    print(f"Found {len(storeys)} storey(s) in '{args.input_ifc}':")
    for st in storeys:
        name = getattr(st, "Name", None) or "unnamed"
        count = len(_collect_direct_elements(model, st))
        print(f"  [{st.GlobalId}]  {name!r}  —  {count} direct element(s)")

    print()
    for i, storey in enumerate(storeys):
        name = getattr(storey, "Name", None) or f"storey_{i}"
        out_path = os.path.join(output_dir, f"{base_name}__{_safe_name(name)}.ifc")

        print(f"Extracting {name!r}  ->  {out_path} ...", end=" ", flush=True)
        extract_storey(args.input_ifc, storey.GlobalId, out_path)
        print("done.")

    print(f"\n{len(storeys)} file(s) written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
