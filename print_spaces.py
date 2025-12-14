#!/usr/bin/env python3
import argparse, sys, csv
from typing import List
try:
    import ifcopenshell
except Exception:
    print("Needs ifcopenshell. pip install ifcopenshell", file=sys.stderr)
    sys.exit(1)

BOUNDARY_TYPES = (
    "IfcRelSpaceBoundary2ndLevel",
    "IfcRelSpaceBoundary1stLevel",
    "IfcRelSpaceBoundary",
)

def safe(a, name, default=None):
    try:
        return getattr(a, name)
    except Exception:
        return default

def guid_of(ent):
    try:
        return ent.GlobalId
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser("List linked space boundaries per IfcSpace")
    ap.add_argument("ifc", help="Path to IFC")
    ap.add_argument("--csv", help="Optional CSV to write")
    args = ap.parse_args()

    ifc = ifcopenshell.open(args.ifc)
    spaces = ifc.by_type("IfcSpace")

    rows: List[List[str]] = []
    any_linked = False

    for s in spaces:
        s_guid = guid_of(s) or f"#{s.id()}"
        s_name = safe(s, "Name", "") or ""
        # gather boundaries that reference this space
        bnds = []
        for t in BOUNDARY_TYPES:
            for rsb in ifc.by_type(t):
                rel_sp = safe(rsb, "RelatingSpace")
                if rel_sp and (rel_sp.id() == s.id()):
                    bnds.append(rsb)

        linked = []
        unlinked = []
        for b in bnds:
            bel = safe(b, "RelatedBuildingElement")
            pv  = safe(b, "PhysicalOrVirtualBoundary", None)
            ie  = safe(b, "InternalOrExternalBoundary", None)
            lvl = b.is_a().replace("IfcRelSpaceBoundary", "L")
            if bel:
                any_linked = True
                linked.append((b, bel, pv, ie, lvl))
            else:
                # could be space-to-space (CorrespondingBoundary) or from virtual separators
                cb = safe(b, "CorrespondingBoundary", None)
                other_space = guid_of(safe(cb, "RelatingSpace")) if cb else None
                unlinked.append((b, pv, ie, lvl, other_space))

        print(f"\nIfcSpace {s} {s_name!s} <{s_guid}>")
        print(f"  Boundaries total: {len(bnds)} | linked to element: {len(linked)} | no element: {len(unlinked)}")

        if linked:
            print("  Linked boundaries:")
            for b, el, pv, ie, lvl in linked:
                print(f"    - {lvl}  #{b.id()}  PV={pv}  IE={ie}  -> {el.is_a()}  {guid_of(el)}  “{safe(el,'Name','') or ''}”")
                # CSV row
                rows.append([
                    s_guid, s_name, str(s.id()),
                    lvl, str(b.id()), str(pv), str(ie),
                    guid_of(el) or "", el.is_a() if el else "", safe(el, "Name", "") or ""
                ])
        else:
            print("  (No boundaries with RelatedBuildingElement in this space.)")

        if unlinked:
            print("  Unlinked (space-to-space / virtual) boundaries:")
            for b, pv, ie, lvl, other in unlinked:
                tail = f" (corresponds to space {other})" if other else ""
                print(f"    - {lvl}  #{b.id()}  PV={pv}  IE={ie}{tail}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "space_guid","space_name","space_express_id",
                "boundary_level","boundary_express_id","physical_or_virtual","internal_or_external",
                "element_guid","element_ifc_type","element_name"
            ])
            w.writerows(rows)
        print(f"\nWrote CSV with {len(rows)} linked boundaries: {args.csv}")

    if not any_linked:
        print("\n⚠ No boundaries in this file point to a RelatedBuildingElement.")
        print("   That means your exporter generated only space-to-space/virtual boundaries.")
        print("   To get element links, ensure in Revit: elements are Room Bounding, Volume Computations on,")
        print("   and IFC export uses Space boundaries = 2nd level.")

if __name__ == "__main__":
    main()
