"""Builds public/data/marta.json from MARTA's public GTFS feed.

Input:  data/raw/marta_gtfs/ (unzipped https://itsmarta.com/google_transit_feed/google_transit.zip)
Output: public/data/marta.json — every route's main path per direction (the
shape most trips use, simplified) and every stop, small enough to load in
the browser.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GTFS = ROOT / "data/raw/marta_gtfs"
OUT = ROOT / "public/data/marta.json"

# ~8 m at Atlanta's latitude: invisible at city zoom, cuts points ~10x.
SIMPLIFY_DEG = 0.00008


def read(name):
    with open(GTFS / name, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def perp_dist(p, a, b):
    if a == b:
        return math.dist(p, a)
    (x, y), (x1, y1), (x2, y2) = p, a, b
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)))
    return math.dist(p, (x1 + t * (x2 - x1), y1 + t * (y2 - y1)))


def simplify(pts, eps):
    # Iterative Douglas-Peucker (shapes run to thousands of points; no recursion limit issues).
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        best, idx = 0.0, -1
        for k in range(i + 1, j):
            d = perp_dist(pts[k], pts[i], pts[j])
            if d > best:
                best, idx = d, k
        if best > eps:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(pts, keep) if k]


def main():
    routes = {r["route_id"]: r for r in read("routes.txt")}

    # The shape most trips use, per route and direction, is that route's real path.
    shape_use = defaultdict(Counter)
    for t in read("trips.txt"):
        if t["shape_id"]:
            shape_use[(t["route_id"], t.get("direction_id", "0"))][t["shape_id"]] += 1
    wanted = {c.most_common(1)[0][0]: key for key, c in shape_use.items()}

    shape_pts = defaultdict(list)
    for s in read("shapes.txt"):
        if s["shape_id"] in wanted:
            shape_pts[s["shape_id"]].append((int(s["shape_pt_sequence"]), float(s["shape_pt_lat"]), float(s["shape_pt_lon"])))

    out_routes = []
    for shape_id, (route_id, direction) in wanted.items():
        r = routes.get(route_id)
        pts = [(lat, lon) for _, lat, lon in sorted(shape_pts[shape_id])]
        if not r or len(pts) < 2:
            continue
        pts = simplify(pts, SIMPLIFY_DEG)
        out_routes.append({
            "id": route_id,
            "dir": int(direction or 0),
            "short": r["route_short_name"],
            "name": r["route_long_name"],
            # GTFS route_type: 1 = subway/rail, 3 = bus
            "type": "rail" if r["route_type"] in ("0", "1", "2") else "bus",
            "color": "#" + (r["route_color"] or "4fd1c5"),
            "path": [[round(lat, 5), round(lon, 5)] for lat, lon in pts],
        })

    out_stops = []
    for s in read("stops.txt"):
        if s.get("location_type") not in ("", "0", None):
            continue  # station parents/entrances; the boarding stops are enough
        out_stops.append([
            round(float(s["stop_lat"]), 5),
            round(float(s["stop_lon"]), 5),
            s["stop_name"].title(),
            1 if "STATION" in s["stop_name"].upper() else 0,
        ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "MARTA GTFS", "routes": out_routes, "stops": out_stops}, separators=(",", ":")))
    rail = sum(1 for r in out_routes if r["type"] == "rail")
    print(f"{len(out_routes)} route paths ({rail} rail), {len(out_stops)} stops, "
          f"{sum(len(r['path']) for r in out_routes)} path points -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
