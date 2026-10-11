"""Builds public/data/places.json: Atlanta's public places and responders
(police, fire, hospitals, schools, parks, venues...) from OpenStreetMap via
the Overpass API. Bounding box covers MARTA's service area.
"""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public/data/places.json"
BBOX = "33.55,-84.62,34.08,-84.13"  # south, west, north, east

# category -> Overpass tag filters
CATEGORIES = {
    "police": ['["amenity"="police"]'],
    "fire": ['["amenity"="fire_station"]'],
    "hospital": ['["amenity"="hospital"]'],
    "library": ['["amenity"="library"]'],
    "university": ['["amenity"~"^(university|college)$"]'],
    "school": ['["amenity"="school"]'],
    "stadium": ['["leisure"="stadium"]'],
    "park": ['["leisure"="park"]'],
    "mall": ['["shop"="mall"]'],
    "venue": ['["amenity"~"^(theatre|events_venue|arts_centre|community_centre)$"]', '["tourism"~"^(museum|attraction)$"]'],
}


# The main instance is often overloaded (504); these are the public mirrors
# listed on the OSM wiki.
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def overpass(query: str) -> dict:
    # curl rather than urllib: this machine's python.org build has no CA bundle
    # installed, and verification should stay on.
    errors = []
    for attempt in range(2):
        for url in ENDPOINTS:
            out = subprocess.run(
                ["curl", "-sS", "--fail", "--max-time", "180",
                 "-A", "eyes-hacklanta/0.1 (hackathon demo)", "-H", "Accept: application/json",
                 "--data-urlencode", f"data={query}", url],
                capture_output=True, text=True,
            )
            if out.returncode == 0:
                try:
                    return json.loads(out.stdout)
                except json.JSONDecodeError:
                    pass
            errors.append(f"{url}: {out.stderr.strip()[:120]}")
        time.sleep(10)
    raise RuntimeError("all Overpass endpoints failed:\n" + "\n".join(errors))


CACHE = ROOT / "data/raw/places"


def tiles(bbox: str, n: int):
    s, w, nth, e = map(float, bbox.split(","))
    for i in range(n):
        for j in range(n):
            yield f"{s + (nth - s) * i / n:.4f},{w + (e - w) * j / n:.4f},{s + (nth - s) * (i + 1) / n:.4f},{w + (e - w) * (j + 1) / n:.4f}"


def fetch_category(cat: str, filters: list[str]) -> list[dict]:
    # Cached per category so a busy-server failure doesn't redo finished work.
    # Big categories (thousands of schools/parks) time out as one query, so
    # those go in 2x2 tiles.
    cached = CACHE / f"{cat}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    elements = []
    for box in tiles(BBOX, 2 if cat in ("school", "park") else 1):
        body = "".join(f"nwr{f}({box});" for f in filters)
        elements += overpass(f"[out:json][timeout:120];({body});out center tags;")["elements"]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(elements))
    return elements


def main():
    places, seen = [], set()
    for cat, filters in CATEGORIES.items():
        n = 0
        for el in fetch_category(cat, filters):
            name = el.get("tags", {}).get("name")
            lat = el.get("lat", el.get("center", {}).get("lat"))
            lon = el.get("lon", el.get("center", {}).get("lon"))
            if not name or lat is None:
                continue  # unnamed features are mostly mapping fragments
            key = (cat, name, round(lat, 3), round(lon, 3))
            if key in seen:
                continue
            seen.add(key)
            places.append([round(lat, 5), round(lon, 5), name, cat])
            n += 1
        print(f"{cat:11s} {n}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "OpenStreetMap contributors (ODbL), via Overpass", "places": places}, separators=(",", ":")))
    print(f"{len(places)} places -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
