// Turns a typed place ("Five Points", "Grady hospital", "55 Gilmer St") into
// a point on the map. Our own data first (public places + MARTA stops, no
// network needed); street addresses fall back to OpenStreetMap's Nominatim,
// limited to the Atlanta area.

import { loadMarta, loadPlaces, type MartaData, type Place, type PlaceCategory } from "./scenario";

export interface GeoResult {
  lat: number;
  lng: number;
  name: string;
  kind: PlaceCategory | "stop" | "station" | "address";
  source: "local" | "osm";
}

// Same bounds as the places data (scripts/build_places.py).
const VIEWBOX = "-84.62,34.08,-84.13,33.55";

const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/\b(st|street)\b/g, "st")
    .replace(/\b(ave|avenue)\b/g, "ave")
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

// Share of query words found in the name, nudged toward names close in length
// (so "five points" picks "Five Points Station", not a long name that
// happens to contain both words).
function score(query: string[], name: string): number {
  const words = norm(name).split(" ");
  const hits = query.filter((q) => words.some((w) => w === q || (q.length >= 3 && w.startsWith(q)))).length;
  if (hits < query.length) return 0;
  return 1 - Math.min(0.5, Math.abs(words.length - query.length) * 0.05);
}

async function searchLocal(q: string): Promise<GeoResult | null> {
  const [places, marta] = await Promise.all([loadPlaces(), loadMarta()]);
  return matchLocal(q, places, marta.stops);
}

// The local search on its own, for callers that already have the data (the
// agent service reads it from disk).
export function matchLocal(q: string, places: Place[], stops: MartaData["stops"]): GeoResult | null {
  const query = norm(q).split(" ").filter(Boolean);
  if (!query.length) return null;
  const marta = { stops };
  let best: GeoResult | null = null;
  let bestScore = 0;
  const consider = (r: GeoResult, s: number) => {
    if (s > bestScore) {
      best = r;
      bestScore = s;
    }
  };
  for (const p of places) consider({ lat: p.lat, lng: p.lng, name: p.name, kind: p.cat, source: "local" }, score(query, p.name));
  for (const [lat, lng, name, isStation] of marta.stops) {
    // A small bump so "Five Points" lands on the station, not a bus stop named after it.
    consider({ lat, lng, name, kind: isStation ? "station" : "stop", source: "local" }, score(query, name) * (isStation ? 1.02 : 0.98));
  }
  return best;
}

export async function searchOsm(q: string): Promise<GeoResult | null> {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&bounded=1&viewbox=${VIEWBOX}&q=${encodeURIComponent(q)}`;
  try {
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    const [hit] = (await r.json()) as { lat: string; lon: string; display_name: string }[];
    if (!hit) return null;
    return { lat: Number(hit.lat), lng: Number(hit.lon), name: hit.display_name.split(",").slice(0, 3).join(","), kind: "address", source: "osm" };
  } catch {
    return null;
  }
}

// Anything with a house number reads as an address: OSM first. Otherwise our
// data first, OSM only if nothing matched.
export async function geocode(q: string): Promise<GeoResult | null> {
  const looksLikeAddress = /^\s*\d+\s+\w/.test(q);
  if (looksLikeAddress) return (await searchOsm(q)) ?? (await searchLocal(q));
  return (await searchLocal(q)) ?? (await searchOsm(q));
}
