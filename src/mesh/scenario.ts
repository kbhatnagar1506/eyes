// Where things are on the map. The scene is fixed at GSU Student Center East;
// ?loc=gps instead puts the incident wherever the camera device really is
// (browser geolocation) and each device at its own real position. Responders
// are the real nearest police / fire / hospital from OpenStreetMap.

import { metersBetween } from "./radio";
import { dataUrl } from "./media";

export type PlaceCategory =
  | "police"
  | "fire"
  | "hospital"
  | "library"
  | "university"
  | "school"
  | "stadium"
  | "park"
  | "mall"
  | "venue";

export interface Place {
  lat: number;
  lng: number;
  name: string;
  cat: PlaceCategory;
}

export interface MartaRoute {
  id: string;
  dir: number;
  short: string;
  name: string;
  type: "rail" | "bus";
  color: string;
  path: [number, number][];
}

export interface MartaData {
  routes: MartaRoute[];
  stops: [number, number, string, number][]; // lat, lng, name, isStation
}

export interface LiveBus {
  id: string;
  route: string | null;
  lat: number;
  lng: number;
  bearing: number | null;
  ts: number;
}

export interface Scene {
  lat: number;
  lng: number;
  label: string;
  source: "gps" | "fixed";
  accuracyM?: number;
}

// The demo is pinned to GSU Student Center East (55 Gilmer St SE, from
// OpenStreetMap). ?loc=gps uses each device's real location instead.
const FIXED_SCENE: Scene = { lat: 33.75258, lng: -84.38479, label: "GSU Student Center East · 55 Gilmer St SE", source: "fixed" };
const USE_GPS = typeof location !== "undefined" && new URLSearchParams(location.search).get("loc") === "gps";
let currentScene: Scene = FIXED_SCENE;
export const scene = () => currentScene;

// MARTA headquarters, 2424 Piedmont Rd NE: where the dispatch gateway "sits".
export const GATEWAY_POS = { lat: 33.8205, lng: -84.3687 };

// A fix farther than this from downtown is outside the data we have
// (MARTA + Atlanta places), e.g. testing from home: use the default scene.
const ATLANTA = { lat: 33.749, lng: -84.388 };
const MAX_FROM_ATLANTA_M = 40_000;

export interface Fix {
  lat: number;
  lng: number;
  accuracyM: number;
}

// Browsers only give location on a secure origin (https or localhost) and
// after the user allows it; any failure resolves to null, never throws.
export function getFix(timeoutMs = 6000): Promise<Fix | null> {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lng: p.coords.longitude, accuracyM: p.coords.accuracy }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: timeoutMs, maximumAge: 10_000 },
    );
  });
}

export function watchFix(cb: (f: Fix) => void): () => void {
  if (!("geolocation" in navigator)) return () => {};
  const id = navigator.geolocation.watchPosition(
    (p) => cb({ lat: p.coords.latitude, lng: p.coords.longitude, accuracyM: p.coords.accuracy }),
    () => {},
    { enableHighAccuracy: true, maximumAge: 5_000 },
  );
  return () => navigator.geolocation.clearWatch(id);
}

export const inAtlanta = (f: { lat: number; lng: number }) => metersBetween(f, ATLANTA) <= MAX_FROM_ATLANTA_M;

// "Near North Ave Station" style label from the closest MARTA stop: no
// geocoding service needed, and it's how riders name places anyway.
export async function describeLocation(at: { lat: number; lng: number }): Promise<string> {
  const { stops } = await loadMarta();
  let best: [number, number, string, number] | null = null;
  let bestM = Infinity;
  for (const s of stops) {
    const m = metersBetween({ lat: s[0], lng: s[1] }, at);
    if (m < bestM) {
      best = s;
      bestM = m;
    }
  }
  return best ? `Near ${best[2]} (${Math.round(bestM)} m)` : "Atlanta";
}

export function moveScene(f: Fix) {
  if (currentScene.source === "gps") currentScene = { ...currentScene, lat: f.lat, lng: f.lng, accuracyM: f.accuracyM };
}

// With ?loc=gps the camera's real position becomes the scene; otherwise the
// fixed scene stands. Returns the scene used.
export async function locateScene(): Promise<Scene> {
  if (!USE_GPS) return currentScene;
  const fix = await getFix();
  if (fix && inAtlanta(fix)) {
    currentScene = { lat: fix.lat, lng: fix.lng, accuracyM: fix.accuracyM, source: "gps", label: await describeLocation(fix) };
  }
  return currentScene;
}

let martaP: Promise<MartaData> | null = null;
let placesP: Promise<Place[]> | null = null;

// Set when a dataset couldn't load (expired signed link, blocked request),
// so the map can say so instead of silently drawing nothing.
export const dataProblems: string[] = [];

async function fetchJson<T>(name: "marta" | "places"): Promise<T> {
  const r = await fetch(await dataUrl(name));
  if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`);
  return (await r.json()) as T;
}

// Never rejects: a failed load yields empty data (and is retried next call),
// so the map, the devices and a phone joining the mesh keep working.
export function loadMarta(): Promise<MartaData> {
  return (martaP ??= fetchJson<MartaData>("marta").catch((err) => {
    martaP = null;
    dataProblems.push(`MARTA routes and stops: ${String(err)}`);
    return { routes: [], stops: [] };
  }));
}

// OSM often maps one building twice (a node and an outline); keep one per spot.
export function loadPlaces(): Promise<Place[]> {
  return (placesP ??= fetchJson<{ places: [number, number, string, PlaceCategory][] }>("places")
    .then((d) => {
      const kept: Place[] = [];
      for (const [lat, lng, name, cat] of d.places) {
        const p = { lat, lng, name, cat };
        if (kept.some((k) => k.cat === cat && metersBetween(k, p) < 60)) continue;
        kept.push(p);
      }
      return kept;
    })
    .catch((err) => {
      placesP = null;
      dataProblems.push(`Public places: ${String(err)}`);
      return [] as Place[];
    }));
}

export function nearest(places: Place[], cat: PlaceCategory, from: { lat: number; lng: number } = currentScene): Place | null {
  let best: Place | null = null;
  let bestM = Infinity;
  for (const p of places) {
    if (p.cat !== cat) continue;
    const m = metersBetween(p, from);
    if (m < bestM) {
      best = p;
      bestM = m;
    }
  }
  return best;
}

// Problems the simulator can stage at a pin, and who responds to each
// (nearest first). `words` lets the chat read "fire at Five Points".
export type ProblemId = "assault" | "robbery" | "medical" | "fire" | "harassment" | "package";

export const PROBLEMS: Record<ProblemId, { label: string; icon: string; responders: PlaceCategory[]; words: string[] }> = {
  assault: { label: "Assault", icon: "🥊", responders: ["police", "hospital"], words: ["assault", "fight", "punch", "attack", "beating"] },
  robbery: { label: "Robbery", icon: "💰", responders: ["police"], words: ["robbery", "rob", "theft", "mugging", "stolen", "steal"] },
  medical: { label: "Medical emergency", icon: "🚑", responders: ["hospital", "fire"], words: ["medical", "collapse", "faint", "injured", "ambulance", "heart", "overdose"] },
  fire: { label: "Fire", icon: "🔥", responders: ["fire", "hospital"], words: ["fire", "smoke", "burning", "flames"] },
  harassment: { label: "Harassment", icon: "⚠️", responders: ["police"], words: ["harassment", "harass", "stalking", "following", "threat"] },
  package: { label: "Suspicious package", icon: "🎒", responders: ["police", "fire"], words: ["package", "bag", "suspicious", "unattended", "bomb"] },
};

export function problemLabel(kind: unknown): string {
  return PROBLEMS[kind as ProblemId]?.label ?? String(kind ?? "incident");
}

export function problemFromText(text: string): ProblemId | null {
  const t = text.toLowerCase();
  for (const [id, p] of Object.entries(PROBLEMS) as [ProblemId, (typeof PROBLEMS)[ProblemId]][]) {
    if (p.words.some((w) => new RegExp(`\\b${w}`).test(t))) return id;
  }
  return null;
}

export type Role = "camera" | "driver" | "rider" | "police" | "gateway";

export const ROLE_LABEL: Record<Role, string> = {
  camera: "CCTV camera",
  driver: "Bus driver",
  rider: "Rider",
  police: "Police",
  gateway: "Dispatch AI",
};

export interface Position {
  lat: number;
  lng: number;
  place?: string;
  source: "gps" | "fixed";
}

// Where each real device sits on the map. Fixed by default: camera at the
// scene, driver and rider right beside it, police at the real police station
// nearest the scene. With ?loc=gps, each device's own fix (if in Atlanta).
export async function positionFor(role: Role): Promise<Position> {
  if (role === "gateway") return { ...GATEWAY_POS, source: "fixed" };
  if (role === "camera") {
    const s = await locateScene();
    return { lat: s.lat, lng: s.lng, source: s.source };
  }
  if (USE_GPS) {
    const fix = await getFix();
    if (fix && inAtlanta(fix)) return { lat: fix.lat, lng: fix.lng, source: "gps" };
  }
  const s = currentScene;
  switch (role) {
    case "driver":
      return { lat: s.lat + 0.00003, lng: s.lng - 0.00004, source: "fixed" };
    case "rider":
      return { lat: s.lat - 0.00004, lng: s.lng + 0.00005, source: "fixed" };
    case "police": {
      const p = nearest(await loadPlaces(), "police");
      return p ? { lat: p.lat, lng: p.lng, place: p.name, source: "fixed" } : { lat: 33.7544, lng: -84.389, source: "fixed" };
    }
  }
}

export async function loadLiveBuses(url: string): Promise<LiveBus[]> {
  try {
    const r = await fetch(url);
    return (await r.json()).vehicles ?? [];
  } catch {
    return [];
  }
}
