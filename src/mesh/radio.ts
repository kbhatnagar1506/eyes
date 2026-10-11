// Simulated short-range radio between city nodes (buses, stations, precincts)
// that don't have a real device in the room. The nodes run the real protocol
// in node.ts; only the radio is simulated: an in-memory link with
// distance-based delay, formed only between nodes within range of each other.

import type { MeshNode, Wire } from "./node";

const METERS_PER_DEG_LAT = 111_320;

export function metersBetween(a: { lat: number; lng: number }, b: { lat: number; lng: number }): number {
  const dy = (a.lat - b.lat) * METERS_PER_DEG_LAT;
  const dx = (a.lng - b.lng) * METERS_PER_DEG_LAT * Math.cos(((a.lat + b.lat) / 2) * (Math.PI / 180));
  return Math.hypot(dx, dy);
}

// Per-hop delay of a store-and-forward mesh (receive, check, re-broadcast):
// real BLE / Wi-Fi Direct mesh hops take tens to a few hundred ms. Grows
// with distance (weaker signal, retries), with jitter. Also slow enough that
// the wave is visible on the stage map.
function hopDelayMs(m: number) {
  return 80 + m / 15 + Math.random() * 60;
}

export function radioLink(a: MeshNode, b: MeshNode, opts: { exclude?: ReadonlySet<string> } = {}) {
  const m = metersBetween(a.info, b.info);
  const deliver = (from: MeshNode, to: MeshNode) => (w: Wire) => {
    const copy = JSON.parse(JSON.stringify(w)) as Wire; // nothing shared between "devices"
    setTimeout(() => to.receive(from.id, "radio", copy), hopDelayMs(m));
  };
  a.addLink(b.id, "radio", deliver(a, b), opts);
  b.addLink(a.id, "radio", deliver(b, a), opts);
}

// Links every node to its nearest neighbors within range. Capping the degree
// keeps the city mesh readable and the traffic sane.
export function buildRadioMesh(nodes: MeshNode[], rangeM: number, maxDegree = 4): [MeshNode, MeshNode][] {
  const pairs: [MeshNode, MeshNode][] = [];
  const degree = new Map<string, number>();
  const linked = new Set<string>();
  for (const a of nodes) {
    const near = nodes
      .filter((b) => b !== a)
      .map((b) => ({ b, m: metersBetween(a.info, b.info) }))
      .filter((x) => x.m <= rangeM)
      .sort((x, y) => x.m - y.m);
    for (const { b } of near) {
      if ((degree.get(a.id) ?? 0) >= maxDegree) break;
      if ((degree.get(b.id) ?? 0) >= maxDegree + 2) continue;
      const key = a.id < b.id ? `${a.id}|${b.id}` : `${b.id}|${a.id}`;
      if (linked.has(key)) continue;
      linked.add(key);
      radioLink(a, b);
      pairs.push([a, b]);
      degree.set(a.id, (degree.get(a.id) ?? 0) + 1);
      degree.set(b.id, (degree.get(b.id) ?? 0) + 1);
    }
  }
  return pairs;
}
