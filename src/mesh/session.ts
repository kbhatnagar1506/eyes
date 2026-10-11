// This page's device on the mesh: one node, with the hub relay and direct
// P2P links, plus the mesh-wide outage switches used on stage.

import { MeshNode, type TransportKind } from "./node";
import { stampTime } from "./hlc";
import type { Op } from "./store";
import { HubClient } from "./hub-client";
import { P2PManager } from "./p2p";
import { inAtlanta, moveScene, positionFor, ROLE_LABEL, watchFix, type Position, type Role } from "./scenario";

// Where the mesh hub is. Locally it's proxied through this page's own origin
// (vite.config.ts). On the hosted site (Vercel) it's the laptop's hub behind a
// tunnel: the join links carry it as ?hub=wss://…, and each device remembers
// it, so a reload or a later visit still finds the mesh.
const HUB_KEY = "eyes-hub";

function hubBase(): string | null {
  const fromLink = new URLSearchParams(location.search).get("hub");
  if (fromLink && /^(wss?|https?):\/\//.test(fromLink)) {
    try {
      localStorage.setItem(HUB_KEY, fromLink);
    } catch {
      /* not persisted: fine for this visit */
    }
    return fromLink;
  }
  try {
    const saved = localStorage.getItem(HUB_KEY);
    if (saved) return saved;
  } catch {
    /* storage unavailable */
  }
  return (import.meta.env.VITE_HUB_URL as string | undefined) || null;
}

const hubWs = () => {
  const base = hubBase();
  if (!base) return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/mesh`;
  return `${base.replace(/^http/, "ws").replace(/\/+$/, "")}/mesh`;
};

export const hubUrl = () => hubBase();

// Live MARTA buses: a Vercel function on the hosted site, the hub locally.
export const busesUrl = () => (import.meta.env.PROD ? "/api/buses" : `${location.origin}/hub/buses`);

export const OUTAGE_DOC = { c: "control", id: "outage" };
export const SWITCHABLE: TransportKind[] = ["relay", "p2p", "radio"];

export interface Session {
  node: MeshNode;
  hub: HubClient;
  p2p: P2PManager;
  role: Role;
  position: Position;
}

// Same identity across reloads of one tab, so a reload doesn't leave a ghost.
function stableId(role: Role): string {
  const key = `eyes-node-${role}`;
  try {
    const existing = sessionStorage.getItem(key);
    if (existing) return existing;
    const id = `${role}-${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem(key, id);
    return id;
  } catch {
    return `${role}-${Math.random().toString(36).slice(2, 8)}`;
  }
}

export async function joinMesh(role: Role, name?: string): Promise<Session> {
  const pos = await positionFor(role);
  const node = new MeshNode(
    { name: name ?? (pos.place ? `${ROLE_LABEL[role]} · ${pos.place}` : ROLE_LABEL[role]), role, lat: pos.lat, lng: pos.lng },
    { id: stableId(role) },
  );
  restore(node, role);
  persist(node, role);
  const hub = new HubClient(node, hubWs());
  const p2p = new P2PManager(node, hub);

  // A device with a real fix keeps reporting where it is (the bus moves).
  if (pos.source === "gps") {
    watchFix((f) => {
      if (!inAtlanta(f)) return;
      node.info.lat = f.lat;
      node.info.lng = f.lng;
      if (role === "camera") moveScene(f);
    });
  }

  // Outages are a property of the network, not one device: a switch flipped
  // on any device syncs to all of them, and each stops using that transport.
  const applyOutage = () => {
    const doc = node.store.get(OUTAGE_DOC.c, OUTAGE_DOC.id);
    for (const k of SWITCHABLE) node.setDisabled(k, !!doc?.[k]);
  };
  node.store.onChange((c, id) => {
    if (c === OUTAGE_DOC.c && id === OUTAGE_DOC.id) applyOutage();
  });
  return { node, hub, p2p, role, position: pos };
}

// Each device keeps its copy of the mesh data across reloads, the way a
// Ditto-style store persists on disk: a refresh (or a dead battery) doesn't
// lose incidents, and whatever this device knew syncs back out on reconnect.
// Presence is live-only and isn't kept. Storage may be unavailable (private
// mode, quota): then it's simply not persisted.
const STORE_KEY = (role: Role) => `eyes-mesh-${role}`;
const SAVE_DEBOUNCE_MS = 1000;
const KEEP_MS = 24 * 3600_000;

function restore(node: MeshNode, role: Role) {
  try {
    const raw = localStorage.getItem(STORE_KEY(role));
    if (!raw) return;
    for (const op of JSON.parse(raw) as Op[]) {
      node.hlc.observe(op.t);
      node.store.apply(op);
    }
  } catch {
    /* unreadable or unavailable: start empty */
  }
}

function persist(node: MeshNode, role: Role) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const save = () => {
    timer = null;
    const cutoff = Date.now() - KEEP_MS;
    const ops = node.store.snapshot().filter((op) => op.c !== "presence" && stampTime(op.t) > cutoff);
    try {
      localStorage.setItem(STORE_KEY(role), JSON.stringify(ops));
    } catch {
      // Over quota: frames are the bulk, keep everything else.
      try {
        localStorage.setItem(STORE_KEY(role), JSON.stringify(ops.filter((op) => op.c !== "frames")));
      } catch {
        /* storage unavailable */
      }
    }
  };
  node.store.onChange((c) => {
    if (c === "presence" || timer) return;
    timer = setTimeout(save, SAVE_DEBOUNCE_MS);
  });
}

export function setOutage(node: MeshNode, kind: TransportKind, down: boolean) {
  node.write(OUTAGE_DOC.c, OUTAGE_DOC.id, { [kind]: down });
}
