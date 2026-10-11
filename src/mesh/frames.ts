// Evidence frames: one JPEG per detected punch, each its own mesh document
// ("frames"/"<incident>#<seq>") so later punches add to the same incident.

import type { Store } from "./store";

// Frames come from other devices: only ever render a real base64 JPEG, so a
// rogue device can't inject markup through an image attribute.
export function safeJpeg(v: unknown): string | null {
  return typeof v === "string" && /^data:image\/jpeg;base64,[A-Za-z0-9+/]+=*$/.test(v) ? v : null;
}

// Frames for an incident, oldest first (also reads the older single-frame layout).
export function framesFor(store: Store, incidentId: string) {
  return store
    .list("frames")
    .filter((f) => f.incident === incidentId || f.id === incidentId)
    .sort((a, b) => Number(a.seq ?? 1) - Number(b.seq ?? 1));
}
