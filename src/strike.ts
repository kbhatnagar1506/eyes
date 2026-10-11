// Strike detection straight off the pose mesh: each person gets a body
// hitbox (torso capsule + head circle). A strike fires when one person's hand
// enters ANOTHER person's hitbox while moving fast. Deterministic, per-frame,
// no model calls.
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

const LM = { nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lHip: 23, rHip: 24 };

// Everything is scaled by torso length (shoulder-mid to hip-mid), not
// shoulder width: shoulder width collapses to ~0 when someone stands side-on
// to the camera, torso length doesn't. Speeds are torso-lengths per second.
// Calm motion (walking, guards up, gesturing) measured ≤2.4 on labeled
// footage; controlled sparring jabs peaked ~3.2–3.6. Contact with another
// body is also required, so 3.0 doesn't fire on fast-but-harmless motion.
const SPEED_THRESH = 3.0;
// New tracks jitter for a few frames while landmarks settle.
const WARMUP_MS = 300;
// A punch is the elbow snapping from bent to straight. That signal survives
// when the wrist landmark lags or blurs, and when the punch goes toward the
// camera (little 2D travel). Calm arm motion stays well under this.
const EXT_RATE_THRESH = 350; // degrees per second of elbow opening
const EXT_MIN_SPEED = 1.5; // hand still has to be moving, not just a flicker
// A real punch moves ~0.5 torso-lengths per frame at 25fps at most; a bigger
// jump between samples is a tracking glitch (hand swapped to another person,
// landmark flicker), so history restarts instead of reading as speed.
const MAX_STEP = 1.0;
const MAX_PLAUSIBLE_SPEED = 25;
// A fist decelerates as it lands, so the speed AT contact understates the
// punch. The motion gate uses the peak over this lookback instead (measured:
// a chest punch read 2.6 torso/s one frame before contact, 2.2 on contact).
const PEAK_MS = 200;
// How much deeper (in hitbox radii) a just-appeared hand must move to count
// as arriving rather than already resting there.
const APPROACH_MIN = 0.1;
// 2 samples (one frame-to-frame step) so speed still reads on slower machines;
// MAX_STEP above guards against single-frame glitches.
const MIN_WINDOW_SAMPLES = 2;
const VIS = 0.5;
// Fast-moving wrists blur, and MediaPipe reports them at low visibility exactly
// mid-punch — a stricter cutoff drops the punching hand at the worst moment.
const HAND_VIS = 0.25;
const HISTORY_MS = 400;
const SPEED_WINDOW_MS = 250;
const TRACK_TIMEOUT_MS = 600;
// At the moment of impact the two bodies overlap and pose detection often
// drops the person being hit for a few frames. Their last hitbox stays live
// this long so the incoming hand still has something to hit.
const COAST_MS = 450;
const PAIR_COOLDOWN_MS = 1500;
const HIT_FLASH_MS = 700;

type P = { x: number; y: number };
// p = absolute screen position (for contact); rel = position relative to the
// person's own shoulders (for speed, so camera pans and walking cancel out).
type Sample = { t: number; p: P; rel: P; elbowDeg: number | null; speed: number; extRate: number };
type Side = "left" | "right";
type Capsule = { a: P; b: P; r: number };
type Target = "head" | "torso" | "arm";

export interface Hitbox {
  personId: number;
  // Torso is a capsule around the spine segment, so it keeps real width at
  // any body rotation (a 4-point shoulder/hip polygon goes flat side-on).
  torso: Capsule;
  head: { c: P; r: number };
  // Upper arms: a punch that lands on the shoulder or arm is still a hit, and
  // side-on the near arm covers most of the torso from the camera's view.
  arms: Capsule[];
  centroid: P;
  scale: number;
}

export interface HandState {
  personId: number;
  side: Side;
  p: P;
  speed: number;
  extRate: number;
  peakSpeed: number;
  peakExtRate: number;
}

export interface Strike {
  attackerId: number;
  victimId: number;
  side: Side;
  speed: number;
  target: Target;
  at: P;
}

interface Track {
  id: number;
  centroid: P;
  scale: number;
  firstSeen: number;
  lastSeen: number;
  lastBox: Hitbox | null;
  hands: Record<Side, Sample[]>;
}

const dist = (a: P, b: P) => Math.hypot(a.x - b.x, a.y - b.y);

function angleDeg(a: P, vertex: P, b: P): number {
  const v1 = { x: a.x - vertex.x, y: a.y - vertex.y };
  const v2 = { x: b.x - vertex.x, y: b.y - vertex.y };
  const cos = (v1.x * v2.x + v1.y * v2.y) / (Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

function distToSegment(p: P, a: P, b: P): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  const u = len2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2)) : 0;
  return dist(p, { x: a.x + u * dx, y: a.y + u * dy });
}

// How deep a point sits in a body: distance over radius for the nearest part,
// so ≤ 1 is inside and smaller is deeper.
function depth(pt: P, box: Hitbox): number {
  return Math.min(
    dist(pt, box.head.c) / box.head.r,
    distToSegment(pt, box.torso.a, box.torso.b) / box.torso.r,
    ...box.arms.map((c) => distToSegment(pt, c.a, c.b) / c.r),
  );
}

function hitTarget(pt: P, box: Hitbox): Target | null {
  if (dist(pt, box.head.c) <= box.head.r) return "head";
  if (distToSegment(pt, box.torso.a, box.torso.b) <= box.torso.r) return "torso";
  if (box.arms.some((c) => distToSegment(pt, c.a, c.b) <= c.r)) return "arm";
  return null;
}

export class StrikeDetector {
  private tracks: Track[] = [];
  private nextId = 1;
  private pairCooldown = new Map<string, number>();
  private recentHits = new Map<number, number>();

  hitboxes: Hitbox[] = [];
  hands: HandState[] = [];

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.pairCooldown.clear();
    this.recentHits.clear();
    this.hitboxes = [];
    this.hands = [];
  }

  update(landmarksByPerson: NormalizedLandmark[][], W: number, H: number, t: number): Strike[] {
    const px = (l: NormalizedLandmark): P => ({ x: l.x * W, y: l.y * H });
    const vis = (l: NormalizedLandmark) => (l.visibility ?? 0) >= VIS;

    // Build per-person geometry for anyone with both shoulders visible.
    const people: { lm: NormalizedLandmark[]; box: Omit<Hitbox, "personId"> }[] = [];
    for (const lm of landmarksByPerson) {
      // Side-on, one shoulder is usually occluded, so accept either.
      const shPts = [lm[LM.lSh], lm[LM.rSh]].filter(vis).map(px);
      if (shPts.length === 0) continue;
      const shMid = { x: shPts.reduce((s, p) => s + p.x, 0) / shPts.length, y: shPts.reduce((s, p) => s + p.y, 0) / shPts.length };
      const shWidth = shPts.length === 2 ? dist(shPts[0], shPts[1]) : 0;
      const nose = px(lm[LM.nose]);
      const hipPts = [lm[LM.lHip], lm[LM.rHip]].filter(vis).map(px);
      const hipMid = hipPts.length
        ? { x: hipPts.reduce((s, p) => s + p.x, 0) / hipPts.length, y: hipPts.reduce((s, p) => s + p.y, 0) / hipPts.length }
        : { x: shMid.x, y: shMid.y + 2.2 * Math.max(10, dist(nose, shMid)) };
      const scale = Math.max(30, dist(shMid, hipMid));
      const torso = { a: shMid, b: hipMid, r: Math.max(shWidth * 0.5, scale * 0.28) * 1.25 };
      const centroid = { x: (shMid.x + hipMid.x) / 2, y: (shMid.y + hipMid.y) / 2 };
      const arms: Capsule[] = [];
      for (const [s, e] of [[LM.lSh, LM.lEl], [LM.rSh, LM.rEl]]) {
        if (vis(lm[s]) && vis(lm[e])) arms.push({ a: px(lm[s]), b: px(lm[e]), r: scale * 0.2 });
      }
      // Head zone includes the guard (raised hands in front of the face): a punch
      // that lands on someone's guard is still a strike on them.
      people.push({ lm, box: { torso, head: { c: nose, r: scale * 0.45 }, arms, centroid, scale } });
    }

    // Greedy nearest-centroid matching keeps person IDs stable across frames,
    // which hand velocity depends on (MediaPipe's per-frame order can swap).
    const unmatched = new Set(this.tracks.filter((tr) => t - tr.lastSeen < TRACK_TIMEOUT_MS));
    const assigned: Track[] = [];
    for (const person of people) {
      let best: Track | null = null;
      let bestD = Infinity;
      for (const tr of unmatched) {
        const d = dist(tr.centroid, person.box.centroid);
        if (d < bestD && d < person.box.scale) {
          best = tr;
          bestD = d;
        }
      }
      if (best) unmatched.delete(best);
      else {
        best = { id: this.nextId++, centroid: person.box.centroid, scale: person.box.scale, firstSeen: t, lastSeen: t, lastBox: null, hands: { left: [], right: [] } };
        this.tracks.push(best);
      }
      best.centroid = person.box.centroid;
      best.scale = person.box.scale;
      best.lastSeen = t;
      best.lastBox = { personId: best.id, ...person.box };
      assigned.push(best);
    }
    this.tracks = this.tracks.filter((tr) => t - tr.lastSeen < TRACK_TIMEOUT_MS);

    const seenNow = new Set(assigned.map((tr) => tr.id));
    const live = assigned.map((tr) => tr.lastBox!);
    // A coasting box sitting on top of a live one is the same person re-detected
    // under a new ID — drop it, or one hand "hits" two copies of one body.
    const coasting = this.tracks
      .filter((tr) => tr.lastBox && !seenNow.has(tr.id) && t - tr.lastSeen < COAST_MS)
      .map((tr) => tr.lastBox!)
      .filter((cb) => !live.some((lb) => dist(lb.centroid, cb.centroid) < 0.5 * lb.scale));
    this.hitboxes = [...live, ...coasting];

    // Hand point = wrist pushed out along the forearm, approximating the fist.
    this.hands = [];
    people.forEach((person, i) => {
      const track = assigned[i];
      (["left", "right"] as Side[]).forEach((side) => {
        const wrist = person.lm[side === "left" ? LM.lWr : LM.rWr];
        const elbow = person.lm[side === "left" ? LM.lEl : LM.rEl];
        const history = track.hands[side];
        if ((wrist.visibility ?? 0) < HAND_VIS) {
          history.length = 0;
          return;
        }
        const w = px(wrist);
        const p = (elbow.visibility ?? 0) >= HAND_VIS ? { x: w.x + (w.x - px(elbow).x) * 0.3, y: w.y + (w.y - px(elbow).y) * 0.3 } : w;
        const sh = person.box.torso.a;
        const rel = { x: p.x - sh.x, y: p.y - sh.y };
        const shoulder = person.lm[side === "left" ? LM.lSh : LM.rSh];
        const elbowDeg =
          (elbow.visibility ?? 0) >= HAND_VIS && (shoulder.visibility ?? 0) >= HAND_VIS
            ? angleDeg(px(shoulder), px(elbow), w)
            : null;
        const last = history[history.length - 1];
        if (last && dist(last.rel, rel) > MAX_STEP * track.scale) history.length = 0;
        if (!last || dist(last.p, p) > 0) history.push({ t, p, rel, elbowDeg, speed: 0, extRate: 0 }); // skip duplicate (unchanged) video frames
        while (history.length && t - history[0].t > HISTORY_MS) history.shift();

        // Net displacement over the window, not max per-step: robust to jitter.
        const win = history.filter((s) => t - s.t <= SPEED_WINDOW_MS);
        let speed = 0;
        if (win.length >= MIN_WINDOW_SAMPLES) {
          const dt = (win[win.length - 1].t - win[0].t) / 1000;
          if (dt > 0) speed = dist(win[win.length - 1].rel, win[0].rel) / dt / track.scale;
          if (speed > MAX_PLAUSIBLE_SPEED) speed = 0;
        }
        const angles = win.filter((s) => s.elbowDeg !== null);
        let extRate = 0;
        if (angles.length >= 2) {
          const dtA = (angles[angles.length - 1].t - angles[0].t) / 1000;
          if (dtA > 0) extRate = Math.max(0, (angles[angles.length - 1].elbowDeg! - angles[0].elbowDeg!) / dtA);
        }
        const newest = history[history.length - 1];
        if (newest && newest.t === t) {
          newest.speed = speed;
          newest.extRate = extRate;
        }
        const recent = history.filter((s) => t - s.t <= PEAK_MS);
        const peakSpeed = Math.max(speed, ...recent.map((s) => s.speed));
        const peakExtRate = Math.max(extRate, ...recent.map((s) => s.extRate));
        this.hands.push({ personId: track.id, side, p, speed, extRate, peakSpeed, peakExtRate });
      });
    });

    // A strike = fast hand that is inside someone else's hitbox now and arrived
    // there: outside it at the start of the speed window, not resting on it.
    // A punching arm often only becomes visible (unblurred) a frame before
    // contact, so a hand whose history doesn't cover the whole window counts as
    // arriving if it's moving deeper into the body.
    const strikes: Strike[] = [];
    const warm = (id: number) => {
      const tr = this.tracks.find((x) => x.id === id);
      return !!tr && t - tr.firstSeen >= WARMUP_MS;
    };
    for (const hand of this.hands) {
      const punchMotion =
        hand.peakSpeed >= SPEED_THRESH || (hand.peakExtRate >= EXT_RATE_THRESH && hand.peakSpeed >= EXT_MIN_SPEED);
      if (!punchMotion || !warm(hand.personId)) continue;
      const history = this.tracks.find((tr) => tr.id === hand.personId)?.hands[hand.side] ?? [];
      const windowStart = history.find((s) => t - s.t <= SPEED_WINDOW_MS) ?? history[0];
      for (const box of this.hitboxes) {
        if (box.personId === hand.personId || !warm(box.personId)) continue;
        const target = hitTarget(hand.p, box);
        if (!target || !windowStart) continue;
        const startDepth = depth(windowStart.p, box);
        const justAppeared = history[0].t > t - SPEED_WINDOW_MS;
        const arrived = startDepth > 1 || (justAppeared && startDepth - depth(hand.p, box) >= APPROACH_MIN);
        if (!arrived) continue;
        const key = `${hand.personId}->${box.personId}`;
        if (t - (this.pairCooldown.get(key) ?? -Infinity) < PAIR_COOLDOWN_MS) continue;
        this.pairCooldown.set(key, t);
        this.recentHits.set(box.personId, t);
        strikes.push({ attackerId: hand.personId, victimId: box.personId, side: hand.side, speed: hand.peakSpeed, target, at: hand.p });
      }
    }
    return strikes;
  }

  draw(ctx: CanvasRenderingContext2D, t: number, mirrored = false) {
    const text = (s: string, x: number, y: number) => {
      if (!mirrored) return ctx.fillText(s, x, y);
      ctx.save();
      ctx.translate(x, y);
      ctx.scale(-1, 1);
      ctx.fillText(s, 0, 0);
      ctx.restore();
    };
    ctx.save();
    for (const box of this.hitboxes) {
      const hit = t - (this.recentHits.get(box.personId) ?? -Infinity) < HIT_FLASH_MS;
      ctx.strokeStyle = hit ? "#ff2b2b" : "rgba(79, 209, 197, 0.9)";
      ctx.fillStyle = hit ? "rgba(255, 43, 43, 0.35)" : "rgba(79, 209, 197, 0.08)";
      ctx.lineWidth = hit ? 4 : 2;
      ctx.setLineDash(hit ? [] : [8, 6]);
      for (const { a, b, r } of [box.torso, ...box.arms]) {
        const ang = Math.atan2(b.y - a.y, b.x - a.x);
        ctx.beginPath();
        ctx.arc(b.x, b.y, r, ang - Math.PI / 2, ang + Math.PI / 2);
        ctx.arc(a.x, a.y, r, ang + Math.PI / 2, ang + (3 * Math.PI) / 2);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(box.head.c.x, box.head.c.y, box.head.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#fff";
      ctx.font = "600 14px ui-monospace, monospace";
      text(`P${box.personId}`, box.head.c.x - 10, box.head.c.y - box.head.r - 6);
    }
    for (const hand of this.hands) {
      const fast = hand.speed >= SPEED_THRESH;
      ctx.fillStyle = fast ? "#ff2b2b" : hand.speed >= SPEED_THRESH * 0.6 ? "#f5c344" : "#2dd4a7";
      ctx.beginPath();
      ctx.arc(hand.p.x, hand.p.y, fast ? 10 : 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "12px ui-monospace, monospace";
      text(hand.speed.toFixed(1), hand.p.x + 10, hand.p.y - 8);
    }
    ctx.restore();
  }
}
