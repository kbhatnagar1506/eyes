// Full-body pose extraction + a security-monitor-style render: body-region
// coloring, bounding boxes, and live classification labels — all computed from
// real MediaPipe landmarks (33-point, per person), not faked. The body fill is
// thick colored limb segments along real bone positions, not true pixel-level
// segmentation (DensePose) — that needs a GPU Python backend we don't have time
// for this weekend. This gets the same visual read from data we actually have.

import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from "@mediapipe/tasks-vision";

let landmarker: PoseLandmarker | null = null;

export async function loadPoseLandmarker(): Promise<void> {
  if (landmarker) return;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        // "full", not "lite": lite missed the person being hit in wide frames.
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU",
    },
    // IMAGE, not VIDEO: VIDEO mode skips the person detector while it is
    // tracking someone, so with two people close together it locks onto one
    // and flips between them. Measured on a real two-person punch recording:
    // VIDEO saw both people in almost no frames; IMAGE saw both in most.
    runningMode: "IMAGE",
    numPoses: 4,
    // Defaults (0.5) drop the person being hit when two bodies overlap at impact.
    minPoseDetectionConfidence: 0.3,
    minPosePresenceConfidence: 0.3,
  });
}

export function isReady(): boolean {
  return landmarker !== null;
}

// When the full frame finds fewer people than it just saw, the missing one is
// usually tangled up with the other at the moment of contact. Each half of the
// frame on its own holds one body, which the detector finds reliably, so
// detect on two overlapping crops and merge.
const TILE_FRAC = 0.6; // each crop covers 60% of the width; middle 20% overlaps
const RECENT_MS = 1000;
// With one person in view there's nobody missing to look for, but still check
// the halves now and then so a second person who walks in gets picked up.
const DISCOVERY_EVERY_N_FRAMES = 10;
const tileCanvas = document.createElement("canvas");
let recentCounts: { t: number; n: number }[] = [];
let frameIndex = 0;

type Px = { x: number; y: number };

function meanVisibility(lm: NormalizedLandmark[]) {
  return lm.reduce((s, p) => s + (p.visibility ?? 0), 0) / lm.length;
}

// Upper-body center and size in pixels. Hips are often out of a webcam's
// frame, so this leans on shoulders and the nose.
function bodyAnchor(lm: NormalizedLandmark[], W: number, H: number): { c: Px; size: number } | null {
  const vis = (i: number) => (lm[i].visibility ?? 0) > 0.3;
  const px = (i: number): Px => ({ x: lm[i].x * W, y: lm[i].y * H });
  const sh = [11, 12].filter(vis).map(px);
  if (!sh.length) return null;
  const c = { x: sh.reduce((s, p) => s + p.x, 0) / sh.length, y: sh.reduce((s, p) => s + p.y, 0) / sh.length };
  const hips = [23, 24].filter(vis).map(px);
  const torso = hips.length ? Math.hypot(hips[0].x - c.x, hips[0].y - c.y) : 0;
  const shW = sh.length === 2 ? Math.hypot(sh[0].x - sh[1].x, sh[0].y - sh[1].y) : 0;
  const neck = vis(0) ? 2 * Math.hypot(px(0).x - c.x, px(0).y - c.y) : 0;
  return { c, size: Math.max(torso, shW, neck, 30) };
}

// Adds a detection unless it's a second copy of someone already in the list
// (the detector sometimes returns one body several times); keeps the clearer copy.
function mergeInto(merged: NormalizedLandmark[][], lm: NormalizedLandmark[], W: number, H: number) {
  const a = bodyAnchor(lm, W, H);
  if (!a) return;
  const dupIdx = merged.findIndex((m) => {
    const b = bodyAnchor(m, W, H);
    return !!b && Math.hypot(a.c.x - b.c.x, a.c.y - b.c.y) < 0.5 * Math.max(a.size, b.size);
  });
  if (dupIdx < 0) merged.push(lm);
  else if (meanVisibility(lm) > meanVisibility(merged[dupIdx])) merged[dupIdx] = lm;
}

function detectPeople(video: HTMLVideoElement, tMs: number): NormalizedLandmark[][] {
  const W = video.videoWidth;
  const H = video.videoHeight;
  const merged: NormalizedLandmark[][] = [];
  for (const lm of landmarker!.detect(video).landmarks) mergeInto(merged, lm, W, H);

  recentCounts = recentCounts.filter((r) => tMs - r.t < RECENT_MS && r.t <= tMs);
  const recentMax = Math.max(0, ...recentCounts.map((r) => r.n));
  frameIndex++;
  if (merged.length < recentMax || frameIndex % DISCOVERY_EVERY_N_FRAMES === 0) {
    const cropW = Math.round(W * TILE_FRAC);
    tileCanvas.width = cropW;
    tileCanvas.height = H;
    const tctx = tileCanvas.getContext("2d")!;
    [0, W - cropW].forEach((x0, i) => {
      tctx.drawImage(video, x0, 0, cropW, H, 0, 0, cropW, H);
      const seamX = i === 0 ? x0 + cropW : x0; // the crop's inner edge, in pixels
      for (const lm of landmarker!.detect(tileCanvas).landmarks) {
        const fullLm = lm.map((p) => ({ ...p, x: (x0 + p.x * cropW) / W }));
        const a = bodyAnchor(fullLm, W, H);
        // A body centered near the crop's inner edge is probably cut in half by it.
        if (!a || Math.abs(a.c.x - seamX) < 0.05 * W) continue;
        mergeInto(merged, fullLm, W, H);
      }
    });
  }
  recentCounts.push({ t: tMs, n: merged.length });
  return merged;
}

export interface PoseVectorRecord {
  tMs: number;
  personIndex: number;
  vector: number[]; // 33 landmarks x (x, y, z, visibility), flattened = 132 numbers
}

export function flattenLandmarks(landmarks: NormalizedLandmark[]): number[] {
  const out: number[] = [];
  for (const lm of landmarks) out.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
  return out;
}

// MediaPipe Pose landmark indices we care about.
const L = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
};

// Bone segments to render as thick colored capsules, approximating the
// DensePose region palette: head = magenta, torso = blue/olive, arms =
// blue->red. Legs (hip-knee, knee-ankle) are deliberately not rendered —
// at normal webcam distance they're usually out of frame or low-confidence,
// which made the leg segments jitter and draw to wrong/guessed positions.
const BONES: Array<{ a: number; b: number; color: string; widthMul: number }> = [
  { a: L.leftShoulder, b: L.rightShoulder, color: "#2b4fe0", widthMul: 1.0 }, // shoulder line
  { a: L.leftHip, b: L.rightHip, color: "#8a7a24", widthMul: 1.0 }, // hip line
  { a: L.leftShoulder, b: L.leftElbow, color: "#2b5ee0", widthMul: 0.85 },
  { a: L.leftElbow, b: L.leftWrist, color: "#d43b5e", widthMul: 0.65 },
  { a: L.rightShoulder, b: L.rightElbow, color: "#2b5ee0", widthMul: 0.85 },
  { a: L.rightElbow, b: L.rightWrist, color: "#d43b5e", widthMul: 0.65 },
];

function mid(a: NormalizedLandmark, b: NormalizedLandmark) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

export interface PersonMetrics {
  bbox: { x0: number; y0: number; x1: number; y1: number }; // pixel space
  motionPct: number; // 0-100
  walking: boolean;
  alertPct: number; // 0-100, proximity+motion heuristic
}

const prevByPerson = new Map<number, NormalizedLandmark[]>();

function landmarkMotion(personIdx: number, landmarks: NormalizedLandmark[]): number {
  const prev = prevByPerson.get(personIdx);
  prevByPerson.set(personIdx, landmarks);
  if (!prev) return 0;
  let total = 0;
  let n = 0;
  for (let i = 0; i < landmarks.length; i++) {
    if ((landmarks[i].visibility ?? 0) < 0.4) continue;
    total += Math.hypot(landmarks[i].x - prev[i].x, landmarks[i].y - prev[i].y);
    n++;
  }
  return n > 0 ? total / n : 0; // normalized 0-1-ish units already (landmarks are 0-1 normalized)
}

/**
 * Draws the security-monitor-style overlay for one video frame and returns
 * per-person metrics computed from real landmark data.
 */
export function renderFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  timestampMs?: number,
): { people: PersonMetrics[]; tMs: number; landmarksByPerson: NormalizedLandmark[][] } | null {
  if (!landmarker) return null;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width;
  const H = canvas.height;
  const tMs = timestampMs ?? performance.now();
  const allLandmarks = detectPeople(video, tMs);

  ctx.clearRect(0, 0, W, H);

  const people: PersonMetrics[] = [];
  const seenPersons = new Set<number>();

  allLandmarks.forEach((landmarks, personIdx) => {
    seenPersons.add(personIdx);
    const px = (p: NormalizedLandmark) => ({ x: p.x * W, y: p.y * H });

    // Body size reference (shoulder-to-hip distance) for proportional limb width.
    const shoulderMid = mid(landmarks[L.leftShoulder], landmarks[L.rightShoulder]);
    const hipMid = mid(landmarks[L.leftHip], landmarks[L.rightHip]);
    const torsoLen = Math.hypot((shoulderMid.x - hipMid.x) * W, (shoulderMid.y - hipMid.y) * H) || 40;
    const baseWidth = Math.max(6, torsoLen * 0.22);

    ctx.save();
    ctx.globalAlpha = 0.82;
    for (const bone of BONES) {
      const a = landmarks[bone.a];
      const b = landmarks[bone.b];
      if ((a.visibility ?? 0) < 0.3 || (b.visibility ?? 0) < 0.3) continue;
      const pa = px(a);
      const pb = px(b);
      ctx.strokeStyle = bone.color;
      ctx.lineWidth = baseWidth * bone.widthMul;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    // Head as a filled magenta ellipse, sized off shoulder width.
    const nose = landmarks[L.nose];
    const shoulderW = Math.hypot(
      (landmarks[L.leftShoulder].x - landmarks[L.rightShoulder].x) * W,
      (landmarks[L.leftShoulder].y - landmarks[L.rightShoulder].y) * H,
    );
    const headR = Math.max(10, shoulderW * 0.42);
    const headP = px(nose);
    ctx.fillStyle = "#e0379c";
    ctx.beginPath();
    ctx.ellipse(headP.x, headP.y - headR * 0.3, headR * 0.72, headR, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Bounding box from visible landmark extents.
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    for (const lm of landmarks) {
      if ((lm.visibility ?? 0) < 0.25) continue;
      const p = px(lm);
      x0 = Math.min(x0, p.x);
      y0 = Math.min(y0, p.y);
      x1 = Math.max(x1, p.x);
      y1 = Math.max(y1, p.y);
    }
    const pad = baseWidth * 0.8;
    x0 -= pad;
    y0 -= headR * 1.6;
    x1 += pad;
    y1 += pad;

    const motion = landmarkMotion(personIdx, landmarks);
    const motionPct = Math.min(100, (motion / 0.012) * 100);
    const walking = motionPct > 35;

    people.push({ bbox: { x0, y0, x1, y1 }, motionPct, walking, alertPct: 0 });
  });

  // Clean up motion history for people who left the frame.
  for (const idx of Array.from(prevByPerson.keys())) {
    if (!seenPersons.has(idx)) prevByPerson.delete(idx);
  }

  // Proximity-based alert heuristic across all currently-visible people.
  let minDist = Infinity;
  for (let i = 0; i < people.length; i++) {
    for (let j = i + 1; j < people.length; j++) {
      const ci = { x: (people[i].bbox.x0 + people[i].bbox.x1) / 2, y: (people[i].bbox.y0 + people[i].bbox.y1) / 2 };
      const cj = { x: (people[j].bbox.x0 + people[j].bbox.x1) / 2, y: (people[j].bbox.y0 + people[j].bbox.y1) / 2 };
      const d = Math.hypot(ci.x - cj.x, ci.y - cj.y) / W;
      minDist = Math.min(minDist, d);
    }
  }
  const proximityPct = people.length >= 2 && minDist < 0.4 ? Math.min(100, (1 - minDist / 0.4) * 100) : 0;

  people.forEach((p, i) => {
    const maxMotion = Math.max(...people.map((q) => q.motionPct));
    p.alertPct = Math.min(100, 0.6 * proximityPct + 0.4 * (people[i].motionPct / Math.max(1, maxMotion)) * maxMotion);
  });

  // Draw boxes + labels now that alertPct is known for all people.
  people.forEach((p, i) => {
    const isPrimary = p.alertPct === Math.max(...people.map((q) => q.alertPct)) && p.alertPct > 0;
    const boxColor = isPrimary && people.length > 1 ? "#ef2b2b" : i === 0 ? "#ef2b2b" : "#2bd65a";
    ctx.save();
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.strokeRect(p.bbox.x0, p.bbox.y0, p.bbox.x1 - p.bbox.x0, p.bbox.y1 - p.bbox.y0);
    ctx.restore();

    if (i === 0) {
      const labels: Array<{ text: string; color: string }> = [
        { text: `${p.walking ? "Walking" : "Standing"}: ${(p.walking ? p.motionPct : 100 - p.motionPct).toFixed(2)}%`, color: p.walking ? "#2b5ee0" : "#2b5ee0" },
        { text: `${p.alertPct > 50 ? "Alert" : "Normal"}: ${(p.alertPct > 50 ? p.alertPct : 100 - p.alertPct).toFixed(2)}%`, color: p.alertPct > 50 ? "#ef2b2b" : "#2bd65a" },
      ];
      drawLabelStack(ctx, labels, W - 10, 10, "right");
    }
  });

  // CCTV monitor chrome: grayscale handled via CSS on the video element;
  // here we just stamp a camera tag like the reference image.
  ctx.save();
  ctx.font = "600 16px ui-monospace, monospace";
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.textAlign = "right";
  ctx.fillText("CAM 03", W - 10, H - 14);
  ctx.restore();

  return { people, tMs, landmarksByPerson: allLandmarks };
}

function drawLabelStack(
  ctx: CanvasRenderingContext2D,
  labels: Array<{ text: string; color: string }>,
  rightX: number,
  topY: number,
  align: "right" | "left",
) {
  ctx.save();
  ctx.font = "700 20px ui-monospace, monospace";
  ctx.textAlign = align;
  ctx.textBaseline = "top";
  let y = topY;
  for (const { text, color } of labels) {
    const metrics = ctx.measureText(text);
    const padX = 10;
    const padY = 6;
    const w = metrics.width + padX * 2;
    const h = 20 + padY * 2;
    const x = align === "right" ? rightX - w : rightX;
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, align === "right" ? rightX - padX : rightX + padX, y + padY);
    y += h + 6;
  }
  ctx.restore();
}
