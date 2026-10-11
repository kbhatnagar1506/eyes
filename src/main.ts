import { logoSvg } from "./logo";
import "./style.css";
import { loadPoseLandmarker, renderFrame, flattenLandmarks, isReady, type PoseVectorRecord } from "./pose";
import { StrikeDetector, type Strike } from "./strike";
import { joinMesh } from "./mesh/session";
import { addEvidenceFrame, MeshView, raiseIncident } from "./meshview";
import type { SessionLabels } from "./sessions-dev";
import { attackUrl, clipUrl, denseposeUrl } from "./mesh/media";

type Clip = {
  id: string;
  label: string;
  file: string;
  group: string;
  labels?: SessionLabels;
  // Rendered on the NVIDIA GPU (DensePose): plays as-is, with its detector
  // scores and tracking summary instead of the MediaPipe tools.
  gpu?: { scores: string | null; summary: string | null; attack: string | null; key: string };
};

// Recorded sessions (real people's faces) load in local dev only: a
// production build never even resolves the module, so no recording can end
// up on the hosted site.
const SESSION_CLIPS: Clip[] = import.meta.env.DEV ? (await import("./sessions-dev")).SESSION_CLIPS.map((c) => ({ ...c, group: "Live demo recordings" })) : [];

// Clear, well-lit, close-up stock footage — much better pose-tracking signal
// than grainy real CCTV. Use these for training data quality.
const STOCK_CLIPS = [
  { id: "sparring", label: "Sparring (clear)", file: "sparring_6296158.mp4" },
  { id: "woman_sparring", label: "Woman sparring w/ man (clear)", file: "woman_sparring_6296385.mp4" },
  { id: "boxing", label: "Boxing (clear)", file: "boxing_6296696.mp4" },
  { id: "sparring2", label: "Sparring 2 (clear)", file: "sparring2_6296587.mp4" },
  { id: "busstop_calm", label: "Bus stop, calm (clear baseline)", file: "busstop_calm_36467681.mp4" },
  // Original real UCF-Crime surveillance footage — grainier, worse pose signal,
  // kept for the pitch narrative. Research-licensed: local only (clipUrl
  // returns null on the hosted site).
  { id: "assault", label: "Assault049 (real CCTV, grainy)", file: "Assault049_x264.mp4" },
  { id: "fighting", label: "Fighting049 (real CCTV, grainy)", file: "Fighting049_x264.mp4" },
  { id: "normal", label: "Normal_929 (real CCTV, grainy)", file: "Normal_929_x264.mp4" },
];

// The NVIDIA T4 runs (Detectron2 DensePose on Google Cloud): every visible
// person's body surface mapped and tracked. Stock clips at full resolution;
// the real CCTV clips with multi-person tracking and per-frame detector scores.
const GPU_GROUP = "NVIDIA GPU · DensePose";
const GPU_STOCK = [
  ["sparring_6296158", "Sparring"],
  ["woman_sparring_6296385", "Woman sparring w/ man"],
  ["boxing_6296696", "Boxing"],
  ["sparring2_6296587", "Sparring 2"],
  ["busstop_calm_36467681", "Bus stop, calm"],
];
const GPU_CCTV = [
  ["Fighting042", "Fighting 042"],
  ["Fighting049", "Fighting 049"],
  ["Assault049", "Assault 049"],
  ["Assault050", "Assault 050"],
  ["Assault051", "Assault 051"],
  ["Normal_929", "Normal 929 (no incident)"],
];

async function gpuClips(): Promise<Clip[]> {
  const stock = GPU_STOCK.map(async ([base, label]) => {
    const [file, attack] = await Promise.all([denseposeUrl(`${base}_densepose.mp4`), attackUrl(`${base}_attack.json`)]);
    return file ? [{ id: `gpu-${base}`, label: `${label} · DensePose`, file, group: GPU_GROUP, gpu: { scores: null, summary: null, attack, key: base } }] : [];
  });
  const cctv = GPU_CCTV.map(async ([base, label]) => {
    const [file, scores, summary, attack] = await Promise.all([
      denseposeUrl(`${base}_x264_multi_tracked.mp4`),
      denseposeUrl(`${base}_x264_scores.json`),
      denseposeUrl(`${base}_x264_multi.json`),
      attackUrl(`${base}_x264_attack.json`),
    ]);
    return file ? [{ id: `gpu-${base}`, label: `${label} · real CCTV · DensePose`, file, group: GPU_GROUP, gpu: { scores, summary, attack, key: `${base}_x264` } }] : [];
  });
  return (await Promise.all([...stock, ...cctv])).flat();
}

// The demo shows one clip: the clearest real CCTV attack. The rest stay
// wired up; if its links are missing, fall back to the full list.
const SHOWCASE = "gpu-Assault050";
const ALL_CLIPS: Clip[] = [
  ...(await gpuClips()),
  ...SESSION_CLIPS,
  ...(await Promise.all(STOCK_CLIPS.map(async (c) => ({ ...c, file: await clipUrl(c.file) }))))
    .flatMap((c) => (c.file ? [{ ...c, file: c.file, group: c.id === "assault" || c.id === "fighting" || c.id === "normal" ? "Real CCTV (UCF-Crime) · MediaPipe" : "Stock footage · MediaPipe" }] : [])),
];
const showcase = ALL_CLIPS.filter((c) => c.id === SHOWCASE);
const CLIPS: Clip[] = showcase.length ? showcase : ALL_CLIPS;

const clipOptions = () => {
  const groups = [...new Set(CLIPS.map((c) => c.group))];
  return groups
    .map((g) => `<optgroup label="${g}">${CLIPS.filter((c) => c.group === g).map((c) => `<option value="${c.id}">${c.label}</option>`).join("")}</optgroup>`)
    .join("");
};

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <header class="lab-hud">
    <a class="brand" href="/" aria-label="Eyes home">
      ${logoSvg(36, "brand-logo")}
      <span class="brand-text"><b>Eyes</b><i>Cameras that watch. Eyes that act.</i></span>
    </a>
    <nav class="tabnav">
      <button id="tabDataset" class="tab-btn active">Dataset tool</button>
      <button id="tabLive" class="tab-btn">Live demo</button>
      <button id="tabMesh" class="tab-btn">Mesh</button>
      <button id="tabModel" class="tab-btn">Model</button>
    </nav>
  </header>

  <main class="lab-main" id="datasetView">
    <div class="video-stage">
      <video id="clipVideo" controls muted playsinline></video>
      <canvas id="overlay"></canvas>
    </div>

    <aside class="lab-panel">
      <div class="panel-row">
        <label class="panel-label">Clip</label>
        <select id="clipSelect">${clipOptions()}</select>
      </div>

      <div class="gpu-panel hidden" id="gpuPanel">
        <div class="gpu-badge">NVIDIA T4 GPU · DensePose (Detectron2)</div>
        <p class="hint">
          Rendered on a Google Cloud NVIDIA T4. DensePose maps the body surface of
          every visible person, each color a region of the body, and tracks people
          across frames: the dense body model Eyes' detection is built on.
        </p>
        <div class="stat-grid" id="gpuStats"></div>
        <div class="gpu-chart hidden" id="gpuChartWrap">
          <div class="stat-label" id="gpuChartLabel">GPU person-detection confidence, per frame</div>
          <canvas id="gpuChart"></canvas>
        </div>
      </div>

      <div id="mpTools">
      <div class="panel-row">
        <button id="startBtn">Start tracking</button>
        <button id="stopBtn" class="secondary" disabled>Stop</button>
      </div>
      <div class="panel-row">
        <button id="analyzeBtn" class="secondary">Analyze frame-by-frame</button>
      </div>

      <div class="stat-grid">
        <div class="stat"><div class="stat-label">Status</div><div id="statStatus" class="stat-value">idle</div></div>
        <div class="stat"><div class="stat-label">People now</div><div id="statPeople" class="stat-value">0</div></div>
        <div class="stat"><div class="stat-label">Strikes detected</div><div id="statStrikes" class="stat-value">0</div></div>
        <div class="stat"><div class="stat-label">Vectors captured</div><div id="statVectors" class="stat-value">0</div></div>
      </div>

      <button id="exportBtn" class="export-btn" disabled>Export vectors (JSON)</button>
      <p class="hint">
        Each captured frame records one 132-number vector per detected person
        (33 MediaPipe pose landmarks × x, y, z, visibility), tagged with the
        clip label and timestamp — that's the training set.
      </p>
      </div>

      <div class="log" id="logPanel"><div class="log-title">Log</div><div id="logLines"></div></div>
    </aside>
  </main>

  <div class="live-fullscreen hidden" id="liveView">
    <div class="live-mirror-wrap">
      <video id="liveVideo" autoplay muted playsinline></video>
      <canvas id="liveOverlay"></canvas>
    </div>
    <div id="threatBanner" class="threat-banner hidden">⚠ THREAT DETECTED ⚠</div>
    <div id="cueOverlay" class="cue-overlay hidden">
      <div class="cue-action" id="cueAction"></div>
      <div class="cue-count" id="cueCount"></div>
      <div class="cue-progress" id="cueProgress"></div>
    </div>

    <div class="live-controls">
      <div class="panel-row">
        <button id="liveStartBtn">Start camera</button>
        <button id="liveStopBtn" class="secondary" disabled>Stop</button>
      </div>
      <div class="panel-row">
        <button id="recordBtn" class="secondary" disabled><span class="rec-dot">●</span> Record training session</button>
      </div>
      <div class="stat-grid">
        <div class="stat"><div class="stat-label">Status</div><div id="liveStatus" class="stat-value">idle</div></div>
        <div class="stat"><div class="stat-label">People now</div><div id="livePeople" class="stat-value">0</div></div>
        <div class="stat"><div class="stat-label">Punches detected</div><div id="livePunches" class="stat-value">0</div></div>
        <div class="stat"><div class="stat-label">Fastest hand</div><div id="liveSpeed" class="stat-value">—</div></div>
      </div>
    </div>

    <div class="live-log-float" id="livePanel"><div class="log-title">Live log</div><div id="liveLogLines"></div></div>
  </div>

  <div class="mesh-fullscreen hidden" id="meshView">
  </div>
  <div class="model-view hidden" id="modelView"></div>
`;

const clipVideo = document.querySelector<HTMLVideoElement>("#clipVideo")!;
// Clips come from the media bucket on the hosted site: pose detection reads
// their pixels, which a cross-origin video only allows in CORS mode.
clipVideo.crossOrigin = "anonymous";
const overlay = document.querySelector<HTMLCanvasElement>("#overlay")!;
const clipSelect = document.querySelector<HTMLSelectElement>("#clipSelect")!;
const startBtn = document.querySelector<HTMLButtonElement>("#startBtn")!;
const stopBtn = document.querySelector<HTMLButtonElement>("#stopBtn")!;
const exportBtn = document.querySelector<HTMLButtonElement>("#exportBtn")!;
const statStatus = document.querySelector<HTMLDivElement>("#statStatus")!;
const statPeople = document.querySelector<HTMLDivElement>("#statPeople")!;
const statStrikes = document.querySelector<HTMLDivElement>("#statStrikes")!;
const statVectors = document.querySelector<HTMLDivElement>("#statVectors")!;
const logLinesEl = document.querySelector<HTMLDivElement>("#logLines")!;

function log(text: string) {
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `<span class="log-t">${t}</span> ${text}`;
  logLinesEl.prepend(line);
}

function currentClip() {
  return CLIPS.find((c) => c.id === clipSelect.value)!;
}

// ---- NVIDIA GPU (DensePose) clips -------------------------------------------
const gpuPanel = document.querySelector<HTMLDivElement>("#gpuPanel")!;
const gpuStats = document.querySelector<HTMLDivElement>("#gpuStats")!;
const gpuChartWrap = document.querySelector<HTMLDivElement>("#gpuChartWrap")!;
const gpuChart = document.querySelector<HTMLCanvasElement>("#gpuChart")!;
const mpTools = document.querySelector<HTMLDivElement>("#mpTools")!;
const gpuChartLabel = document.querySelector<HTMLDivElement>("#gpuChartLabel")!;

type GpuScores = { n_sampled_frames: number; raw_scores: number[]; confirmed_frame_count: number };
type GpuSummary = { max_people_single_frame: number; num_tracks: number; frames_with_any_detection: number; n_sampled_frames: number };
const jsonCache = new Map<string, Promise<unknown>>();
const getJson = <T,>(url: string) => (jsonCache.get(url) ?? jsonCache.set(url, fetch(url).then((r) => r.json())).get(url)!) as Promise<T>;
let gpuScores: number[] | null = null;
let gpuRaf = 0;

// AttackNet (trained on the GCP T4): per frame, every person's box and how
// likely they're part of an attack; plus where humans labeled attacks.
type AttackPerson = { id: number; box: [number, number, number, number]; p: number };
type AttackData = { clip: string; fps: number; threshold: number; frames: { t: number; p: number; label: number; people: AttackPerson[] }[] };
type ClipMetrics = { f1: number; accuracy: number; auc: number | null; recall: number };
type StageResult = { ours: { per_clip: Record<string, ClipMetrics> }; sparring: { per_clip: Record<string, ClipMetrics> } };
// metrics.json from scripts/attack/train_rlvs.py (AttackNet v2).
type AttackMetrics = {
  weights: number;
  shown: "A" | "B";
  rlvs: { train: number; stage_A_test: { clip_accuracy: number; clips: number } };
  stage_A: StageResult;
  stage_B: StageResult;
};
let gpuAttack: AttackData | null = null;
// A box turns red only when AttackNet is above this; under it, green.
const RED_AT = 0.3;

const statBox = (label: string, value: string) => `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div></div>`;

async function showGpuInfo(c: Clip) {
  gpuScores = null;
  gpuAttack = null;
  gpuChartWrap.classList.add("hidden");
  gpuStats.innerHTML = statBox("Model", "DensePose R50") + statBox("Hardware", "NVIDIA T4");
  try {
    if (c.gpu?.summary) {
      const s = await getJson<GpuSummary>(c.gpu.summary);
      gpuStats.innerHTML =
        statBox("Most people in one frame", String(s.max_people_single_frame)) +
        statBox("People tracked", String(s.num_tracks)) +
        statBox("Frames with people", `${s.frames_with_any_detection} / ${s.n_sampled_frames}`) +
        statBox("Hardware", "NVIDIA T4");
    }
    if (c.gpu?.scores) {
      const s = await getJson<GpuScores>(c.gpu.scores);
      if (currentClip().id !== c.id) return;
      gpuScores = s.raw_scores;
      gpuChartWrap.classList.remove("hidden");
    }
    if (c.gpu?.attack) {
      const [a, metricsUrl] = await Promise.all([getJson<AttackData>(c.gpu.attack), attackUrl("metrics.json")]);
      if (currentClip().id !== c.id) return;
      gpuAttack = a;
      gpuChartLabel.textContent = "AttackNet: attack probability (shaded = labeled attack)";
      gpuChartWrap.classList.remove("hidden");
      if (metricsUrl) {
        const m = await getJson<AttackMetrics>(metricsUrl);
        const stage = m.shown === "B" ? m.stage_B : m.stage_A;
        const r = stage.ours.per_clip[c.gpu.key] ?? stage.sparring.per_clip[c.gpu.key];
        // Every clip here is scored by a model that never trained on it.
        // A calm clip has no attack to find: what matters is false alarms.
        const calm = a.frames.every((f) => !f.label);
        gpuStats.innerHTML +=
          (r ? statBox(calm ? "Unseen calm clip: frames flagged" : "Unseen clip: attack F1", `${Math.round((calm ? 1 - r.accuracy : r.f1) * 100)}%`) : "") +
          statBox("AttackNet", `${(m.weights / 1e6).toFixed(1)}M weights`) +
          statBox("Pretrained on", `${m.rlvs.train.toLocaleString()} real fight / normal clips`) +
          statBox("Unseen test clips", `${Math.round(m.rlvs.stage_A_test.clip_accuracy * 100)}% right (${m.rlvs.stage_A_test.clips})`);
      }
    } else {
      gpuChartLabel.textContent = "GPU person-detection confidence, per frame";
    }
  } catch {
    /* stats are extra; the render itself still plays */
  }
}

// Boxes over each person, drawn in video pixels on the overlay (which
// letterboxes exactly like the video): red with the probability when
// AttackNet puts them in an attack, green otherwise.
function drawAttackBoxes(a: AttackData) {
  const W = clipVideo.videoWidth;
  const H = clipVideo.videoHeight;
  if (!W || !H) return;
  if (overlay.width !== W || overlay.height !== H) {
    overlay.width = W;
    overlay.height = H;
  }
  const g = overlay.getContext("2d")!;
  g.clearRect(0, 0, W, H);
  const i = Math.min(a.frames.length - 1, Math.max(0, Math.round(clipVideo.currentTime * a.fps)));
  const fr = a.frames[i];
  if (!fr) return;
  const lw = Math.max(2, Math.round(W / 320));
  const font = Math.max(11, Math.round(W / 60));
  g.font = `800 ${font}px ui-monospace, Menlo, monospace`;
  for (const p of fr.people) {
    const [x1, y1, x2, y2] = [p.box[0] * W, p.box[1] * H, p.box[2] * W, p.box[3] * H];
    const hot = p.p >= RED_AT;
    g.lineWidth = hot ? lw * 2 : lw;
    g.strokeStyle = hot ? "#ef4444" : "rgba(74, 222, 128, 0.9)";
    if (hot) {
      g.fillStyle = "rgba(239, 68, 68, 0.12)";
      g.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = hot ? `ATTACK ${Math.round(p.p * 100)}%` : `${Math.round(p.p * 100)}%`;
    const tw = g.measureText(text).width + 8;
    g.fillStyle = hot ? "#ef4444" : "rgba(22, 101, 52, 0.85)";
    g.fillRect(x1, Math.max(0, y1 - font - 6), tw, font + 6);
    g.fillStyle = "#fff";
    g.fillText(text, x1 + 4, Math.max(font, y1 - 4));
  }
}

function drawAttackChart(a: AttackData, g: CanvasRenderingContext2D, w: number, h: number) {
  const n = a.frames.length;
  const x = (i: number) => (i / Math.max(1, n - 1)) * w;
  const y = (v: number) => h - 6 - v * (h - 12);
  g.fillStyle = "rgba(255, 255, 255, 0.08)";
  a.frames.forEach((f, i) => f.label && g.fillRect(x(i), 0, w / n + 1, h));
  g.strokeStyle = "rgba(255, 255, 255, 0.35)";
  g.setLineDash([4, 4]);
  g.beginPath();
  g.moveTo(0, y(RED_AT));
  g.lineTo(w, y(RED_AT));
  g.stroke();
  g.setLineDash([]);
  g.lineWidth = 2;
  for (let i = 1; i < n; i++) {
    g.strokeStyle = a.frames[i].p >= RED_AT ? "#ef4444" : "#4ade80";
    g.beginPath();
    g.moveTo(x(i - 1), y(a.frames[i - 1].p));
    g.lineTo(x(i), y(a.frames[i].p));
    g.stroke();
  }
  return (t: number) => a.frames[Math.min(n - 1, Math.round(t * (n - 1)))]?.p ?? 0;
}

// Scores along the clip, the playhead on top: follows the video live.
function drawGpuChart() {
  gpuRaf = requestAnimationFrame(drawGpuChart);
  if (gpuAttack) drawAttackBoxes(gpuAttack);
  if (!gpuScores?.length && !gpuAttack) return;
  const dpr = window.devicePixelRatio || 1;
  const w = gpuChart.clientWidth;
  const h = 84;
  if (gpuChart.width !== Math.round(w * dpr)) {
    gpuChart.width = Math.round(w * dpr);
    gpuChart.height = Math.round(h * dpr);
  }
  const g = gpuChart.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const tNow = clipVideo.duration ? clipVideo.currentTime / clipVideo.duration : 0;
  if (gpuAttack) {
    const valueAt = drawAttackChart(gpuAttack, g, w, h);
    g.strokeStyle = "#fff";
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(tNow * w, 0);
    g.lineTo(tNow * w, h);
    g.stroke();
    g.fillStyle = "#fff";
    g.font = "600 11px ui-monospace, monospace";
    g.fillText(`${Math.round(valueAt(tNow) * 100)}%`, Math.min(w - 34, tNow * w + 4), 12);
    return;
  }
  if (!gpuScores) return;
  const n = gpuScores.length;
  const x = (i: number) => (i / Math.max(1, n - 1)) * w;
  const y = (v: number) => h - 6 - v * (h - 12);
  g.fillStyle = "rgba(79, 209, 197, 0.15)";
  g.beginPath();
  g.moveTo(0, h);
  gpuScores.forEach((v, i) => g.lineTo(x(i), y(v)));
  g.lineTo(w, h);
  g.fill();
  g.strokeStyle = "#4fd1c5";
  g.lineWidth = 1.5;
  g.beginPath();
  gpuScores.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
  g.stroke();
  const t = clipVideo.duration ? clipVideo.currentTime / clipVideo.duration : 0;
  const i = Math.min(n - 1, Math.round(t * (n - 1)));
  g.strokeStyle = "#ef4444";
  g.beginPath();
  g.moveTo(x(i), 0);
  g.lineTo(x(i), h);
  g.stroke();
  g.fillStyle = "#fff";
  g.font = "600 11px ui-monospace, monospace";
  g.fillText(`${Math.round(gpuScores[i] * 100)}%`, Math.min(w - 34, x(i) + 4), 12);
}

function applyClip() {
  stopTracking();
  const c = currentClip();
  if (!c) return;
  const gpu = !!c.gpu;
  overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  mpTools.classList.toggle("hidden", gpu);
  clipVideo.parentElement!.classList.toggle("gpu-mode", gpu);
  gpuPanel.classList.toggle("hidden", !gpu);
  clipVideo.loop = gpu;
  clipVideo.src = c.file;
  cancelAnimationFrame(gpuRaf);
  if (gpu) {
    void clipVideo.play().catch(() => {});
    void showGpuInfo(c);
    drawGpuChart();
  }
}

clipSelect.addEventListener("change", applyClip);

let tracking = false;
let rafId: number | null = null;
let captured: PoseVectorRecord[] = [];
let framesCaptured = 0;
let clipStrikes = 0;
const clipDetector = new StrikeDetector();

function describeStrike(s: Strike) {
  return `P${s.attackerId} ${s.side} hand → P${s.victimId} ${s.target} at ${s.speed.toFixed(1)} torso/s`;
}

function updateStats(numPeople: number) {
  statPeople.textContent = String(numPeople);
  statStrikes.textContent = String(clipStrikes);
  statVectors.textContent = String(captured.length);
}

function loop() {
  if (!tracking) return;
  if (!clipVideo.paused && !clipVideo.ended && clipVideo.readyState >= 2) {
    const result = renderFrame(clipVideo, overlay);
    if (result) {
      framesCaptured++;
      for (let i = 0; i < result.landmarksByPerson.length; i++) {
        captured.push({
          tMs: result.tMs,
          personIndex: i,
          vector: flattenLandmarks(result.landmarksByPerson[i]),
        });
      }
      const strikes = clipDetector.update(result.landmarksByPerson, overlay.width, overlay.height, result.tMs);
      clipDetector.draw(overlay.getContext("2d")!, result.tMs);
      for (const s of strikes) {
        clipStrikes++;
        log(`<strong style="color:#ff2b2b">STRIKE</strong> @${clipVideo.currentTime.toFixed(2)}s — ${describeStrike(s)}`);
      }
      updateStats(result.people.length);
      exportBtn.disabled = captured.length === 0;
    }
  }
  rafId = requestAnimationFrame(loop);
}

function stopTracking() {
  tracking = false;
  if (rafId !== null) cancelAnimationFrame(rafId);
  rafId = null;
  clipVideo.pause();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statStatus.textContent = "stopped";
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  if (!isReady()) {
    statStatus.textContent = "loading model...";
    log("Loading MediaPipe PoseLandmarker (full-body, 33 landmarks, WASM)...");
    await loadPoseLandmarker();
    log("Model loaded.");
  }
  captured = [];
  framesCaptured = 0;
  clipStrikes = 0;
  clipDetector.reset();
  updateStats(0);
  exportBtn.disabled = true;

  clipVideo.currentTime = 0;
  await clipVideo.play().catch(() => {});
  tracking = true;
  statStatus.textContent = `tracking "${currentClip().label}"`;
  stopBtn.disabled = false;
  log(`Tracking started on "${currentClip().label}".`);
  loop();
});

stopBtn.addEventListener("click", () => {
  stopTracking();
  analyzing = false;
  log(`Tracking stopped. ${captured.length} vectors captured this run.`);
});

// Seeks through the clip frame by frame and timestamps detection with the
// video's own clock, so results don't depend on how fast this machine runs
// the pose model (real-time playback drops frames on slower hardware).
const ANALYZE_FPS = 15;
const analyzeBtn = document.querySelector<HTMLButtonElement>("#analyzeBtn")!;
let analyzing = false;
let tsBase = 0; // MediaPipe VIDEO mode needs strictly increasing timestamps across runs

// Setting currentTime to where the video already is doesn't always fire
// "seeked", so don't wait on it forever.
function seekTo(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener("seeked", done);
      resolve();
    };
    const timer = setTimeout(done, 1000);
    video.addEventListener("seeked", done);
    video.currentTime = t;
  });
}

async function analyzeClip(): Promise<{ t: number; strike: Strike }[]> {
  stopTracking();
  if (!isReady()) {
    statStatus.textContent = "loading model...";
    await loadPoseLandmarker();
  }
  analyzing = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  clipStrikes = 0;
  clipDetector.reset();
  updateStats(0);
  log(`Analyzing "${currentClip().label}" at ${ANALYZE_FPS} fps...`);
  clipVideo.pause();
  if (clipVideo.readyState < 1) await new Promise((r) => clipVideo.addEventListener("loadedmetadata", r, { once: true }));
  // MediaRecorder files carry no duration until the browser has seen the end.
  if (!isFinite(clipVideo.duration)) await seekTo(clipVideo, 1e9);
  const found: { t: number; strike: Strike }[] = [];
  const base = tsBase;
  for (let t = 0; t < clipVideo.duration && analyzing; t += 1 / ANALYZE_FPS) {
    await seekTo(clipVideo, t);
    const ts = base + t * 1000;
    const result = renderFrame(clipVideo, overlay, ts);
    if (!result) continue;
    const strikes = clipDetector.update(result.landmarksByPerson, overlay.width, overlay.height, ts);
    clipDetector.draw(overlay.getContext("2d")!, ts);
    statStatus.textContent = `analyzing ${t.toFixed(1)}s / ${clipVideo.duration.toFixed(1)}s`;
    for (const s of strikes) {
      clipStrikes++;
      found.push({ t, strike: s });
      log(`<strong style="color:#ff2b2b">STRIKE</strong> @${t.toFixed(2)}s — ${describeStrike(s)}`);
    }
    updateStats(result.people.length);
  }
  tsBase = base + clipVideo.duration * 1000 + 1000;
  analyzing = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  const labels = currentClip().labels;
  if (labels) scoreSession(labels, found.map((f) => f.t));
  statStatus.textContent = "analysis done";
  log(`Analysis done: ${found.length} strike(s).`);
  return found;
}

// A cue's window opens a beat before "NOW" (people jump the countdown) and
// closes before the next cue's countdown ends, so windows never overlap.
const CUE_WINDOW_BEFORE_S = 1.0;
const CUE_WINDOW_AFTER_S = 2.5;

function scoreSession(labels: SessionLabels, strikeTimes: number[]) {
  const before = labels.window?.before ?? CUE_WINDOW_BEFORE_S;
  const after = labels.window?.after ?? CUE_WINDOW_AFTER_S;
  const inWindow = (t: number, cueT: number) => t >= cueT - before && t <= cueT + after;
  const strikes = labels.cues.filter((c) => c.kind === "strike");
  const others = labels.cues.filter((c) => c.kind !== "strike");
  const missed = strikes.filter((c) => !strikeTimes.some((t) => inWindow(t, c.t)));
  const falseAlarms = others.filter((c) => strikeTimes.some((t) => inWindow(t, c.t)));
  const stray = strikeTimes.filter((t) => !labels.cues.some((c) => inWindow(t, c.t)));
  const fmt = (cs: { action: string; t: number }[]) => cs.map((c) => `${c.action}@${c.t.toFixed(1)}s`).join(", ") || "none";
  log(
    `<strong>SCORE</strong> punches caught ${strikes.length - missed.length}/${strikes.length} · ` +
      `false alarms ${falseAlarms.length}/${others.length} · stray ${stray.length}`,
  );
  log(`missed: ${fmt(missed)}`);
  log(`false alarms: ${fmt(falseAlarms)}`);
  if (stray.length) log(`stray: ${stray.map((t) => `${t.toFixed(1)}s`).join(", ")}`);
}

analyzeBtn.addEventListener("click", () => {
  void analyzeClip();
});

clipVideo.addEventListener("ended", () => {
  if (tracking) {
    stopTracking();
    log(`Clip ended. ${captured.length} vectors captured.`);
  }
});

exportBtn.addEventListener("click", () => {
  const payload = {
    clip: currentClip().label,
    landmarkFormat: "33 MediaPipe pose landmarks, [x, y, z, visibility] per landmark, flattened to 132 numbers",
    capturedAt: new Date().toISOString(),
    frameCount: framesCaptured,
    records: captured,
  };
  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${currentClip().label}-pose-vectors.json`;
  a.click();
  URL.revokeObjectURL(url);
  log(`Exported ${captured.length} vectors to ${a.download}.`);
});

log("Pose extraction ready. Pick a clip and hit Start tracking.");

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
type Tab = "dataset" | "live" | "mesh" | "model";
const tabs: Record<Tab, { btn: HTMLButtonElement; view: HTMLElement }> = {
  dataset: { btn: document.querySelector("#tabDataset")!, view: document.querySelector("#datasetView")! },
  live: { btn: document.querySelector("#tabLive")!, view: document.querySelector("#liveView")! },
  mesh: { btn: document.querySelector("#tabMesh")!, view: document.querySelector("#meshView")! },
  model: { btn: document.querySelector("#tabModel")!, view: document.querySelector("#modelView")! },
};
let modelView: import("./modelview").ModelView | null = null;
let activeTab: Tab = "dataset";

// This laptop is the bus camera on the mesh. Joining asks for location once:
// the incident is stamped wherever this device really is.
const sessionP = joinMesh("camera", "CCTV · Student Center East");
let meshView: MeshView | null = null;

function showTab(tab: Tab) {
  if (tab === activeTab) return;
  // The camera keeps running when the mesh view takes over the screen, so a
  // second punch mid-demo still fires.
  if (tab === "dataset") stopLive();
  if (activeTab === "dataset") stopTracking();
  if (activeTab === "mesh") meshView?.hide();
  activeTab = tab;
  for (const [name, { btn, view }] of Object.entries(tabs) as [Tab, (typeof tabs)[Tab]][]) {
    btn.classList.toggle("active", name === tab);
    // A hidden <video> can get paused by the browser, so a running camera stays
    // rendered underneath the map (which covers it) instead of being hidden.
    const keepCameraUnderMap = name === "live" && tab === "mesh" && liveActive;
    view.classList.toggle("hidden", name !== tab && !keepCameraUnderMap);
  }
  if (tab === "model") {
    // Chart.js loads only when someone opens the tab.
    void import("./modelview").then((m) => {
      modelView ??= new m.ModelView(tabs.model.view);
      return modelView.show();
    });
  }
  if (tab === "mesh") {
    void sessionP.then((s) => {
      meshView ??= new MeshView(tabs.mesh.view, s);
      return meshView.show();
    });
  }
}

for (const [name, { btn }] of Object.entries(tabs) as [Tab, (typeof tabs)[Tab]][]) btn.addEventListener("click", () => showTab(name));
// Stage shortcuts: 1 dataset, 2 live camera, 3 mesh map, 4 model report.
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  const tab = ({ "1": "dataset", "2": "live", "3": "mesh", "4": "model" } as Record<string, Tab>)[e.key];
  if (tab) showTab(tab);
});

// ---------------------------------------------------------------------------
// Live webcam demo: pose mesh + StrikeDetector — fires the instant one
// person's fast-moving hand enters another person's body hitbox.
// ---------------------------------------------------------------------------
const liveVideo = document.querySelector<HTMLVideoElement>("#liveVideo")!;
const liveOverlay = document.querySelector<HTMLCanvasElement>("#liveOverlay")!;
const threatBanner = document.querySelector<HTMLDivElement>("#threatBanner")!;
const liveStartBtn = document.querySelector<HTMLButtonElement>("#liveStartBtn")!;
const liveStopBtn = document.querySelector<HTMLButtonElement>("#liveStopBtn")!;
const liveStatus = document.querySelector<HTMLDivElement>("#liveStatus")!;
const livePeople = document.querySelector<HTMLDivElement>("#livePeople")!;
const livePunches = document.querySelector<HTMLDivElement>("#livePunches")!;
const liveSpeed = document.querySelector<HTMLDivElement>("#liveSpeed")!;
const liveLogLines = document.querySelector<HTMLDivElement>("#liveLogLines")!;

function liveLog(text: string) {
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `<span class="log-t">${t}</span> ${text}`;
  liveLogLines.prepend(line);
}

let liveActive = false;
let liveRafId: number | null = null;
let mediaStream: MediaStream | null = null;
const liveDetector = new StrikeDetector();
let punchCount = 0;
let lastAlertAt = 0;
// The detector already spaces strikes 1.5 s apart per pair; this only keeps
// the banner and beep from stacking.
const ALERT_COOLDOWN_MS = 1000;

let audioCtx: AudioContext | null = null;
function beep() {
  try {
    audioCtx = audioCtx ?? new AudioContext();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.18);
  } catch {
    /* ignore — audio is a bonus, not load-bearing */
  }
}

function triggerThreat(reason: string, strike?: Strike) {
  const now = performance.now();
  if (now - lastAlertAt < ALERT_COOLDOWN_MS) return;
  lastAlertAt = now;
  punchCount++;
  livePunches.textContent = String(punchCount);
  threatBanner.classList.remove("hidden");
  threatBanner.classList.add("threat-flash");
  beep();
  liveLog(`<strong>THREAT DETECTED</strong> — ${reason}`);
  setTimeout(() => {
    threatBanner.classList.add("hidden");
    threatBanner.classList.remove("threat-flash");
  }, 1200);
  void escalate(reason, strike ? captureEvidence(strike, reason) : null, strike);
}

// The frame the strike was detected on, as evidence: camera image + the pose
// mesh and hitboxes drawn on it, a ring at the point of contact, and a
// caption. Mirrored like the live view, so it shows what the room saw.
// Encoded small enough to travel the mesh as one message.
const EVIDENCE_WIDTH = 560;
const EVIDENCE_MAX_CHARS = 44_000;

function captureEvidence(s: Strike, reason: string): string | null {
  const W = liveVideo.videoWidth;
  const H = liveVideo.videoHeight;
  if (!W || !H) return null;
  const w = EVIDENCE_WIDTH;
  const h = Math.round((H * w) / W);
  const k = w / W;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  g.save();
  g.translate(w, 0);
  g.scale(-1, 1);
  g.drawImage(liveVideo, 0, 0, w, h);
  g.drawImage(liveOverlay, 0, 0, w, h);
  g.restore();
  const x = w - s.at.x * k;
  const y = s.at.y * k;
  g.strokeStyle = "#ff2b2b";
  g.lineWidth = 4;
  g.beginPath();
  g.arc(x, y, 28, 0, Math.PI * 2);
  g.stroke();
  g.fillStyle = "rgba(0, 0, 0, 0.65)";
  g.fillRect(0, h - 28, w, 28);
  g.fillStyle = "#fff";
  g.font = "600 13px ui-monospace, Menlo, monospace";
  g.fillText(`EYES · CCTV Student Center East · ${new Date().toLocaleTimeString([], { hour12: false })} · ${reason}`.slice(0, 80), 8, h - 10);
  let url = "";
  for (const q of [0.62, 0.5, 0.4, 0.3, 0.22]) {
    url = c.toDataURL("image/jpeg", q);
    if (url.length <= EVIDENCE_MAX_CHARS) break;
  }
  return url;
}

// A detected strike becomes an incident on the mesh, then the screen hands
// over to the city map so the room watches it spread. Every later punch in
// the same fight adds its own evidence frame to that incident: each one goes
// to the police, none are dropped.
const ESCALATE_TO_MAP_MS = 1800;
// Punches this close together are one fight (one incident, many frames).
const SAME_FIGHT_MS = 2 * 60_000;
let fight: { id: string; lastAt: number; punches: number } | null = null;

// What a responder reads; the detector's own description stays as evidence.
const TARGET_WORDS = { head: "the head", torso: "the body", arm: "the arm" } as const;

async function escalate(reason: string, frame: string | null, strike?: Strike) {
  const s = await sessionP;
  const now = Date.now();
  const summary = strike ? `Punch to ${TARGET_WORDS[strike.target]}, caught on CCTV` : reason;
  const ongoing = fight && now - fight.lastAt < SAME_FIGHT_MS && s.node.store.get("incidents", fight.id)?.status !== "resolved";
  if (!ongoing) {
    const id = raiseIncident(s.node, summary, frame, reason);
    fight = { id, lastAt: now, punches: 1 };
    liveLog("Incident sent to the mesh, frame dispatched to police.");
    setTimeout(() => showTab("mesh"), ESCALATE_TO_MAP_MS);
    return;
  }
  const f = fight!;
  f.punches++;
  f.lastAt = now;
  s.node.write("incidents", f.id, { punches: f.punches, lastPunchAt: now, detail: `${summary} (punch #${f.punches})`, evidence: reason });
  if (frame) addEvidenceFrame(s.node, f.id, f.punches, frame, `Punch #${f.punches}: ${summary}`);
  liveLog(`Punch #${f.punches}: frame dispatched to police.`);
}

function liveLoop() {
  if (!liveActive) return;
  if (liveVideo.readyState >= 2) {
    const result = renderFrame(liveVideo, liveOverlay);
    if (result) {
      livePeople.textContent = String(result.people.length);
      const strikes = liveDetector.update(result.landmarksByPerson, liveOverlay.width, liveOverlay.height, result.tMs);
      liveDetector.draw(liveOverlay.getContext("2d")!, result.tMs, true);
      const fastest = Math.max(0, ...liveDetector.hands.map((h) => h.speed));
      liveSpeed.textContent = `${fastest.toFixed(1)} torso/s`;
      for (const s of strikes) {
        if (recording) sessionLiveStrikes.push({ t: (performance.now() - recordStartedAt) / 1000, desc: describeStrike(s) });
        triggerThreat(describeStrike(s), s);
      }
    }
  }
  liveRafId = requestAnimationFrame(liveLoop);
}

function stopLive() {
  liveActive = false;
  if (liveRafId !== null) cancelAnimationFrame(liveRafId);
  liveRafId = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  liveVideo.pause();
  liveVideo.srcObject = null;
  liveStartBtn.disabled = false;
  liveStopBtn.disabled = true;
  recordBtn.disabled = true;
  liveStatus.textContent = "stopped";
}

liveStartBtn.addEventListener("click", async () => {
  liveStartBtn.disabled = true;
  if (!isReady()) {
    liveStatus.textContent = "loading model...";
    liveLog("Loading MediaPipe PoseLandmarker...");
    await loadPoseLandmarker();
    liveLog("Model loaded.");
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  } catch (e) {
    liveLog(`Camera access denied or unavailable: ${e}`);
    liveStartBtn.disabled = false;
    return;
  }
  liveVideo.srcObject = mediaStream;
  await liveVideo.play().catch(() => {});
  punchCount = 0;
  livePunches.textContent = "0";
  liveSpeed.textContent = "—";
  liveDetector.reset();
  liveActive = true;
  liveStatus.textContent = "live";
  liveStopBtn.disabled = false;
  recordBtn.disabled = false;
  liveLog("Camera live. A fast hand entering another person's body hitbox triggers the alert.");
  liveLoop();
});

liveStopBtn.addEventListener("click", () => {
  stopLive();
  liveLog(`Stopped. ${punchCount} threats detected this session.`);
});

// ---------------------------------------------------------------------------
// Guided training-session recorder. Records the raw camera at the real demo
// setup while on-screen cues tell the two people what to do, so every punch
// and every harmless contact in the video comes with its own timestamp. The
// Dataset tab replays these sessions frame by frame and scores the detector.
// ---------------------------------------------------------------------------
type CueKind = "strike" | "harmless" | "idle";
type Cue = { action: string; kind: CueKind; text: string };

const SESSION_CUES: Cue[] = [
  ...Array(6).fill({ action: "punch_head", kind: "strike", text: "PUNCH → FACE (stop short!)" }),
  ...Array(4).fill({ action: "punch_body", kind: "strike", text: "PUNCH → STOMACH" }),
  ...Array(3).fill({ action: "high_five", kind: "harmless", text: "HIGH-FIVE" }),
  ...Array(3).fill({ action: "handshake", kind: "harmless", text: "HANDSHAKE" }),
  ...Array(3).fill({ action: "shoulder_pat", kind: "harmless", text: "PAT SHOULDER" }),
  ...Array(2).fill({ action: "wave", kind: "harmless", text: "WAVE AT CAMERA" }),
  ...Array(2).fill({ action: "point", kind: "harmless", text: "POINT AT THEM" }),
  { action: "reach_past", kind: "harmless", text: "REACH PAST THEM" },
  ...Array(2).fill({ action: "idle", kind: "idle", text: "BOTH STAND STILL" }),
];
const COUNTDOWN_S = 3;
const NOW_HOLD_MS = 1500;
const REST_MS = 1000;

const recordBtn = document.querySelector<HTMLButtonElement>("#recordBtn")!;
const cueOverlay = document.querySelector<HTMLDivElement>("#cueOverlay")!;
const cueAction = document.querySelector<HTMLDivElement>("#cueAction")!;
const cueCount = document.querySelector<HTMLDivElement>("#cueCount")!;
const cueProgress = document.querySelector<HTMLDivElement>("#cueProgress")!;

let recording = false;
let recordStartedAt = 0;
let sessionLiveStrikes: { t: number; desc: string }[] = [];

function shuffled<T>(xs: T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

function showCue(action: string, count: string, progress: string, now = false) {
  cueOverlay.classList.remove("hidden");
  cueOverlay.classList.toggle("cue-now", now);
  cueAction.textContent = action;
  cueCount.textContent = count;
  cueProgress.textContent = progress;
}

function downloadBlob(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

function pickRecorderMime() {
  for (const m of ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

recordBtn.addEventListener("click", async () => {
  if (!mediaStream || recording) return;
  recording = true;
  recordBtn.disabled = true;
  liveStopBtn.disabled = true;

  const mimeType = pickRecorderMime();
  const recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType, videoBitsPerSecond: 6_000_000 } : undefined);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const stopped = new Promise<void>((r) => (recorder.onstop = () => r()));

  const cues = shuffled(SESSION_CUES);
  const cueLog: { action: string; kind: CueKind; t: number }[] = [];
  sessionLiveStrikes = [];

  showCue("GET IN FRAME", "", "both people visible, side by side");
  await wait(3000);
  recorder.start(1000);
  recordStartedAt = performance.now();
  liveLog(`<span class="rec-dot">●</span> Recording ${cues.length} cues (${recorder.mimeType})`);
  await wait(1500); // calm lead-in so the detector warms up on both people

  for (let i = 0; i < cues.length; i++) {
    const c = cues[i];
    const progress = `${i + 1} / ${cues.length}`;
    for (let n = COUNTDOWN_S; n > 0; n--) {
      showCue(c.text, String(n), progress);
      await wait(1000);
    }
    showCue(c.text, "NOW", progress, true);
    cueLog.push({ action: c.action, kind: c.kind, t: (performance.now() - recordStartedAt) / 1000 });
    await wait(NOW_HOLD_MS);
    showCue("", "", progress);
    await wait(REST_MS);
  }

  showCue("DONE", "", "saving…");
  await wait(1000);
  recorder.stop();
  await stopped;
  const durationS = (performance.now() - recordStartedAt) / 1000;
  recording = false;
  cueOverlay.classList.add("hidden");

  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const ext = recorder.mimeType.includes("mp4") ? "mp4" : "webm";
  const name = `eyes-session-${stamp}`;
  downloadBlob(new Blob(chunks, { type: recorder.mimeType }), `${name}.${ext}`);
  const labels = {
    name,
    video: `${name}.${ext}`,
    durationS,
    width: liveVideo.videoWidth,
    height: liveVideo.videoHeight,
    cues: cueLog,
    liveStrikes: sessionLiveStrikes,
  };
  downloadBlob(new Blob([JSON.stringify(labels, null, 2)], { type: "application/json" }), `${name}.json`);
  liveLog(`Saved ${name}.${ext} + ${name}.json — drop both into the project's sessions/ folder.`);
  recordBtn.disabled = false;
  liveStopBtn.disabled = false;
});

// First clip (the GPU DensePose set leads) once everything above exists.
applyClip();
