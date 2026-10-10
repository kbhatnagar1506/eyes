import "./style.css";
import { loadPoseLandmarker, detectFrame, flattenLandmarks, isReady, type PoseVectorRecord } from "./pose";

const CLIPS = [
  // Clear, well-lit, close-up stock footage — much better pose-tracking signal
  // than grainy real CCTV. Use these for training data quality.
  { id: "sparring", label: "Sparring (clear)", file: "/demo-clips/sparring_6296158.mp4" },
  { id: "woman_sparring", label: "Woman sparring w/ man (clear)", file: "/demo-clips/woman_sparring_6296385.mp4" },
  { id: "boxing", label: "Boxing (clear)", file: "/demo-clips/boxing_6296696.mp4" },
  { id: "sparring2", label: "Sparring 2 (clear)", file: "/demo-clips/sparring2_6296587.mp4" },
  { id: "busstop_calm", label: "Bus stop, calm (clear baseline)", file: "/demo-clips/busstop_calm_36467681.mp4" },
  // Original real UCF-Crime surveillance footage — grainier, worse pose signal,
  // kept for the actual pitch narrative / authenticity once the model is trained.
  { id: "assault", label: "Assault049 (real CCTV, grainy)", file: "/demo-clips/Assault049_x264.mp4" },
  { id: "fighting", label: "Fighting049 (real CCTV, grainy)", file: "/demo-clips/Fighting049_x264.mp4" },
  { id: "normal", label: "Normal_929 (real CCTV, grainy)", file: "/demo-clips/Normal_929_x264.mp4" },
];

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <header class="lab-hud">
    <span class="eye">👁</span> EYES — pose extraction
  </header>

  <main class="lab-main">
    <div class="video-stage">
      <video id="clipVideo" controls muted playsinline></video>
      <canvas id="overlay"></canvas>
    </div>

    <aside class="lab-panel">
      <div class="panel-row">
        <label class="panel-label">Clip</label>
        <select id="clipSelect">
          ${CLIPS.map((c) => `<option value="${c.id}">${c.label}</option>`).join("")}
        </select>
      </div>
      <div class="panel-row">
        <button id="startBtn">Start tracking</button>
        <button id="stopBtn" class="secondary" disabled>Stop</button>
      </div>

      <div class="stat-grid">
        <div class="stat"><div class="stat-label">Status</div><div id="statStatus" class="stat-value">idle</div></div>
        <div class="stat"><div class="stat-label">People now</div><div id="statPeople" class="stat-value">0</div></div>
        <div class="stat"><div class="stat-label">Frames captured</div><div id="statFrames" class="stat-value">0</div></div>
        <div class="stat"><div class="stat-label">Vectors captured</div><div id="statVectors" class="stat-value">0</div></div>
      </div>

      <button id="exportBtn" class="export-btn" disabled>Export vectors (JSON)</button>
      <p class="hint">
        Each captured frame records one 132-number vector per detected person
        (33 MediaPipe pose landmarks × x, y, z, visibility), tagged with the
        clip label and timestamp — that's the training set.
      </p>

      <div class="log" id="logPanel"><div class="log-title">Log</div><div id="logLines"></div></div>
    </aside>
  </main>
`;

const clipVideo = document.querySelector<HTMLVideoElement>("#clipVideo")!;
const overlay = document.querySelector<HTMLCanvasElement>("#overlay")!;
const clipSelect = document.querySelector<HTMLSelectElement>("#clipSelect")!;
const startBtn = document.querySelector<HTMLButtonElement>("#startBtn")!;
const stopBtn = document.querySelector<HTMLButtonElement>("#stopBtn")!;
const exportBtn = document.querySelector<HTMLButtonElement>("#exportBtn")!;
const statStatus = document.querySelector<HTMLDivElement>("#statStatus")!;
const statPeople = document.querySelector<HTMLDivElement>("#statPeople")!;
const statFrames = document.querySelector<HTMLDivElement>("#statFrames")!;
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

clipVideo.src = currentClip().file;
clipSelect.addEventListener("change", () => {
  stopTracking();
  clipVideo.src = currentClip().file;
});

let tracking = false;
let rafId: number | null = null;
let captured: PoseVectorRecord[] = [];
let framesCaptured = 0;

function updateStats(numPeople: number) {
  statPeople.textContent = String(numPeople);
  statFrames.textContent = String(framesCaptured);
  statVectors.textContent = String(captured.length);
}

function loop() {
  if (!tracking) return;
  if (!clipVideo.paused && !clipVideo.ended && clipVideo.readyState >= 2) {
    const result = detectFrame(clipVideo, overlay);
    if (result) {
      framesCaptured++;
      for (let i = 0; i < result.landmarksByPerson.length; i++) {
        captured.push({
          tMs: result.tMs,
          personIndex: i,
          vector: flattenLandmarks(result.landmarksByPerson[i]),
        });
      }
      updateStats(result.numPeople);
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
  log(`Tracking stopped. ${captured.length} vectors captured this run.`);
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
