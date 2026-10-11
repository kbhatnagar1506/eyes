// Live AttackNet: the webcam streams to our GPU server (YOLOv8x-pose +
// AttackNet on an NVIDIA T4) at 10 fps, the rate the model was trained at;
// every person's box and attack probability comes back. Reconnects on its
// own; when the server is unreachable the live demo carries on with the
// in-browser StrikeDetector alone.

export interface AttackPerson {
  id: number;
  box: [number, number, number, number]; // normalized x1, y1, x2, y2
  p: number;
}
export interface AttackResult {
  p: number; // frame score: most violent person, last 3 frames
  people: AttackPerson[];
  ms: number; // GPU time
  rtt: number; // network + GPU, from the moment the frame was sent
  encodeMs: number; // JPEG encoding in this browser
  at: number;
}

const KEY = "eyes-attacknet";
const SEND_EVERY_MS = 100;
const SEND_WIDTH = 960;

// ?attacknet=wss://… (saved for next time), else the last one used.
export function attackNetUrl(): string | null {
  const fromLink = new URLSearchParams(location.search).get("attacknet");
  if (fromLink && /^wss?:\/\//.test(fromLink)) {
    try {
      localStorage.setItem(KEY, fromLink);
    } catch {
      /* not persisted: fine for this visit */
    }
    return fromLink;
  }
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export class LiveAttackNet {
  latest: AttackResult | null = null;
  status: "connecting" | "live" | "offline" = "connecting";
  private ws: WebSocket | null = null;
  private timer = 0;
  private sentAt: number[] = [];
  private inflight = 0;
  private lastSend = 0;
  private encodeMs = 0;
  private canvas = document.createElement("canvas");
  private stopped = false;
  private retry = 1000;
  private url: string;
  private video: HTMLVideoElement;
  private onResult: (r: AttackResult) => void;

  constructor(url: string, video: HTMLVideoElement, onResult: (r: AttackResult) => void) {
    this.url = url;
    this.video = video;
    this.onResult = onResult;
  }

  start() {
    this.stopped = false;
    this.connect();
    this.timer = window.setInterval(() => this.send(), SEND_EVERY_MS);
  }

  stop() {
    this.stopped = true;
    clearInterval(this.timer);
    this.ws?.close();
    this.ws = null;
    this.latest = null;
  }

  private connect() {
    if (this.stopped) return;
    this.status = "connecting";
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.status = "live";
      this.retry = 1000;
      this.sentAt = [];
      this.inflight = 0;
    };
    ws.onmessage = (e) => {
      const r = JSON.parse(String(e.data)) as Omit<AttackResult, "rtt" | "at">;
      this.inflight = Math.max(0, this.inflight - 1);
      const sent = this.sentAt.shift() ?? performance.now();
      const out = { ...r, rtt: Math.round(performance.now() - sent), encodeMs: Math.round(this.encodeMs), at: performance.now() };
      this.latest = out;
      this.onResult(out);
    };
    ws.onclose = () => {
      this.ws = null;
      if (this.stopped) return;
      this.status = "offline";
      setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 2, 8000);
    };
    ws.onerror = () => ws.close();
    this.ws = ws;
  }

  private send() {
    const v = this.video;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !v.videoWidth) return;
    // Never queue up: if two frames are still in flight, skip this one.
    if (this.inflight >= 2) {
      if (performance.now() - this.lastSend > 2000) {
        this.inflight = 0;
        this.sentAt = [];
      }
      return;
    }
    const w = SEND_WIDTH;
    const h = Math.round((v.videoHeight * w) / v.videoWidth);
    if (this.canvas.width !== w) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const t0 = performance.now();
    this.canvas.getContext("2d")!.drawImage(v, 0, 0, w, h);
    this.inflight++;
    this.canvas.toBlob(
      (blob) => {
        const now = performance.now();
        this.encodeMs = now - t0;
        if (blob && this.ws?.readyState === WebSocket.OPEN) {
          this.sentAt.push(now);
          this.lastSend = now;
          this.ws.send(blob);
        } else this.inflight = Math.max(0, this.inflight - 1);
      },
      "image/jpeg",
      0.72,
    );
  }
}

// AttackNet's boxes on the live overlay (video pixels; the overlay is
// mirrored by CSS, so labels are drawn flipped back to read normally).
export function drawAttackBoxes(ctx: CanvasRenderingContext2D, r: AttackResult, W: number, H: number, redAt: number) {
  for (const p of r.people) {
    const [x1, y1, x2, y2] = [p.box[0] * W, p.box[1] * H, p.box[2] * W, p.box[3] * H];
    const hot = p.p >= redAt;
    const col = hot ? "#ef4444" : "#4ade80";
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = hot ? 6 : 2.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = hot ? `ATTACK ${Math.round(p.p * 100)}%` : `${Math.round(p.p * 100)}%`;
    ctx.font = `700 ${hot ? 26 : 18}px ui-monospace, Menlo, monospace`;
    const tw = ctx.measureText(label).width + 16;
    const th = hot ? 36 : 28;
    // Top-left as the audience sees it = top-right in video pixels.
    ctx.translate(x2, Math.max(th, y1));
    ctx.scale(-1, 1);
    ctx.fillStyle = col;
    ctx.fillRect(0, -th, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, 8, -10);
    ctx.restore();
  }
}
