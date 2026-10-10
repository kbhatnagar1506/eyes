// Real, in-browser detection signal — not a trained anomaly classifier, but not
// faked either: live frame-differencing motion analysis on the actual video
// pixels, computed every sampled frame. This is the honest "fallback" path
// called out in the design doc's CV-approach decision: no hidden network call,
// no canned number — the score genuinely comes from what's on screen.
//
// Heuristic: sudden, large, sustained motion (a fight/assault breaking out)
// scores high; static or slow, steady motion (a normal clip, a calm shot)
// stays low. It's a heuristic, not semantic understanding of "violence" — that's
// stated plainly in the design doc rather than oversold here.

export interface DetectionFrame {
  score: number; // smoothed 0-1, this is what drives the UI + threshold
  rawScore: number; // unsmoothed 0-1
  motion: number; // normalized frame-to-frame pixel delta, 0-1ish
  baseline: number; // rolling "calm" motion level for this clip
}

export interface LiveDetector {
  stop: () => void;
}

const SAMPLE_W = 48;
const SAMPLE_H = 27;

export async function loadDetector(): Promise<void> {
  // No model to download for the frame-diff heuristic — kept as an async no-op
  // so main.ts's "loading model..." UI step and call site don't need to change
  // if/when a real pretrained model replaces this later.
  return;
}

/**
 * Runs live motion-based scoring against a <video> element while it plays.
 */
export function runLiveDetection(
  video: HTMLVideoElement,
  onFrame: (f: DetectionFrame) => void,
  opts: { sampleMs?: number } = {},
): LiveDetector {
  const sampleMs = opts.sampleMs ?? 120;
  let stopped = false;

  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE_W;
  canvas.height = SAMPLE_H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

  let prevGray: Float32Array | null = null;
  let smoothed = 0;
  let baseline = 0.02; // rolling "calm" motion estimate, adapts per clip
  let frameCount = 0;

  const tick = () => {
    if (stopped) return;
    if (video.paused || video.ended || video.readyState < 2) {
      if (!stopped) setTimeout(tick, sampleMs);
      return;
    }

    ctx.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
    const { data } = ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);

    const gray = new Float32Array(SAMPLE_W * SAMPLE_H);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      // standard luminance weighting
      gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    let motion = 0;
    if (prevGray) {
      let sum = 0;
      for (let p = 0; p < gray.length; p++) {
        sum += Math.abs(gray[p] - prevGray[p]);
      }
      motion = sum / gray.length / 255; // ~0 (still) to ~1 (total change)
    }
    prevGray = gray;
    frameCount++;

    // Adapt baseline slowly toward observed motion (so camera shake / gentle
    // handheld footage doesn't permanently read as "anomalous"), but only
    // once we have a few frames of context, and never upward during an active
    // spike (so the spike itself doesn't get absorbed into the baseline).
    if (frameCount > 5 && motion < baseline * 2.5) {
      baseline = baseline * 0.95 + motion * 0.05;
    }

    const spike = Math.max(0, motion - baseline * 1.4);
    const rawScore = Math.min(1, spike / 0.055); // tuned against sample clips

    smoothed = smoothed * 0.65 + rawScore * 0.35;

    onFrame({ score: smoothed, rawScore, motion, baseline });

    if (!stopped) setTimeout(tick, sampleMs);
  };

  tick();

  return {
    stop: () => {
      stopped = true;
    },
  };
}
