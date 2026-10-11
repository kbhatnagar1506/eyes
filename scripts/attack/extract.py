"""Step 1 of the attack model: detect and track every person in the source
clips with YOLOv8-pose, sampled at exactly the DensePose renders' frame times
(10 fps), so boxes line up frame for frame with the GPU videos.

Output: data/attack/<clip>_tracks.json
  {clip, fps, width, height, frames: [{i, t, persons: [{id, box, conf, kpts}]}]}
  box = [x1, y1, x2, y2] and kpts = 17 x [x, y, conf], all normalized 0..1.

Run: python3 scripts/attack/extract.py [clip ...]
"""

import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/attack"
SAMPLE_FPS = 10.0
# NVIDIA GPU when present (the GCP T4 VM). Never Apple MPS: Ultralytics
# documents a pose bug there (wrong keypoints), and keypoints are what the
# attack model learns from.
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "yolov8x-pose.pt" if DEVICE == "cuda" else "yolov8m-pose.pt"

# clip id -> (source video, DensePose render it must align with)
CLIPS = {
    "sparring_6296158": ("public/demo-clips/sparring_6296158.mp4", "densepose-output/sparring_6296158_densepose.mp4"),
    "woman_sparring_6296385": ("public/demo-clips/woman_sparring_6296385.mp4", "densepose-output/woman_sparring_6296385_densepose.mp4"),
    "boxing_6296696": ("public/demo-clips/boxing_6296696.mp4", "densepose-output/boxing_6296696_densepose.mp4"),
    "sparring2_6296587": ("public/demo-clips/sparring2_6296587.mp4", "densepose-output/sparring2_6296587_densepose.mp4"),
    "busstop_calm_36467681": ("public/demo-clips/busstop_calm_36467681.mp4", "densepose-output/busstop_calm_36467681_densepose.mp4"),
    "Assault049_x264": ("public/demo-clips/Assault049_x264.mp4", "densepose-multi/Assault049_x264_multi_tracked.mp4"),
    "Assault050_x264": ("public/demo-clips/Assault050_x264.mp4", "densepose-multi/Assault050_x264_multi_tracked.mp4"),
    "Assault051_x264": ("public/demo-clips/Assault051_x264.mp4", "densepose-multi/Assault051_x264_multi_tracked.mp4"),
    "Fighting042_x264": ("public/demo-clips/Fighting042_x264.mp4", "densepose-multi/Fighting042_x264_multi_tracked.mp4"),
    "Fighting049_x264": ("public/demo-clips/Fighting049_x264.mp4", "densepose-multi/Fighting049_x264_multi_tracked.mp4"),
    "Normal_929_x264": ("public/demo-clips/Normal_929_x264.mp4", "densepose-multi/Normal_929_x264_multi_tracked.mp4"),
    # Real punches at the demo setup (no DensePose render: sampled over its full length).
    "gf-punch-1": ("sessions/gf-punch-1.mov", None),
}


class IoUTracker:
    """Gives each person a stable id: match to recent tracks by box overlap
    (greedy, best overlap first). At 10 fps people move little between frames."""

    def __init__(self, min_iou=0.25, max_age=5):
        self.tracks = {}  # id -> (box, last_frame)
        self.next_id = 1
        self.min_iou, self.max_age = min_iou, max_age

    @staticmethod
    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def update(self, boxes, frame):
        live = {tid: v for tid, v in self.tracks.items() if frame - v[1] <= self.max_age}
        pairs = sorted(((self.iou(b, tb), i, tid) for i, b in enumerate(boxes) for tid, (tb, _) in live.items()), reverse=True)
        ids, used = [None] * len(boxes), set()
        for score, i, tid in pairs:
            if score < self.min_iou:
                break
            if ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = (b, frame)
        return ids


def render_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def extract(clip: str, model: YOLO):
    src_rel, render_rel = CLIPS[clip]
    src = ROOT / src_rel
    cap = cv2.VideoCapture(str(src))
    if render_rel:
        n = render_frame_count(ROOT / render_rel)
    else:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * SAMPLE_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    tracker = IoUTracker()
    for i in range(n):
        t = i / SAMPLE_FPS
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            break
        # Upscale small CCTV frames so people aren't a few pixels tall.
        imgsz = 1280 if DEVICE == "cuda" else 640
        res = model.predict(img, imgsz=imgsz, conf=0.25, device=DEVICE, verbose=False)[0]
        persons = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxyn.tolist()
            confs = res.boxes.conf.tolist()
            ids = tracker.update(boxes, i)
            kpts = res.keypoints.data.tolist() if res.keypoints is not None else [[]] * len(boxes)
            for b, c, pid, k in zip(boxes, confs, ids, kpts):
                persons.append({
                    "id": int(pid) if pid is not None else -1,
                    "box": [round(v, 4) for v in b],
                    "conf": round(c, 3),
                    "kpts": [[round(x / W, 4), round(y / H, 4), round(kc, 3)] for x, y, kc in k],
                })
        frames.append({"i": i, "t": round(t, 2), "persons": persons})
    cap.release()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{clip}_tracks.json").write_text(json.dumps({"clip": clip, "fps": SAMPLE_FPS, "width": W, "height": H, "frames": frames}))
    people = max((len(f["persons"]) for f in frames), default=0)
    print(f"{clip}: {len(frames)} frames (render has {n}), up to {people} people in a frame", flush=True)


def main():
    clips = sys.argv[1:] or [c for c in CLIPS if not (OUT / f"{c}_tracks.json").exists()]
    model = YOLO(MODEL)
    for clip in clips:
        extract(clip, model)


if __name__ == "__main__":
    main()
