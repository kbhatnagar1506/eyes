"""Pose tracks for the Real Life Violence Situations dataset (RLVS: 1,000 fight
and 1,000 normal clips from real footage), in the same format as extract.py,
so the attack model can learn body language from far more than our own clips.

Each clip: sampled at 10 fps (up to MAX_S seconds), every person detected and
tracked with YOLOv8x-pose on the GPU, batched in half precision.

Output: data/attack/rlvs/<id>_tracks.json  plus data/attack/rlvs/index.json
  index = [{id, label (1 fight / 0 normal), source}]

Run: python3 scripts/attack/extract_rlvs.py <dir with the unzipped dataset>
"""

import json
import re
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import IoUTracker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/attack/rlvs"
SAMPLE_FPS = 10.0
MAX_S = 8.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO = re.compile(r"\.(mp4|avi|mov|mkv|webm)$", re.I)


def label_of(path: Path):
    """1 fight, 0 normal, None unknown. RLVS folders are Violence/NonViolence,
    files V_123 / NV_123."""
    parts = [p.lower() for p in path.parts]
    if any("nonviolence" in p or "non_violence" in p or "non-violence" in p for p in parts) or path.name.upper().startswith("NV_"):
        return 0
    if any("violence" in p for p in parts) or path.name.upper().startswith("V_"):
        return 1
    return None


def frames_of(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = fps / SAMPLE_FPS
    imgs, i, nxt = [], 0, 0.0
    while len(imgs) < MAX_S * SAMPLE_FPS:
        ok = cap.grab()
        if not ok:
            break
        if i >= nxt:
            ok, img = cap.retrieve()
            if not ok:
                break
            imgs.append(img)
            nxt += step
        i += 1
    cap.release()
    return imgs, W, H


def main():
    src = Path(sys.argv[1])
    videos = sorted(p for p in src.rglob("*") if VIDEO.search(p.name))
    labeled = [(p, label_of(p)) for p in videos]
    unknown = [p for p, y in labeled if y is None]
    n1 = sum(1 for _, y in labeled if y == 1)
    n0 = sum(1 for _, y in labeled if y == 0)
    print(f"{len(videos)} videos: {n1} fight, {n0} normal, {len(unknown)} unlabeled", flush=True)
    if unknown:
        print("unlabeled examples:", [str(p) for p in unknown[:5]], flush=True)
    if n1 < 100 or n0 < 100:
        sys.exit("labels not found: check the dataset layout")
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8x-pose.pt")
    index = []
    for n, (path, y) in enumerate(labeled):
        if y is None:
            continue
        cid = re.sub(r"[^\w-]", "_", path.stem)
        out = OUT / f"{cid}_tracks.json"
        index.append({"id": cid, "label": y, "source": path.name})
        if out.exists():
            continue
        imgs, W, H = frames_of(path)
        if len(imgs) < 5 or W == 0:
            index.pop()
            continue
        tracker = IoUTracker()
        frames = []
        for b in range(0, len(imgs), 32):
            results = model.predict(imgs[b:b + 32], imgsz=640, conf=0.25, device=DEVICE, half=DEVICE == "cuda", verbose=False)
            for j, res in enumerate(results):
                i = b + j
                persons = []
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.xyxyn.tolist()
                    confs = res.boxes.conf.tolist()
                    ids = tracker.update(boxes, i)
                    kpts = res.keypoints.data.tolist() if res.keypoints is not None else [[]] * len(boxes)
                    for bx, c, pid, k in zip(boxes, confs, ids, kpts):
                        persons.append({
                            "id": int(pid),
                            "box": [round(v, 4) for v in bx],
                            "conf": round(c, 3),
                            "kpts": [[round(x / W, 4), round(yy / H, 4), round(kc, 3)] for x, yy, kc in k],
                        })
                frames.append({"i": i, "t": round(i / SAMPLE_FPS, 2), "persons": persons})
        out.write_text(json.dumps({"clip": cid, "fps": SAMPLE_FPS, "width": W, "height": H, "frames": frames}, separators=(",", ":")))
        if n % 100 == 0:
            print(f"{n}/{len(labeled)} {cid}: {len(frames)} frames", flush=True)
    (OUT / "index.json").write_text(json.dumps(index))
    print(f"done: {len(index)} clips", flush=True)


if __name__ == "__main__":
    main()
