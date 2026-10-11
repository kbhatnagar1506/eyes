"""AttackNet, live: the same pipeline it was trained on, one webcam frame at a time.

The laptop sends a JPEG every 100 ms (10 fps, the rate AttackNet was trained
at) over a WebSocket. For each frame: YOLOv8x-pose on the GPU, the IoU
tracker, the 84 body-language features (train.py's own code), each person's
last 2 s, then AttackNet. The reply is every person's box and attack
probability, plus the frame's score (its most violent person, averaged over
the last 3 frames).

Run on the GPU VM: python3 live_server.py   (listens on 127.0.0.1:8765;
run_live.sh puts a public wss:// address in front of it)
"""

import asyncio
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import websockets
from ultralytics import YOLO

from extract import IoUTracker
from train import NF, P, T, AttackNet, frame_features, person_geom

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 8765
SMOOTH = 3  # frames averaged for the frame score

pose = YOLO("yolov8x-pose.pt")
ckpt = torch.load(ROOT / "data/attack/out/attacknet.pt", map_location=DEVICE)
net = AttackNet().to(DEVICE)
net.load_state_dict(ckpt["state_dict"])
net.eval()


class Stream:
    """One camera: its own tracker and each person's recent features."""

    def __init__(self):
        self.tracker = IoUTracker()
        self.prev = {}
        self.feats = deque(maxlen=T)
        self.recent = deque(maxlen=SMOOTH)
        self.i = 0

    @torch.no_grad()
    def step(self, img):
        t0 = time.perf_counter()
        H, W = img.shape[:2]
        res = pose.predict(img, imgsz=640, conf=0.25, device=DEVICE, half=DEVICE == "cuda", verbose=False)[0]
        persons = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxyn.tolist()
            ids = self.tracker.update(boxes, self.i)
            kpts = res.keypoints.data.tolist() if res.keypoints is not None else [[]] * len(boxes)
            for b, pid, k in zip(boxes, ids, kpts):
                persons.append({"id": int(pid), "box": b, "kpts": [[x / W, y / H, c] for x, y, c in k]})
        # The P largest people, as in training.
        persons = sorted(persons, key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))[:P]
        geoms = {p["id"]: person_geom(p, W, H) for p in persons}
        f = frame_features(geoms, self.prev, W, H)
        self.prev = geoms
        self.feats.append(f)
        self.i += 1

        # Each present person's last T frames, exactly like train.windows().
        hist = list(self.feats)
        ids = list(f)[:P]
        x = np.zeros((1, P, T, NF), np.float32)
        pm = np.zeros((1, P), np.float32)
        for k, tid in enumerate(ids):
            pm[0, k] = 1
            for j in range(T):
                src = len(hist) - T + j
                if src >= 0 and tid in hist[src]:
                    x[0, k, j] = hist[src][tid]
        if ids:
            per = torch.sigmoid(net(torch.from_numpy(x).to(DEVICE), torch.from_numpy(pm).to(DEVICE)))[0].cpu().numpy() * pm[0]
        else:
            per = np.zeros(P, np.float32)
        frame_p = float(per.max()) if ids else 0.0
        self.recent.append(frame_p)
        box_of = {p["id"]: p["box"] for p in persons}
        return {
            "p": round(float(np.mean(self.recent)), 4),
            "raw": round(frame_p, 4),
            "people": [{"id": int(tid), "box": [round(v, 4) for v in box_of[tid]], "p": round(float(per[k]), 4)} for k, tid in enumerate(ids)],
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }


async def handler(ws):
    stream = Stream()
    latest = {"buf": None}
    ready = asyncio.Event()
    print(f"camera connected ({ws.remote_address})", flush=True)

    async def receive():
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                latest["buf"] = msg
                ready.set()

    async def work():
        while True:
            await ready.wait()
            ready.clear()
            img = cv2.imdecode(np.frombuffer(latest["buf"], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out = await asyncio.to_thread(stream.step, img)
            await ws.send(json.dumps(out))

    tasks = [asyncio.create_task(receive()), asyncio.create_task(work())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        print("camera disconnected", flush=True)


async def main():
    # Warm up the GPU so the first real frame is fast.
    Stream().step(np.zeros((720, 1280, 3), np.uint8))
    print(f"AttackNet live on {DEVICE}, ws://127.0.0.1:{PORT}", flush=True)
    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=4_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
