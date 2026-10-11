"""Test live AttackNet on a recorded video, through the real GPU server.

Sends the video to the server exactly like the live demo does (10 fps,
960-px JPEGs over the WebSocket), records every reply, scores it against
hand labels when there are any, and writes an annotated video: a red box
over anyone the model puts above 30%, green otherwise.

Run: python3 live_test.py <video> <wss url> <out dir> [labels key]
"""

import asyncio
import json
import ssl
import sys
import time
from pathlib import Path

import certifi
import cv2
import numpy as np
import websockets

FPS = 10
RED = 0.30   # box turns red
ALERT = 0.50  # frame score that would raise the alert (2 frames in a row)


def frames(path, width=960):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps * FPS)
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, i / FPS * 1000)
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        yield i / FPS, cv2.resize(img, (width, round(h * width / w)))


async def run(video, url):
    out = []
    tls = ssl.create_default_context(cafile=certifi.where()) if url.startswith("wss") else None
    async with websockets.connect(url, max_size=4_000_000, ssl=tls) as ws:
        for t, img in frames(video):
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            t0 = time.perf_counter()
            await ws.send(jpg.tobytes())
            r = json.loads(await ws.recv())
            r["t"], r["rtt"], r["img"] = t, round((time.perf_counter() - t0) * 1000), img
            out.append(r)
    return out


def annotate(results, path):
    h, w = results[0]["img"].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
    for r in results:
        img = r["img"].copy()
        for p in r["people"]:
            x1, y1, x2, y2 = (np.array(p["box"]) * [w, h, w, h]).astype(int)
            hot = p["p"] >= RED
            col = (59, 59, 239) if hot else (128, 222, 74)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 4 if hot else 2)
            label = f"ATTACK {round(p['p'] * 100)}%" if hot else f"{round(p['p'] * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - th - 14), (x1 + tw + 12, y1), col, -1)
            cv2.putText(img, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        bar = f"t={r['t']:.1f}s  frame score {r['p']:.2f}  gpu {r['ms']:.0f} ms  round trip {r['rtt']} ms"
        cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
        cv2.putText(img, bar, (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        vw.write(img)
    vw.release()


def score(results, labels):
    """Per labeled punch: did the alert fire inside it? Plus false alarms."""
    ts = np.array([r["t"] for r in results])
    p = np.array([r["p"] for r in results])
    y = np.array([any(a <= t <= b for a, b in labels.get("attack", [])) for t in ts])
    ign = np.array([any(a <= t <= b for a, b in labels.get("ignore", [])) for t in ts])
    fire = np.zeros_like(p, bool)
    fire[1:] = (p[1:] >= ALERT) & (p[:-1] >= ALERT)
    hits = []
    for a, b in labels.get("attack", []):
        idx = np.where((ts >= a) & (ts <= b + 0.5) & fire)[0]
        hits.append({"punch": [a, b], "caught": bool(len(idx)), "delay_s": round(float(ts[idx[0]] - a), 2) if len(idx) else None})
    calm = (~y) & (~ign)
    out = {"punches": hits, "caught": f"{sum(h['caught'] for h in hits)}/{len(hits)}",
           "false_alarm_frames": f"{int((fire & calm).sum())}/{int(calm.sum())}"}
    try:
        from sklearn.metrics import roc_auc_score
        k = ~ign
        out["frame_auc"] = round(float(roc_auc_score(y[k], p[k])), 3)
    except Exception:
        pass
    return out


def main():
    video, url, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    key = sys.argv[4] if len(sys.argv) > 4 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run(video, url))
    name = Path(video).stem.replace(" ", "_")
    annotate(results, out_dir / f"{name}_attacknet.mp4")
    ms = [r["ms"] for r in results]
    rtt = [r["rtt"] for r in results]
    summary = {"video": name, "frames": len(results), "gpu_ms_median": float(np.median(ms)), "round_trip_ms_median": float(np.median(rtt)),
               "round_trip_ms_p90": float(np.percentile(rtt, 90)), "max_frame_score": round(max(r["p"] for r in results), 3),
               "frames_with_red_box": f"{sum(any(p['p'] >= RED for p in r['people']) for r in results)}/{len(results)}"}
    if key:
        labels = json.loads((Path(__file__).resolve().parents[2] / "data/attack/labels.json").read_text())[key]
        summary.update(score(results, labels))
    (out_dir / f"{name}_attacknet.json").write_text(json.dumps({"summary": summary, "frames": [{k: v for k, v in r.items() if k != "img"} for r in results]}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
