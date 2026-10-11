"""Step 2 of the attack model: from tracked people (extract.py) to a trained
network that says, per person and per moment, whether they're in an attack.

It judges body language and reactions, not just punches:
  posture   torso lean, lying vs upright, head below hips (fallen), crouch
  defense   arms raised, hands shielding the head
  motion    body speed, fast wrist/ankle strikes (punches, kicks), jerks
  contact   distance to the nearest person, closing in, fist/foot inside them
  crowd     how many people, others converging on or fleeing from someone

Model (~1.2M weights): a temporal transformer reads each person's last 2 s,
then a set transformer lets everyone in the frame attend to each other (an
attacker, a victim and a bystander only make sense together). Labels say
*when* an attack happens, not *who*: trained as multiple-instance learning,
frame score = max over people, so per-person scores learn who's involved.

Honest evaluation: leave-one-clip-out over the real-world clips, plus the
sparring clips that are never trained on. A gradient-boosting baseline on the
same features is scored the same way.

Run: python3 train.py   (expects data/attack/*_tracks.json and labels.json)
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/attack"
OUT = ROOT / "data/attack/out"
T = 20            # window: 2 s at 10 fps
P = 8             # people per frame (largest boxes)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7

# COCO-17 keypoints
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LANK, RANK = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# ---------------------------------------------------------------- features --

def angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if n < 1e-6 else math.acos(max(-1.0, min(1.0, float(v1 @ v2) / n))) / math.pi


def person_geom(p, W, H):
    k = np.array(p["kpts"], dtype=np.float32) if p["kpts"] else np.zeros((17, 3), np.float32)
    xy = k[:, :2] * np.array([W, H], np.float32)
    conf = k[:, 2]
    x1, y1, x2, y2 = p["box"]
    box = np.array([x1 * W, y1 * H, x2 * W, y2 * H], np.float32)
    bw, bh = box[2] - box[0], box[3] - box[1]
    s = max(bw, bh, 8.0)  # person scale; max() so a lying person keeps their size
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
    ok = lambda i: conf[i] > 0.3
    hip = (xy[LHIP] + xy[RHIP]) / 2 if ok(LHIP) and ok(RHIP) else c
    sh = (xy[LSH] + xy[RSH]) / 2 if ok(LSH) and ok(RSH) else c - np.array([0, s * 0.25], np.float32)
    return dict(xy=xy, conf=conf, box=box, bw=bw, bh=bh, s=s, c=c, hip=hip, sh=sh, ok=ok)


def signed_box_dist(pt, box, s):
    # < 0 inside the box, > 0 outside; in person-scale units
    dx = max(box[0] - pt[0], 0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0, pt[1] - box[3])
    if dx == 0 and dy == 0:
        return -min(pt[0] - box[0], box[2] - pt[0], pt[1] - box[1], box[3] - pt[1]) / s
    return math.hypot(dx, dy) / s


# Feature layout (index: meaning) — augment() depends on it.
#  0-33 pose (x,y per keypoint, relative to hips, / person scale)
# 34-50 keypoint confidence   51 lying-ness (box w/h)   52 size in frame
# 53-54 body velocity (x,y)   55 body speed
# 56-65 limb velocity relative to body: nose, L wrist, R wrist, L ankle, R ankle (x,y each)
# 66 torso lean   67 head below hips   68-69 L/R arm raised   70 hands shielding head
# 71-72 L/R knee bend   73 distance to nearest person   74 closing in
# 75 fist/foot inside nearest person (signed)   76 their lying-ness   77 their speed
# 78-79 where they are (x,y)   80-81 others converging (mean, max)   82 crowd size   83 present
NF = 84


def frame_features(geoms, prev, W, H):
    """geoms: {track_id: geom} this frame; prev: {track_id: geom} previous frame."""
    out = {}
    ids = list(geoms)
    for tid in ids:
        g = geoms[tid]
        s, xy, conf = g["s"], g["xy"], g["conf"]
        f = []
        rel = (xy - g["hip"]) / s
        f += list(np.clip(rel.flatten(), -3, 3))                       # 34 pose
        f += list(conf)                                                # 17 keypoint confidence
        f += [g["bw"] / max(g["bh"], 1), g["bh"] / H]                  # 2 lying-ness, size in frame
        pg = prev.get(tid)
        v = (g["c"] - pg["c"]) / s if pg else np.zeros(2, np.float32)
        f += list(np.clip(v, -3, 3)) + [float(np.linalg.norm(v))]      # 3 body motion
        for j in (NOSE, LWR, RWR, LANK, RANK):                         # 10 limb motion (relative to body)
            if pg and g["ok"](j) and pg["ok"](j):
                lv = ((xy[j] - g["c"]) - (pg["xy"][j] - pg["c"])) / s
            else:
                lv = np.zeros(2, np.float32)
            f += list(np.clip(lv, -3, 3))
        torso = g["sh"] - g["hip"]
        f += [math.atan2(abs(torso[0]), -torso[1] + 1e-6) / math.pi]   # 1 lean (0 upright)
        f += [float((xy[NOSE][1] - g["hip"][1]) / s) if g["ok"](NOSE) else 0.0]  # 1 head below hips
        f += [float((g["sh"][1] - xy[j][1]) / s) if g["ok"](j) else 0.0 for j in (LWR, RWR)]  # 2 arms up
        f += [min(float(np.linalg.norm(xy[j] - xy[NOSE]) / s) if g["ok"](j) and g["ok"](NOSE) else 2.0 for j in (LWR, RWR))]  # 1 shielding head
        f += [angle(xy[LHIP], xy[LKN], xy[LANK]) if all(g["ok"](j) for j in (LHIP, LKN, LANK)) else 1.0,
              angle(xy[RHIP], xy[RKN], xy[RANK]) if all(g["ok"](j) for j in (RHIP, RKN, RANK)) else 1.0]  # 2 crouch
        others = [o for o in ids if o != tid]
        if others:
            dist = {o: float(np.linalg.norm(geoms[o]["c"] - g["c"]) / s) for o in others}
            o = min(dist, key=dist.get)
            og = geoms[o]
            approach = 0.0
            if pg and o in prev:
                approach = float(np.linalg.norm(prev[o]["c"] - pg["c"]) / s) - dist[o]
            limb = min(signed_box_dist(xy[j], og["box"], s) for j in (LWR, RWR, LANK, RANK) if g["ok"](j)) if any(g["ok"](j) for j in (LWR, RWR, LANK, RANK)) else 3.0
            opv = (og["c"] - prev[o]["c"]) / og["s"] if o in prev else np.zeros(2, np.float32)
            rel_o = (og["c"] - g["c"]) / s
            # others moving toward (+) or away from (-) this person
            conv = []
            for q in others:
                if q in prev:
                    d0 = float(np.linalg.norm(prev[q]["c"] - g["c"]))
                    d1 = float(np.linalg.norm(geoms[q]["c"] - g["c"]))
                    conv.append((d0 - d1) / s)
            f += [min(dist[o], 6.0), float(np.clip(approach, -2, 2)), float(np.clip(limb, -1, 3)),
                  og["bw"] / max(og["bh"], 1), float(np.linalg.norm(opv)),
                  float(np.clip(rel_o[0], -6, 6)), float(np.clip(rel_o[1], -6, 6)),
                  float(np.mean(conv)) if conv else 0.0, float(np.max(np.abs(conv))) if conv else 0.0]  # 9
        else:
            f += [6.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        f += [len(ids) / P, 1.0]                                       # 2 crowd size, present
        assert len(f) == NF, len(f)
        out[tid] = np.array(f, np.float32)
    return out


def load_clip(name, labels):
    d = json.loads((DATA / f"{name}_tracks.json").read_text())
    W, H = d["width"], d["height"]
    frames = d["frames"]
    lab = labels.get(name, {})
    feats, boxes, ys, masks = [], [], [], []
    prev = {}
    for fr in frames:
        persons = sorted(fr["persons"], key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))[:P]
        geoms = {p["id"]: person_geom(p, W, H) for p in persons}
        feats.append(frame_features(geoms, prev, W, H))
        boxes.append({p["id"]: p["box"] for p in persons})
        prev = geoms
        t = fr["t"]
        ys.append(1.0 if any(a <= t <= b for a, b in lab.get("attack", [])) else 0.0)
        masks.append(0.0 if any(a <= t <= b for a, b in lab.get("ignore", [])) else 1.0)
    return dict(name=name, feats=feats, boxes=boxes, y=np.array(ys, np.float32), m=np.array(masks, np.float32), t=[fr["t"] for fr in frames])


def windows(clip):
    """Per frame: [P, T, NF] of each present person's last T frames, plus masks."""
    X, PM, ids_all = [], [], []
    for i, fr in enumerate(clip["feats"]):
        ids = list(fr)[:P]
        x = np.zeros((P, T, NF), np.float32)
        pm = np.zeros(P, np.float32)
        for k, tid in enumerate(ids):
            pm[k] = 1
            for j in range(T):
                src = i - (T - 1 - j)
                if src >= 0 and tid in clip["feats"][src]:
                    x[k, j] = clip["feats"][src][tid]
        X.append(x)
        PM.append(pm)
        ids_all.append(ids)
    return np.stack(X), np.stack(PM), ids_all


# ------------------------------------------------------------------- model --

class AttackNet(nn.Module):
    def __init__(self, nf=NF, d=160, heads=4, t_layers=4, p_layers=2, drop=0.25):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(nf), nn.Linear(nf, d))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc(), t_layers)
        self.people = nn.TransformerEncoder(enc(), p_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, pm):
        B, Pn, Tn, Fn = x.shape
        h = self.inp(x.reshape(B * Pn, Tn, Fn))
        h = torch.cat([self.cls.expand(B * Pn, -1, -1), h], 1) + self.pos
        tmask = torch.cat([torch.zeros(B * Pn, 1, dtype=torch.bool, device=x.device), x.reshape(B * Pn, Tn, Fn)[..., -1] == 0], 1)
        h = self.temporal(h, src_key_padding_mask=tmask)[:, 0].reshape(B, Pn, -1)
        pmask = pm == 0
        safe = pmask.clone()
        safe[safe.all(1)] = False  # a frame with nobody: keep attention defined
        h = self.people(h, src_key_padding_mask=safe)
        return self.head(h).squeeze(-1)  # [B, P] per-person logits


def frame_logit(logits, pm):
    return logits.masked_fill(pm == 0, -1e4).max(1).values


# ------------------------------------------------------------ augmentation --

# Left/right feature pairs swapped when a sample is mirrored.
LR_SWAP = [(58, 60), (59, 61), (62, 64), (63, 65), (68, 69), (71, 72)]
X_SIGNED = [53, 56, 58, 60, 62, 64, 78]  # x components that change sign
GEOMETRIC = list(range(0, 34)) + list(range(53, 66)) + list(range(73, 82))


def augment(x):
    """Mirror (with left/right swapped), scale jitter on geometric features,
    noise, and random dropped timesteps. Never touches the presence flag
    except to drop steps."""
    x = x.clone()
    B = x.shape[0]
    present = x[..., -1:].clone()
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        xf = x[flip]
        pose = xf[..., :34].reshape(*xf.shape[:-1], 17, 2)[..., FLIP, :]
        pose[..., 0] *= -1
        xf[..., :34] = pose.reshape(*xf.shape[:-1], 34)
        xf[..., 34:51] = xf[..., 34:51][..., FLIP]
        for a, b in LR_SWAP:
            xf[..., [a, b]] = xf[..., [b, a]]
        xf[..., X_SIGNED] *= -1
        x[flip] = xf
    g = torch.tensor(GEOMETRIC, device=x.device)
    scale = 1 + 0.12 * torch.randn(B, 1, 1, 1, device=x.device)
    x[..., g] = x[..., g] * scale + 0.03 * torch.randn_like(x[..., g])
    keep = (torch.rand(*x.shape[:-1], 1, device=x.device) >= 0.1).float()
    x = x * present * keep
    x[..., -1:] = present * keep
    return x


# ------------------------------------------------------------------- train --

def train_net(train_sets, epochs=60, seed=SEED, verbose=False):
    torch.manual_seed(seed)
    random.seed(seed)
    X = torch.tensor(np.concatenate([s["X"] for s in train_sets]))
    PM = torch.tensor(np.concatenate([s["PM"] for s in train_sets]))
    Y = torch.tensor(np.concatenate([s["y"] for s in train_sets]))
    M = torch.tensor(np.concatenate([s["m"] for s in train_sets]))
    keep = M > 0
    X, PM, Y = X[keep], PM[keep], Y[keep]
    pos = float(Y.mean())
    model = AttackNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps = epochs * math.ceil(len(X) / 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    pw = torch.tensor((1 - pos) / max(pos, 1e-3), device=DEVICE)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for b in range(0, len(X), 64):
            idx = perm[b:b + 64]
            xb, pb, yb = augment(X[idx].to(DEVICE)), PM[idx].to(DEVICE), Y[idx].to(DEVICE)
            pb = pb * (xb[:, :, -1, -1] > 0).float()
            loss = F.binary_cross_entropy_with_logits(frame_logit(model(xb, pb), pb), yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} loss {tot / len(X):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, s):
    model.eval()
    X, PM = torch.tensor(s["X"]).to(DEVICE), torch.tensor(s["PM"]).to(DEVICE)
    out = []
    for b in range(0, len(X), 256):
        out.append(torch.sigmoid(model(X[b:b + 256], PM[b:b + 256])).cpu())
    per = torch.cat(out).numpy() * s["PM"]
    return per, per.max(1)


def smooth(p, k=5):
    return np.convolve(np.pad(p, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def metrics(y, p, m, thr):
    keep = m > 0
    y, p = y[keep], p[keep]
    pred = p >= thr
    tp, fp, fn, tn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, len(y))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    except Exception:
        auc = float("nan")
    return dict(f1=round(f1, 3), precision=round(prec, 3), recall=round(rec, 3), accuracy=round(acc, 3), auc=round(auc, 3) if auc == auc else None, frames=int(len(y)))


def baseline_features(s):
    """Frame-level: per-feature max and mean over people, plus 1 s change."""
    fx = np.where(s["PM"][:, :, None] > 0, s["X"][:, :, -1, :], np.nan)
    mx, mn = np.nan_to_num(np.nanmax(fx, 1)), np.nan_to_num(np.nanmean(fx, 1))
    prev = np.where(s["PM"][:, :, None] > 0, s["X"][:, :, -11, :], np.nan)
    d = np.nan_to_num(np.nanmax(np.abs(fx - prev), 1))
    return np.concatenate([mx, mn, d], 1)


def main():
    t0 = time.time()
    labels = {k: v for k, v in json.loads((DATA / "labels.json").read_text()).items() if not k.startswith("_")}
    clips = {}
    for name in labels:
        if not (DATA / f"{name}_tracks.json").exists():
            print(f"skip {name}: no tracks")
            continue
        c = load_clip(name, labels)
        c["X"], c["PM"], c["ids"] = windows(c)
        c["split"] = labels[name]["split"]
        clips[name] = c
        print(f"{name}: {len(c['y'])} frames, {int(c['y'].sum())} attack, people/frame {c['PM'].sum(1).mean():.1f}", flush=True)
    train_names = [n for n, c in clips.items() if c["split"] == "train"]
    test_only = [n for n, c in clips.items() if c["split"] != "train"]
    n_params = sum(p.numel() for p in AttackNet().parameters())
    print(f"\nAttackNet: {n_params:,} weights, device {DEVICE}\n", flush=True)

    # Leave-one-clip-out: every real-world clip is scored by a model that never saw it.
    from sklearn.ensemble import HistGradientBoostingClassifier
    loco = {}
    for held in train_names:
        rest = [clips[n] for n in train_names if n != held]
        model = train_net(rest)
        _, pf = predict(model, clips[held])
        Xb = np.concatenate([baseline_features(c) for c in rest])
        yb = np.concatenate([c["y"] for c in rest])
        mb = np.concatenate([c["m"] for c in rest]) > 0
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)
        gb.fit(Xb[mb], yb[mb])
        pb = gb.predict_proba(baseline_features(clips[held]))[:, 1]
        loco[held] = dict(net=smooth(pf), gb=smooth(pb))
        print(f"held out {held}: done ({time.time() - t0:.0f}s)", flush=True)

    # One threshold for everything, chosen on the pooled held-out predictions.
    def pooled(key):
        return (np.concatenate([clips[n]["y"] for n in train_names]), np.concatenate([loco[n][key] for n in train_names]), np.concatenate([clips[n]["m"] for n in train_names]))

    results = {}
    for key in ("net", "gb"):
        y, p, m = pooled(key)
        best = max((metrics(y, p, m, t)["f1"], t) for t in np.linspace(0.1, 0.9, 33))
        thr = float(best[1])
        results[key] = dict(threshold=round(thr, 3), pooled=metrics(y, p, m, thr), per_clip={n: metrics(clips[n]["y"], loco[n][key], clips[n]["m"], thr) for n in train_names})
    print("\nLeave-one-clip-out (real-world clips, each scored by a model that never saw it):")
    for key, name in (("net", "AttackNet"), ("gb", "Gradient boosting")):
        r = results[key]
        print(f"  {name:18s} F1 {r['pooled']['f1']:.3f}  precision {r['pooled']['precision']:.3f}  recall {r['pooled']['recall']:.3f}  AUC {r['pooled']['auc']}  (threshold {r['threshold']})")
        for n, mm in r["per_clip"].items():
            print(f"      {n:24s} F1 {mm['f1']:.3f}  acc {mm['accuracy']:.3f}  AUC {mm['auc']}")

    # Final model on all real-world clips; sparring clips are a true held-out test.
    final = train_net([clips[n] for n in train_names], epochs=60, verbose=True)
    thr = results["net"]["threshold"]
    OUT.mkdir(parents=True, exist_ok=True)
    test_metrics = {}
    for name, c in clips.items():
        per, pf = predict(final, c)
        if name in loco:
            # Real-world clips: show the held-out model's frame score (honest), per-person from the final model.
            pf_show = loco[name]["net"]
        else:
            pf_show = smooth(pf)
            test_metrics[name] = metrics(c["y"], pf_show, c["m"], thr)
        frames = []
        for i, ids in enumerate(c["ids"]):
            people = [{"id": int(tid), "box": c["boxes"][i][tid], "p": round(float(per[i, k]), 3)} for k, tid in enumerate(ids)]
            frames.append({"t": c["t"][i], "p": round(float(pf_show[i]), 3), "label": int(c["y"][i]), "people": people})
        (OUT / f"{name}_attack.json").write_text(json.dumps({"clip": name, "fps": 10, "threshold": thr, "frames": frames}, separators=(",", ":")))
    if test_only:
        print("\nNever-trained-on sparring clips (final model):")
        for n, mm in test_metrics.items():
            print(f"      {n:24s} F1 {mm['f1']:.3f}  acc {mm['accuracy']:.3f}  AUC {mm['auc']}")
    torch.save({"state_dict": final.state_dict(), "nf": NF, "T": T, "P": P, "threshold": thr}, OUT / "attacknet.pt")
    summary = dict(model="AttackNet (temporal transformer + people transformer, multiple-instance learning)", weights=n_params, features=NF, window_s=T / 10,
                   train_clips=train_names, test_only_clips=test_only, loco=results, test_only=test_metrics, seconds=round(time.time() - t0))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
