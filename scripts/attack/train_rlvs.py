"""AttackNet v2: learn body language from Real Life Violence Situations (RLVS),
then adapt to our footage, scored only on clips the model never trained on.

Stage A  Pretrain on RLVS (1,000 fight + 1,000 normal clips, one label per
         clip). Multiple-instance learning twice over: a frame's score is
         its most violent person, a clip's score is its most violent moment.
         Split 70/10/20 by clip: train / pick thresholds / final test.
Stage B  Fine-tune on our own frame-labeled clips (CCTV + the demo setup),
         leave-one-clip-out: each of our clips is scored by a model that
         never saw it. The sparring clips are never trained on at all.

Whichever stage does better on our held-out clips is what the Dataset tool
shows, and every box it draws comes from a model that never saw that clip.

Run: python3 train_rlvs.py   (expects data/attack/rlvs/ from extract_rlvs.py,
     data/attack/*_tracks.json and labels.json)
"""

import json
import math
import random
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import DATA, DEVICE, NF, OUT, P, SEED, T, AttackNet, augment, frame_logit, load_clip, metrics, smooth, windows

K = 4  # moments sampled per clip per step (clip score = the most violent)

# Feature groups (see the layout in train.py), for "what body language matters".
GROUPS = {
    "Posture (pose, lean, fallen, crouch)": list(range(0, 34)) + [51, 66, 67, 71, 72],
    "Body speed": [53, 54, 55],
    "Strikes (wrist / ankle speed)": list(range(56, 66)),
    "Defense (arms up, shielding head)": [68, 69, 70],
    "Contact (distance, closing in, limb inside)": [73, 74, 75, 76, 77],
    "Crowd (converging, size)": [80, 81, 82],
    "Keypoint visibility": list(range(34, 51)) + [52],
}
REPORT = {}


def load_rlvs(item):
    c = load_clip(f"rlvs/{item['id']}", {f"rlvs/{item['id']}": {"attack": [[0, 1e9]]} if item["label"] else {}})
    X, PM, ids = windows(c)
    head = json.loads((DATA / f"rlvs/{item['id']}_tracks.json").read_text())
    return dict(id=item["id"], label=item["label"], X=X.astype(np.float16), PM=PM, n=len(c["y"]), W=head["width"], H=head["height"],
                tracks=len({p["id"] for f in head["frames"] for p in f["persons"]}))


def load_ours(name, labels):
    c = load_clip(name, labels)
    c["X"], c["PM"], c["ids"] = windows(c)
    c["split"] = labels[name]["split"]
    return c


@torch.no_grad()
def predict(model, X, PM):
    model.eval()
    out = []
    for b in range(0, len(X), 256):
        xb = torch.tensor(X[b:b + 256], dtype=torch.float32, device=DEVICE)
        out.append(torch.sigmoid(model(xb, torch.tensor(PM[b:b + 256], device=DEVICE))).cpu())
    per = torch.cat(out).numpy() * PM
    return per, per.max(1)


def clip_score(pf):
    # Mean of the top quarter of moments: one blip doesn't make a fight.
    s = np.sort(smooth(pf))[::-1]
    return float(s[: max(1, len(s) // 4)].mean())


def pretrain(clips, epochs=50, seed=SEED, val=None, curve=None):
    torch.manual_seed(seed)
    random.seed(seed)
    model = AttackNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    bs = 32
    steps = epochs * math.ceil(len(clips) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    for ep in range(epochs):
        model.train()
        order = list(range(len(clips)))
        random.shuffle(order)
        tot = 0.0
        for b in range(0, len(order), bs):
            batch = [clips[i] for i in order[b:b + bs]]
            xs, ps = [], []
            for c in batch:
                # Prefer moments with someone in frame and a full 2 s behind them.
                cand = [i for i in range(len(c["PM"])) if c["PM"][i].sum() > 0 and i >= T // 2] or list(range(len(c["PM"])))
                pick = [random.choice(cand) for _ in range(K)]
                xs.append(c["X"][pick])
                ps.append(c["PM"][pick])
            xb = augment(torch.tensor(np.concatenate(xs), dtype=torch.float32, device=DEVICE))
            pb = torch.tensor(np.concatenate(ps), device=DEVICE)
            pb = pb * (xb[:, :, -1, -1] > 0).float()
            fl = frame_logit(model(xb, pb), pb).reshape(len(batch), K)
            yb = torch.tensor([float(c["label"]) for c in batch], device=DEVICE)
            loss = F.binary_cross_entropy_with_logits(fl.max(1).values, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
        point = {"epoch": ep + 1, "loss": round(tot / len(clips), 4)}
        if val is not None and (ep % 5 == 4 or ep == epochs - 1):
            point.update(quick_eval(model, val))
        if curve is not None:
            curve.append(point)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d} loss {tot / len(clips):.4f} {point}", flush=True)
    return model


def clip_scores(model, clips):
    return np.array([clip_score(predict(model, c["X"], c["PM"])[1]) for c in clips]), np.array([c["label"] for c in clips], np.float32)


def quick_eval(model, clips):
    from sklearn.metrics import roc_auc_score
    s, y = clip_scores(model, clips)
    acc = max((((s >= t) == (y == 1)).mean()) for t in np.linspace(0.05, 0.95, 19))
    return {"val_auc": round(float(roc_auc_score(y, s)), 4), "val_accuracy": round(float(acc), 4)}


def curve_points(y, s, kind="roc", n=60):
    from sklearn.metrics import precision_recall_curve, roc_curve
    if not (0 < y.sum() < len(y)):
        return []
    if kind == "roc":
        a, b, _ = roc_curve(y, s)
    else:
        b, a, _ = precision_recall_curve(y, s)  # x = recall, y = precision
    idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
    return [[round(float(a[i]), 4), round(float(b[i]), 4)] for i in idx]


def confusion(y, s, thr, m=None):
    keep = np.ones_like(y, bool) if m is None else m > 0
    y, pred = y[keep] == 1, s[keep] >= thr
    return {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()), "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum())}


def hist(v, bins=20):
    return [int(x) for x in np.histogram(v, bins=bins, range=(0, 1))[0]]


def importance(model, clips):
    """Shuffle one group of features across all moments and see how much the
    clip AUC drops: the bigger the drop, the more the model relies on it."""
    from sklearn.metrics import roc_auc_score
    base_s, y = clip_scores(model, clips)
    base = float(roc_auc_score(y, base_s))
    sizes = [len(c["X"]) for c in clips]
    allX = np.concatenate([c["X"] for c in clips])
    rng = np.random.default_rng(SEED)
    out = []
    for name, cols in GROUPS.items():
        Xs = allX.copy()
        perm = rng.permutation(len(Xs))
        Xs[:, :, :, cols] = allX[perm][:, :, :, cols]
        s, at = [], 0
        for c, n in zip(clips, sizes):
            s.append(clip_score(predict(model, Xs[at:at + n], c["PM"])[1]))
            at += n
        out.append({"group": name, "auc_drop": round(base - float(roc_auc_score(y, np.array(s))), 4)})
    return {"base_auc": round(base, 4), "groups": sorted(out, key=lambda g: -g["auc_drop"])}


def finetune(base_state, sets, epochs=15, seed=SEED, curve=None):
    """Frame-labeled clips: frame score = most violent person (as train.py)."""
    torch.manual_seed(seed)
    model = AttackNet().to(DEVICE)
    model.load_state_dict(base_state)
    X = np.concatenate([s["X"] for s in sets])
    PM = np.concatenate([s["PM"] for s in sets])
    Y = np.concatenate([s["y"] for s in sets])
    keep = np.concatenate([s["m"] for s in sets]) > 0
    X, PM, Y = X[keep], PM[keep], Y[keep]
    pos = float(Y.mean())
    pw = torch.tensor((1 - pos) / max(pos, 1e-3), device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X))
        tot = 0.0
        for b in range(0, len(X), 64):
            idx = perm[b:b + 64]
            xb = augment(torch.tensor(X[idx], dtype=torch.float32, device=DEVICE))
            pb = torch.tensor(PM[idx], device=DEVICE) * (xb[:, :, -1, -1] > 0).float()
            yb = torch.tensor(Y[idx], device=DEVICE)
            loss = F.binary_cross_entropy_with_logits(frame_logit(model(xb, pb), pb), yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        if curve is not None:
            curve.append({"epoch": ep + 1, "loss": round(tot / len(X), 4)})
    return model


def best_threshold(y, p, m):
    return float(max((metrics(y, p, m, t)["f1"], t) for t in np.linspace(0.05, 0.95, 37))[1])


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    index = json.loads((DATA / "rlvs/index.json").read_text())
    with Pool(8) as pool:
        rlvs = pool.map(load_rlvs, index, chunksize=16)
    rlvs = [c for c in rlvs if c["n"] >= 5]
    by_label = {0: [c for c in rlvs if c["label"] == 0], 1: [c for c in rlvs if c["label"] == 1]}
    split = {"train": [], "val": [], "test": []}
    for group in by_label.values():
        rng.shuffle(group)
        a, b = int(len(group) * 0.7), int(len(group) * 0.8)
        split["train"] += group[:a]
        split["val"] += group[a:b]
        split["test"] += group[b:]
    print(f"RLVS: {len(rlvs)} clips ({len(by_label[1])} fight, {len(by_label[0])} normal) -> "
          f"train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} ({time.time() - t0:.0f}s)", flush=True)

    REPORT["data"] = {
        "rlvs": {
            "clips": {k: {"fight": sum(c["label"] for c in v), "normal": sum(1 - c["label"] for c in v)} for k, v in split.items()},
            "frames": int(sum(c["n"] for c in rlvs)),
            "tracks": int(sum(c["tracks"] for c in rlvs)),
            "seconds_hist": {lbl: [int(x) for x in np.histogram([c["n"] / 10 for c in rlvs if c["label"] == lbl], bins=16, range=(0, 8))[0]] for lbl in (0, 1)},
            "people_hist": {lbl: [int(x) for x in np.bincount(np.concatenate([c["PM"].sum(1).astype(int) for c in rlvs if c["label"] == lbl]), minlength=P + 1)[: P + 1]] for lbl in (0, 1)},
            "resolution": {k: int(v) for k, v in zip(*np.unique([("≤360p" if c["H"] <= 360 else "480p" if c["H"] <= 480 else "720p" if c["H"] <= 720 else "1080p+") for c in rlvs], return_counts=True))},
        }
    }
    labels = {k: v for k, v in json.loads((DATA / "labels.json").read_text()).items() if not k.startswith("_")}
    ours = {n: load_ours(n, labels) for n in labels if (DATA / f"{n}_tracks.json").exists()}
    ours_train = [n for n, c in ours.items() if c["split"] == "train"]
    sparring = [n for n, c in ours.items() if c["split"] != "train"]
    n_params = sum(p.numel() for p in AttackNet().parameters())
    _m = AttackNet()
    REPORT["params"] = {name: int(sum(p.numel() for p in mod.parameters())) for name, mod in _m.named_children()}
    REPORT["params"]["pos + cls tokens"] = int(_m.pos.numel() + _m.cls.numel())
    print(f"AttackNet: {n_params:,} weights, device {DEVICE}\n\nStage A: pretraining on RLVS", flush=True)

    # ---- stage A ----------------------------------------------------------------
    REPORT["data"]["ours"] = [{"clip": n, "split": c["split"], "frames": len(c["y"]), "attack_frames": int(c["y"].sum()), "people_per_frame": round(float(c["PM"].sum(1).mean()), 2)} for n, c in ours.items()]
    REPORT["training"] = {"pretrain": [], "finetune": {}}
    A = pretrain(split["train"], val=split["val"], curve=REPORT["training"]["pretrain"])
    A_state = {k: v.detach().clone() for k, v in A.state_dict().items()}

    def rlvs_eval(model, clips, clip_thr=None, frame_thr=None):
        ys, ss, fy, fp = [], [], [], []
        for c in clips:
            _, pf = predict(model, c["X"], c["PM"])
            ys.append(c["label"])
            ss.append(clip_score(pf))
            fy.append(np.full(len(pf), float(c["label"])))
            fp.append(smooth(pf))
        ys, ss = np.array(ys, np.float32), np.array(ss)
        fy, fp = np.concatenate(fy), np.concatenate(fp)
        if clip_thr is None:
            clip_thr = float(max(((((ss >= t) == (ys == 1)).mean()), t) for t in np.linspace(0.05, 0.95, 37))[1])
        if frame_thr is None:
            frame_thr = best_threshold(fy, fp, np.ones_like(fy))
        from sklearn.metrics import roc_auc_score
        return dict(clip_threshold=round(clip_thr, 3), frame_threshold=round(frame_thr, 3), clip_accuracy=round(float(((ss >= clip_thr) == (ys == 1)).mean()), 3),
                    clip_auc=round(float(roc_auc_score(ys, ss)), 3), frames=metrics(fy, fp, np.ones_like(fy), frame_thr), clips=len(clips))

    val = rlvs_eval(A, split["val"])
    test = rlvs_eval(A, split["test"], val["clip_threshold"], val["frame_threshold"])
    print(f"  RLVS test ({test['clips']} clips never trained on): clip accuracy {test['clip_accuracy']}, AUC {test['clip_auc']}", flush=True)
    thr_A = val["frame_threshold"]
    ts, ty = clip_scores(A, split["test"])
    REPORT["rlvs_test"] = {"roc": curve_points(ty, ts), "pr": curve_points(ty, ts, "pr"), "confusion": confusion(ty, ts, val["clip_threshold"]),
                           "threshold": val["clip_threshold"], "hist": {"fight": hist(ts[ty == 1]), "normal": hist(ts[ty == 0])}}
    print("  feature importance on RLVS val...", flush=True)
    REPORT["importance"] = importance(A, split["val"])

    def ours_eval(preds, thr, names):
        y = np.concatenate([ours[n]["y"] for n in names])
        p = np.concatenate([preds[n] for n in names])
        m = np.concatenate([ours[n]["m"] for n in names])
        return dict(threshold=round(thr, 3), pooled=metrics(y, p, m, thr), per_clip={n: metrics(ours[n]["y"], preds[n], ours[n]["m"], thr) for n in names})

    out_A = {}
    for n, c in ours.items():
        per, pf = predict(A, c["X"], c["PM"])
        out_A[n] = (per, smooth(pf))
    res_A = {"ours": ours_eval({n: v[1] for n, v in out_A.items()}, thr_A, ours_train), "sparring": ours_eval({n: v[1] for n, v in out_A.items()}, thr_A, sparring)}
    print(f"  our clips, never seen: F1 {res_A['ours']['pooled']['f1']}, AUC {res_A['ours']['pooled']['auc']}; sparring F1 {res_A['sparring']['pooled']['f1']}, AUC {res_A['sparring']['pooled']['auc']}", flush=True)

    # ---- stage B ----------------------------------------------------------------
    print("\nStage B: fine-tuning on our clips, leave-one-clip-out", flush=True)
    out_B = {}
    for held in ours_train:
        mdl = finetune(A_state, [ours[n] for n in ours_train if n != held], curve=REPORT["training"]["finetune"].setdefault(held, []))
        per, pf = predict(mdl, ours[held]["X"], ours[held]["PM"])
        out_B[held] = (per, smooth(pf))
        print(f"  held out {held} ({time.time() - t0:.0f}s)", flush=True)
    thr_B = best_threshold(np.concatenate([ours[n]["y"] for n in ours_train]), np.concatenate([out_B[n][1] for n in ours_train]), np.concatenate([ours[n]["m"] for n in ours_train]))
    B_final = finetune(A_state, [ours[n] for n in ours_train])
    for n in sparring:
        per, pf = predict(B_final, ours[n]["X"], ours[n]["PM"])
        out_B[n] = (per, smooth(pf))
    res_B = {"ours": ours_eval({n: v[1] for n, v in out_B.items()}, thr_B, ours_train), "sparring": ours_eval({n: v[1] for n, v in out_B.items()}, thr_B, sparring)}
    rlvs_B = rlvs_eval(B_final, split["test"])
    print(f"  our clips, held out: F1 {res_B['ours']['pooled']['f1']}, AUC {res_B['ours']['pooled']['auc']}; sparring F1 {res_B['sparring']['pooled']['f1']}, AUC {res_B['sparring']['pooled']['auc']}", flush=True)

    # ---- what the Dataset tool shows ----------------------------------------------
    score = lambda r: (r["ours"]["pooled"]["auc"] or 0) + (r["sparring"]["pooled"]["auc"] or 0)
    use = "B" if score(res_B) > score(res_A) else "A"
    shown, thr = (out_B, thr_B) if use == "B" else (out_A, thr_A)
    print(f"\nShowing stage {use}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, c in ours.items():
        per, pf = shown[n]
        frames = []
        for i, ids in enumerate(c["ids"]):
            people = [{"id": int(tid), "box": c["boxes"][i][tid], "p": round(float(per[i, k]), 3)} for k, tid in enumerate(ids)]
            frames.append({"t": c["t"][i], "p": round(float(pf[i]), 3), "label": int(c["y"][i]), "people": people})
        (OUT / f"{n}_attack.json").write_text(json.dumps({"clip": n, "fps": 10, "threshold": thr, "seen": False, "frames": frames}, separators=(",", ":")))
    def ours_curves(out, thr):
        res = {}
        for key, names in (("ours", ours_train), ("sparring", sparring)):
            y = np.concatenate([ours[n]["y"] for n in names])
            pr = np.concatenate([out[n][1] for n in names])
            m = np.concatenate([ours[n]["m"] for n in names])
            k = m > 0
            res[key] = {"roc": curve_points(y[k], pr[k]), "confusion": confusion(y, pr, thr, m)}
        return res
    REPORT["ours"] = {"A": ours_curves(out_A, thr_A), "B": ours_curves(out_B, thr_B)}
    final = A if use == "A" else B_final
    torch.save({"state_dict": final.state_dict(), "nf": NF, "T": T, "P": P, "threshold": thr}, OUT / "attacknet.pt")
    old = {}
    try:
        old = json.loads((OUT / "metrics.json").read_text()).get("loco", {}).get("net", {}).get("pooled", {})
    except Exception:
        pass
    summary = dict(
        model="AttackNet v2 (temporal transformer + people transformer, multiple-instance learning), pretrained on RLVS",
        weights=n_params, features=NF, window_s=T / 10, shown=use, threshold=round(thr, 3),
        rlvs=dict(clips=len(rlvs), train=len(split["train"]), val=len(split["val"]), test=len(split["test"]), stage_A_test=test, stage_B_test=rlvs_B),
        stage_A=res_A, stage_B=res_B, v1_ours_pooled=old,
        train_clips=ours_train, test_only_clips=sparring, seconds=round(time.time() - t0),
    )
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=1))
    REPORT["summary"] = summary
    (OUT / "report.json").write_text(json.dumps(REPORT, separators=(",", ":")))
    print(json.dumps({k: summary[k] for k in ("shown", "threshold")}), f"\ndone in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
