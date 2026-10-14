"""Optional cross-scanner check: DvXray-trained models applied, without retraining, to SIXray.

    python cross_scanner.py           # CPU-subsample checkpoints
    python cross_scanner.py --full    # full-split checkpoints

Uses the SIXray (Kaggle YOLO subset) *test* split images that have >= 1 box. This subset has
almost no negatives, so only threat recall is meaningful (no precision / false-alarm rate).
Recall is reported at the 0.5 threshold and at the DvXray-validation thresholds chosen for
90% / 95% recall. CIs are an image-level bootstrap (SIXray images are single-view).
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from common import CKPT, DATA, EVAL_TF, RESULTS, SEED, build_model, make_loader, predict, read_split

MODELS = ["baseline", "tuned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    suffix = "full" if args.full else "sub"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco = json.loads((DATA / "annotations" / "sixray_yolo_subset_coco.json").read_text())
    classes = {}
    for a in coco["annotations"]:
        classes.setdefault(a["image_id"], set()).add(a["attributes"]["original_label"])
    imgs = pd.DataFrame(coco["images"])
    imgs["n_classes"] = imgs.id.map(lambda i: len(classes.get(i, ())))
    print("SIXray subset images by split:", imgs.split.value_counts().to_dict(),
          " without boxes:", int((imgs.n_classes == 0).sum()))
    df = imgs[(imgs.split == "test") & (imgs.n_classes > 0)].sort_values("file_name").reset_index(drop=True)
    df["label"] = 1
    df["classes"] = df.id.map(lambda i: "|".join(sorted(classes[i])))
    print(f"evaluating on {len(df)} boxed SIXray test images; classes:",
          dict(Counter(c for s in df.classes for c in s.split("|"))))

    val_df = read_split("val", args.full)
    loader = make_loader(df, EVAL_TF, 16, shuffle=False, workers=args.workers)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, len(df), len(df)) for _ in range(2000)]
    out = {"split": suffix, "source": "sixray_yolo_subset test split, images with >= 1 box",
           "n_images": int(len(df)), "models": {}}
    for m in MODELS:
        tag = f"{m}_{suffix}"
        model = build_model(m)
        model.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=True))
        p = predict(model.to(device), loader, m, device)
        pd.DataFrame({"file_name": df.file_name, "classes": df.classes, "p_dangerous": p}).to_csv(
            RESULTS / f"sixray_preds_{tag}.csv", index=False)
        p_val = np.load(CKPT / f"{tag}_valprobs.npy")
        fpr, tpr, thr = roc_curve(val_df.label.values, p_val)
        thresholds = {"0.5": 0.5}
        for t in (0.90, 0.95):
            thresholds[f"dvxray_val_recall_{int(t * 100)}"] = float(thr[int(np.argmax(tpr >= t))])
        res = {}
        for name, th in thresholds.items():
            hit = (p >= th).astype(float)
            boots = [hit[i].mean() for i in boot_idx]
            per_class = {}
            for c in sorted({c for s in df.classes for c in s.split("|")}):
                mask = df.classes.str.split("|").apply(lambda s: c in s).values
                per_class[c] = {"n": int(mask.sum()), "recall": float(hit[mask].mean())}
            res[name] = {"threshold": th, "threat_recall": float(hit.mean()),
                         "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "per_class_recall": per_class}
        out["models"][m] = res
        print(m, json.dumps({k: (v["threat_recall"], v["ci95"]) for k, v in res.items()}))

    path = RESULTS / "metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[f"cross_scanner_sixray_{suffix}"] = out
    path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
