"""Shortcut probe: can global image statistics alone (no spatial / object information) tell
dangerous from safe? Logistic regression on a 3x32-bin colour histogram plus image-level
summary stats, fit on train and scored on val (the test set is not touched).

A high AUROC would mean the classes differ in global appearance (e.g. acquisition or packing
differences between DvXray's separately collected positive and negative bags), which a CNN/ViT
could exploit instead of finding the object. It cannot prove absence of shortcuts.

    python shortcut_probe.py [--full]
"""
import argparse
import json

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, RESULTS, read_split


def features(rel):
    with Image.open(DATA / rel) as im:
        a = np.asarray(im.convert("RGB").resize((200, 150)), dtype=np.float32)
    fg = a.mean(-1) < 235  # non-background pixels
    hist = [np.histogram(a[..., c], bins=32, range=(0, 255))[0] / a[..., c].size for c in range(3)]
    return np.concatenate(hist + [[fg.mean(), a.mean() / 255, a.std() / 255]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    tr, va = read_split("train", args.full), read_split("val", args.full)
    Xtr = np.stack([features(f) for f in tr.file_name])
    Xva = np.stack([features(f) for f in va.file_name])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)).fit(Xtr, tr.label)
    auc = float(roc_auc_score(va.label, clf.predict_proba(Xva)[:, 1]))
    fg_auc = float(roc_auc_score(va.label, Xva[:, -3]))
    out = {"split": "full" if args.full else "sub", "fit_on": "train", "scored_on": "val",
           "n_train": len(tr), "n_val": len(va), "histogram_logreg_val_auroc": auc,
           "foreground_fraction_only_val_auroc": fg_auc}
    print(json.dumps(out, indent=2))
    path = RESULTS / "metrics.json"
    m = json.loads(path.read_text())
    m[f"shortcut_probe_{out['split']}"] = out
    path.write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
