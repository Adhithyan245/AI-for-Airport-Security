"""Diagnostic for the cross-scanner test: how often do the models flag SIXray images that have
NO boxes? The subset has only ~119 of them, from every split, and an unboxed image is not
guaranteed to be threat-free, so this is NOT a reliable false-alarm rate. It only helps tell
"detects threats on a new scanner" apart from "flags anything from a new scanner".

    python sixray_unboxed_diagnostic.py [--full]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from common import CKPT, DATA, EVAL_TF, RESULTS, build_model, make_loader, predict, read_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    suffix = "full" if args.full else "sub"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = json.loads((DATA / "annotations" / "sixray_yolo_subset_coco.json").read_text())
    boxed = {a["image_id"] for a in coco["annotations"]}
    df = pd.DataFrame([i for i in coco["images"] if i["id"] not in boxed]).sort_values("file_name")
    df["label"] = 0
    print(f"SIXray unboxed images: {len(df)} by split {df.split.value_counts().to_dict()}")
    loader = make_loader(df, EVAL_TF, 16, shuffle=False)
    val_y = read_split("val", args.full).label.values
    out = {"n_images": int(len(df)), "splits": df.split.value_counts().to_dict(), "models": {}}
    for m in ("baseline", "tuned"):
        tag = f"{m}_{suffix}"
        model = build_model(m)
        model.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=True))
        p = predict(model.to(device), loader, m, device)
        fpr, tpr, thr = roc_curve(val_y, np.load(CKPT / f"{tag}_valprobs.npy"))
        ths = {"0.5": 0.5, **{f"dvxray_val_recall_{int(t * 100)}": float(thr[int(np.argmax(tpr >= t))]) for t in (0.90, 0.95)}}
        out["models"][m] = {k: {"threshold": v, "flag_rate": float((p >= v).mean())} for k, v in ths.items()}
        print(m, json.dumps(out["models"][m]))
    path = RESULTS / "metrics.json"
    allm = json.loads(path.read_text())
    allm[f"cross_scanner_sixray_{suffix}"]["diagnostic_unboxed_images_flag_rate"] = out
    path.write_text(json.dumps(allm, indent=2))


if __name__ == "__main__":
    main()
