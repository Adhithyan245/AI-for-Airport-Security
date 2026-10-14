"""Evaluate both trained models on the held-out test split (run once).

    python evaluate.py            # CPU-subsample checkpoints on splits/sub_test.csv
    python evaluate.py --full     # full-split checkpoints on splits/test.csv

Step 1 (once): run each checkpoint on the test images and save per-image probabilities to
results/test_preds_<tag>.csv. If those files already exist the script refuses to run the
models again (pass --rerun-test to override, and say so in the README if you do).
Step 2: compute every metric from the saved probabilities -> results/metrics.json + plots.
Bootstrap CIs resample whole bags, because the two views of a bag are correlated.
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from common import CKPT, EVAL_TF, RESULTS, SEED, build_model, make_loader, predict, read_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["baseline", "tuned"]
N_BOOT = 2000
TARGET_RECALLS = [0.90, 0.95]


def point_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "accuracy": float((yhat == y).mean()),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "threat_recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "false_alarm_rate": float(fp / (fp + tn)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def far_at_recall_from_roc(y, p, target):
    """Lowest false-alarm rate on this set whose threshold reaches >= target threat recall."""
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= target))
    return {"threshold": float(thr[i]), "threat_recall": float(tpr[i]), "false_alarm_rate": float(fpr[i])}


def threshold_from_val(y_val, p_val, target):
    """Highest threshold that gives >= target recall on validation."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    return float(thr[int(np.argmax(tpr >= target))])


def bag_bootstrap(df, prob_cols, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap over bags. Returns {model: {metric: [samples]}} plus paired deltas."""
    rng = np.random.default_rng(seed)
    bags = df.bag.unique()
    idx_by_bag = df.groupby("bag").indices
    y = df.label.values
    out = {m: {"accuracy": [], "threat_recall": [], "precision": [], "f1": [], "auroc": []} for m in prob_cols}
    deltas = {"auroc": [], "threat_recall": [], "accuracy": []}
    for _ in range(n_boot):
        pick = rng.choice(bags, size=len(bags), replace=True)
        idx = np.concatenate([idx_by_bag[b] for b in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals = {}
        for m, col in prob_cols.items():
            pb = df[col].values[idx]
            yhat = (pb >= 0.5).astype(int)
            vals[m] = {"accuracy": (yhat == yb).mean(),
                       "threat_recall": yhat[yb == 1].mean(),
                       "precision": yb[yhat == 1].mean() if yhat.sum() else 0.0,
                       "f1": f1_score(yb, yhat, zero_division=0),
                       "auroc": roc_auc_score(yb, pb)}
            for k, v in vals[m].items():
                out[m][k].append(v)
        if len(prob_cols) == 2:
            for k in deltas:
                deltas[k].append(vals["tuned"][k] - vals["baseline"][k])
    return out, deltas


def ci(samples):
    a = np.asarray(samples)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def run_models_on_test(test_df, suffix, device, workers):
    loader = make_loader(test_df, EVAL_TF, 16, shuffle=False, workers=workers)
    for m in MODELS:
        tag = f"{m}_{suffix}"
        model = build_model(m)
        model.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=True))
        model.to(device)
        p = predict(model, loader, m, device)
        out = test_df[["file_name", "bag", "label", "folder"]].copy()
        out["p_dangerous"] = p
        out.to_csv(RESULTS / f"test_preds_{tag}.csv", index=False)
        print(f"saved test predictions for {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--rerun-test", action="store_true")
    args = ap.parse_args()
    suffix = "full" if args.full else "sub"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = read_split("test", args.full)
    val_df = read_split("val", args.full)

    pred_files = [RESULTS / f"test_preds_{m}_{suffix}.csv" for m in MODELS]
    if all(f.exists() for f in pred_files) and not args.rerun_test:
        print("test predictions already exist; computing metrics from them (test set NOT re-run)")
    else:
        run_models_on_test(test_df, suffix, device, args.workers)

    df = test_df[["file_name", "bag", "label", "folder"]].copy()
    for m, f in zip(MODELS, pred_files):
        pr = pd.read_csv(f)
        assert (pr.file_name.values == df.file_name.values).all()
        df[f"p_{m}"] = pr.p_dangerous.values
    y = df.label.values

    metrics = {"split": suffix, "n_test_images": int(len(df)), "n_test_bags": int(df.bag.nunique()),
               "n_test_dangerous": int(y.sum()), "decision_threshold": 0.5, "bootstrap": {
                   "n_resamples": N_BOOT, "unit": "bag (both views resampled together)", "ci": "percentile 95%"},
               "models": {}}
    boot, deltas = bag_bootstrap(df, {m: f"p_{m}" for m in MODELS})
    for m in MODELS:
        p = df[f"p_{m}"].values
        res = point_metrics(y, p)
        res["ci95"] = {k: ci(v) for k, v in boot[m].items()}
        res["ci95_n_valid_resamples"] = len(boot[m]["accuracy"])
        train_info = json.loads((RESULTS / f"train_{m}_{suffix}.json").read_text())
        p_val = np.load(CKPT / f"{m}_{suffix}_valprobs.npy")
        res["best_epoch"] = train_info["best_epoch"]
        res["best_val_auroc"] = train_info["best_val_auroc"]
        res["threshold_analysis"] = {}
        for t in TARGET_RECALLS:
            thr = threshold_from_val(val_df.label.values, p_val, t)
            pm = point_metrics(y, p, thr)
            res["threshold_analysis"][f"recall_{int(t * 100)}"] = {
                "test_roc_operating_point": far_at_recall_from_roc(y, p, t),
                "val_chosen_threshold": {"threshold": thr, "test_threat_recall": pm["threat_recall"],
                                         "test_false_alarm_rate": pm["false_alarm_rate"]},
            }
        # sensitivity: drop images from the positive/ folder that have no box (label is ambiguous)
        keep = ~((df.folder == "positive") & (df.label == 0)).values
        res["sensitivity_excluding_unboxed_positive_folder_images"] = {
            "n_excluded": int((~keep).sum()), **{k: v for k, v in point_metrics(y[keep], p[keep]).items()}}
        metrics["models"][m] = res
    metrics["paired_delta_tuned_minus_baseline_ci95"] = {k: {"point": float(
        metrics["models"]["tuned"][k] - metrics["models"]["baseline"][k]), "ci95": ci(v)} for k, v in deltas.items()}

    path = RESULTS / "metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[f"dvxray_test_{suffix}"] = metrics
    path.write_text(json.dumps(existing, indent=2))
    print(json.dumps(metrics, indent=2))

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, MODELS):
        cm = metrics["models"][m]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > mat.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["safe", "dangerous"])
        ax.set_yticks([0, 1], ["safe", "dangerous"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{m} (threshold 0.5)")
    fig.suptitle(f"DvXray test confusion matrices ({suffix}, n={len(df)})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_matrix_{suffix}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 5))
    for m in MODELS:
        fpr, tpr, _ = roc_curve(y, df[f"p_{m}"].values)
        ax.plot(fpr, tpr, label=f"{m} (AUROC {metrics['models'][m]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for t in TARGET_RECALLS:
        ax.axhline(t, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("threat recall")
    ax.legend(loc="lower right")
    ax.set_title(f"DvXray test ROC ({suffix})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"roc_{suffix}.png", dpi=150)


if __name__ == "__main__":
    main()
