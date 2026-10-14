"""Train one model (baseline or tuned) and keep the checkpoint with the best validation AUROC.

    python train.py --model baseline            # CPU subsample (splits/sub_*.csv)
    python train.py --model tuned
    python train.py --model baseline --full     # full pair-grouped split (needs a GPU)

The test split is never read here.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR

from common import (CKPT, EVAL_TF, RESULTS, SEED, TUNED_TRAIN_TF, autocast_ctx, build_model,
                    make_loader, predict, read_split, seed_everything)


def class_weights(labels):
    """Inverse-frequency weights, normalised so a balanced set would get 1.0 per class."""
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "tuned"], required=True)
    ap.add_argument("--full", action="store_true", help="use the full split instead of the CPU subsample")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (use 2-4 on Colab)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{args.model}_{'full' if args.full else 'sub'}"
    print(f"[{tag}] device={device} torch={torch.__version__}")

    train_df, val_df = read_split("train", args.full), read_split("val", args.full)
    print(f"train {len(train_df)} imgs ({train_df.label.sum()} dangerous), val {len(val_df)} imgs ({val_df.label.sum()} dangerous)")

    train_tf = EVAL_TF if args.model == "baseline" else TUNED_TRAIN_TF
    train_loader = make_loader(train_df, train_tf, args.batch_size, shuffle=True, seed=args.seed, workers=args.workers)
    val_loader = make_loader(val_df, EVAL_TF, 16, shuffle=False, workers=args.workers)

    torch.manual_seed(args.seed)  # identical backbone/head init RNG state for both models
    model = build_model(args.model).to(device)
    w = class_weights(train_df.label.values).to(device)
    print("class weights [safe, dangerous]:", w.tolist())
    criterion = nn.CrossEntropyLoss(weight=w)

    if args.model == "baseline":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler, scaler, clip = None, None, None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        clip = 0.8

    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    history, best = [], {"auroc": -1.0}
    y_val = val_df.label.values
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, (x, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(args.model, device):
                loss = criterion(model(x).float(), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            run_loss += loss.item() * len(y)
            n += len(y)
            if step % 10 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {run_loss / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if scheduler is not None:
            scheduler.step()

        p_val = predict(model, val_loader, args.model, device)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.tensor(p_val).clamp(1e-7, 1 - 1e-7), torch.tensor(y_val, dtype=torch.float64)))
        rec = {"epoch": epoch, "train_loss": run_loss / n, "val_auroc": float(roc_auc_score(y_val, p_val)),
               "val_acc@0.5": float(((p_val >= 0.5) == y_val).mean()), "val_bce": val_loss,
               "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - t0}
        history.append(rec)
        print(f"[{tag}] {json.dumps(rec)}", flush=True)
        if rec["val_auroc"] > best["auroc"]:
            best = {"auroc": rec["val_auroc"], "epoch": epoch}
            torch.save(model.state_dict(), CKPT / f"{tag}.pt")
            np.save(CKPT / f"{tag}_valprobs.npy", p_val)
            print(f"  -> new best val AUROC, checkpoint saved")

    out = {"model": args.model, "full": args.full, "seed": args.seed, "epochs": args.epochs,
           "batch_size": args.batch_size, "device": str(device), "best_epoch": best["epoch"],
           "best_val_auroc": best["auroc"], "class_weights": w.tolist(), "history": history}
    (RESULTS / f"train_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[{tag}] done. best epoch {best['epoch']} val AUROC {best['auroc']:.4f}")


if __name__ == "__main__":
    main()
