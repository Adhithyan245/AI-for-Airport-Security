"""Does the model look at the threat? Box-occlusion test on dangerous validation images.

For each dangerous image: (a) original, (b) every annotated threat box painted white
(white = no material in these X-ray renderings), (c) control: the same boxes shifted to a
random position inside the bag's non-background bounding box, also painted white.
If the model relies on the object, P(dangerous) should drop much more for (b) than (c).

Images come from the full val split but exclude the val subsample used for checkpoint
selection; the test set is not touched.

    python occlusion_test.py [--n 300]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from common import CKPT, DATA, EVAL_TF, RESULTS, SEED, autocast_ctx, build_model, read_split


def boxes_by_file():
    coco = json.loads((DATA / "annotations" / "dvxray_coco.json").read_text())
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    out = {}
    for a in coco["annotations"]:
        out.setdefault(fname[a["image_id"]], []).append(a["bbox"])
    return out


def paint(im, boxes):
    im = im.copy()
    d = ImageDraw.Draw(im)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
    return im


def shifted(im, boxes, rng):
    g = np.asarray(im.convert("L"))
    ys, xs = np.nonzero(g < 235)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    out = []
    for x, y, w, h in boxes:
        nx = rng.uniform(x0, max(x0, x1 - w))
        ny = rng.uniform(y0, max(y0, y1 - h))
        out.append([nx, ny, w, h])
    return out


@torch.no_grad()
def prob(model, kind, ims):
    x = torch.stack([EVAL_TF(i) for i in ims])
    with autocast_ctx(kind, torch.device("cpu")):
        return torch.softmax(model(x).float(), 1)[:, 1].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    val, sub_val = read_split("val", True), read_split("val", False)
    pool = val[(val.label == 1) & ~val.file_name.isin(sub_val.file_name)]
    pick = pool.iloc[rng.permutation(len(pool))[:args.n]].sort_values("file_name")
    bxs = boxes_by_file()
    conds = {"original": [], "threat_boxes_masked": [], "control_shifted_masks": []}
    for f in pick.file_name:
        with Image.open(DATA / f) as im:
            im = im.convert("RGB")
        conds["original"].append(im)
        conds["threat_boxes_masked"].append(paint(im, bxs[f]))
        conds["control_shifted_masks"].append(paint(im, shifted(im, bxs[f], rng)))
    out = {"n_images": int(len(pick)), "source": "full val positives not in val subsample", "models": {}}
    for kind in ("baseline", "tuned"):
        model = build_model(kind)
        model.load_state_dict(torch.load(CKPT / f"{kind}_sub.pt", map_location="cpu", weights_only=True))
        model.eval()
        res = {}
        for c, ims in conds.items():
            p = np.concatenate([prob(model, kind, ims[i:i + 16]) for i in range(0, len(ims), 16)])
            res[c] = {"mean_p_dangerous": float(p.mean()), "flagged_at_0.5": float((p >= 0.5).mean())}
            pd.DataFrame({"file_name": pick.file_name, "p": p}).to_csv(RESULTS / f"occlusion_{kind}_{c}.csv", index=False)
        out["models"][kind] = res
        print(kind, json.dumps(res))
    m = json.loads((RESULTS / "metrics.json").read_text())
    m["occlusion_test_sub_checkpoints"] = out
    (RESULTS / "metrics.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
