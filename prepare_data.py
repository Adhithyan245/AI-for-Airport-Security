"""Build image-level labels, a pair-grouped stratified split, and perceptual-hash dedup for DvXray.

Run:  python prepare_data.py            (writes splits/*.csv and results/data_report.json)

Steps
1. Label = 1 (dangerous) if the image has >= 1 bounding box in the COCO file, else 0 (safe).
2. Pair id: DvXray file names are <folder>/<P|N><5 digits>_<OL|SD>.png; the two views of one
   bag share the prefix. We assert every bag has exactly one OL and one SD view.
3. Check whether the official split in MANIFEST.csv keeps both views of a bag together.
4. pHash every image. Images whose hashes are within HAMMING_THRESHOLD bits are duplicates.
   Bags linked by a duplicate are merged into one group (union-find) so duplicates can never
   be split across train/val/test. Within a group only the first copy of each duplicate
   cluster is kept.
5. Stratified (by group label), group-preserving 70/15/15 split, then assert zero
   cross-split pHash overlap.
6. A stratified, pair-preserving CPU subsample is drawn from inside each split.
"""
import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "xray_baggage"
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
SEED = 42
NAME_RE = re.compile(r"^images/dvxray/(positive|negative)/([PN]\d{5})_(OL|SD)\.png$")


HASH_SIZE = 16          # 256-bit pHash
BACKGROUND_LEVEL = 235  # grey level above which a pixel counts as empty belt / background


def phash_one(rel_path):
    """pHash of the bag itself. DvXray scans are mostly white background, and a plain 64-bit
    pHash of the full frame collides across unrelated bags (verified by eye), so crop to the
    non-background bounding box first and use a 256-bit hash."""
    with Image.open(DATA / rel_path) as im:
        g = im.convert("L")
    mask = g.point(lambda v: 255 if v < BACKGROUND_LEVEL else 0)
    box = mask.getbbox()
    if box is not None:
        g = g.crop(box)
    return str(imagehash.phash(g, hash_size=HASH_SIZE))


def load_labels():
    coco = json.loads((DATA / "annotations" / "dvxray_coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    boxes = Counter(a["image_id"] for a in coco["annotations"])
    df = pd.DataFrame(coco["images"])[["id", "file_name", "split"]]
    df = df.rename(columns={"id": "image_id", "split": "official_split"})
    df["n_boxes"] = df.image_id.map(boxes).fillna(0).astype(int)
    df["label"] = (df.n_boxes > 0).astype(int)

    m = df.file_name.str.extract(NAME_RE)
    assert m.notna().all().all(), "unexpected DvXray file name"
    df["folder"], df["bag"], df["view"] = m[0], m[1], m[2]
    views = df.groupby("bag").view.agg(lambda v: tuple(sorted(v)))
    assert (views == ("OL", "SD")).all(), "some bag lacks exactly one OL + one SD view"

    # per-image threat classes (unified + original names)
    per_img = {}
    for a in coco["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a["attributes"]["original_label"])
    df["threat_classes"] = df.image_id.map(lambda i: "|".join(sorted(set(per_img.get(i, [])))))
    orig = Counter(a["attributes"]["original_label"] for a in coco["annotations"])
    unified = Counter(cats[a["category_id"]] for a in coco["annotations"])
    return df, orig, unified


def hamming_pairs(hashes, threshold):
    """All (i, j), i < j, with Hamming(hash_i, hash_j) <= threshold. Exact, chunked."""
    h = np.array([[int(x[k:k + 16], 16) for k in range(0, len(x), 16)] for x in hashes], dtype=np.uint64)
    pairs, hist = [], np.zeros(65, dtype=np.int64)
    for start in range(0, len(h), 64):
        block = h[start:start + 64]
        d = np.bitwise_count(block[:, None, :] ^ h[None, :, :]).sum(-1, dtype=np.int64)
        ii, jj = np.nonzero(d <= 64)
        ii = ii + start
        upper = ii < jj
        ii, jj = ii[upper], jj[upper]
        dd = d[ii - start, jj]
        hist += np.bincount(dd, minlength=257)[:65]
        sel = dd <= threshold
        pairs += list(zip(ii[sel].tolist(), jj[sel].tolist()))
    return pairs, {k: int(v) for k, v in enumerate(hist) if v}


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def stratified_group_split(groups, rng, fracs=(0.70, 0.15, 0.15)):
    """groups: DataFrame [group, glabel, n_images]. Assign whole groups to splits so that each
    label stratum is split ~70/15/15 by image count."""
    assign = {}
    for _, g in groups.groupby("glabel"):
        g = g.iloc[rng.permutation(len(g))]
        cum = g.n_images.cumsum() / g.n_images.sum()
        for grp, c in zip(g.group, cum):
            assign[grp] = "train" if c <= fracs[0] else ("val" if c <= fracs[0] + fracs[1] else "test")
    return assign


def subsample(split_df, n_bags, rng):
    """Stratified (by bag label) sample of whole bags, both views kept."""
    bags = split_df.groupby("bag").agg(glabel=("label", "max")).reset_index()
    frac = n_bags / len(bags)
    keep = []
    for _, b in bags.groupby("glabel"):
        k = int(round(len(b) * frac))
        keep += list(b.bag.iloc[rng.permutation(len(b))[:k]])
    return split_df[split_df.bag.isin(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-only", action="store_true", help="compute hashes + distance histogram, then stop")
    ap.add_argument("--threshold", type=int, default=44,
                    help="max Hamming distance (of 256 bits) for duplicates; chosen by inspecting pairs, see README")
    ap.add_argument("--sub-bags", type=int, nargs=3, default=[200, 100, 400],
                    help="bags (pairs) in the CPU subsample for train/val/test")
    args = ap.parse_args()
    SPLITS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    report = {}

    df, orig, unified = load_labels()
    print(f"images: {len(df)}  bags: {df.bag.nunique()}")
    print("class balance (image level):", df.label.value_counts().rename({1: "dangerous", 0: "safe"}).to_dict())
    print("threat classes (boxes, original DvXray label):", dict(orig.most_common()))
    print("threat classes (boxes, unified label):", dict(unified.most_common()))
    no_box_pos = df[(df.folder == "positive") & (df.label == 0)]
    print(f"images in positive/ folder with no box (labelled safe by the box rule): {len(no_box_pos)}")
    report.update(n_images=len(df), n_bags=int(df.bag.nunique()),
                  class_balance=df.label.value_counts().rename({1: "dangerous", 0: "safe"}).to_dict(),
                  threat_boxes_original=dict(orig.most_common()),
                  threat_boxes_unified=dict(unified.most_common()),
                  positive_folder_images_without_boxes=len(no_box_pos))

    # official split check
    span = df.groupby("bag").official_split.nunique()
    n_broken = int((span > 1).sum())
    print(f"official MANIFEST split: {n_broken} of {len(span)} bags have their two views in different splits")
    report["official_split_bags_broken"] = n_broken
    use_official = n_broken == 0
    print("-> using official split" if use_official else "-> official split rejected; building pair-grouped 70/15/15")

    # perceptual hashing
    hash_csv = SPLITS / f"phash{HASH_SIZE}_crop_cache.csv"
    if hash_csv.exists():
        cache = pd.read_csv(hash_csv, dtype=str).set_index("file_name").phash
        df["phash"] = df.file_name.map(cache)
    if "phash" not in df or df.phash.isna().any():
        with Pool(3, maxtasksperchild=500) as pool:
            df["phash"] = pool.map(phash_one, df.file_name.tolist(), chunksize=64)
        df[["file_name", "phash"]].to_csv(hash_csv, index=False)

    pairs, hist = hamming_pairs(df.phash.tolist(), args.threshold)
    print("pairs of images by pHash Hamming distance (0..64, nonzero):", hist)
    report["phash_pair_distance_hist_0_64"] = hist
    report["dedup_threshold_bits"] = args.threshold
    if args.hash_only:
        return

    # duplicate clusters at image level
    uf_img = UnionFind(range(len(df)))
    for i, j in pairs:
        uf_img.union(i, j)
    df["dup_cluster"] = [uf_img.find(i) for i in range(len(df))]
    cross_label = df.groupby("dup_cluster").label.nunique()
    n_conflict = int((cross_label > 1).sum())
    print(f"duplicate clusters with conflicting labels: {n_conflict}")
    report["dup_clusters_with_conflicting_labels"] = n_conflict

    # group bags linked by duplicates
    uf = UnionFind(df.bag.unique())
    for i, j in pairs:
        uf.union(df.bag.iat[i], df.bag.iat[j])
    df["group"] = df.bag.map(uf.find)
    gsize = df.groupby("group").bag.nunique()
    print(f"bag groups: {len(gsize)}  largest group: {gsize.max()} bags  groups merging >1 bag: {(gsize > 1).sum()}")
    report["n_bag_groups"] = int(len(gsize))
    report["largest_bag_group"] = int(gsize.max())

    rng = np.random.default_rng(SEED)
    if use_official:
        df["split"] = df.official_split
    else:
        groups = df.groupby("group").agg(glabel=("label", "max"), n_images=("label", "size")).reset_index()
        df["split"] = df.group.map(stratified_group_split(groups, rng))

    # remove duplicates within each split: keep first image of each duplicate cluster
    before = len(df)
    df = df.sort_values("file_name")
    dup_mask = df.duplicated(["split", "dup_cluster"], keep="first")
    removed = df[dup_mask]
    df = df[~dup_mask].copy()
    print(f"duplicates removed: {len(removed)} (of {before}); by split: {removed.split.value_counts().to_dict()}")
    report["duplicates_removed"] = int(len(removed))
    report["duplicates_removed_by_split"] = removed.split.value_counts().to_dict()

    # leakage assertions
    assert df.groupby("bag").split.nunique().max() == 1, "a bag spans splits"
    assert df.groupby("dup_cluster").split.nunique().max() == 1, "a duplicate cluster spans splits"
    split_of = dict(zip(range(len(df)), df.split))
    h_all = df.phash.tolist()
    xpairs, _ = hamming_pairs(h_all, args.threshold)
    cross = [(i, j) for i, j in xpairs if split_of[i] != split_of[j]]
    assert len(cross) == 0, f"{len(cross)} near-duplicate pairs cross splits"
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not set(df[df.split == a].file_name) & set(df[df.split == b].file_name)
    print(f"ASSERT OK: zero cross-split overlap (file names, bags, pHash <= {args.threshold} bits)")
    report["cross_split_overlap"] = 0

    cols = ["file_name", "bag", "view", "label", "n_boxes", "threat_classes", "folder", "official_split", "split", "phash"]
    summary = {}
    for s in ("train", "val", "test"):
        part = df[df.split == s][cols].sort_values("file_name")
        part.to_csv(SPLITS / f"{s}.csv", index=False)
        summary[s] = {"images": len(part), "bags": int(part.bag.nunique()),
                      "dangerous": int(part.label.sum()), "safe": int((part.label == 0).sum())}
    print("final split:", summary)
    report["split"] = summary

    # CPU subsample (subset of each split, so it inherits the no-leak guarantees)
    sub = {}
    for s, nb in zip(("train", "val", "test"), args.sub_bags):
        part = subsample(pd.read_csv(SPLITS / f"{s}.csv"), nb, np.random.default_rng(SEED))
        part.to_csv(SPLITS / f"sub_{s}.csv", index=False)
        sub[s] = {"images": len(part), "bags": int(part.bag.nunique()),
                  "dangerous": int(part.label.sum()), "safe": int((part.label == 0).sum())}
    print("CPU subsample:", sub)
    report["cpu_subsample"] = sub
    (RESULTS / "data_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
