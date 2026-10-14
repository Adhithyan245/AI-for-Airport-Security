"""Shared pieces for train.py / evaluate.py / cross_scanner.py."""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "xray_baggage"
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
SEED = 42

# torchvision ViT-B/16 IMAGENET1K_V1 was trained with ImageNet statistics.
# (The old notebook normalised with mean=std=0.5, which does not match the weights.)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# DvXray images are 800x600 and the threat can be anywhere, so resize the whole bag
# instead of centre-cropping.
EVAL_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
TUNED_TRAIN_TF = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_csv(split, full):
    return SPLITS / (f"{split}.csv" if full else f"sub_{split}.csv")


class XrayDataset(Dataset):
    def __init__(self, df, transform, root=DATA):
        self.files = df.file_name.tolist()
        self.labels = df.label.astype(int).tolist()
        self.transform, self.root = transform, Path(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.root / self.files[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i], i


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df, transform, batch_size, shuffle, seed=SEED, root=DATA, workers=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(XrayDataset(df, transform, root), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, worker_init_fn=_seed_worker, generator=g,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def build_model(kind):
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    d = model.heads.head.in_features
    if kind == "baseline":
        model.heads.head = nn.Linear(d, 2)
    elif kind == "tuned":
        model.heads.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2))
    else:
        raise ValueError(kind)
    return model


def autocast_ctx(kind, device):
    """Tuned model uses mixed precision: fp16 on CUDA (with GradScaler), bf16 on CPU
    (CPU autocast only supports bf16, whose fp32-sized exponent needs no loss scaling)."""
    enabled = kind == "tuned"
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def predict(model, loader, kind, device):
    """Returns P(dangerous) for every item in loader order."""
    model.eval()
    probs = np.zeros(len(loader.dataset), dtype=np.float64)
    for x, _, idx in loader:
        with autocast_ctx(kind, device):
            logits = model(x.to(device))
        probs[idx.numpy()] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
    return probs


def read_split(split, full):
    return pd.read_csv(split_csv(split, full))
