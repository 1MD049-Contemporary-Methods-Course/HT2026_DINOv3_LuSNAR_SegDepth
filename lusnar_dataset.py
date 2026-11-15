"""
lusnar_dataset.py - PyTorch loading for the LuSNAR lunar dataset
(monocular depth estimation + semantic segmentation, e.g. with a DINOv3 backbone).

Expected layout per scene (as found on disk):

    <ROOT>/<scene>/image0/color/<timestamp>.png   left RGB         -> model input
    <ROOT>/<scene>/image0/depth/<timestamp>.pfm   left depth (f32) -> depth target
    <ROOT>/<scene>/image0/label/<timestamp>.png   colour-coded seg -> segmentation target
    <ROOT>/<scene>/image1/color/<timestamp>.png   right RGB        -> optional (stereo / self-supervision)

Folder names are matched regardless of upper/lower case.

What a sample looks like (after resizing to image_size x image_size):
    image       float32 [3, H, W]  normalised with ImageNet mean/std (DINOv3 web backbones)
    depth       float32 [1, H, W]  metres (as stored in the PFM); invalid pixels set to 0
    depth_valid bool    [1, H, W]  True where depth is finite, > 0, not sky (65504) and <= max_depth
    label       int64   [H, W]     class index 0..4, IGNORE_INDEX (255) for unknown colours
    image_right float32 [3, H, W]  only if return_right=True
    scene, stem str

Quick check from the command line:
    python lusnar_dataset.py --root /proj/<project>/task1_groupc_dataset
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------
CLASS_NAMES = ["Lunar Regolith", "Impact Crater", "Rock", "Mountain", "Sky"]
CLASS_COLORS = ["#BB469C", "#7800C8", "#E8FA50", "#AD451F", "#22C9F8"]   # from the LuSNAR README
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255
# LuSNAR stores "no surface hit" (the sky) as the float16 maximum 65504 m -> treated as invalid depth.
DEPTH_SKY_VALUE = 65504.0

# ImageNet statistics - used by the DINOv3 web-image (LVD-1689M) backbones.
# The satellite DINOv3 variants (SAT-493M) use different values; change these if you use one.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUBDIRS = {
    "rgb": "image0/color",
    "depth": "image0/depth",
    "label": "image0/label",
    "rgb_right": "image1/color",
}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEPTH_EXTS = (".pfm", ".npy", ".png", ".tif", ".tiff")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# 24-bit colour code -> class index, as sorted arrays for a fast vectorised lookup
_CODES = np.array([(r << 16) | (g << 8) | b for r, g, b in map(_hex_to_rgb, CLASS_COLORS)], dtype=np.int64)
_ORDER = np.argsort(_CODES)
_SORTED_CODES, _SORTED_CLASSES = _CODES[_ORDER], np.arange(NUM_CLASSES)[_ORDER]


# --------------------------------------------------------------------------------------
# File readers
# --------------------------------------------------------------------------------------
def read_pfm(path: str | Path, bottom_to_top: bool = False) -> np.ndarray:
    """Read a PFM file as float32. 'Pf' = 1 channel, 'PF' = 3 channels.

    The PFM spec stores rows bottom-to-top, but LuSNAR writes them top-to-bottom
    (checked: the 65504 "sky" pixels line up exactly with the Sky label only without a flip).
    Set bottom_to_top=True for spec-conforming PFMs from other sources."""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"Pf", b"PF"):
            raise ValueError(f"{path} is not a PFM file (header {kind!r})")
        dims = f.readline().split()
        while len(dims) < 2:
            dims += f.readline().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(f.readline().strip())
        data = np.fromfile(f, ("<" if scale < 0 else ">") + "f4")
    shape = (h, w, 3) if kind == b"PF" else (h, w)
    data = data.reshape(shape)
    if bottom_to_top:
        data = np.flipud(data)
    return np.ascontiguousarray(data).astype(np.float32)


def load_depth(path: str | Path) -> np.ndarray:
    p = str(path).lower()
    if p.endswith(".pfm"):
        d = read_pfm(path)
    elif p.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = np.asarray(Image.open(path)).astype(np.float32)
    return d[..., 0] if d.ndim == 3 else d


def rgb_to_class(label_rgb: np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Colour-coded label image [H, W, 3] -> class-index mask [H, W] (uint8). Unknown colours -> ignore_index."""
    lab = label_rgb[..., :3].astype(np.int64)
    codes = (lab[..., 0] << 16) | (lab[..., 1] << 8) | lab[..., 2]
    pos = np.clip(np.searchsorted(_SORTED_CODES, codes), 0, NUM_CLASSES - 1)
    hit = _SORTED_CODES[pos] == codes
    return np.where(hit, _SORTED_CLASSES[pos], ignore_index).astype(np.uint8)


def class_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Class-index mask -> colour image (for plotting). Ignore pixels become black."""
    palette = np.zeros((256, 3), np.uint8)
    for i, h in enumerate(CLASS_COLORS):
        palette[i] = _hex_to_rgb(h)
    return palette[np.asarray(mask, dtype=np.uint8)]


# --------------------------------------------------------------------------------------
# Index (file paths only - nothing heavy is read here)
# --------------------------------------------------------------------------------------
def _resolve_dir(base: Path, rel: str) -> Path | None:
    cur = base
    for part in rel.split("/"):
        if not cur.is_dir():
            return None
        hits = [p for p in cur.iterdir() if p.is_dir() and p.name.lower() == part.lower()]
        if not hits:
            return None
        cur = hits[0]
    return cur


def _list_stems(folder: Path | None, exts: Sequence[str]) -> dict[str, str]:
    if folder is None:
        return {}
    with os.scandir(folder) as it:
        return {os.path.splitext(e.name)[0]: e.path for e in it
                if e.is_file() and e.name.lower().endswith(tuple(exts))}


def find_scenes(root: str | Path) -> list[Path]:
    """Scene = folder containing image0/. ROOT may be a single scene or a folder of scenes."""
    root = Path(root)
    if _resolve_dir(root, "image0"):
        return [root]
    cands = [p for p in root.glob("*") if p.is_dir()] + [p for p in root.glob("*/*") if p.is_dir()]
    scenes = sorted(p for p in cands if not p.name.startswith(".") and _resolve_dir(p, "image0"))
    if not scenes:
        raise FileNotFoundError(f"No scene folders (containing image0/) under {root}")
    return scenes


def build_index(root: str | Path, require: Sequence[str] = ("rgb", "depth", "label"),
                cache: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """One row per timestamp with the file path of each modality. Rows missing any
    `require`d modality are dropped. Optionally cached to a CSV file."""
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, dtype={"stem": str})
    rows = []
    for scene in find_scenes(root):
        files = {
            "rgb": _list_stems(_resolve_dir(scene, SUBDIRS["rgb"]), IMG_EXTS),
            "depth": _list_stems(_resolve_dir(scene, SUBDIRS["depth"]), DEPTH_EXTS),
            "label": _list_stems(_resolve_dir(scene, SUBDIRS["label"]), IMG_EXTS),
            "rgb_right": _list_stems(_resolve_dir(scene, SUBDIRS["rgb_right"]), IMG_EXTS),
        }
        for stem in sorted(set().union(*[f.keys() for f in files.values()])):
            rows.append({"scene": scene.name, "stem": stem, **{k: v.get(stem) for k, v in files.items()}})
    idx = pd.DataFrame(rows, columns=["scene", "stem", *SUBDIRS])
    n_all = len(idx)
    idx = idx.dropna(subset=list(require)).reset_index(drop=True)
    idx["t_ns"] = pd.to_numeric(idx["stem"].str.extract(r"(\d+)")[0], errors="coerce")
    idx = idx.sort_values(["scene", "t_ns", "stem"]).reset_index(drop=True)
    if verbose:
        print(f"Indexed {len(idx):,} usable frames in {idx.scene.nunique()} scenes "
              f"({n_all - len(idx):,} dropped for missing {list(require)})")
    if idx.empty:
        raise FileNotFoundError(f"No complete samples found under {root}")
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        idx.to_csv(cache, index=False)
    return idx


# --------------------------------------------------------------------------------------
# Scene-based split (avoids leakage between near-identical consecutive frames)
# --------------------------------------------------------------------------------------
def split_by_scene(index: pd.DataFrame, val_scenes: Sequence[str] = (), test_scenes: Sequence[str] = (),
                   n_val: int = 1, n_test: int = 1, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Split whole scenes into train/val/test. Give scene names explicitly, or let it pick
    n_val / n_test scenes at random (reproducible with `seed`)."""
    scenes = sorted(index.scene.unique())
    rng = random.Random(seed)
    if not val_scenes and not test_scenes:
        picked = rng.sample(scenes, n_val + n_test)
        val_scenes, test_scenes = picked[:n_val], picked[n_val:]
    unknown = (set(val_scenes) | set(test_scenes)) - set(scenes)
    if unknown:
        raise ValueError(f"Unknown scene(s): {sorted(unknown)}. Available: {scenes}")
    train_scenes = [s for s in scenes if s not in set(val_scenes) | set(test_scenes)]
    parts = {"train": train_scenes, "val": list(val_scenes), "test": list(test_scenes)}
    return {k: index[index.scene.isin(v)].reset_index(drop=True) for k, v in parts.items()}


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    hflip: float = 0.5                 # probability of a horizontal flip (applied to all outputs)
    brightness: float = 0.2            # +- fraction, RGB only
    contrast: float = 0.2              # +- fraction, RGB only
    random_crop_scale: tuple[float, float] | None = (0.6, 1.0)   # square crop area fraction, None = off


class LuSNARDataset(Dataset):
    """Lazy LuSNAR dataset: only paths are held in memory; files are read in __getitem__."""

    def __init__(self, index: pd.DataFrame, image_size: int | tuple[int, int] = 512,
                 max_depth: float | None = None, augment: AugmentConfig | bool = False,
                 return_right: bool = False, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 ignore_index: int = IGNORE_INDEX):
        self.idx = index.reset_index(drop=True)
        self.size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)  # (H, W)
        if self.size[0] % 16 or self.size[1] % 16:
            print(f"Warning: image_size {self.size} is not a multiple of 16 (DINOv3 patch size).")
        self.max_depth = max_depth
        self.aug = AugmentConfig() if augment is True else (augment or None)
        self.return_right = return_right
        if return_right and self.idx["rgb_right"].isna().any():
            raise ValueError("return_right=True but some frames have no image1/color file; "
                             "build the index with require=('rgb','depth','label','rgb_right').")
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.idx)

    # -- helpers ---------------------------------------------------------------------
    def _crop_box(self, w: int, h: int):
        if not (self.aug and self.aug.random_crop_scale):
            return None
        lo, hi = self.aug.random_crop_scale
        side = int(round(min(w, h) * np.sqrt(random.uniform(lo, hi))))
        x0, y0 = random.randint(0, w - side), random.randint(0, h - side)
        return (x0, y0, x0 + side, y0 + side)

    def _rgb_tensor(self, img: Image.Image, box, flip, jitter) -> torch.Tensor:
        if box: img = img.crop(box)
        img = img.resize(self.size[::-1], Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        if flip: x = x.flip(-1)
        if jitter:
            b, c = jitter
            x = x * b
            x = (x - x.mean()) * c + x.mean()
            x = x.clamp(0, 1)
        return (x - self.mean) / self.std

    # -- main ------------------------------------------------------------------------
    def __getitem__(self, i: int) -> dict:
        r = self.idx.iloc[i]
        rgb = Image.open(r["rgb"]).convert("RGB")
        depth = load_depth(r["depth"])
        label = rgb_to_class(np.asarray(Image.open(r["label"]).convert("RGB")), self.ignore_index)

        box = self._crop_box(*rgb.size)
        flip = bool(self.aug and random.random() < self.aug.hflip)
        jitter = None
        if self.aug and (self.aug.brightness or self.aug.contrast):
            jitter = (1 + random.uniform(-self.aug.brightness, self.aug.brightness),
                      1 + random.uniform(-self.aug.contrast, self.aug.contrast))

        image = self._rgb_tensor(rgb, box, flip, jitter)

        # depth & label: nearest-neighbour resize so invalid values / classes are never blended
        if box:
            x0, y0, x1, y1 = box
            depth, label = depth[y0:y1, x0:x1], label[y0:y1, x0:x1]
        depth = np.asarray(Image.fromarray(np.ascontiguousarray(depth, dtype=np.float32)).resize(self.size[::-1], Image.NEAREST))
        label = np.asarray(Image.fromarray(np.ascontiguousarray(label, dtype=np.uint8)).resize(self.size[::-1], Image.NEAREST))
        valid = np.isfinite(depth) & (depth > 0) & (depth < DEPTH_SKY_VALUE)
        if self.max_depth is not None:
            valid &= depth <= self.max_depth
        depth = np.where(valid, depth, 0.0).astype(np.float32)

        depth_t = torch.from_numpy(depth)[None]
        valid_t = torch.from_numpy(valid)[None]
        label_t = torch.from_numpy(label.astype(np.int64))
        if flip:
            depth_t, valid_t, label_t = depth_t.flip(-1), valid_t.flip(-1), label_t.flip(-1)

        out = {"image": image, "depth": depth_t, "depth_valid": valid_t, "label": label_t,
               "scene": r["scene"], "stem": str(r["stem"])}
        if self.return_right:
            # NOTE: with a horizontal flip, left/right roles swap geometrically - disable hflip for stereo losses.
            out["image_right"] = self._rgb_tensor(Image.open(r["rgb_right"]).convert("RGB"), box, flip, jitter)
        return out


def denormalize(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Undo normalisation for plotting: [3,H,W] or [B,3,H,W] -> values in [0, 1]."""
    m = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    s = torch.tensor(std, device=image.device).view(-1, 1, 1)
    return (image * s + m).clamp(0, 1)


# --------------------------------------------------------------------------------------
# DataLoaders
# --------------------------------------------------------------------------------------
def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class LoaderConfig:
    root: str
    image_size: int = 512
    batch_size: int = 8
    num_workers: int = 8
    max_depth: float | None = None
    val_scenes: Sequence[str] = ()
    test_scenes: Sequence[str] = ()
    n_val: int = 1
    n_test: int = 1
    augment_train: bool = True
    return_right: bool = False
    index_cache: str | None = None
    seed: int = 0
    extra: dict = field(default_factory=dict)


def make_dataloaders(cfg: LoaderConfig) -> tuple[dict[str, DataLoader], dict[str, LuSNARDataset]]:
    require = ("rgb", "depth", "label", "rgb_right") if cfg.return_right else ("rgb", "depth", "label")
    index = build_index(cfg.root, require=require, cache=cfg.index_cache)
    splits = split_by_scene(index, cfg.val_scenes, cfg.test_scenes, cfg.n_val, cfg.n_test, cfg.seed)

    datasets = {
        name: LuSNARDataset(df, image_size=cfg.image_size, max_depth=cfg.max_depth,
                            augment=(cfg.augment_train and name == "train"),
                            return_right=cfg.return_right)
        for name, df in splits.items() if len(df)
    }
    g = torch.Generator().manual_seed(cfg.seed)
    loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(name == "train"), drop_last=(name == "train"),
                         num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=cfg.num_workers > 0, worker_init_fn=_seed_worker, generator=g)
        for name, ds in datasets.items()
    }
    for name, ds in datasets.items():
        print(f"{name:5s}: {len(ds):6,} frames | scenes {sorted(ds.idx.scene.unique())}")
    return loaders, datasets


# --------------------------------------------------------------------------------------
# Utilities for training
# --------------------------------------------------------------------------------------
def class_pixel_counts(dataset: LuSNARDataset, n_samples: int = 200, seed: int = 0) -> np.ndarray:
    """Pixel count per class on a random subset of the (un-augmented) labels."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    counts = np.zeros(NUM_CLASSES, np.int64)
    for i in ids:
        lab = rgb_to_class(np.asarray(Image.open(dataset.idx.iloc[int(i)]["label"]).convert("RGB")))
        counts += np.bincount(lab[lab != IGNORE_INDEX].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts


def class_weights(counts: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights (softened with `power`), normalised to mean 1. Absent classes get 0."""
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12) ** power, 0.0)
    w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


def masked_depth_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """L1 loss over valid depth pixels only."""
    v = valid.bool()
    return (pred[v] - target[v]).abs().mean() if v.any() else pred.sum() * 0


def silog_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               lam: float = 0.85, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant log loss (common for monocular depth). pred must be > 0 (e.g. softplus/exp output)."""
    v = valid.bool() & (target > eps)
    if not v.any():
        return pred.sum() * 0
    g = torch.log(pred[v].clamp_min(eps)) - torch.log(target[v])
    return torch.sqrt((g ** 2).mean() - lam * g.mean() ** 2 + eps)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sanity-check the LuSNAR loader")
    ap.add_argument("--root", required=True)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args()
    loaders, _ = make_dataloaders(LoaderConfig(root=a.root, image_size=a.image_size,
                                               batch_size=a.batch_size, num_workers=a.num_workers))
    b = next(iter(loaders["train"]))
    for k, v in b.items():
        print(f"{k:12s}", tuple(v.shape), v.dtype) if torch.is_tensor(v) else print(f"{k:12s}", v[:2], "...")
