"""
train_unfreeze.py - DINOv3-B + DPT on LuSNAR, one task (seg or depth) per run, with the last N backbone blocks unfrozen.

Model, losses, training loop and metrics are copied verbatim from Andreas's baseline notebook
(dinov3_dpt_seg_depth.ipynb, commit 89b1762) so results are directly comparable; only the configuration
(command-line arguments) and the main block are new. UNFREEZE_LAST_N=0 reproduces the frozen baseline.

Example:
    python jim/train_unfreeze.py --task seg --unfreeze 4 --run-dir /proj/uppmax2026-1-299/jimcai/runs/seg_unf4
    python jim/train_unfreeze.py --task depth --unfreeze 4 --max-steps 50 --max-val-batches 20 --epochs 1 ...  # smoke test
"""
import argparse
import contextlib
import functools
import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm as _tqdm

REPO_DIR = Path(__file__).resolve().parents[1]            # this clone, not the shared copy of lusnar_dataset.py
sys.path.insert(0, str(REPO_DIR))
import lusnar_dataset as L

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", choices=["seg", "depth"], required=True)
ap.add_argument("--unfreeze", type=int, default=4, help="number of last ViT blocks to fine-tune (0 = frozen, 12 = all)")
ap.add_argument("--backbone-lr", type=float, default=1e-5)
ap.add_argument("--head-lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--num-workers", type=int, default=8)
ap.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (smoke test)")
ap.add_argument("--max-val-batches", type=int, default=None, help="cap validation batches (smoke test)")
ap.add_argument("--seed", type=int, default=0, help="init / shuffling / augmentation seed (split and depth bins stay fixed)")
ap.add_argument("--skip-test", action="store_true", help="do not evaluate on the test scenes at the end")
ap.add_argument("--run-dir", type=Path, required=True)
args = ap.parse_args()

# --- data (same as the baseline) ---
ROOT = Path("/proj/uppmax2026-1-299/task1_groupc_dataset/lusnar/extracted")
IMAGE_SIZE = 512
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
MAX_DEPTH = None
VAL_SCENES = ["Moon_8"]
TEST_SCENES = ["Moon_9", "Moon_3"]
INDEX_CACHE = Path.home() / "lusnar_cache" / "index.csv"
SEED = args.seed

# --- model (same as the baseline) ---
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_WEIGHTS = Path("/proj/uppmax2026-1-299/task1_groupc_dataset/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
OUT_LAYERS = (2, 5, 8, 11)
DPT_FEATURES = 256
USE_CLS_READOUT = True
N_DEPTH_BINS = 256

# --- optimisation ---
UNFREEZE_LAST_N = args.unfreeze
HEAD_LR = args.head_lr
BACKBONE_LR = args.backbone_lr
WEIGHT_DECAY = 1e-2
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
MAX_STEPS_PER_EPOCH = args.max_steps
MAX_VAL_BATCHES = args.max_val_batches
DICE_WEIGHT = 0.5
GRAD_LOSS_WEIGHT = 0.5

RUN_DIR = args.run_dir
RUN_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = (torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16)
print("args:", vars(args))
print("torch", torch.__version__, "| device:", DEVICE,
      f"({torch.cuda.get_device_name()})" if DEVICE.type == "cuda" else "", "| AMP:", AMP_DTYPE)

# Progress bars only in a terminal; batch logs get the per-epoch summary lines.
tqdm = functools.partial(_tqdm, disable=not sys.stderr.isatty())


def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if DEVICE.type == "cuda" else contextlib.nullcontext()



# ===== from baseline notebook, cell 9 (verbatim) =====
import re
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel

VITB16_CFG = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                  intermediate_size=3072, num_register_tokens=4, patch_size=16)


def convert_meta_dinov3(sd):
    """Original Meta DINOv3 state dict (blocks.N.attn.qkv, ls1.gamma, storage_tokens, ...) -> HF DINOv3ViTModel keys."""
    out = {}
    for k, v in sd.items():
        v = v.float()
        if k == "rope_embed.periods" or k.endswith("bias_mask"):   # RoPE is recomputed by HF; mask applied below
            continue
        m = re.match(r"blocks\.(\d+)\.(.*)", k)
        if m is None:
            k2 = {"cls_token": "embeddings.cls_token", "storage_tokens": "embeddings.register_tokens",
                  "mask_token": "embeddings.mask_token", "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
                  "patch_embed.proj.bias": "embeddings.patch_embeddings.bias"}.get(k, k)
            out[k2] = v.reshape(1, 1, -1) if k == "mask_token" else v
            continue
        i, rest = m.groups()
        p = f"model.layer.{i}."
        if rest.startswith("attn.qkv."):                            # fused qkv -> q/k/v (HF k_proj has no bias)
            kind = rest.split(".")[-1]
            if kind == "bias":
                v = v * sd[k + "_mask"].float()
            q, kk, vv = v.chunk(3, dim=0)
            out[p + f"attention.q_proj.{kind}"], out[p + f"attention.v_proj.{kind}"] = q, vv
            if kind == "weight":
                out[p + "attention.k_proj.weight"] = kk
            continue
        for a, b in [("attn.proj", "attention.o_proj"), ("mlp.fc1", "mlp.up_proj"), ("mlp.fc2", "mlp.down_proj"),
                     ("ls1.gamma", "layer_scale1.lambda1"), ("ls2.gamma", "layer_scale2.lambda1")]:
            rest = rest.replace(a, b)
        out[p + rest] = v
    return out


def load_dinov3(dino_id=DINO_ID, weights=DINO_WEIGHTS):
    if weights is not None:                      # local .pth on disk (no HF access needed)
        m = DINOv3ViTModel(DINOv3ViTConfig(**VITB16_CFG))
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        m.load_state_dict(convert_meta_dinov3(sd), strict=True)
        print("Loaded pretrained weights from", weights)
        return m, True
    try:
        m = AutoModel.from_pretrained(dino_id)
        print("Loaded pretrained", dino_id)
        return m, True
    except Exception as e:                       # gated repo without access, no internet, ...
        print(f"!! Could not load {dino_id}: {str(e).splitlines()[0]}\n"
              "!! Using RANDOM ViT-B/16 weights - plumbing test only, results will be meaningless.")
        return DINOv3ViTModel(DINOv3ViTConfig(**VITB16_CFG)), False


def _find_blocks(hf_model):
    """The ModuleList of transformer blocks (its location differs between transformers versions)."""
    n = hf_model.config.num_hidden_layers
    for name, mod in hf_model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n:
            return list(mod)
    raise RuntimeError("Could not find the transformer blocks in the DINOv3 model")


class DinoFeatures(nn.Module):
    """DINOv3 -> list of (patch feature map [B, C, H/16, W/16], cls token [B, C]) for each block in out_layers."""

    def __init__(self, hf_model, out_layers=OUT_LAYERS, norm=True):
        super().__init__()
        self.model = hf_model
        c = hf_model.config
        self.patch, self.embed_dim = c.patch_size, c.hidden_size
        self.n_prefix = 1 + c.num_register_tokens
        self.out_layers, self.norm = tuple(out_layers), norm
        self.blocks = _find_blocks(hf_model)    # plain list -> not registered twice in state_dict
        self._cache = {}
        for i in self.out_layers:
            self.blocks[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, i):
        def hook(module, inputs, output):
            self._cache[i] = output[0] if isinstance(output, (tuple, list)) else output
        return hook

    def forward(self, x):
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        frozen = not any(p.requires_grad for p in self.model.parameters())
        with torch.no_grad() if frozen else contextlib.nullcontext():
            self._cache = {}
            self.model(pixel_values=x)
            outs = []
            for i in self.out_layers:
                t = self._cache[i]
                if self.norm:
                    t = self.model.norm(t)
                patches = t[:, self.n_prefix:]
                assert patches.shape[1] == h * w, (patches.shape, h, w)
                outs.append((patches.transpose(1, 2).reshape(B, -1, h, w), t[:, 0]))
        self._cache = {}
        return outs

# ===== from baseline notebook, cell 10 (verbatim) =====
class ResidualConvUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class FusionBlock(nn.Module):
    """x (coarser path) [+ RCU(skip)] -> RCU -> upsample to `size` (or ×2) -> 1×1 conv."""

    def __init__(self, ch):
        super().__init__()
        self.rcu1, self.rcu2 = ResidualConvUnit(ch), ResidualConvUnit(ch)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = (F.interpolate(x, size=size, mode="bilinear", align_corners=True) if size is not None
             else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))
        return self.out(x)


class DPTNeck(nn.Module):
    """4 ViT feature maps at 1/16 -> fused [B, features, H/2, W/2]."""

    def __init__(self, embed_dim, features=DPT_FEATURES, out_channels=(96, 192, 384, 768), use_cls=USE_CLS_READOUT):
        super().__init__()
        self.use_cls = use_cls
        if use_cls:
            self.readout = nn.ModuleList(nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.GELU())
                                         for _ in out_channels)
        self.project = nn.ModuleList(nn.Conv2d(embed_dim, c, 1) for c in out_channels)
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),   # 1/16 -> 1/4
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),   # 1/16 -> 1/8
            nn.Identity(),                                                        # 1/16
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),  # 1/16 -> 1/32
        ])
        self.to_feat = nn.ModuleList(nn.Conv2d(c, features, 3, padding=1, bias=False) for c in out_channels)
        self.fuse = nn.ModuleList(FusionBlock(features) for _ in out_channels)

    def forward(self, feats):
        layers = []
        for i, (f, cls) in enumerate(feats):
            if self.use_cls:
                B, C, h, w = f.shape
                tok = f.flatten(2).transpose(1, 2)                                   # [B, hw, C]
                tok = self.readout[i](torch.cat([tok, cls[:, None].expand_as(tok)], -1))
                f = tok.transpose(1, 2).reshape(B, C, h, w)
            layers.append(self.to_feat[i](self.resample[i](self.project[i](f))))
        l1, l2, l3, l4 = layers                                                       # 1/4, 1/8, 1/16, 1/32
        p = self.fuse[3](l4, size=l3.shape[-2:])
        p = self.fuse[2](p, l3, size=l2.shape[-2:])
        p = self.fuse[1](p, l2, size=l1.shape[-2:])
        return self.fuse[0](p, l1)                                                    # 1/2


class SegHead(nn.Module):
    def __init__(self, in_ch=DPT_FEATURES, n_classes=L.NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch),
                                 nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.Conv2d(in_ch, n_classes, 1))

    def forward(self, x, out_size):
        return F.interpolate(self.net(x), size=out_size, mode="bilinear", align_corners=False)   # logits


class DepthHead(nn.Module):
    """Classification-regression: softmax over log-spaced depth bins, log depth = Σ p_k · log c_k (always in range, > 0).
    Bins are evaluated at 1/2 resolution, then the 1-channel depth is upsampled (saves memory)."""

    def __init__(self, in_ch=DPT_FEATURES, n_bins=N_DEPTH_BINS, d_min=1.0, d_max=100.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(in_ch // 2, 32, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, n_bins, 1))
        self.register_buffer("log_centers", torch.linspace(math.log(d_min), math.log(d_max), n_bins).view(1, -1, 1, 1))

    def forward(self, x, out_size):
        prob = self.net(x).float().softmax(1)
        depth = (prob * self.log_centers).sum(1, keepdim=True).exp()
        return F.interpolate(depth, size=out_size, mode="bilinear", align_corners=False)   # metres


class DPTModel(nn.Module):
    def __init__(self, backbone: DinoFeatures, head: nn.Module, features=DPT_FEATURES):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.neck = DPTNeck(backbone.embed_dim, features)

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)), x.shape[-2:])


def build_model(head):
    hf, pretrained = load_dinov3()
    model = DPTModel(DinoFeatures(hf), head)
    model.backbone.model.requires_grad_(False)
    if UNFREEZE_LAST_N > 0:
        for blk in model.backbone.blocks[-UNFREEZE_LAST_N:]:
            blk.requires_grad_(True)
        model.backbone.model.norm.requires_grad_(True)
    n = lambda ps: sum(p.numel() for p in ps) / 1e6
    print(f"backbone {n(model.backbone.parameters()):.1f}M (trainable {n(p for p in model.backbone.parameters() if p.requires_grad):.1f}M)"
          f" | neck {n(model.neck.parameters()):.1f}M | head {n(model.head.parameters()):.2f}M")
    return model.to(DEVICE), pretrained

# ===== from baseline notebook, cell 12 (verbatim) =====
def to_device(b):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def make_optimizer(model):
    groups = {"head_decay": [], "head_no_decay": [], "bb_decay": [], "bb_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = ("bb" if name.startswith("backbone.") else "head") + ("_no_decay" if p.ndim <= 1 else "_decay")
        groups[g].append(p)
    lr = {"head": HEAD_LR, "bb": BACKBONE_LR}
    return torch.optim.AdamW([{"params": ps, "lr": lr[k.split("_")[0]], "weight_decay": 0.0 if "no_decay" in k else WEIGHT_DECAY}
                              for k, ps in groups.items() if ps])


def trainable_state_dict(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def fit(model, task, train_loader, val_loader, epochs, ckpt_path):
    opt = make_optimizer(model)
    steps_per_epoch = len(train_loader) if MAX_STEPS_PER_EPOCH is None else min(len(train_loader), MAX_STEPS_PER_EPOCH)
    total, warmup = epochs * steps_per_epoch, max(1, int(WARMUP_FRAC * epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)
    params = [p for p in model.parameters() if p.requires_grad]
    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, run = time.time(), {}
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}/{epochs}", unit="batch")
        for step, b in enumerate(pbar):
            if step >= steps_per_epoch:
                break
            b = to_device(b)
            with autocast():
                out = model(b["image"])
            loss, parts = task.loss(out.float(), b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            pbar.set_postfix({**{k: f"{v / (step + 1):.4f}" for k, v in run.items()}, "lr": f"{sched.get_last_lr()[0]:.1e}"})
        pbar.close()
        train_parts = {f"train_{k}": v / steps_per_epoch for k, v in run.items()}
        metrics = task.evaluate(model, val_loader, max_batches=MAX_VAL_BATCHES)
        score = metrics[task.main_metric]
        improved = best is None or (score > best if task.higher_is_better else score < best)
        if improved:
            best = score
            torch.save({"model": trainable_state_dict(model), "epoch": epoch, "metrics": metrics}, ckpt_path)
        history.append({"epoch": epoch, **train_parts, **metrics, "time_s": time.time() - t0})
        print(f"epoch {epoch}/{epochs} | " + " ".join(f"{k} {v:.4f}" for k, v in train_parts.items()) +
              f" | val {task.main_metric} {score:.4f}{'  *best*' if improved else ''} | {time.time() - t0:.0f}s")
    return pd.DataFrame(history)


def load_best(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, unexpected
    print(f"loaded {ckpt_path.name} (epoch {ck['epoch']}, val {ck['metrics']})")
    return model


def plot_history(hist, keys, title):
    fig, ax = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3))
    for a, k in zip(np.atleast_1d(ax), keys):
        a.plot(hist["epoch"], hist[k], marker="o"); a.set_title(k); a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(title); plt.tight_layout(); plt.show()

# ===== from baseline notebook, cell 14 (verbatim) =====
class SegTask:
    main_metric, higher_is_better = "mIoU", True

    def __init__(self, class_weights, dice_weight=DICE_WEIGHT, n_classes=L.NUM_CLASSES, ignore=L.IGNORE_INDEX):
        self.ce = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), ignore_index=ignore)
        self.dice_w, self.K, self.ignore = dice_weight, n_classes, ignore

    def dice(self, logits, label):
        valid = (label != self.ignore)
        prob = logits.softmax(1) * valid[:, None]
        onehot = F.one_hot(label.clamp(max=self.K - 1) * valid, self.K).permute(0, 3, 1, 2).float() * valid[:, None]
        inter, denom = (prob * onehot).sum((0, 2, 3)), (prob + onehot).sum((0, 2, 3))
        return 1 - ((2 * inter + 1) / (denom + 1)).mean()

    def loss(self, logits, b):
        ce, dice = self.ce(logits, b["label"]), self.dice(logits, b["label"])
        return ce + self.dice_w * dice, {"ce": ce.item(), "dice": dice.item()}

    @torch.no_grad()
    def evaluate(self, model, loader, max_batches=None):
        model.eval()
        conf = torch.zeros(self.K, self.K, dtype=torch.long, device=DEVICE)
        total = len(loader) if max_batches is None else min(len(loader), max_batches)
        for i, b in enumerate(tqdm(loader, total=total, desc="eval", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            b = to_device(b)
            with autocast():
                pred = model(b["image"]).argmax(1)
            m = b["label"] != self.ignore
            conf += torch.bincount(b["label"][m] * self.K + pred[m], minlength=self.K ** 2).view(self.K, self.K)
        conf = conf.double()
        tp = conf.diag()
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)          # NaN for classes absent from GT and prediction
        out = {"mIoU": float(torch.nanmean(iou)), "pixel_acc": float(tp.sum() / conf.sum().clamp_min(1))}
        out.update({f"IoU_{n}": float(v) for n, v in zip(L.CLASS_NAMES, iou)})
        return out

# ===== from baseline notebook, cell 21 (verbatim) =====
def gradient_loss(pred, gt, valid, eps=1e-6):
    v = valid.bool() & (gt > eps)
    d = torch.where(v, torch.log(pred.clamp_min(eps)) - torch.log(gt.clamp_min(eps)), torch.zeros_like(pred))
    mx, my = v[..., :, 1:] & v[..., :, :-1], v[..., 1:, :] & v[..., :-1, :]
    gx, gy = (d[..., :, 1:] - d[..., :, :-1]).abs(), (d[..., 1:, :] - d[..., :-1, :]).abs()
    return (gx[mx].sum() + gy[my].sum()) / (mx.sum() + my.sum()).clamp_min(1)


class DepthTask:
    main_metric, higher_is_better = "AbsRel", False

    def __init__(self, grad_weight=GRAD_LOSS_WEIGHT):
        self.grad_w = grad_weight

    def loss(self, pred, b):
        si = L.silog_loss(pred, b["depth"], b["depth_valid"])
        gr = gradient_loss(pred, b["depth"], b["depth_valid"])
        return si + self.grad_w * gr, {"silog": si.item(), "grad": gr.item()}

    @torch.no_grad()
    def evaluate(self, model, loader, max_batches=None):
        model.eval()
        keys = ["AbsRel", "SqRel", "RMSE", "RMSE_log", "SILog", "d1", "d2", "d3"]
        acc, n = dict.fromkeys(keys, 0.0), 0
        total = len(loader) if max_batches is None else min(len(loader), max_batches)
        for i, b in enumerate(tqdm(loader, total=total, desc="eval", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            b = to_device(b)
            with autocast():
                pred = model(b["image"]).float()
            for j in range(pred.shape[0]):
                v = b["depth_valid"][j, 0]
                if v.sum() < 10:
                    continue
                p, g = pred[j, 0][v].clamp_min(1e-3), b["depth"][j, 0][v]
                r, e = torch.maximum(p / g, g / p), torch.log(p) - torch.log(g)
                m = {"AbsRel": ((p - g).abs() / g).mean(), "SqRel": ((p - g) ** 2 / g).mean(),
                     "RMSE": ((p - g) ** 2).mean().sqrt(), "RMSE_log": (e ** 2).mean().sqrt(),
                     "SILog": 100 * ((e ** 2).mean() - e.mean() ** 2).clamp_min(0).sqrt(),
                     "d1": (r < 1.25).float().mean(), "d2": (r < 1.25 ** 2).float().mean(), "d3": (r < 1.25 ** 3).float().mean()}
                for k in keys:
                    acc[k] += float(m[k])
                n += 1
        return {k: v / max(n, 1) for k, v in acc.items()}



def save_history_plot(hist, keys, title, path):
    fig, ax = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3))
    for a, k in zip(np.atleast_1d(ax), keys):
        a.plot(hist["epoch"], hist[k], marker="o"); a.set_title(k); a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(title); plt.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ===== main (new) =====
cfg = L.LoaderConfig(
    root=str(ROOT), image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    max_depth=MAX_DEPTH, val_scenes=VAL_SCENES, test_scenes=TEST_SCENES, augment_train=True,
    index_cache=str(INDEX_CACHE), seed=SEED,
)
loaders, datasets = L.make_dataloaders(cfg)
train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
print({k: f"{len(v.idx)} frames / {len(loaders[k])} batches" for k, v in datasets.items()})

if args.task == "seg":
    counts = L.class_pixel_counts(datasets["train"], n_samples=200)
    task = SegTask(L.class_weights(counts))
    head = SegHead()
    plot_keys = ["train_ce", "train_dice", "mIoU", "pixel_acc"]
else:
    # Depth bin limits from 50 training frames, exactly as in the baseline (same seed -> same limits)
    vals = []
    for i in np.random.default_rng(0).choice(len(datasets["train"]), size=min(50, len(datasets["train"])), replace=False):
        d = L.load_depth(datasets["train"].idx.iloc[int(i)]["depth"])
        d = d[np.isfinite(d) & (d > 0) & (d < L.DEPTH_SKY_VALUE)]
        vals.append(d[:: max(1, d.size // 20000)])
    vals = np.concatenate(vals)
    pct = {p: float(np.percentile(vals, p)) for p in (0.1, 99.9)}
    DEPTH_MIN, DEPTH_MAX = max(0.05, 0.8 * pct[0.1]), 1.2 * pct[99.9]
    print(f"depth bins: {N_DEPTH_BINS} log-spaced in [{DEPTH_MIN:.2f}, {DEPTH_MAX:.1f}] m")
    task = DepthTask()
    head = DepthHead(n_bins=N_DEPTH_BINS, d_min=DEPTH_MIN, d_max=DEPTH_MAX)
    plot_keys = ["train_silog", "train_grad", "AbsRel", "RMSE", "d1"]

model, pretrained = build_model(head)
assert pretrained
ckpt = RUN_DIR / f"{args.task}_best.pt"

t_start = time.time()
hist = fit(model, task, train_loader, val_loader, args.epochs, ckpt)
hist.to_csv(RUN_DIR / "history.csv", index=False)
save_history_plot(hist, plot_keys, f"{args.task}, unfreeze last {UNFREEZE_LAST_N}", RUN_DIR / "history.png")

load_best(model, ckpt)
results = {"val": task.evaluate(model, val_loader, max_batches=MAX_VAL_BATCHES)}
if not args.skip_test:
    results["test"] = task.evaluate(model, test_loader)
summary = {"task": args.task, "backbone": DINO_ID, "image_size": IMAGE_SIZE, "args": {k: str(v) for k, v in vars(args).items()},
           "best_epoch": int(hist[task.main_metric].idxmax() if task.higher_is_better else hist[task.main_metric].idxmin()) + 1,
           "train_time_min": round((time.time() - t_start) / 60, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE.type == "cuda" else None,
           "results": results}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
