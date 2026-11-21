# HT2026 DINOv3 LuSNAR SegDepth

Semantic segmentation and monocular depth estimation on the LuSNAR lunar dataset, using a DINOv3 ViT-B/16 backbone with a DPT neck.

Segmentation classes: Lunar Regolith, Impact Crater, Rock, Mountain, Sky.

## Files

| File | What it does |
|------|--------------|
| `lusnar_dataset.py` | PyTorch dataset and dataloaders for LuSNAR (RGB, depth, labels), scene-based split, augmentation, losses |
| `lusnar_dataloader.ipynb` | Example of using the dataloader |
| `dinov3_dpt_seg_depth.ipynb` | Experiment 1: separate models for segmentation and depth |
| `multitask/` | Experiment 2: one shared multitask model |
| `train_method4.py` / `train_method4.sh` | Experiment 3: fine-tuning the backbone (SLURM job script included) |

## Experiments

All experiments use the same setup so the results can be compared:
- **Backbone:** DINOv3 ViT-B/16, pretrained on LVD-1689M. Features are taken from blocks 2, 5, 8 and 11.
- **Neck:** a DPT neck (256 channels, with CLS-token readout) fuses those four feature maps into one map at 1/2 resolution.
- **Data:** 512×512 images. The split is by scene: validation is Moon_8, test is Moon_3 and Moon_9, training is the rest.
- **Training:** 10 epochs, AdamW with warmup and a cosine schedule, mixed precision.
- **Segmentation head:** trained with class-weighted cross-entropy plus Dice loss. Scored with mIoU.
- **Depth head:** predicts a softmax over 256 log-spaced depth bins. Trained with SILog plus a gradient-matching loss. Scored with AbsRel, RMSE and δ<1.25.

### 1. DINOv3 + DPT, separate tasks (baseline)

`dinov3_dpt_seg_depth.ipynb` trains two independent models. One does segmentation and the other does depth, and each has its own DPT neck. The backbone is frozen, so only the neck and head are trained.

### 2. Shared multitask model

`multitask/shared_multitask_dinov3_dpt_poc.ipynb` uses a single frozen backbone and a single DPT neck, with two heads (segmentation and depth) trained together. The total loss is `seg_loss + depth_loss`, with each weighted 1.0. Everything else is the same as the baseline. `shared_multitask_results.ipynb` reloads the checkpoint, evaluates it and makes the plots. Results are in `multitask/results/` and `multitask/run/`.

### 3. Fine-tuning the backbone (method 4)

`train_method4.py` is a script version of the baseline that trains one task per run (`--task seg` or `--task depth`). The model, losses and metrics are the same, but here the backbone is also trained:
- `--unfreeze N` trains the last N ViT blocks (0 = frozen, 12 = whole backbone).
- Layer-wise LR decay (`--lld 0.9`): blocks closer to the input get a smaller learning rate.
- An EMA of the weights is used for validation and checkpoints.
- Stronger augmentation: wider random crops, more brightness and contrast change.
- At final evaluation, results are also reported with horizontal-flip test-time augmentation.
- `--image-size` allows higher resolutions, e.g. 768.

`train_method4.sh` runs it on UPPMAX with the whole backbone unfrozen and a backbone LR of 4e-5, for example `sbatch -J method4_seg768 train_method4.sh seg 768`. Each run writes `history.csv`, `summary.json` (val/test metrics, with and without TTA), a checkpoint and a figure of example predictions to `--run-dir`.

## Setup

```bash
pip install -r requirements.txt
```

You also need:
- The LuSNAR data, laid out as `<root>/<scene>/image0/{color,depth,label}/` (e.g. `Moon_1/`).
- DINOv3 ViT-B/16 weights: either the original Meta `.pth` file or access to `facebook/dinov3-vitb16-pretrain-lvd1689m` on Hugging Face.

## Usage

Check that the data loads:

```bash
python lusnar_dataset.py --root /path/to/lusnar
```

Fine-tune a single-task model:

```bash
python train_method4.py --task seg   --unfreeze 12 --run-dir runs/seg
python train_method4.py --task depth --unfreeze 12 --run-dir runs/depth
```

Data and weight paths in the script and notebooks point to UPPMAX (`/proj/uppmax2026-1-299/...`); change them to run elsewhere.
