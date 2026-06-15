# LBB Competition Pipeline (Scheme C)

## Files
- `config.yaml`: unified configuration (paths, training, inference, hybrid branches)
- `config.dinov3.yaml`: DINOv3-ConvNeXt + FasterRCNN training config
- `dataset.py`: entry re-export of dataset module
- `model.py`: entry re-export of model module
- `train.py`: training + validation + early stopping + checkpointing
- `inference.py`: TTA inference + JSON export + ZIP packaging
- `src/dataset.py`: data parsing/augmentation/loader implementation
- `src/model.py`: detector + classical proposal + metric rescoring implementation
- `src/utils.py`: logging, config loader, hardware detection, helpers

## Install
```bash
python3 -m pip install --user -r requirements.txt
```

## Train
```bash
python3 train.py --config ./config.yaml
```

### Train With DINOv3
1. Clone DINOv3 repo (only needed once):
```bash
git clone --depth 1 https://github.com/facebookresearch/dinov3.git ./third_party/dinov3
```
2. (Optional but recommended) put official DINOv3 weights path into `config.dinov3.yaml` at `model.dinov3.weights`.
3. Launch training:
```bash
python3 train.py --config ./config.dinov3.yaml
```

## Inference + Submission ZIP
```bash
python3 inference.py --config ./config.yaml --ckpt ./checkpoints/best_model.pt
```

The output ZIP is generated under `outputs/`.

## Reproduce Best Online Score
Current best public submission:

- ZIP name: `少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123.zip`
- Online mAP@0.5: `0.283265`
- Key inference settings: no classical proposal branch, score threshold `0.01`, class-wise NMS IoU `0.445`, two-checkpoint ensemble, TTA scales `[1.0, 1.1, 0.9]` plus horizontal flip.

Run all commands from `LBB_competition/`:

```bash
cd /home/heqing/LBB_competition
conda activate lbb
```

Train the seed-42 model:

```bash
python3 train.py --config ./config.no_classic.thr001_nms0445.ensemble_seed123.yaml
```

Keep the seed-42 best checkpoint:

```bash
mkdir -p ./checkpoints/seed42
cp ./checkpoints/best_model.pt ./checkpoints/seed42/best_model.pt
```

Train a second model with seed 123. This keeps the same architecture/training recipe and only changes the random seed plus checkpoint directory:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

base_path = Path("config.no_classic.thr001_nms0445.ensemble_seed123.yaml")
cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
cfg["seed"] = 123
cfg["system"]["checkpoints_dir"] = "./checkpoints/seed123"

out_path = Path("config.no_classic.thr001_nms0445.seed123.train.yaml")
out_path.write_text(
    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
print(out_path)
PY

python3 train.py --config ./config.no_classic.thr001_nms0445.seed123.train.yaml
```

Generate the best-score submission ZIP with the two checkpoints:

```bash
python3 inference.py \
  --config ./config.no_classic.thr001_nms0445.ensemble_seed123.yaml \
  --ckpt ./checkpoints/seed42/best_model.pt,./checkpoints/seed123/best_model.pt
```

The final file is:

```text
./outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123.zip
```

For reference, the nearby `nms=0.455` config is also tracked because it was explicitly preserved, but its online score was slightly lower (`0.283147`). The current best known config is `config.no_classic.thr001_nms0445.ensemble_seed123.yaml`.

## Notes
- Polygon labels are converted to bounding boxes for detection training.
- Training includes strong augmentation (mixup + copy-paste + photometric + cutout + flips).
- Hybrid branch includes:
  - classical high-frequency proposals
  - prototype-based metric re-scoring
- Validation metric is mAP@0.5 (custom implementation aligned with task rule).
