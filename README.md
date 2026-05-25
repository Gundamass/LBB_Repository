# LBB Competition Pipeline (Scheme C)

## Files
- `config.yaml`: unified configuration (paths, training, inference, hybrid branches)
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

## Inference + Submission ZIP
```bash
python3 inference.py --config ./config.yaml --ckpt ./checkpoints/best_model.pt
```

The output ZIP is generated under `outputs/`.

## Notes
- Polygon labels are converted to bounding boxes for detection training.
- Training includes strong augmentation (mixup + copy-paste + photometric + cutout + flips).
- Hybrid branch includes:
  - classical high-frequency proposals
  - prototype-based metric re-scoring
- Validation metric is mAP@0.5 (custom implementation aligned with task rule).
