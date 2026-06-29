import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import (
    bbox_iou_xyxy,
    copy_best_as_latest,
    detect_hardware,
    ensure_dirs,
    load_config,
    seed_everything,
    setup_logger,
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


class DetectionDataParallel(torch.nn.DataParallel):
    """DataParallel variant that splits detection batches by image list item.

    The default PyTorch scatter recursively chunks every tensor. For detection
    inputs shaped as List[Tensor[C,H,W]], that would incorrectly split channels.
    This wrapper keeps each image intact and distributes whole samples.
    """

    @staticmethod
    def _move_target(target: Dict, device: torch.device) -> Dict:
        return {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in target.items()
        }

    @staticmethod
    def _chunk_bounds(n_items: int, n_chunks: int) -> List[Tuple[int, int]]:
        n_chunks = max(1, min(n_chunks, n_items))
        base = n_items // n_chunks
        rem = n_items % n_chunks
        bounds = []
        start = 0
        for idx in range(n_chunks):
            size = base + (1 if idx < rem else 0)
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

    def scatter(self, inputs, kwargs, device_ids):
        if not inputs:
            return super().scatter(inputs, kwargs, device_ids)

        images = inputs[0]
        targets = inputs[1] if len(inputs) > 1 else None
        if not isinstance(images, list):
            return super().scatter(inputs, kwargs, device_ids)

        bounds = self._chunk_bounds(len(images), len(device_ids))
        scattered_inputs = []
        scattered_kwargs = []
        for dev_id, (start, end) in zip(device_ids, bounds):
            device = torch.device("cuda", int(dev_id))
            image_chunk = [img.to(device, non_blocking=True) for img in images[start:end]]
            if targets is None:
                scattered_inputs.append((image_chunk,))
            else:
                target_chunk = [self._move_target(t, device) for t in targets[start:end]]
                scattered_inputs.append((image_chunk, target_chunk))
            scattered_kwargs.append(dict(kwargs))

        return tuple(scattered_inputs), tuple(scattered_kwargs)


def strip_targets_for_model(targets: List[Dict], device: Optional[torch.device]) -> List[Dict]:
    out = []
    for t in targets:
        def maybe_to(x):
            if device is None:
                return x
            return x.to(device)

        out.append(
            {
                "boxes": maybe_to(t["boxes"]),
                "labels": maybe_to(t["labels"]),
                "image_id": maybe_to(t["image_id"]),
                "area": maybe_to(t["area"]),
                "iscrowd": maybe_to(t["iscrowd"]),
            }
        )
    return out


def evaluate_map50(pred_records: List[Dict], gt_records: List[Dict], num_classes: int) -> Dict:
    per_cls_ap = {}
    eps = 1e-9

    for cls in range(1, num_classes):
        # Collect all GT per image for one class.
        gts = {}
        npos = 0
        for i, gt in enumerate(gt_records):
            boxes = []
            labels = gt["labels"]
            for b, l in zip(gt["boxes"], labels):
                if int(l) == cls:
                    boxes.append(list(map(float, b)))
            gts[i] = {"boxes": boxes, "used": [False] * len(boxes)}
            npos += len(boxes)

        # Collect predictions for one class.
        preds = []
        for i, pred in enumerate(pred_records):
            for b, l, s in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if int(l) == cls:
                    preds.append((float(s), i, list(map(float, b))))

        preds.sort(key=lambda x: x[0], reverse=True)

        tp = [0.0] * len(preds)
        fp = [0.0] * len(preds)

        for pi, (_, img_idx, pbox) in enumerate(preds):
            gt_img = gts[img_idx]
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(gt_img["boxes"]):
                iou = bbox_iou_xyxy(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.5 and best_j >= 0:
                if not gt_img["used"][best_j]:
                    tp[pi] = 1.0
                    gt_img["used"][best_j] = True
                else:
                    fp[pi] = 1.0
            else:
                fp[pi] = 1.0

        if len(preds) == 0:
            per_cls_ap[cls] = 0.0
            continue

        tp_cum, fp_cum = [], []
        t, f = 0.0, 0.0
        for ti, fi in zip(tp, fp):
            t += ti
            f += fi
            tp_cum.append(t)
            fp_cum.append(f)

        rec = [x / max(float(npos), eps) for x in tp_cum]
        prec = [tp_cum[i] / max(tp_cum[i] + fp_cum[i], eps) for i in range(len(tp_cum))]

        # VOC-style area under precision-recall curve.
        mrec = [0.0] + rec + [1.0]
        mpre = [0.0] + prec + [0.0]
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

        per_cls_ap[cls] = float(ap)

    mAP = sum(per_cls_ap.values()) / max(len(per_cls_ap), 1)
    return {"mAP50": mAP, "ap_per_class": per_cls_ap}


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best = -1.0
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def validate(model, loader, device, amp_enabled: bool, logger):
    model = unwrap_model(model)
    model.eval()

    pred_records = []
    gt_records = []

    for images, targets in tqdm(loader, desc="validate", leave=False):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            with autocast(enabled=amp_enabled):
                preds = model(images)

        for pred, gt in zip(preds, targets):
            pred_records.append(
                {
                    "boxes": pred["boxes"].detach().cpu().tolist(),
                    "labels": pred["labels"].detach().cpu().tolist(),
                    "scores": pred["scores"].detach().cpu().tolist(),
                }
            )
            gt_records.append(
                {
                    "boxes": gt["boxes"].detach().cpu().tolist(),
                    "labels": gt["labels"].detach().cpu().tolist(),
                }
            )

    metrics = evaluate_map50(pred_records, gt_records, num_classes=unwrap_model(model).num_classes)
    logger.info("Validation mAP@0.5 = %.6f", metrics["mAP50"])
    return metrics


def main(config_path: str):
    cfg = load_config(config_path)

    ensure_dirs(
        [
            cfg["system"]["checkpoints_dir"],
            cfg["system"]["logs_dir"],
            cfg["inference"]["output_dir"],
        ]
    )

    logger = setup_logger(cfg["system"]["logs_dir"], name="train")

    try:
        seed_everything(int(cfg.get("seed", 42)))

        hw = detect_hardware()
        device = torch.device(hw.device)
        logger.info(
            "Hardware: device=%s gpu_count=%d gpu_names=%s total_vram_gb=%.2f",
            hw.device,
            hw.gpu_count,
            hw.gpu_names,
            hw.total_vram_gb,
        )

        train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg, hw, logger=logger)
        model = build_model(cfg, logger=logger).to(device)

        init_checkpoint = cfg["training"].get("init_checkpoint")
        if init_checkpoint:
            init_path = Path(str(init_checkpoint)).expanduser()
            if not init_path.exists():
                raise FileNotFoundError(f"training.init_checkpoint not found: {init_path}")
            init_ckpt = torch.load(init_path, map_location=device)
            state = init_ckpt.get("model_state", init_ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(
                "Initialized model from %s | missing=%d unexpected=%d",
                init_path,
                len(missing),
                len(unexpected),
            )

        use_data_parallel = (
            bool(cfg["training"].get("data_parallel", False))
            and device.type == "cuda"
            and torch.cuda.device_count() > 1
        )
        if use_data_parallel:
            model = DetectionDataParallel(model)
            logger.info("DataParallel enabled over %d visible CUDA devices", torch.cuda.device_count())

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )

        sched_name = cfg["training"]["lr_scheduler"]["name"]
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(cfg["training"]["epochs"]),
                eta_min=float(cfg["training"]["lr_scheduler"].get("min_lr", 1e-6)),
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")
        scaler = GradScaler(enabled=amp_enabled)

        ckpt_dir = Path(cfg["system"]["checkpoints_dir"])
        best_ckpt = ckpt_dir / cfg["system"]["best_ckpt_name"]
        latest_ckpt = ckpt_dir / "latest.pt"

        stopper = EarlyStopper(patience=int(cfg["training"].get("early_stopping_patience", 8)))

        best_map = -1.0
        epochs = int(cfg["training"]["epochs"])
        grad_clip = float(cfg["training"].get("grad_clip_norm", 5.0))
        center_w = float(cfg["training"].get("center_loss_weight", 0.0))

        for epoch in range(1, epochs + 1):
            model.train()
            loss_running = 0.0

            pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", leave=False)
            for images, targets in pbar:
                if use_data_parallel:
                    model_images = images
                    model_targets = strip_targets_for_model(targets, device=None)
                else:
                    model_images = [img.to(device, non_blocking=True) for img in images]
                    model_targets = strip_targets_for_model(targets, device=device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled):
                    loss_dict = model(model_images, model_targets)
                    loss_dict = {
                        k: (v.mean() if torch.is_tensor(v) and v.ndim > 0 else v)
                        for k, v in loss_dict.items()
                    }
                    det_loss = sum(loss_dict.values())
                    if center_w > 0:
                        center_loss = unwrap_model(model).compute_center_loss(model_images, model_targets)
                        loss = det_loss + center_w * center_loss
                    else:
                        center_loss = torch.zeros((), device=device)
                        loss = det_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                # Update class prototypes with GT patches.
                with torch.no_grad():
                    unwrap_model(model).update_prototypes(model_images, model_targets)

                loss_running += float(loss.item())
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    det=f"{det_loss.item():.4f}",
                    ctr=f"{center_loss.item():.4f}",
                )

            scheduler.step()
            train_loss = loss_running / max(len(train_loader), 1)
            logger.info("Epoch %d train_loss=%.6f lr=%.8f", epoch, train_loss, optimizer.param_groups[0]["lr"])

            metrics = validate(unwrap_model(model), val_loader, device, amp_enabled, logger)
            val_map = float(metrics["mAP50"])

            if val_map > best_map:
                best_map = val_map
                ckpt = {
                    "epoch": epoch,
                    "model_state": unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_map": best_map,
                    "cfg": cfg,
                }
                torch.save(ckpt, best_ckpt)
                logger.info("Saved new best checkpoint: %s", best_ckpt)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_map": best_map,
                    "cfg": cfg,
                },
                latest_ckpt,
            )

            if stopper.step(val_map):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        copy_best_as_latest(str(best_ckpt), str(latest_ckpt))
        logger.info("Training finished. best_mAP50=%.6f", best_map)

    except Exception as exc:
        logger.error("Training failed: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    main(args.config)
