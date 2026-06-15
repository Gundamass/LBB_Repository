import argparse
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import build_test_loader
from src.masking import load_mask_for_image, should_keep_box_by_mask
from src.model import build_model
from src.utils import detect_hardware, ensure_dirs, load_config, save_json, setup_logger, zip_submission


def class_wise_nms(boxes, labels, scores, iou_thr=0.5):
    from torchvision.ops import nms

    keep_all = []
    uniq = labels.unique()
    for c in uniq:
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            continue
        k = nms(boxes[idx], scores[idx], iou_threshold=iou_thr)
        keep_all.append(idx[k])

    if not keep_all:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep = torch.cat(keep_all, dim=0)
    order = torch.argsort(scores[keep], descending=True)
    return keep[order]


def merge_predictions(preds: List[Dict], score_thr: float = 0.05, nms_thr: float = 0.5) -> Dict:
    if len(preds) == 1:
        p = preds[0]
        keep = p["scores"] >= score_thr
        return {
            "boxes": p["boxes"][keep],
            "labels": p["labels"][keep],
            "scores": p["scores"][keep],
        }

    boxes = torch.cat([p["boxes"] for p in preds], dim=0)
    labels = torch.cat([p["labels"] for p in preds], dim=0)
    scores = torch.cat([p["scores"] for p in preds], dim=0)

    keep = scores >= score_thr
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if boxes.numel() == 0:
        return {
            "boxes": boxes.reshape(0, 4),
            "labels": labels.reshape(0),
            "scores": scores.reshape(0),
        }

    keep = class_wise_nms(boxes, labels, scores, iou_thr=nms_thr)
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
    }


def apply_mask_filter_to_original_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    mask: np.ndarray,
    cfg: Dict,
) -> Dict:
    mask_cfg = cfg.get("mask", {})
    infer_cfg = mask_cfg.get("inference", {})
    min_coverage = float(infer_cfg.get("min_box_coverage", 0.10))
    keep_rule = str(infer_cfg.get("keep_rule", "center_or_coverage"))
    mode = str(infer_cfg.get("mode", "filter")).lower()
    background_score_factor = float(infer_cfg.get("background_score_factor", 0.15))
    min_score_after_rescore = float(infer_cfg.get("min_score_after_rescore", 0.0))

    if boxes.numel() == 0:
        return {"boxes": boxes, "labels": labels, "scores": scores}

    keep_indices = []
    new_scores = scores.clone()
    for idx, box in enumerate(boxes):
        keep, _, _ = should_keep_box_by_mask(
            mask,
            box.detach().cpu().tolist(),
            min_coverage=min_coverage,
            keep_rule=keep_rule,
        )
        if mode == "rescore":
            if not keep:
                new_scores[idx] = new_scores[idx] * background_score_factor
            if float(new_scores[idx].item()) >= min_score_after_rescore:
                keep_indices.append(idx)
        else:
            if keep:
                keep_indices.append(idx)

    if not keep_indices:
        empty_boxes = boxes.new_zeros((0, 4))
        empty_labels = labels.new_zeros((0,), dtype=labels.dtype)
        empty_scores = scores.new_zeros((0,), dtype=scores.dtype)
        return {"boxes": empty_boxes, "labels": empty_labels, "scores": empty_scores}

    keep_t = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
    return {
        "boxes": boxes[keep_t],
        "labels": labels[keep_t],
        "scores": new_scores[keep_t],
    }


@torch.no_grad()
def predict_tta(model, image: torch.Tensor, cfg: Dict, device: torch.device):
    tta_cfg = cfg["inference"]["tta"]
    score_thr = float(cfg["model"].get("score_threshold", 0.05))
    nms_thr = float(cfg["model"].get("nms_iou_threshold", 0.5))

    base_h, base_w = image.shape[1], image.shape[2]
    preds = []

    scales = tta_cfg.get("scales", [1.0]) if tta_cfg.get("enabled", True) else [1.0]
    for s in scales:
        if abs(float(s) - 1.0) < 1e-6:
            img_s = image
        else:
            img_s = F.interpolate(
                image.unsqueeze(0),
                scale_factor=float(s),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        pred = model([img_s.to(device)])[0]
        if abs(float(s) - 1.0) >= 1e-6 and pred["boxes"].numel() > 0:
            pred["boxes"][:, [0, 2]] /= float(s)
            pred["boxes"][:, [1, 3]] /= float(s)

        preds.append(pred)

        if tta_cfg.get("horizontal_flip", False):
            img_f = torch.flip(img_s, dims=[2])
            pred_f = model([img_f.to(device)])[0]
            if pred_f["boxes"].numel() > 0:
                x1 = pred_f["boxes"][:, 0].clone()
                x2 = pred_f["boxes"][:, 2].clone()
                w = img_s.shape[2]
                pred_f["boxes"][:, 0] = w - x2
                pred_f["boxes"][:, 2] = w - x1

                if abs(float(s) - 1.0) >= 1e-6:
                    pred_f["boxes"][:, [0, 2]] /= float(s)
                    pred_f["boxes"][:, [1, 3]] /= float(s)

            preds.append(pred_f)

    merged = merge_predictions(preds, score_thr=score_thr, nms_thr=nms_thr)

    if merged["boxes"].numel() > 0:
        merged["boxes"][:, [0, 2]] = merged["boxes"][:, [0, 2]].clamp(0, base_w - 1)
        merged["boxes"][:, [1, 3]] = merged["boxes"][:, [1, 3]].clamp(0, base_h - 1)

    return merged


@torch.no_grad()
def predict_ensemble_tta(models: List[torch.nn.Module], image: torch.Tensor, cfg: Dict, device: torch.device):
    score_thr = float(cfg["model"].get("score_threshold", 0.05))
    nms_thr = float(cfg["model"].get("nms_iou_threshold", 0.5))
    preds = []
    for m in models:
        preds.append(predict_tta(m, image, cfg, device))
    return merge_predictions(preds, score_thr=score_thr, nms_thr=nms_thr)


def main(config_path: str, ckpt_path: str):
    cfg = load_config(config_path)

    ensure_dirs([cfg["system"]["logs_dir"], cfg["inference"]["output_dir"]])
    logger = setup_logger(cfg["system"]["logs_dir"], name="inference")

    try:
        hw = detect_hardware()
        device = torch.device(hw.device)
        logger.info(
            "Hardware: device=%s gpu_count=%d gpu_names=%s total_vram_gb=%.2f",
            hw.device,
            hw.gpu_count,
            hw.gpu_names,
            hw.total_vram_gb,
        )

        ckpt_paths = [p.strip() for p in ckpt_path.split(",") if p.strip()]
        if not ckpt_paths:
            raise ValueError("No valid checkpoint path provided.")

        models = []
        for cp in ckpt_paths:
            model = build_model(cfg, logger=logger).to(device)
            ckpt = torch.load(cp, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            model.eval()
            models.append(model)
            logger.info("Loaded checkpoint: %s", cp)

        logger.info("Ensemble size: %d", len(models))

        test_loader, _ = build_test_loader(cfg, hw, logger=logger)

        zip_name = cfg["inference"]["zip_name"]
        if not zip_name.endswith(".zip"):
            zip_name += ".zip"
        team_folder_name = zip_name[:-4]

        output_root = Path(cfg["inference"]["output_dir"]).resolve()
        submit_folder = output_root / team_folder_name
        submit_folder.mkdir(parents=True, exist_ok=True)

        id_to_class = {int(v): k for k, v in cfg["classes"].items() if k != "background"}
        image_size = int(cfg["data"]["image_size"])
        mask_cfg = cfg.get("mask", {})
        mask_enabled = bool(mask_cfg.get("enabled", False))
        mask_test_dir = Path(mask_cfg.get("test_dir", "")).resolve() if mask_enabled else None
        missing_masks = 0
        mask_filtered = 0
        mask_before = 0
        mask_after = 0
        if mask_enabled:
            if mask_test_dir is None or not mask_test_dir.exists():
                raise FileNotFoundError(f"Mask test_dir not found: {mask_test_dir}")
            logger.info("Mask filtering enabled: test_dir=%s", mask_test_dir)

        for images, metas in tqdm(test_loader, desc="inference"):
            for img, meta in zip(images, metas):
                pred = predict_ensemble_tta(models, img.to(device), cfg, device)

                ow = int(meta["orig_width"])
                oh = int(meta["orig_height"])

                sx = ow / float(image_size)
                sy = oh / float(image_size)

                boxes_orig = pred["boxes"].detach().clone()
                if boxes_orig.numel() > 0:
                    boxes_orig[:, [0, 2]] *= sx
                    boxes_orig[:, [1, 3]] *= sy

                labels_orig = pred["labels"]
                scores_orig = pred["scores"]

                if mask_enabled:
                    mask = load_mask_for_image(
                        mask_test_dir,
                        meta["image_name"],
                        expected_size=(ow, oh),
                    )
                    if mask is None:
                        missing_masks += 1
                    else:
                        before = int(boxes_orig.shape[0])
                        filtered = apply_mask_filter_to_original_boxes(
                            boxes_orig,
                            labels_orig,
                            scores_orig,
                            mask,
                            cfg,
                        )
                        boxes_orig = filtered["boxes"]
                        labels_orig = filtered["labels"]
                        scores_orig = filtered["scores"]
                        after = int(boxes_orig.shape[0])
                        mask_before += before
                        mask_after += after
                        mask_filtered += max(0, before - after)

                annos = []
                for b, l, s in zip(boxes_orig, labels_orig, scores_orig):
                    cls_id = int(l.item())
                    if cls_id <= 0:
                        continue
                    label = id_to_class.get(cls_id)
                    if label is None:
                        continue

                    x1, y1, x2, y2 = b.tolist()

                    x1 = max(0.0, min(float(ow - 1), x1))
                    y1 = max(0.0, min(float(oh - 1), y1))
                    x2 = max(0.0, min(float(ow - 1), x2))
                    y2 = max(0.0, min(float(oh - 1), y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    annos.append(
                        {
                            "label": label,
                            "bbox": [
                                round(x1, 3),
                                round(y1, 3),
                                round(x2, 3),
                                round(y2, 3),
                            ],
                            "confidence": round(float(s.item()), 6),
                        }
                    )

                payload = {
                    "image_id": meta["image_name"],
                    "annotations": annos,
                }

                out_name = Path(meta["image_name"]).with_suffix(".json").name
                save_json(
                    str(submit_folder / out_name),
                    payload,
                    encoding=cfg["inference"].get("json_encoding", "utf-8"),
                )

        zip_path = output_root / zip_name
        zip_submission(str(submit_folder), str(zip_path))
        if mask_enabled:
            logger.info(
                "Mask filtering stats: missing_masks=%d boxes_before=%d boxes_after=%d filtered=%d",
                missing_masks,
                mask_before,
                mask_after,
                mask_filtered,
            )
        logger.info("Submission exported: %s", zip_path)

    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/best_model.pt")
    args = parser.parse_args()
    main(args.config, args.ckpt)
