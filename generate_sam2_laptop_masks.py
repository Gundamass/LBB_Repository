import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.masking import (
    clean_binary_mask,
    mask_to_box,
    sample_background_points,
    sample_points_from_mask,
    save_mask,
    segment_laptop_mask,
)
from src.utils import load_config


def _resolve_path(path_value: str, base: Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _collect_images(cfg: Dict, split: str) -> List[Path]:
    if split == "train":
        roots = [Path(cfg["data"]["train_pos_dir"]), Path(cfg["data"]["train_neg_dir"])]
        paths: List[Path] = []
        for root in roots:
            paths.extend(sorted(root.rglob("*.jpg")))
        return sorted(set(paths))
    if split == "test":
        return sorted(Path(cfg["data"]["test_dir"]).rglob("*.jpg"))
    raise ValueError(f"Unsupported split: {split}")


def _import_sam2(cfg: Dict):
    import sys

    sam2_repo = Path(cfg["sam2"]["repo_dir"]).resolve()
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return build_sam2, SAM2ImagePredictor


def _draw_preview(
    image: Image.Image,
    coarse_mask: np.ndarray,
    sam_mask: np.ndarray,
    out_path: Path,
    box: Optional[np.ndarray] = None,
    fg_points: Optional[np.ndarray] = None,
    bg_points: Optional[np.ndarray] = None,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # coarse mask in red
    coarse_arr = np.zeros((coarse_mask.shape[0], coarse_mask.shape[1], 4), dtype=np.uint8)
    coarse_arr[..., 0] = 255
    coarse_arr[..., 3] = coarse_mask.astype(np.uint8) * 70
    coarse_img = Image.fromarray(coarse_arr, mode="RGBA")
    overlay = Image.alpha_composite(overlay, coarse_img)

    # SAM mask in green
    sam_arr = np.zeros((sam_mask.shape[0], sam_mask.shape[1], 4), dtype=np.uint8)
    sam_arr[..., 1] = 255
    sam_arr[..., 3] = sam_mask.astype(np.uint8) * 100
    sam_img = Image.fromarray(sam_arr, mode="RGBA")
    overlay = Image.alpha_composite(overlay, sam_img)

    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 255), width=4)

    if fg_points is not None:
        for x, y in fg_points.astype(int):
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 255, 0, 255))

    if bg_points is not None:
        for x, y in bg_points.astype(int):
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 0, 0, 255))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=92)


def _pick_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    coarse_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    if masks.ndim == 2:
        masks = masks[None, ...]
    if scores.ndim == 0:
        scores = np.asarray([float(scores)])

    coarse = coarse_mask.astype(bool)
    coarse_area = float(coarse.mean()) + 1e-9
    best_idx = 0
    best_score = -1e9
    best_stats = {}

    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        inter = float(np.logical_and(m, coarse).sum())
        union = float(np.logical_or(m, coarse).sum()) + 1e-9
        iou = inter / union
        coverage = float(np.logical_and(m, coarse).sum()) / (float(m.sum()) + 1e-9)
        coarse_recall = float(np.logical_and(m, coarse).sum()) / (float(coarse.sum()) + 1e-9)
        balance = 0.0
        if coarse_area > 0:
            balance = 1.0 - min(1.0, abs(float(m.mean()) - coarse_area) / coarse_area)
        score = float(scores[i]) + 1.5 * iou + 0.5 * coverage + 0.5 * coarse_recall + 0.2 * balance
        if score > best_score:
            best_score = score
            best_idx = i
            best_stats = {
                "iou": iou,
                "coverage": coverage,
                "coarse_recall": coarse_recall,
                "sam_score": float(scores[i]),
                "combined": score,
            }
    return masks[best_idx], best_stats


def _generate_mask_with_sam2(
    predictor,
    image: Image.Image,
    cfg: Dict,
    coarse_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    gen_cfg = cfg["mask"]["generator"]
    sam_cfg = cfg["mask"]["sam2"]

    box_pad = int(sam_cfg.get("box_pad", 16))
    fg_point_count = int(sam_cfg.get("fg_point_count", 8))
    bg_point_count = int(sam_cfg.get("bg_point_count", 8))
    use_box = bool(sam_cfg.get("use_box", True))
    use_points = bool(sam_cfg.get("use_points", True))
    multimask_output = bool(sam_cfg.get("multimask_output", True))
    box_from = str(sam_cfg.get("box_from", "coarse_mask"))
    erode_for_fg = int(sam_cfg.get("erode_for_fg", 6))

    box = None
    if use_box:
        if box_from == "full_image":
            box = np.asarray([0, 0, image.width - 1, image.height - 1], dtype=np.float32)
        else:
            box = mask_to_box(coarse_mask, padding=box_pad)
            if box is None:
                box = np.asarray([0, 0, image.width - 1, image.height - 1], dtype=np.float32)

    fg_points = None
    bg_points = None
    point_coords = None
    point_labels = None
    if use_points:
        fg_points = sample_points_from_mask(coarse_mask, fg_point_count, margin=erode_for_fg)
        bg_points = sample_background_points(coarse_mask, bg_point_count)
        pts = []
        labels = []
        if fg_points.size > 0:
            pts.append(fg_points)
            labels.append(np.ones((fg_points.shape[0],), dtype=np.int32))
        if bg_points.size > 0:
            pts.append(bg_points)
            labels.append(np.zeros((bg_points.shape[0],), dtype=np.int32))
        if pts:
            point_coords = np.concatenate(pts, axis=0)
            point_labels = np.concatenate(labels, axis=0)

    predictor.set_image(np.asarray(image.convert("RGB")))
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask_output,
        return_logits=False,
    )

    best_mask, stats = _pick_best_mask(masks, scores, coarse_mask)
    cleaned = clean_binary_mask(best_mask, gen_cfg)
    return cleaned, {
        **stats,
        "box": box.tolist() if box is not None else None,
        "fg_points": int(fg_points.shape[0]) if fg_points is not None else 0,
        "bg_points": int(bg_points.shape[0]) if bg_points is not None else 0,
        "coarse_ratio": float(coarse_mask.mean()),
        "sam_ratio": float(cleaned.mean()),
    }


def _generate_for_split(
    cfg: Dict,
    split: str,
    output_dir: Path,
    overwrite: bool,
    preview_count: int,
    limit: Optional[int] = None,
) -> None:
    image_paths = _collect_images(cfg, split)
    if limit is not None and int(limit) > 0:
        image_paths = image_paths[: int(limit)]
    split_dir = output_dir / split
    preview_dir = output_dir / "preview" / split
    gen_cfg = cfg["mask"]["generator"]

    build_sam2, SAM2ImagePredictor = _import_sam2(cfg)
    sam2_cfg = cfg["sam2"]
    predictor = SAM2ImagePredictor(
        build_sam2(
            sam2_cfg["config_file"],
            ckpt_path=sam2_cfg["checkpoint"],
            device=sam2_cfg.get("device", "cuda"),
            mode="eval",
        )
    )

    written = 0
    skipped = 0
    fallback = 0
    stats_list = []

    for idx, image_path in enumerate(tqdm(image_paths, desc=f"sam2/{split}")):
        out_path = split_dir / f"{image_path.stem}.png"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        coarse_mask, coarse_stats = segment_laptop_mask(image, gen_cfg)
        sam_mask, sam_stats = _generate_mask_with_sam2(predictor, image, cfg, coarse_mask)
        save_mask(sam_mask, out_path)

        if idx < preview_count:
            preview_path = preview_dir / f"{image_path.stem}.jpg"
            _draw_preview(
                image=image,
                coarse_mask=coarse_mask,
                sam_mask=sam_mask,
                out_path=preview_path,
                box=np.asarray(sam_stats.get("box")) if sam_stats.get("box") is not None else None,
                fg_points=sample_points_from_mask(coarse_mask, int(cfg["mask"]["sam2"].get("fg_point_count", 8)), margin=int(cfg["mask"]["sam2"].get("erode_for_fg", 6))),
                bg_points=sample_background_points(coarse_mask, int(cfg["mask"]["sam2"].get("bg_point_count", 8))),
            )

        written += 1
        if coarse_stats.fallback:
            fallback += 1
        stats_list.append((coarse_stats.foreground_ratio, sam_stats["sam_ratio"], sam_stats["combined"]))

    if stats_list:
        arr = np.asarray(stats_list, dtype=np.float32)
        print(
            f"{split}: written={written} skipped={skipped} fallback={fallback} | "
            f"coarse_ratio mean={arr[:,0].mean():.4f} sam_ratio mean={arr[:,1].mean():.4f} "
            f"combined mean={arr[:,2].mean():.4f}"
        )
    else:
        print(f"{split}: written={written} skipped={skipped} fallback={fallback}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.sam2_mask.yaml")
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images per split.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    base = config_path.parent

    if args.output_dir is not None:
        output_dir = _resolve_path(args.output_dir, base)
    else:
        output_dir = _resolve_path(
            cfg["mask"]["generator"].get("output_dir", "./masks/sam2_laptop_object"),
            base,
        )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    invalid = [s for s in splits if s not in {"train", "test"}]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}. Use train,test.")

    print(f"SAM2 mask output: {output_dir}")
    for split in splits:
        _generate_for_split(
            cfg=cfg,
            split=split,
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
            preview_count=max(0, int(args.preview_count)),
            limit=int(args.limit) if int(args.limit) > 0 else None,
        )


if __name__ == "__main__":
    main()
