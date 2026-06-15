import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.masking import save_mask, segment_laptop_mask
from src.utils import load_config


def _resolve_path(path_value: str, base: Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _collect_images(cfg: Dict, split: str) -> List[Path]:
    if split == "train":
        roots = [
            Path(cfg["data"]["train_pos_dir"]),
            Path(cfg["data"]["train_neg_dir"]),
        ]
        paths: List[Path] = []
        for root in roots:
            paths.extend(sorted(root.rglob("*.jpg")))
        return sorted(set(paths))

    if split == "test":
        return sorted(Path(cfg["data"]["test_dir"]).rglob("*.jpg"))

    raise ValueError(f"Unsupported split: {split}")


def _save_preview(image: Image.Image, mask: np.ndarray, out_path: Path) -> None:
    base = image.convert("RGB").convert("RGBA")
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 1] = 255
    rgba[..., 3] = (mask.astype(np.uint8) * 80)
    overlay = Image.fromarray(rgba, mode="RGBA")
    preview = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=92)


def generate_for_split(
    cfg: Dict,
    split: str,
    output_dir: Path,
    overwrite: bool,
    preview_count: int,
) -> Tuple[int, int, List[float], int]:
    image_paths = _collect_images(cfg, split)
    split_dir = output_dir / split
    preview_dir = output_dir / "preview" / split
    gen_cfg = cfg.get("mask", {}).get("generator", {})

    written = 0
    skipped = 0
    fallback = 0
    ratios: List[float] = []

    pbar = tqdm(image_paths, desc=f"masks/{split}")
    for idx, image_path in enumerate(pbar):
        out_path = split_dir / f"{image_path.stem}.png"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        mask, stats = segment_laptop_mask(image, gen_cfg)
        save_mask(mask, out_path)

        if idx < preview_count:
            _save_preview(image, mask, preview_dir / f"{image_path.stem}.jpg")

        written += 1
        ratios.append(stats.foreground_ratio)
        if stats.fallback:
            fallback += 1

    return written, skipped, ratios, fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.mask.yaml")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits: train,test",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-count", type=int, default=16)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    base = config_path.parent

    if args.output_dir is not None:
        output_dir = _resolve_path(args.output_dir, base)
    else:
        output_dir_value = (
            cfg.get("mask", {})
            .get("generator", {})
            .get("output_dir", "./masks/laptop_object")
        )
        output_dir = _resolve_path(str(output_dir_value), base)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    invalid = [s for s in splits if s not in {"train", "test"}]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}. Use train,test.")

    print(f"Mask output: {output_dir}")
    for split in splits:
        written, skipped, ratios, fallback = generate_for_split(
            cfg=cfg,
            split=split,
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
            preview_count=max(0, int(args.preview_count)),
        )
        if ratios:
            arr = np.asarray(ratios, dtype=np.float32)
            ratio_msg = (
                f"foreground ratio min/mean/max="
                f"{arr.min():.4f}/{arr.mean():.4f}/{arr.max():.4f}"
            )
        else:
            ratio_msg = "foreground ratio: no new masks"
        print(
            f"{split}: written={written} skipped={skipped} "
            f"fallback={fallback} | {ratio_msg}"
        )


if __name__ == "__main__":
    main()
