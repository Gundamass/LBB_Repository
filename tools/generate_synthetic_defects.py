import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


LABELS = ["plain particle", "dirt", "scratch", "collision"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_mask(mask_dir: Path, split: str, image_name: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    path = mask_dir / split / f"{Path(image_name).stem}.png"
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            if img.size != size:
                img = img.resize(size, Image.NEAREST)
            arr = np.asarray(img, dtype=np.uint8)
        mask = arr > 127
        if mask.mean() < 0.03:
            return None
        return mask
    except Exception:
        return None


def erode_for_sampling(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = np.ones((radius, radius), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return eroded if eroded.any() else mask


def choose_point(mask: np.ndarray, rng: random.Random, margin: int = 0) -> Optional[Tuple[int, int]]:
    h, w = mask.shape[:2]
    valid = mask
    if margin > 0 and h > 2 * margin and w > 2 * margin:
        valid = np.zeros_like(mask, dtype=bool)
        valid[margin : h - margin, margin : w - margin] = mask[margin : h - margin, margin : w - margin]
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return None
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def mask_coverage(mask: np.ndarray, box: Sequence[float]) -> float:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def blend_with_alpha(image: np.ndarray, color_layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha3 = np.clip(alpha[..., None].astype(np.float32), 0.0, 1.0)
    out = image.astype(np.float32) * (1.0 - alpha3) + color_layer.astype(np.float32) * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def local_mean_color(image: np.ndarray, x: int, y: int, radius: int = 12) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([128, 128, 128], dtype=np.float32)
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


def random_contrast_color(base: np.ndarray, rng: random.Random, label: str) -> np.ndarray:
    base = base.astype(np.float32)
    if label in {"scratch", "collision"}:
        delta = rng.choice([-1.0, 1.0]) * rng.uniform(28, 85)
        color = base + delta
    elif label == "dirt":
        tint = np.array([rng.uniform(-18, 18), rng.uniform(-22, 8), rng.uniform(-30, 0)], dtype=np.float32)
        color = base * rng.uniform(0.45, 0.82) + tint
    else:
        delta = rng.choice([-1.0, 1.0]) * rng.uniform(35, 100)
        color = base + delta
    return np.clip(color, 0, 255).astype(np.uint8)


def draw_scratch(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> Optional[Tuple[np.ndarray, List[float]]]:
    h, w = image.shape[:2]
    point = choose_point(erode_for_sampling(mask, 11), rng, margin=8)
    if point is None:
        return None
    cx, cy = point
    length = rng.uniform(35, min(360, max(45, 0.30 * min(w, h))))
    thickness = rng.uniform(1.2, 5.5)
    angle = rng.uniform(0, math.tau)
    n_pts = rng.randint(3, 7)
    perp = angle + math.pi * 0.5
    pts = []
    for i in range(n_pts):
        t = (i / max(n_pts - 1, 1) - 0.5) * length
        jitter = rng.uniform(-0.14, 0.14) * length
        x = cx + math.cos(angle) * t + math.cos(perp) * jitter * 0.22
        y = cy + math.sin(angle) * t + math.sin(perp) * jitter * 0.22
        pts.append([int(round(x)), int(round(y))])

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = max(4.0, thickness * 3.0)
    box = clip_box([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad], w, h)
    if box is None or mask_coverage(mask, box) < 0.45:
        return None

    alpha = np.zeros((h, w), dtype=np.float32)
    line_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(line_mask, [np.array(pts, dtype=np.int32)], False, 255, int(round(thickness)), cv2.LINE_AA)
    if rng.random() < 0.55:
        cv2.polylines(line_mask, [np.array(pts, dtype=np.int32)], False, 255, max(1, int(round(thickness * 0.45))), cv2.LINE_AA)
    blur = cv2.GaussianBlur(line_mask, (0, 0), sigmaX=rng.uniform(0.4, 1.4))
    alpha = (blur.astype(np.float32) / 255.0) * rng.uniform(0.35, 0.85)
    alpha *= mask.astype(np.float32)
    color = random_contrast_color(local_mean_color(image, cx, cy), rng, "scratch")
    color_layer = np.zeros_like(image)
    color_layer[:] = color
    return blend_with_alpha(image, color_layer, alpha), box


def draw_dirt(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> Optional[Tuple[np.ndarray, List[float]]]:
    h, w = image.shape[:2]
    point = choose_point(erode_for_sampling(mask, 15), rng, margin=10)
    if point is None:
        return None
    cx, cy = point
    rx = rng.uniform(10, 75)
    ry = rng.uniform(8, 70)
    angle = rng.uniform(0, math.tau)
    n = rng.randint(8, 18)
    pts = []
    for i in range(n):
        a = math.tau * i / n
        rr = rng.uniform(0.55, 1.18)
        x = math.cos(a) * rx * rr
        y = math.sin(a) * ry * rr
        xr = math.cos(angle) * x - math.sin(angle) * y
        yr = math.sin(angle) * x + math.cos(angle) * y
        pts.append([int(round(cx + xr)), int(round(cy + yr))])

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    box = clip_box([min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4], w, h)
    if box is None or mask_coverage(mask, box) < 0.45:
        return None

    blob = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(blob, [np.array(pts, dtype=np.int32)], 255, cv2.LINE_AA)
    blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=rng.uniform(1.0, 4.0))
    noise = cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).uniform(0.75, 1.25, (h, w)).astype(np.float32), (0, 0), 1.2)
    alpha = (blob.astype(np.float32) / 255.0) * noise * rng.uniform(0.18, 0.58)
    alpha *= mask.astype(np.float32)
    color = random_contrast_color(local_mean_color(image, cx, cy), rng, "dirt")
    color_layer = np.zeros_like(image)
    color_layer[:] = color
    return blend_with_alpha(image, color_layer, alpha), box


def draw_particle(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> Optional[Tuple[np.ndarray, List[float]]]:
    h, w = image.shape[:2]
    point = choose_point(erode_for_sampling(mask, 7), rng, margin=4)
    if point is None:
        return None
    cx, cy = point
    rx = rng.uniform(3, 15)
    ry = rng.uniform(3, 15)
    angle = rng.uniform(0, 180)
    box = clip_box([cx - rx - 3, cy - ry - 3, cx + rx + 3, cy + ry + 3], w, h)
    if box is None or mask_coverage(mask, box) < 0.50:
        return None
    spot = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(spot, (cx, cy), (int(round(rx)), int(round(ry))), angle, 0, 360, 255, -1, cv2.LINE_AA)
    if rng.random() < 0.35:
        cv2.circle(spot, (cx + rng.randint(-2, 2), cy + rng.randint(-2, 2)), max(1, int(min(rx, ry) * 0.35)), 255, -1, cv2.LINE_AA)
    spot = cv2.GaussianBlur(spot, (0, 0), sigmaX=rng.uniform(0.35, 1.0))
    alpha = (spot.astype(np.float32) / 255.0) * rng.uniform(0.45, 0.95)
    alpha *= mask.astype(np.float32)
    color = random_contrast_color(local_mean_color(image, cx, cy), rng, "plain particle")
    color_layer = np.zeros_like(image)
    color_layer[:] = color
    return blend_with_alpha(image, color_layer, alpha), box


def draw_collision(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> Optional[Tuple[np.ndarray, List[float]]]:
    h, w = image.shape[:2]
    point = choose_point(erode_for_sampling(mask, 13), rng, margin=8)
    if point is None:
        return None
    cx, cy = point
    rx = rng.uniform(8, 36)
    ry = rng.uniform(7, 34)
    angle = rng.uniform(0, 180)
    box = clip_box([cx - rx * 1.5, cy - ry * 1.5, cx + rx * 1.5, cy + ry * 1.5], w, h)
    if box is None or mask_coverage(mask, box) < 0.45:
        return None

    alpha_dark = np.zeros((h, w), dtype=np.uint8)
    alpha_light = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(alpha_dark, (cx, cy), (int(rx), int(ry)), angle, 20, 320, 255, max(1, int(min(rx, ry) * 0.22)), cv2.LINE_AA)
    cv2.ellipse(alpha_light, (int(cx - rx * 0.18), int(cy - ry * 0.22)), (max(2, int(rx * 0.55)), max(2, int(ry * 0.45))), angle, 190, 340, 255, max(1, int(min(rx, ry) * 0.12)), cv2.LINE_AA)
    alpha_dark = cv2.GaussianBlur(alpha_dark, (0, 0), sigmaX=rng.uniform(0.6, 1.6)).astype(np.float32) / 255.0
    alpha_light = cv2.GaussianBlur(alpha_light, (0, 0), sigmaX=rng.uniform(0.5, 1.2)).astype(np.float32) / 255.0
    alpha_dark *= rng.uniform(0.35, 0.75) * mask.astype(np.float32)
    alpha_light *= rng.uniform(0.20, 0.55) * mask.astype(np.float32)

    base = local_mean_color(image, cx, cy)
    dark = np.clip(base - rng.uniform(35, 90), 0, 255).astype(np.uint8)
    light = np.clip(base + rng.uniform(25, 70), 0, 255).astype(np.uint8)
    layer = np.zeros_like(image)
    layer[:] = dark
    out = blend_with_alpha(image, layer, alpha_dark)
    layer[:] = light
    out = blend_with_alpha(out, layer, alpha_light)
    return out, box


def draw_one(image: np.ndarray, mask: np.ndarray, label: str, rng: random.Random) -> Optional[Tuple[np.ndarray, List[float]]]:
    for _ in range(40):
        if label == "scratch":
            res = draw_scratch(image, mask, rng)
        elif label == "dirt":
            res = draw_dirt(image, mask, rng)
        elif label == "plain particle":
            res = draw_particle(image, mask, rng)
        elif label == "collision":
            res = draw_collision(image, mask, rng)
        else:
            res = None
        if res is not None:
            return res
    return None


def parse_weights(s: str) -> Dict[str, float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != len(LABELS):
        raise ValueError(f"--class-weights must have {len(LABELS)} comma-separated values")
    total = sum(max(0.0, v) for v in vals)
    if total <= 0:
        raise ValueError("--class-weights sum must be positive")
    return {label: max(0.0, val) / total for label, val in zip(LABELS, vals)}


def choose_label(weights: Dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for label in LABELS:
        acc += weights[label]
        if r <= acc:
            return label
    return LABELS[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic defect images from clean LBB positives.")
    parser.add_argument("--clean-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--out-root", default="./outputs/synth_defects_v1")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--min-defects", type=int, default=1)
    parser.add_argument("--max-defects", type=int, default=3)
    parser.add_argument(
        "--class-weights",
        default="0.18,0.25,0.37,0.20",
        help="Weights for plain particle,dirt,scratch,collision.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    mask_dir = Path(args.mask_dir)
    out_root = Path(args.out_root)
    image_dir = out_root / "images"
    ann_dir = out_root / "annotations"
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(clean_dir)
    if int(args.limit_images) > 0:
        images = images[: int(args.limit_images)]
    if not images:
        raise FileNotFoundError(f"No clean images found: {clean_dir}")

    weights = parse_weights(args.class_weights)
    rng = random.Random(int(args.seed))
    counts = Counter()
    image_count = 0
    box_count = 0
    skipped = 0

    for image_path in tqdm(images, desc="synthetic defects"):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            base = np.asarray(img, dtype=np.uint8)
        mask = load_mask(mask_dir, "train", image_path.name, (width, height))
        if mask is None:
            mask = np.ones((height, width), dtype=bool)

        for rep in range(int(args.repeats)):
            image = base.copy()
            anns = []
            n_defects = rng.randint(int(args.min_defects), int(args.max_defects))
            for _ in range(n_defects):
                label = choose_label(weights, rng)
                res = draw_one(image, mask, label, rng)
                if res is None:
                    skipped += 1
                    continue
                image, box = res
                anns.append(
                    {
                        "label": label,
                        "bbox": [round(float(v), 3) for v in box],
                    }
                )
                counts[label] += 1

            if not anns:
                skipped += 1
                continue
            out_name = f"{image_path.stem}_synth{rep:02d}_{int(args.seed)}.jpg"
            out_ann = f"{Path(out_name).stem}.json"
            Image.fromarray(image).save(image_dir / out_name, quality=int(args.jpeg_quality))
            save_json(
                ann_dir / out_ann,
                {
                    "image_id": out_name,
                    "annotations": anns,
                },
            )
            image_count += 1
            box_count += len(anns)

    print(f"Output root: {out_root.resolve()}")
    print(f"Images: {image_count} boxes: {box_count} skipped_attempts: {skipped}")
    print(f"Class counts: {dict(counts)}")
    print(f"Image dir: {image_dir.resolve()}")
    print(f"Annotation dir: {ann_dir.resolve()}")


if __name__ == "__main__":
    main()
