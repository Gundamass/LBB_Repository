import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
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


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_shape_box(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def expand_box(box: Sequence[float], width: int, height: int, factor: float, pad: float) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * float(factor) + 2.0 * float(pad)
    bh = (y2 - y1) * float(factor) + 2.0 * float(pad)
    return clip_box([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], width, height)


def load_product_mask(mask_dir: Path, split: str, image_name: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
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


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = np.ones((radius, radius), np.uint8)
    out = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out if out.any() else mask


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


def choose_location(mask: np.ndarray, box_w: int, box_h: int, rng: random.Random, max_tries: int = 80) -> Optional[Tuple[int, int]]:
    h, w = mask.shape[:2]
    if box_w >= w or box_h >= h:
        return None
    margin = max(4, int(max(box_w, box_h) * 0.15))
    valid = erode_mask(mask, max(5, min(41, margin * 2 + 1)))
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return None
    for _ in range(max_tries):
        idx = rng.randrange(len(xs))
        cx = int(xs[idx])
        cy = int(ys[idx])
        x1 = int(round(cx - box_w * 0.5))
        y1 = int(round(cy - box_h * 0.5))
        x2 = x1 + int(box_w)
        y2 = y1 + int(box_h)
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        if mask_coverage(mask, [x1, y1, x2, y2]) >= 0.80:
            return x1, y1
    return None


def polygon_mask_for_box(points: Sequence[Sequence[float]], crop_box: Sequence[float], crop_size: Tuple[int, int]) -> np.ndarray:
    cw, ch = crop_size
    mask = np.zeros((ch, cw), dtype=np.uint8)
    if len(points) >= 3:
        x0, y0 = crop_box[0], crop_box[1]
        pts = np.array([[int(round(float(x) - x0)), int(round(float(y) - y0))] for x, y in points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
    else:
        mask[:, :] = 255
    return mask


def feather_alpha(alpha: np.ndarray, sigma: float) -> np.ndarray:
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    a = alpha.astype(np.float32) / 255.0
    if sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=float(sigma))
    return np.clip(a, 0.0, 1.0)


def crop_supports(defect_dir: Path, crop_expand: float, crop_pad: float) -> List[Dict]:
    rows: List[Dict] = []
    for ann_path in sorted(defect_dir.rglob("*.json")):
        image_path = ann_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        try:
            data = load_json(ann_path)
        except Exception:
            continue
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            image = np.asarray(img, dtype=np.uint8)

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in LABELS:
                continue
            points = shape.get("points", [])
            box = parse_shape_box(points)
            if box is None:
                continue
            box = clip_box(box, width, height)
            if box is None:
                continue
            crop_box = expand_box(box, width, height, crop_expand, crop_pad)
            if crop_box is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in crop_box]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2].copy()
            alpha = polygon_mask_for_box(points, [x1, y1, x2, y2], (x2 - x1, y2 - y1))
            # Keep a soft rectangular support as context; the polygon itself is
            # often just a two-point rectangle in LabelMe.
            rect_alpha = np.zeros_like(alpha)
            bx1 = max(0, int(round(box[0] - x1)))
            by1 = max(0, int(round(box[1] - y1)))
            bx2 = min(rect_alpha.shape[1], int(round(box[2] - x1)))
            by2 = min(rect_alpha.shape[0], int(round(box[3] - y1)))
            rect_alpha[by1:by2, bx1:bx2] = 255
            alpha = np.maximum(alpha, rect_alpha)
            if crop.shape[0] < 3 or crop.shape[1] < 3:
                continue
            rows.append(
                {
                    "label": label,
                    "crop": crop,
                    "alpha": alpha,
                    "box_in_crop": [box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1],
                    "source": str(image_path),
                }
            )
    return rows


def local_stats(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    flat = arr.reshape(-1, 3).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + eps
    return mean, std


def match_color(src: np.ndarray, dst: np.ndarray, strength: float) -> np.ndarray:
    src_f = src.astype(np.float32)
    dst_f = dst.astype(np.float32)
    sm, ss = local_stats(src_f)
    dm, ds = local_stats(dst_f)
    matched = (src_f - sm) / ss * ds + dm
    out = src_f * (1.0 - strength) + matched * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate_scale(crop: np.ndarray, alpha: np.ndarray, scale: float, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape[:2]
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if abs(angle) < 1e-3:
        return crop, alpha
    center = (new_w * 0.5, new_h * 0.5)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    bound_w = int(new_h * sin + new_w * cos)
    bound_h = int(new_h * cos + new_w * sin)
    mat[0, 2] += bound_w * 0.5 - center[0]
    mat[1, 2] += bound_h * 0.5 - center[1]
    crop_r = cv2.warpAffine(crop, mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha_r = cv2.warpAffine(alpha, mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop_r, alpha_r


def transformed_box_from_alpha(alpha: np.ndarray, pad: int = 1) -> Optional[List[float]]:
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        return None
    h, w = alpha.shape[:2]
    return [
        max(0.0, float(xs.min() - pad)),
        max(0.0, float(ys.min() - pad)),
        min(float(w - 1), float(xs.max() + 1 + pad)),
        min(float(h - 1), float(ys.max() + 1 + pad)),
    ]


def paste_one(base: np.ndarray, product_mask: np.ndarray, support: Dict, rng: random.Random, args) -> Optional[Tuple[np.ndarray, Dict]]:
    height, width = base.shape[:2]
    label = support["label"]
    crop = support["crop"]
    alpha = support["alpha"]

    if label == "plain particle":
        scale = rng.uniform(float(args.particle_scale_min), float(args.particle_scale_max))
        angle = rng.uniform(-25.0, 25.0)
    elif label == "scratch":
        scale = rng.uniform(float(args.scratch_scale_min), float(args.scratch_scale_max))
        angle = rng.uniform(-12.0, 12.0)
    elif label == "collision":
        scale = rng.uniform(float(args.collision_scale_min), float(args.collision_scale_max))
        angle = rng.uniform(-20.0, 20.0)
    else:
        scale = rng.uniform(float(args.dirt_scale_min), float(args.dirt_scale_max))
        angle = rng.uniform(-25.0, 25.0)

    crop_t, alpha_t = rotate_scale(crop, alpha, scale, angle)
    box_local = transformed_box_from_alpha(alpha_t, pad=int(args.box_pad))
    if box_local is None:
        return None
    ch, cw = crop_t.shape[:2]
    if cw < 2 or ch < 2 or cw >= width or ch >= height:
        return None
    loc = choose_location(product_mask, cw, ch, rng)
    if loc is None:
        return None
    x, y = loc
    roi = base[y : y + ch, x : x + cw]
    if roi.shape[:2] != crop_t.shape[:2]:
        return None
    if product_mask[y : y + ch, x : x + cw].mean() < 0.80:
        return None

    crop_t = match_color(crop_t, roi, strength=float(args.color_match_strength))
    a = feather_alpha(alpha_t, sigma=float(args.feather_sigma))
    a *= product_mask[y : y + ch, x : x + cw].astype(np.float32)
    a *= rng.uniform(float(args.alpha_min), float(args.alpha_max))
    if label == "scratch":
        a *= rng.uniform(0.80, 1.10)
    elif label == "plain particle":
        a *= rng.uniform(0.90, 1.20)
    a = np.clip(a, 0.0, 1.0)
    if a.max() < 0.05:
        return None

    out = base.copy()
    out_roi = roi.astype(np.float32) * (1.0 - a[..., None]) + crop_t.astype(np.float32) * a[..., None]
    out[y : y + ch, x : x + cw] = np.clip(out_roi, 0, 255).astype(np.uint8)

    bx1, by1, bx2, by2 = box_local
    box = clip_box([x + bx1, y + by1, x + bx2, y + by2], width, height)
    if box is None:
        return None
    if (box[2] - box[0]) < float(args.min_box_side) or (box[3] - box[1]) < float(args.min_box_side):
        return None
    if mask_coverage(product_mask, box) < 0.45:
        return None
    ann = {"label": label, "bbox": [round(float(v), 3) for v in box]}
    return out, ann


def parse_class_weights(s: str) -> Dict[str, float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} class weights for {LABELS}")
    total = sum(max(v, 0.0) for v in vals)
    if total <= 0:
        raise ValueError("Class weights must sum to a positive value.")
    return {label: max(val, 0.0) / total for label, val in zip(LABELS, vals)}


def choose_label(weights: Dict[str, float], available: Dict[str, List[Dict]], rng: random.Random) -> str:
    labels = [label for label in LABELS if available.get(label)]
    if not labels:
        raise RuntimeError("No available support labels.")
    total = sum(weights.get(label, 0.0) for label in labels)
    if total <= 0:
        return rng.choice(labels)
    r = rng.random() * total
    acc = 0.0
    for label in labels:
        acc += weights.get(label, 0.0)
        if r <= acc:
            return label
    return labels[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real defect copy-paste data from LBB support defects.")
    parser.add_argument("--clean-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--defect-dir", default="./初赛数据/训练集/负样本")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--out-root", default="./outputs/realpaste_defects_v2")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--min-defects", type=int, default=1)
    parser.add_argument("--max-defects", type=int, default=3)
    parser.add_argument("--class-weights", default="0.24,0.26,0.28,0.22")
    parser.add_argument("--crop-expand", type=float, default=1.18)
    parser.add_argument("--crop-pad", type=float, default=4.0)
    parser.add_argument("--particle-scale-min", type=float, default=0.85)
    parser.add_argument("--particle-scale-max", type=float, default=1.35)
    parser.add_argument("--dirt-scale-min", type=float, default=0.75)
    parser.add_argument("--dirt-scale-max", type=float, default=1.45)
    parser.add_argument("--scratch-scale-min", type=float, default=0.55)
    parser.add_argument("--scratch-scale-max", type=float, default=1.25)
    parser.add_argument("--collision-scale-min", type=float, default=0.75)
    parser.add_argument("--collision-scale-max", type=float, default=1.35)
    parser.add_argument("--color-match-strength", type=float, default=0.72)
    parser.add_argument("--feather-sigma", type=float, default=2.0)
    parser.add_argument("--alpha-min", type=float, default=0.55)
    parser.add_argument("--alpha-max", type=float, default=0.95)
    parser.add_argument("--box-pad", type=int, default=1)
    parser.add_argument("--min-box-side", type=float, default=4.0)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    image_dir = out_root / "images"
    ann_dir = out_root / "annotations"
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    supports = crop_supports(Path(args.defect_dir), float(args.crop_expand), float(args.crop_pad))
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for row in supports:
        by_label[row["label"]].append(row)
    if not supports:
        raise RuntimeError("No support crops found.")
    print("Support crops:", {label: len(by_label.get(label, [])) for label in LABELS})

    weights = parse_class_weights(args.class_weights)
    clean_images = find_images(Path(args.clean_dir))
    if int(args.limit_images) > 0:
        clean_images = clean_images[: int(args.limit_images)]
    if not clean_images:
        raise FileNotFoundError(f"No clean images found: {args.clean_dir}")

    counts = Counter()
    image_count = 0
    box_count = 0
    skipped = 0
    for image_path in tqdm(clean_images, desc="realpaste defects"):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            base = np.asarray(img, dtype=np.uint8)
        product_mask = load_product_mask(Path(args.mask_dir), "train", image_path.name, (width, height))
        if product_mask is None:
            product_mask = np.ones((height, width), dtype=bool)

        for rep in range(int(args.repeats)):
            image = base.copy()
            anns: List[Dict] = []
            n_defects = rng.randint(int(args.min_defects), int(args.max_defects))
            for _ in range(n_defects):
                label = choose_label(weights, by_label, rng)
                support = rng.choice(by_label[label])
                pasted = None
                for _try in range(35):
                    pasted = paste_one(image, product_mask, support, rng, args)
                    if pasted is not None:
                        break
                    support = rng.choice(by_label[label])
                if pasted is None:
                    skipped += 1
                    continue
                image, ann = pasted
                anns.append(ann)
                counts[ann["label"]] += 1

            if not anns:
                skipped += 1
                continue
            out_name = f"{image_path.stem}_realpaste{rep:02d}_{int(args.seed)}.jpg"
            Image.fromarray(image).save(image_dir / out_name, quality=int(args.jpeg_quality))
            save_json(
                ann_dir / f"{Path(out_name).stem}.json",
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
