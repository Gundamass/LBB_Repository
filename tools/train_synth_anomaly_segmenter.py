import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LABELS = ["plain particle", "dirt", "scratch", "collision"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_shape_box(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
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
    bw = max(2.0, (x2 - x1) * float(factor) + 2.0 * float(pad))
    bh = max(2.0, (y2 - y1) * float(factor) + 2.0 * float(pad))
    return clip_box([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], width, height)


def load_product_mask(mask_dir: Path, split: str, image_name: str, size: Tuple[int, int]) -> np.ndarray:
    path = mask_dir / split / f"{Path(image_name).stem}.png"
    if path.exists():
        try:
            with Image.open(path) as img:
                img = img.convert("L")
                if img.size != size:
                    img = img.resize(size, Image.NEAREST)
                arr = np.asarray(img, dtype=np.uint8)
            mask = arr > 127
            if mask.mean() >= 0.03:
                return mask
        except Exception:
            pass
    return np.ones((size[1], size[0]), dtype=bool)


def resize_keep_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8).copy()


def resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    out = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return out > 0


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
            rect_alpha = np.zeros_like(alpha)
            bx1 = max(0, int(round(box[0] - x1)))
            by1 = max(0, int(round(box[1] - y1)))
            bx2 = min(rect_alpha.shape[1], int(round(box[2] - x1)))
            by2 = min(rect_alpha.shape[0], int(round(box[3] - y1)))
            rect_alpha[by1:by2, bx1:bx2] = 255
            alpha = np.maximum(alpha, rect_alpha)
            if crop.shape[0] < 3 or crop.shape[1] < 3:
                continue
            rows.append({"label": label, "crop": crop, "alpha": alpha, "source": str(image_path)})
    return rows


def load_real_defect_paths(defect_dir: Path) -> List[Path]:
    return sorted(p for p in defect_dir.rglob("*.json") if p.with_suffix(".jpg").exists())


def real_defect_sample(ann_path: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    image_path = ann_path.with_suffix(".jpg")
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        image = np.asarray(img.resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8).copy()
    data = load_json(ann_path)
    target = np.zeros((len(LABELS), image_size, image_size), dtype=np.float32)
    sx = float(image_size) / max(float(width), 1.0)
    sy = float(image_size) / max(float(height), 1.0)
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in LABELS:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        pts = np.array([[int(round(float(x) * sx)), int(round(float(y) * sy))] for x, y in points], dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, image_size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_size - 1)
        if len(points) >= 3:
            cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
        else:
            x1, y1 = pts.min(axis=0).tolist()
            x2, y2 = pts.max(axis=0).tolist()
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1, cv2.LINE_AA)
        if mask.sum() > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            target[LABELS.index(label)] = np.maximum(target[LABELS.index(label)], (mask > 0).astype(np.float32))
    return image, target


def local_stats(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    flat = arr.reshape(-1, 3).astype(np.float32)
    return flat.mean(axis=0), flat.std(axis=0) + eps


def match_color(src: np.ndarray, dst: np.ndarray, strength: float) -> np.ndarray:
    src_f = src.astype(np.float32)
    dst_f = dst.astype(np.float32)
    sm, ss = local_stats(src_f)
    dm, ds = local_stats(dst_f)
    matched = (src_f - sm) / ss * ds + dm
    out = src_f * (1.0 - float(strength)) + matched * float(strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def choose_location(mask: np.ndarray, box_w: int, box_h: int, rng: random.Random) -> Optional[Tuple[int, int]]:
    h, w = mask.shape[:2]
    if box_w >= w or box_h >= h:
        return None
    kernel = np.ones((max(3, min(31, int(max(box_w, box_h) * 0.25) | 1)),) * 2, np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    valid = eroded if eroded.any() else mask
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return None
    for _ in range(120):
        idx = rng.randrange(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        x1 = int(round(cx - box_w * 0.5))
        y1 = int(round(cy - box_h * 0.5))
        x2, y2 = x1 + box_w, y1 + box_h
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        if mask[y1:y2, x1:x2].mean() >= 0.75:
            return x1, y1
    return None


def rotate_scale(crop: np.ndarray, alpha: np.ndarray, scale: float, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape[:2]
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if abs(angle) < 1e-3:
        return crop, alpha
    center = (nw * 0.5, nh * 0.5)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    bw = int(nh * sin + nw * cos)
    bh = int(nh * cos + nw * sin)
    mat[0, 2] += bw * 0.5 - center[0]
    mat[1, 2] += bh * 0.5 - center[1]
    crop_r = cv2.warpAffine(crop, mat, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha_r = cv2.warpAffine(alpha, mat, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop_r, alpha_r


def paste_real_defect(
    image: np.ndarray,
    target: np.ndarray,
    product_mask: np.ndarray,
    support: Dict,
    rng: random.Random,
    alpha_min: float,
    alpha_max: float,
    color_match_strength: float,
) -> bool:
    label = str(support["label"])
    label_idx = LABELS.index(label)
    if label == "collision":
        scale = rng.uniform(1.05, 2.20)
        angle = rng.uniform(-25.0, 25.0)
    elif label == "scratch":
        scale = rng.uniform(0.75, 1.70)
        angle = rng.uniform(-15.0, 15.0)
    elif label == "dirt":
        scale = rng.uniform(0.85, 1.90)
        angle = rng.uniform(-25.0, 25.0)
    else:
        scale = rng.uniform(0.75, 1.60)
        angle = rng.uniform(-30.0, 30.0)
    crop_t, alpha_t = rotate_scale(support["crop"], support["alpha"], scale, angle)
    h, w = image.shape[:2]
    ch, cw = crop_t.shape[:2]
    if ch < 2 or cw < 2 or ch >= h or cw >= w:
        return False
    loc = choose_location(product_mask, cw, ch, rng)
    if loc is None:
        return False
    x, y = loc
    roi = image[y : y + ch, x : x + cw]
    crop_t = match_color(crop_t, roi, strength=color_match_strength)
    alpha = alpha_t.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)
    alpha *= product_mask[y : y + ch, x : x + cw].astype(np.float32)
    alpha *= rng.uniform(alpha_min, alpha_max)
    alpha = np.clip(alpha, 0.0, 1.0)
    if alpha.max() < 0.04:
        return False
    blended = roi.astype(np.float32) * (1.0 - alpha[..., None]) + crop_t.astype(np.float32) * alpha[..., None]
    image[y : y + ch, x : x + cw] = np.clip(blended, 0, 255).astype(np.uint8)
    mask = alpha > max(0.08, float(alpha.max()) * 0.25)
    if mask.sum() < 4:
        return False
    target[label_idx, y : y + ch, x : x + cw] = np.maximum(target[label_idx, y : y + ch, x : x + cw], mask.astype(np.float32))
    return True


def random_point_in_mask(mask: np.ndarray, rng: random.Random) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def draw_procedural_defect(image: np.ndarray, target: np.ndarray, product_mask: np.ndarray, label: str, rng: random.Random) -> bool:
    h, w = image.shape[:2]
    pt = random_point_in_mask(product_mask, rng)
    if pt is None:
        return False
    cx, cy = pt
    overlay = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    if label == "scratch":
        length = rng.randint(max(18, w // 28), max(28, w // 9))
        angle = rng.uniform(0, math.pi)
        dx = int(math.cos(angle) * length * 0.5)
        dy = int(math.sin(angle) * length * 0.5)
        color_delta = rng.choice([-1, 1]) * rng.randint(18, 55)
        color = tuple(int(np.clip(int(image[cy, cx, c]) + color_delta, 0, 255)) for c in range(3))
        thickness = rng.randint(1, 3)
        cv2.line(overlay, (cx - dx, cy - dy), (cx + dx, cy + dy), color, thickness, cv2.LINE_AA)
        cv2.line(mask, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, max(2, thickness + 1), cv2.LINE_AA)
        alpha = rng.uniform(0.25, 0.65)
    elif label == "collision":
        ax = rng.randint(8, max(12, w // 28))
        ay = rng.randint(6, max(10, h // 30))
        angle = rng.uniform(0, 180)
        base = image[cy, cx].astype(np.int16)
        dark = tuple(int(np.clip(v - rng.randint(20, 65), 0, 255)) for v in base)
        light = tuple(int(np.clip(v + rng.randint(10, 35), 0, 255)) for v in base)
        cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360, dark, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx - ax // 4, cy - ay // 4), (max(2, ax // 2), max(2, ay // 2)), angle, 200, 340, light, 1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), (ax + 2, ay + 2), angle, 0, 360, 255, -1, cv2.LINE_AA)
        alpha = rng.uniform(0.18, 0.48)
    elif label == "dirt":
        pts = []
        radius = rng.randint(7, max(10, w // 32))
        for k in range(rng.randint(8, 15)):
            a = 2 * math.pi * k / 12.0 + rng.uniform(-0.25, 0.25)
            r = radius * rng.uniform(0.45, 1.15)
            pts.append([int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)])
        pts = np.asarray(pts, dtype=np.int32)
        base = image[cy, cx].astype(np.int16)
        color = tuple(int(np.clip(v + rng.choice([-1, 1]) * rng.randint(12, 45), 0, 255)) for v in base)
        cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
        mask[:] = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.4)
        alpha = rng.uniform(0.18, 0.55)
    else:
        radius = rng.randint(3, max(5, w // 95))
        base = image[cy, cx].astype(np.int16)
        color = tuple(int(np.clip(v + rng.choice([-1, 1]) * rng.randint(18, 60), 0, 255)) for v in base)
        cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(mask, (cx, cy), radius + 1, 255, -1, cv2.LINE_AA)
        alpha = rng.uniform(0.25, 0.75)

    valid = (mask > 0) & product_mask
    if valid.sum() < 4:
        return False
    a = (mask.astype(np.float32) / 255.0) * float(alpha)
    a *= product_mask.astype(np.float32)
    image[:] = np.clip(image.astype(np.float32) * (1.0 - a[..., None]) + overlay.astype(np.float32) * a[..., None], 0, 255).astype(np.uint8)
    idx = LABELS.index(label)
    target[idx] = np.maximum(target[idx], valid.astype(np.float32))
    return True


def parse_class_weights(value: str) -> Dict[str, float]:
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(vals) != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} class weights for {LABELS}, got {value!r}")
    total = sum(max(v, 0.0) for v in vals)
    if total <= 0:
        raise ValueError("class weights must sum to > 0")
    return {label: max(v, 0.0) / total for label, v in zip(LABELS, vals)}


def choose_label(weights: Dict[str, float], available: Dict[str, List[Dict]], rng: random.Random) -> str:
    labels = list(LABELS)
    total = sum(weights.get(label, 0.0) for label in labels)
    r = rng.random() * total
    acc = 0.0
    for label in labels:
        acc += weights.get(label, 0.0)
        if r <= acc:
            return label
    return labels[-1]


class SyntheticAnomalyDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        defect_dir: Path,
        mask_dir: Path,
        image_size: int,
        length: int,
        seed: int,
        normal_prob: float,
        min_defects: int,
        max_defects: int,
        class_weights: Dict[str, float],
        procedural_prob: float,
        collision_real_prob: float,
        real_defect_prob: float,
    ):
        self.clean_images = find_images(clean_dir)
        if not self.clean_images:
            raise FileNotFoundError(f"No clean images found in {clean_dir}")
        self.mask_dir = mask_dir
        self.image_size = int(image_size)
        self.length = int(length)
        self.seed = int(seed)
        self.normal_prob = float(normal_prob)
        self.min_defects = int(min_defects)
        self.max_defects = int(max_defects)
        self.class_weights = class_weights
        self.procedural_prob = float(procedural_prob)
        self.collision_real_prob = float(collision_real_prob)
        self.real_defect_prob = float(real_defect_prob)
        self.supports = crop_supports(defect_dir, crop_expand=1.22, crop_pad=5.0)
        if not self.supports:
            raise FileNotFoundError(f"No defect support crops found in {defect_dir}")
        self.real_defects = load_real_defect_paths(defect_dir)
        self.supports_by_label: Dict[str, List[Dict]] = defaultdict(list)
        for row in self.supports:
            self.supports_by_label[str(row["label"])].append(row)
        print("Support counts:", dict(Counter(str(row["label"]) for row in self.supports)))
        print("Real defect images:", len(self.real_defects))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + int(index) * 1009)
        if self.real_defects and rng.random() < self.real_defect_prob:
            image, target = real_defect_sample(self.real_defects[rng.randrange(len(self.real_defects))], self.image_size)
            # Mild photometric jitter keeps the tiny real set from becoming a lookup table.
            gain = rng.uniform(0.85, 1.15)
            bias = rng.uniform(-14.0, 14.0)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            if rng.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                target = np.ascontiguousarray(target[:, :, ::-1])
            if rng.random() < 0.5:
                image = np.ascontiguousarray(image[::-1])
                target = np.ascontiguousarray(target[:, ::-1])
            img = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            tgt = torch.from_numpy(target)
            return img, tgt

        path = self.clean_images[rng.randrange(len(self.clean_images))]
        image = resize_keep_rgb(path, self.image_size)
        with Image.open(path) as img:
            raw_size = img.size
        raw_mask = load_product_mask(self.mask_dir, "train", path.name, raw_size)
        product_mask = resize_mask(raw_mask, self.image_size)
        target = np.zeros((len(LABELS), self.image_size, self.image_size), dtype=np.float32)

        if rng.random() >= self.normal_prob:
            num = rng.randint(self.min_defects, self.max_defects)
            for _ in range(num):
                label = choose_label(self.class_weights, self.supports_by_label, rng)
                use_proc = rng.random() < self.procedural_prob
                if label == "collision" and rng.random() < self.collision_real_prob and self.supports_by_label.get(label):
                    use_proc = False
                ok = False
                if not use_proc and self.supports_by_label.get(label):
                    support = rng.choice(self.supports_by_label[label])
                    ok = paste_real_defect(
                        image,
                        target,
                        product_mask,
                        support,
                        rng,
                        alpha_min=0.35 if label == "collision" else 0.25,
                        alpha_max=0.85,
                        color_match_strength=0.75,
                    )
                if not ok:
                    draw_procedural_defect(image, target, product_mask, label, rng)

        if rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            target = np.ascontiguousarray(target[:, :, ::-1])
        if rng.random() < 0.5:
            image = np.ascontiguousarray(image[::-1])
            target = np.ascontiguousarray(target[:, ::-1])

        img = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        tgt = torch.from_numpy(target)
        return img, tgt


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, out_channels: int = 4, base: int = 32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 8, base * 8)
        self.u4 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d4 = ConvBlock(base * 12, base * 4)
        self.u3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d3 = ConvBlock(base * 6, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d2 = ConvBlock(base * 3, base)
        self.u1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        m = self.mid(self.pool(e4))
        x = self.u4(m)
        x = self.d4(torch.cat([x, e4], dim=1))
        x = self.u3(x)
        x = self.d3(torch.cat([x, e3], dim=1))
        x = self.u2(x)
        x = self.d2(torch.cat([x, e2], dim=1))
        x = self.u1(x)
        x = self.d1(torch.cat([x, e1], dim=1))
        return self.out(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


def focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    weight = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1.0 - alpha))
    return (weight * (1.0 - pt).pow(gamma) * bce).mean()


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, step: int, loss_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "labels": LABELS,
            "image_size": int(args.image_size),
            "base_channels": int(args.base_channels),
            "step": int(step),
            "loss": float(loss_value),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RealNet/DRAEM-lite synthetic anomaly segmenter for LBB.")
    parser.add_argument("--clean-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--defect-dir", default="./初赛数据/训练集/负样本")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--out", default="./checkpoints/synth_anomaly_seg/best.pt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normal-prob", type=float, default=0.22)
    parser.add_argument("--min-defects", type=int, default=1)
    parser.add_argument("--max-defects", type=int, default=3)
    parser.add_argument("--class-weights", default="0.12,0.18,0.20,0.50")
    parser.add_argument("--procedural-prob", type=float, default=0.48)
    parser.add_argument("--collision-real-prob", type=float, default=0.72)
    parser.add_argument("--real-defect-prob", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=400)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    torch.backends.cudnn.benchmark = True

    weights = parse_class_weights(args.class_weights)
    dataset = SyntheticAnomalyDataset(
        clean_dir=Path(args.clean_dir),
        defect_dir=Path(args.defect_dir),
        mask_dir=Path(args.mask_dir),
        image_size=int(args.image_size),
        length=max(int(args.steps) * int(args.batch_size), 1024),
        seed=int(args.seed),
        normal_prob=float(args.normal_prob),
        min_defects=int(args.min_defects),
        max_defects=int(args.max_defects),
        class_weights=weights,
        procedural_prob=float(args.procedural_prob),
        collision_real_prob=float(args.collision_real_prob),
        real_defect_prob=float(args.real_defect_prob),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    model = TinyUNet(out_channels=len(LABELS), base=int(args.base_channels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.steps)), eta_min=float(args.lr) * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    out_path = Path(args.out).resolve()
    best_loss = float("inf")
    ema = None
    step = 0
    model.train()
    pbar = tqdm(total=int(args.steps), desc="train synth anomaly segmenter")
    while step < int(args.steps):
        for image, target in loader:
            step += 1
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(image)
                loss_focal = focal_bce_loss(logits, target)
                loss_dice = dice_loss(logits, target)
                loss = loss_focal + 0.65 * loss_dice
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            lv = float(loss.detach().item())
            ema = lv if ema is None else 0.95 * ema + 0.05 * lv
            if ema < best_loss:
                best_loss = ema
                save_checkpoint(out_path, model, args, step, best_loss)
            if step % int(args.save_interval) == 0:
                save_checkpoint(out_path.with_name(f"step{step}.pt"), model, args, step, ema)
            if step % int(args.log_interval) == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"step={step} loss={lv:.5f} ema={ema:.5f} best={best_loss:.5f} lr={lr:.3e}", flush=True)
            pbar.update(1)
            if step >= int(args.steps):
                break
    pbar.close()
    save_checkpoint(out_path.with_name("latest.pt"), model, args, step, ema if ema is not None else best_loss)
    print(f"Best checkpoint: {out_path}")
    print(f"Best EMA loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
