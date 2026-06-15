import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import auto_tune_loader_params


@dataclass
class Sample:
    image_path: str
    image_name: str
    ann_path: Optional[str]
    split_tag: str


def _polygon_to_xyxy(points: List[List[float]]) -> List[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_labelme_boxes(
    ann_path: str,
    label_to_id: Dict[str, int],
    min_box_size: float,
) -> Tuple[List[List[float]], List[int]]:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes: List[List[float]] = []
    labels: List[int] = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in label_to_id:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue

        box = _polygon_to_xyxy(points)
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w < min_box_size or h < min_box_size:
            continue

        boxes.append(box)
        labels.append(label_to_id[label])

    return boxes, labels


def build_samples(cfg: Dict, seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    pos_dir = Path(cfg["data"]["train_pos_dir"])
    neg_dir = Path(cfg["data"]["train_neg_dir"])
    test_dir = Path(cfg["data"]["test_dir"])
    val_ratio = float(cfg["data"].get("val_ratio", 0.2))

    all_train: List[Sample] = []

    for p in sorted(pos_dir.rglob("*.jpg")):
        all_train.append(Sample(str(p), p.name, None, "train"))

    for p in sorted(neg_dir.rglob("*.jpg")):
        ann = p.with_suffix(".json")
        ann_path = str(ann) if ann.exists() else None
        all_train.append(Sample(str(p), p.name, ann_path, "train"))

    rng = random.Random(seed)
    rng.shuffle(all_train)

    n_val = max(1, int(len(all_train) * val_ratio))
    val_samples = all_train[:n_val]
    train_samples = all_train[n_val:]

    test_samples: List[Sample] = []
    for p in sorted(test_dir.rglob("*.jpg")):
        test_samples.append(Sample(str(p), p.name, None, "test"))

    return train_samples, val_samples, test_samples


class DetectionAugmenter:
    def __init__(self, image_size: int, is_train: bool):
        self.image_size = int(image_size)
        self.is_train = is_train

    def _resize(self, image: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        ow, oh = image.size
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        if boxes.numel() > 0:
            scale_x = self.image_size / float(ow)
            scale_y = self.image_size / float(oh)
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return image, boxes

    def _flip_lr(self, image: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if boxes.numel() > 0:
            boxes = boxes.clone()
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = self.image_size - x2
            boxes[:, 2] = self.image_size - x1
        return image, boxes

    def _flip_ud(self, image: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if boxes.numel() > 0:
            boxes = boxes.clone()
            y1 = boxes[:, 1].clone()
            y2 = boxes[:, 3].clone()
            boxes[:, 1] = self.image_size - y2
            boxes[:, 3] = self.image_size - y1
        return image, boxes

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        enhancers = [
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Color,
            ImageEnhance.Sharpness,
        ]
        for enh in enhancers:
            if random.random() < 0.5:
                factor = random.uniform(0.75, 1.25)
                image = enh(image).enhance(factor)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
        return image

    def _random_cutout(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.3:
            return image
        arr = np.array(image).copy()
        h, w = arr.shape[:2]
        ch = random.randint(max(8, h // 40), max(12, h // 15))
        cw = random.randint(max(8, w // 40), max(12, w // 15))
        y0 = random.randint(0, max(0, h - ch))
        x0 = random.randint(0, max(0, w - cw))
        color = arr.mean(axis=(0, 1)).astype(np.uint8)
        arr[y0 : y0 + ch, x0 : x0 + cw] = color
        return Image.fromarray(arr)

    def __call__(self, image: Image.Image, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image, boxes = self._resize(image, boxes)

        if self.is_train:
            image = self._color_jitter(image)
            image = self._random_cutout(image)
            if random.random() < 0.5:
                image, boxes = self._flip_lr(image, boxes)
            if random.random() < 0.2:
                image, boxes = self._flip_ud(image, boxes)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        boxes = boxes.clamp(min=0.0, max=float(self.image_size))
        return image_tensor, boxes


class LBBDetectionDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        cfg: Dict,
        is_train: bool,
        logger=None,
    ):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train
        self.logger = logger

        self.image_size = int(cfg["data"]["image_size"])
        self.min_box_size = float(cfg["data"].get("min_box_size", 2.0))

        self.label_to_id = {
            k: int(v)
            for k, v in cfg["classes"].items()
            if k != "background"
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.augment = DetectionAugmenter(self.image_size, is_train=is_train)

        self.cache_enabled = bool(cfg["data"].get("cache_annotations", True))
        self.ann_cache: Dict[str, Tuple[List[List[float]], List[int]]] = {}

        self.mixup_prob = 0.25 if is_train else 0.0
        self.copy_paste_prob = 0.25 if is_train else 0.0

        self.sample_has_defect: List[bool] = []
        self.num_defect_samples = 0
        self.num_clean_samples = 0
        if self.is_train:
            self.sample_has_defect = self._precompute_sample_flags()
            self.num_defect_samples = int(sum(1 for x in self.sample_has_defect if x))
            self.num_clean_samples = int(len(self.sample_has_defect) - self.num_defect_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _precompute_sample_flags(self) -> List[bool]:
        flags: List[bool] = []
        for s in self.samples:
            if s.ann_path is None:
                flags.append(False)
                continue
            _, labels = self._read_ann(s.ann_path)
            flags.append(len(labels) > 0)
        return flags

    def get_sampling_weights(self, defect_weight: float, clean_weight: float) -> List[float]:
        if not self.is_train or not self.sample_has_defect:
            return [1.0] * len(self.samples)

        dw = max(float(defect_weight), 0.0)
        cw = max(float(clean_weight), 0.0)
        if dw == 0.0 and cw == 0.0:
            dw = 1.0
            cw = 1.0

        return [dw if has_defect else cw for has_defect in self.sample_has_defect]

    def _read_ann(self, ann_path: Optional[str]) -> Tuple[List[List[float]], List[int]]:
        if ann_path is None:
            return [], []

        if self.cache_enabled and ann_path in self.ann_cache:
            return self.ann_cache[ann_path]

        boxes, labels = load_labelme_boxes(
            ann_path,
            label_to_id=self.label_to_id,
            min_box_size=self.min_box_size,
        )

        if self.cache_enabled:
            self.ann_cache[ann_path] = (boxes, labels)

        return boxes, labels

    def _load_image_target(self, idx: int) -> Tuple[Image.Image, List[List[float]], List[int], str]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        boxes, labels = self._read_ann(sample.ann_path)
        return image, boxes, labels, sample.image_name

    def _copy_paste(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        labels: List[int],
    ) -> Tuple[Image.Image, List[List[float]], List[int]]:
        if random.random() > self.copy_paste_prob:
            return image, boxes, labels

        donor_idx = random.randint(0, len(self.samples) - 1)
        donor_img, donor_boxes, donor_labels, _ = self._load_image_target(donor_idx)
        if len(donor_boxes) == 0:
            return image, boxes, labels

        ridx = random.randrange(len(donor_boxes))
        src_box = donor_boxes[ridx]
        src_label = donor_labels[ridx]

        x1, y1, x2, y2 = [int(v) for v in src_box]
        if x2 <= x1 or y2 <= y1:
            return image, boxes, labels

        patch = donor_img.crop((x1, y1, x2, y2))
        bw, bh = image.size
        pw, ph = patch.size
        if pw < 2 or ph < 2 or pw >= bw or ph >= bh:
            return image, boxes, labels

        tx = random.randint(0, max(0, bw - pw))
        ty = random.randint(0, max(0, bh - ph))

        base = np.array(image).copy()
        patch_arr = np.array(patch)
        base[ty : ty + ph, tx : tx + pw] = patch_arr
        image_new = Image.fromarray(base)

        boxes_new = list(boxes)
        labels_new = list(labels)
        boxes_new.append([float(tx), float(ty), float(tx + pw), float(ty + ph)])
        labels_new.append(int(src_label))

        return image_new, boxes_new, labels_new

    def _mixup(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        labels: List[int],
    ) -> Tuple[Image.Image, List[List[float]], List[int]]:
        if random.random() > self.mixup_prob:
            return image, boxes, labels

        mix_idx = random.randint(0, len(self.samples) - 1)
        mix_img, mix_boxes, mix_labels, _ = self._load_image_target(mix_idx)

        if image.size != mix_img.size:
            ow, oh = mix_img.size
            nw, nh = image.size
            sx = nw / float(ow)
            sy = nh / float(oh)
            mix_img = mix_img.resize((nw, nh), resample=Image.BILINEAR)
            scaled = []
            for b in mix_boxes:
                scaled.append([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
            mix_boxes = scaled

        a = random.uniform(0.35, 0.65)
        img1 = np.asarray(image, dtype=np.float32)
        img2 = np.asarray(mix_img, dtype=np.float32)
        blend = (a * img1 + (1.0 - a) * img2).clip(0, 255).astype(np.uint8)

        boxes_out = list(boxes) + list(mix_boxes)
        labels_out = list(labels) + list(mix_labels)

        return Image.fromarray(blend), boxes_out, labels_out

    def __getitem__(self, idx: int):
        image, boxes, labels, image_name = self._load_image_target(idx)

        if self.is_train:
            image, boxes, labels = self._copy_paste(image, boxes, labels)
            image, boxes, labels = self._mixup(image, boxes, labels)

        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        image_t, boxes_t = self.augment(image, boxes_t)

        if boxes_t.numel() == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        iscrowd = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "image_name": image_name,
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
        }

        return image_t, target


class LBBTestDataset(Dataset):
    def __init__(self, samples: List[Sample], cfg: Dict):
        self.samples = samples
        self.image_size = int(cfg["data"]["image_size"])
        self.augment = DetectionAugmenter(self.image_size, is_train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        h0, w0 = image.height, image.width
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        image_t, _ = self.augment(image, boxes)

        meta = {
            "image_name": sample.image_name,
            "orig_height": h0,
            "orig_width": w0,
        }
        return image_t, meta


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def collate_test_fn(batch):
    images = [b[0] for b in batch]
    metas = [b[1] for b in batch]
    return images, metas


def build_dataloaders(cfg: Dict, hw, logger=None):
    seed = int(cfg.get("seed", 42))
    train_samples, val_samples, _ = build_samples(cfg, seed=seed)

    train_ds = LBBDetectionDataset(train_samples, cfg=cfg, is_train=True, logger=logger)
    val_ds = LBBDetectionDataset(val_samples, cfg=cfg, is_train=False, logger=logger)

    batch_size, num_workers = auto_tune_loader_params(cfg, hw)

    sampling_cfg = cfg.get("sampling", {})
    sampler = None
    shuffle = True
    if bool(sampling_cfg.get("enabled", False)):
        defect_weight = float(sampling_cfg.get("defect_weight", 2.0))
        clean_weight = float(sampling_cfg.get("clean_weight", 1.0))
        replacement = bool(sampling_cfg.get("replacement", True))
        num_samples = int(sampling_cfg.get("num_samples_per_epoch", len(train_ds)))

        sample_weights = train_ds.get_sampling_weights(defect_weight, clean_weight)
        weights_t = torch.tensor(sample_weights, dtype=torch.double)
        if float(weights_t.sum().item()) <= 0.0:
            weights_t = torch.ones_like(weights_t)

        sampler = WeightedRandomSampler(weights_t, num_samples=num_samples, replacement=replacement)
        shuffle = False

    if logger is not None:
        logger.info(
            "Data split: train=%d val=%d | batch=%d workers=%d | defect_imgs=%d clean_imgs=%d",
            len(train_ds),
            len(val_ds),
            batch_size,
            num_workers,
            train_ds.num_defect_samples,
            train_ds.num_clean_samples,
        )
        if sampler is not None:
            logger.info(
                "Sampling enabled: defect_weight=%.3f clean_weight=%.3f num_samples=%d replacement=%s",
                defect_weight,
                clean_weight,
                num_samples,
                replacement,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(hw.device == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=(hw.device == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, train_ds, val_ds


def build_test_loader(cfg: Dict, hw, logger=None):
    seed = int(cfg.get("seed", 42))
    _, _, test_samples = build_samples(cfg, seed=seed)
    ds = LBBTestDataset(test_samples, cfg=cfg)

    batch_size = max(1, min(8, int(cfg["training"].get("batch_size", 4))))
    workers = max(1, min((os.cpu_count() or 4) // 2, 8))

    if logger is not None:
        logger.info("Test set size=%d | batch=%d workers=%d", len(ds), batch_size, workers)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(hw.device == "cuda"),
        collate_fn=collate_test_fn,
    )

    return loader, ds
