#!/usr/bin/env python3
"""Duplicate scratch boxes as collision only when DINOv3 favors collision.

This is a targeted second-stage filter for the best cross-label tail:
many true collision defects are localized by scratch boxes, but copying all
scratch boxes creates too many collision false positives. DINOv3 support
prototypes provide a weak but useful semantic margin:

    sim(crop, collision_proto) - sim(crop, scratch_proto)

Only candidates above the requested margin are duplicated as low-score
collision annotations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


VALID_LABELS = ["plain particle", "dirt", "scratch", "collision"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_box(box: Sequence[float], width: Optional[int] = None, height: Optional[int] = None) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if width is not None:
        x1 = max(0.0, min(float(width - 1), x1))
        x2 = max(0.0, min(float(width - 1), x2))
    if height is not None:
        y1 = max(0.0, min(float(height - 1), y1))
        y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: Sequence[float]) -> float:
    norm = normalize_box(box)
    if norm is None:
        return 0.0
    return max(0.0, norm[2] - norm[0]) * max(0.0, norm[3] - norm[1])


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_box(a)
    bb = normalize_box(b)
    if aa is None or bb is None:
        return 0.0
    x1 = max(aa[0], bb[0])
    y1 = max(aa[1], bb[1])
    x2 = min(aa[2], bb[2])
    y2 = min(aa[3], bb[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(aa) + box_area(bb) - inter
    return float(inter / union) if union > 0 else 0.0


def expand_box(box: Sequence[float], width: int, height: int, factor: float) -> List[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * float(factor)
    bh = (y2 - y1) * float(factor)
    return [
        max(0.0, cx - bw * 0.5),
        max(0.0, cy - bh * 0.5),
        min(float(width - 1), cx + bw * 0.5),
        min(float(height - 1), cy + bh * 0.5),
    ]


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.zf: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, str] = {}
        if path.suffix.lower() == ".zip":
            self.zf = zipfile.ZipFile(path, "r")
            self.members = {Path(n).name: n for n in self.zf.namelist() if n.endswith(".json")}

    def names(self) -> List[str]:
        if self.zf is not None:
            return sorted(self.members)
        return sorted(p.name for p in self.path.glob("*.json"))

    def read(self, name: str) -> Dict:
        if self.zf is not None:
            return json.loads(self.zf.read(self.members[name]).decode("utf-8"))
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def find_images(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def build_dinov3(repo_dir: Path, model_name: str, weights: Path, device: torch.device):
    repo = str(repo_dir.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    backbones = importlib.import_module("dinov3.hub.backbones")
    if not hasattr(backbones, model_name):
        raise ValueError(f"DINOv3 model not found: {model_name}")
    model = getattr(backbones, model_name)(
        pretrained=True,
        weights=str(weights.resolve()),
        check_hash=False,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_support_boxes(root: Path) -> List[Tuple[Path, str, List[float]]]:
    rows: List[Tuple[Path, str, List[float]]] = []
    for ann_path in root.rglob("*.json"):
        image_path = ann_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            pts = shape.get("points", [])
            if label not in VALID_LABELS or len(pts) < 2:
                continue
            try:
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
            except Exception:
                continue
            rows.append((image_path, label, [min(xs), min(ys), max(xs), max(ys)]))
    return rows


def crop_to_tensor(image: Image.Image, box: Sequence[float], input_size: int, expand: float) -> Optional[torch.Tensor]:
    width, height = image.size
    norm = normalize_box(box, width, height)
    if norm is None:
        return None
    norm = expand_box(norm, width, height, expand)
    norm = normalize_box(norm, width, height)
    if norm is None:
        return None
    patch = image.crop(tuple(norm)).resize((int(input_size), int(input_size)), Image.BILINEAR)
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def embed_patches(model, patches: List[torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
    if not patches:
        return torch.empty((0, 0), dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    feats = []
    for start in range(0, len(patches), int(batch_size)):
        x = torch.stack(patches[start : start + int(batch_size)]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model((x - mean) / std, is_training=True)
        feat = out["x_norm_clstoken"] if isinstance(out, dict) else out
        feats.append(F.normalize(feat.float(), dim=-1).detach().cpu())
    return torch.cat(feats, dim=0)


def build_prototypes(model, support_rows: List[Tuple[Path, str, List[float]]], args, device: torch.device) -> Dict[str, torch.Tensor]:
    by_image: Dict[Path, List[Tuple[str, List[float]]]] = defaultdict(list)
    for image_path, label, box in support_rows:
        by_image[image_path].append((label, box))

    feats_by_label: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for image_path, items in tqdm(sorted(by_image.items()), desc="DINO support"):
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            patches = []
            labels = []
            for label, box in items:
                patch = crop_to_tensor(im, box, int(args.input_size), float(args.crop_expand))
                if patch is not None:
                    patches.append(patch)
                    labels.append(label)
        feats = embed_patches(model, patches, device, int(args.batch_size))
        for feat, label in zip(feats, labels):
            feats_by_label[label].append(feat)

    prototypes: Dict[str, torch.Tensor] = {}
    for label, feats in feats_by_label.items():
        prototypes[label] = F.normalize(torch.stack(feats).mean(dim=0), dim=0)
    return prototypes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--support-dir", default="./初赛数据/训练集/负样本")
    parser.add_argument("--repo-dir", default="./third_party/dinov3")
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--weights", default="./dinov3/dinov3_vitl16_from_safetensors-8aa4cbdd.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-expand", type=float, default=1.35)
    parser.add_argument("--source-label", default="scratch")
    parser.add_argument("--target-label", default="collision")
    parser.add_argument("--negative-label", default="scratch")
    parser.add_argument("--topk-per-image", type=int, default=150)
    parser.add_argument("--max-added-per-image", type=int, default=150)
    parser.add_argument("--min-area", type=float, default=250.0)
    parser.add_argument("--max-area", type=float, default=20000.0)
    parser.add_argument("--dedup-iou", type=float, default=0.70)
    parser.add_argument("--self-dedup-iou", type=float, default=0.92)
    parser.add_argument("--margin-thr", type=float, default=0.0)
    parser.add_argument("--score-factor", type=float, default=0.0002)
    parser.add_argument("--score-cap", type=float, default=0.0005)
    parser.add_argument("--score-floor", type=float, default=0.0)
    parser.add_argument("--margin-score-alpha", type=float, default=0.0)
    parser.add_argument(
        "--score-margin-center",
        type=float,
        default=None,
        help="Margin center used for score boosting. Defaults to --margin-thr to preserve old behavior.",
    )
    parser.add_argument("--score-decimals", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pred_path = Path(args.pred).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = build_dinov3(Path(args.repo_dir), str(args.model_name), Path(args.weights), device)
    support_rows = load_support_boxes(Path(args.support_dir).resolve())
    prototypes = build_prototypes(model, support_rows, args, device)
    if args.target_label not in prototypes or args.negative_label not in prototypes:
        raise RuntimeError(f"Missing prototypes: {args.target_label}/{args.negative_label}")
    target_proto = prototypes[str(args.target_label)]
    negative_proto = prototypes[str(args.negative_label)]
    print("Support boxes:", len(support_rows))
    print("Prototypes:", sorted(prototypes))

    images = find_images(Path(args.test_image_dir).resolve())
    store = Store(pred_path)
    names = store.names()
    if int(args.limit) > 0:
        names = names[: int(args.limit)]

    total_added = 0
    considered = 0
    dino_kept = 0
    skipped_existing = 0
    skipped_self = 0
    missing_images = 0
    margin_values: List[float] = []

    try:
        for name in tqdm(names, desc="DINO cross-label collision"):
            payload = store.read(name)
            anns = [dict(x) for x in payload.get("annotations", []) if isinstance(x, dict)]
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = images.get(image_name)
            if image_path is None:
                missing_images += 1
                payload["annotations"] = anns
                (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            target_boxes = [
                normalize_box(a.get("bbox", []))
                for a in anns
                if str(a.get("label", "")) == str(args.target_label)
            ]
            target_boxes = [b for b in target_boxes if b is not None]

            candidates = []
            for ann in anns:
                if str(ann.get("label", "")) != str(args.source_label):
                    continue
                bbox = normalize_box(ann.get("bbox", []))
                if bbox is None:
                    continue
                area = box_area(bbox)
                if area < float(args.min_area) or area > float(args.max_area):
                    continue
                score = safe_float(ann.get("confidence", 0.0), 0.0)
                candidates.append((score, bbox, ann))
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[: max(0, int(args.topk_per_image))]
            considered += len(candidates)

            patches: List[torch.Tensor] = []
            kept_meta = []
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                for score, bbox, ann in candidates:
                    if any(bbox_iou(bbox, existing) >= float(args.dedup_iou) for existing in target_boxes):
                        skipped_existing += 1
                        continue
                    patch = crop_to_tensor(im, bbox, int(args.input_size), float(args.crop_expand))
                    if patch is None:
                        continue
                    patches.append(patch)
                    kept_meta.append((score, bbox, ann))

            feats = embed_patches(model, patches, device, int(args.batch_size))
            appended: List[List[float]] = []
            for feat, (score, bbox, ann) in zip(feats, kept_meta):
                margin = float((feat * target_proto).sum().item() - (feat * negative_proto).sum().item())
                margin_values.append(margin)
                if margin < float(args.margin_thr):
                    continue
                dino_kept += 1
                if len(appended) >= int(args.max_added_per_image):
                    break
                if any(bbox_iou(bbox, existing) >= float(args.self_dedup_iou) for existing in appended):
                    skipped_self += 1
                    continue
                score_boost = 1.0
                if float(args.margin_score_alpha) != 0.0:
                    score_center = (
                        float(args.margin_thr)
                        if args.score_margin_center is None
                        else float(args.score_margin_center)
                    )
                    score_boost += float(args.margin_score_alpha) * max(0.0, margin - score_center)
                new_score = score * float(args.score_factor) * score_boost
                new_score = min(float(args.score_cap), max(float(args.score_floor), new_score))
                if new_score <= 0:
                    continue
                new_ann = dict(ann)
                new_ann["label"] = str(args.target_label)
                new_ann["bbox"] = [round(float(v), 4) for v in bbox]
                new_ann["confidence"] = round(float(new_score), int(args.score_decimals))
                anns.append(new_ann)
                appended.append(bbox)
                target_boxes.append(bbox)
                total_added += 1

            anns.sort(key=lambda a: safe_float(a.get("confidence", 0.0), 0.0), reverse=True)
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Input: {pred_path}")
    print(f"Output zip: {out_zip}")
    print(f"Missing images: {missing_images}")
    print(f"Considered/source candidates: {considered}")
    print(f"DINO kept before per-image cap: {dino_kept}")
    print(f"Added: {total_added}")
    print(f"Skipped existing/self: {skipped_existing}/{skipped_self}")
    if margin_values:
        arr = np.asarray(margin_values, dtype=np.float32)
        print(
            "Margin quantiles 1/5/10/25/50/75/90/95/99:",
            [round(float(x), 5) for x in np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])],
        )


if __name__ == "__main__":
    main()
