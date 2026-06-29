import argparse
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm


VALID_LABELS = ["plain particle", "dirt", "scratch", "collision"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def parse_box_from_points(points: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_box(box: Iterable[float], width: int, height: int) -> Optional[List[float]]:
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


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def expand_box(box: Sequence[float], width: int, height: int, factor: float, pad: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, (x2 - x1) * float(factor) + 2.0 * float(pad))
    bh = max(1.0, (y2 - y1) * float(factor) + 2.0 * float(pad))
    return [
        max(0.0, cx - bw * 0.5),
        max(0.0, cy - bh * 0.5),
        min(float(width - 1), cx + bw * 0.5),
        min(float(height - 1), cy + bh * 0.5),
    ]


def load_support_boxes(defect_dir: Path) -> List[Tuple[Path, str, List[float]]]:
    rows: List[Tuple[Path, str, List[float]]] = []
    for ann_path in sorted(defect_dir.rglob("*.json")):
        image_path = ann_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        try:
            data = load_json(ann_path)
        except Exception:
            continue

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in VALID_LABELS:
                continue
            box = parse_box_from_points(shape.get("points", []))
            if box is not None:
                rows.append((image_path, str(label), box))

        for ann in data.get("annotations", []):
            label = ann.get("label")
            bbox = ann.get("bbox", [])
            if label in VALID_LABELS and isinstance(bbox, list) and len(bbox) == 4:
                rows.append((image_path, str(label), [float(v) for v in bbox]))
    return rows


class ResNetPatchFeatures(nn.Module):
    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        backbone = str(backbone).lower()
        if backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            net = torchvision.models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            net = torchvision.models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.out_channels = (
            (128 + 256) if backbone == "resnet18" else (512 + 1024)
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3], dim=1)
        return F.normalize(feat.float(), dim=1)


def image_to_tensor(path: Path, image_size: int) -> Tuple[torch.Tensor, int, int]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        img = img.resize((int(image_size), int(image_size)), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, width, height


def load_mask(mask_dir: Path, split: str, image_name: str, width: int, height: int) -> Optional[np.ndarray]:
    if not mask_dir:
        return None
    path = mask_dir / split / f"{Path(image_name).stem}.png"
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            if img.size != (width, height):
                img = img.resize((width, height), Image.NEAREST)
            arr = np.asarray(img, dtype=np.uint8)
            return arr > 127
    except Exception:
        return None


def resize_mask_to_grid(mask: Optional[np.ndarray], grid_hw: Tuple[int, int]) -> np.ndarray:
    gh, gw = grid_hw
    if mask is None:
        return np.ones((gh, gw), dtype=bool)
    mask_u8 = mask.astype(np.uint8) * 255
    small = cv2.resize(mask_u8, (gw, gh), interpolation=cv2.INTER_NEAREST)
    out = small > 127
    if out.mean() < 0.03:
        out[:] = True
    # A one-cell erosion avoids many notebook-edge responses while keeping small defects.
    if out.mean() > 0.10:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(out.astype(np.uint8), kernel, iterations=1).astype(bool)
        if eroded.mean() > 0.03:
            out = eroded
    return out


@torch.no_grad()
def extract_feature_map(
    extractor: ResNetPatchFeatures,
    image_path: Path,
    image_size: int,
    device: torch.device,
    channel_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, int]:
    tensor, width, height = image_to_tensor(image_path, image_size)
    x = tensor.unsqueeze(0).to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        feat = extractor(x)[0]
    if channel_idx is not None:
        feat = feat.index_select(0, channel_idx.to(feat.device))
        feat = F.normalize(feat.float(), dim=0)
    return feat.detach(), width, height


def choose_channels(total_channels: int, feature_dim: int, seed: int) -> torch.Tensor:
    feature_dim = min(int(feature_dim), int(total_channels))
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(total_channels, size=feature_dim, replace=False))
    return torch.tensor(idx, dtype=torch.long)


def sample_rows(rows: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if rows.shape[0] <= int(max_rows):
        return rows
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(rows.shape[0], generator=gen)[: int(max_rows)]
    return rows[perm]


def build_memory_bank(args, extractor: ResNetPatchFeatures, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_path = Path(args.memory_cache) if args.memory_cache else None
    if cache_path and cache_path.exists() and not args.rebuild_memory:
        cached = torch.load(cache_path, map_location="cpu")
        memory = cached["memory"].float()
        channel_idx = cached["channel_idx"].long()
        print(f"Loaded memory cache: {cache_path} | memory={tuple(memory.shape)}")
        return memory, channel_idx

    normal_images = find_images(Path(args.normal_dir))
    if not normal_images:
        raise FileNotFoundError(f"No normal images found in {args.normal_dir}")

    with torch.no_grad():
        first_feat, _, _ = extract_feature_map(extractor, normal_images[0], int(args.image_size), device)
    channel_idx = choose_channels(first_feat.shape[0], int(args.feature_dim), int(args.seed))

    all_rows: List[torch.Tensor] = []
    rng = np.random.default_rng(int(args.seed) + 17)
    for image_path in tqdm(normal_images, desc="PatchCore normal memory"):
        feat, width, height = extract_feature_map(
            extractor, image_path, int(args.image_size), device, channel_idx=channel_idx
        )
        gh, gw = feat.shape[-2:]
        mask = load_mask(Path(args.mask_dir), "train", image_path.name, width, height)
        mask_grid = resize_mask_to_grid(mask, (gh, gw))
        rows = feat.permute(1, 2, 0).reshape(-1, feat.shape[0]).detach().cpu()
        keep = torch.from_numpy(mask_grid.reshape(-1))
        rows = rows[keep]
        rows = F.normalize(rows.float(), dim=1)
        if rows.numel() == 0:
            continue
        per_image = int(args.normal_patches_per_image)
        if rows.shape[0] > per_image:
            idx = rng.choice(rows.shape[0], size=per_image, replace=False)
            rows = rows[torch.from_numpy(idx).long()]
        all_rows.append(rows)

    if not all_rows:
        raise RuntimeError("Normal memory is empty after mask filtering.")

    memory = torch.cat(all_rows, dim=0)
    memory = sample_rows(memory, int(args.max_memory_patches), int(args.seed) + 31)
    memory = F.normalize(memory.float(), dim=1).contiguous()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"memory": memory.cpu(), "channel_idx": channel_idx.cpu()}, cache_path)
        print(f"Saved memory cache: {cache_path} | memory={tuple(memory.shape)}")
    return memory, channel_idx


def grid_box_from_original_box(box: Sequence[float], width: int, height: int, gw: int, gh: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    gx1 = int(math.floor(x1 / max(float(width), 1.0) * gw))
    gy1 = int(math.floor(y1 / max(float(height), 1.0) * gh))
    gx2 = int(math.ceil(x2 / max(float(width), 1.0) * gw))
    gy2 = int(math.ceil(y2 / max(float(height), 1.0) * gh))
    gx1 = max(0, min(gw - 1, gx1))
    gy1 = max(0, min(gh - 1, gy1))
    gx2 = max(gx1 + 1, min(gw, gx2))
    gy2 = max(gy1 + 1, min(gh, gy2))
    return gx1, gy1, gx2, gy2


def build_class_prototypes(
    args,
    extractor: ResNetPatchFeatures,
    device: torch.device,
    channel_idx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    support_rows = load_support_boxes(Path(args.defect_dir))
    if not support_rows:
        raise FileNotFoundError(f"No labeled defect boxes found in {args.defect_dir}")

    by_image: Dict[Path, List[Tuple[str, List[float]]]] = defaultdict(list)
    for image_path, label, box in support_rows:
        by_image[image_path].append((label, box))

    feats_by_label: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for image_path, items in tqdm(sorted(by_image.items()), desc="Class prototypes"):
        feat, width, height = extract_feature_map(
            extractor, image_path, int(args.image_size), device, channel_idx=channel_idx
        )
        c, gh, gw = feat.shape
        flat = feat.permute(1, 2, 0).reshape(-1, c)
        for label, raw_box in items:
            box = normalize_box(raw_box, width, height)
            if box is None:
                continue
            gx1, gy1, gx2, gy2 = grid_box_from_original_box(box, width, height, gw, gh)
            patch = feat[:, gy1:gy2, gx1:gx2].reshape(c, -1).T
            if patch.numel() == 0:
                cx = min(gw - 1, max(0, (gx1 + gx2) // 2))
                cy = min(gh - 1, max(0, (gy1 + gy2) // 2))
                patch = flat[cy * gw + cx : cy * gw + cx + 1]
            vec = F.normalize(patch.float(), dim=1).mean(dim=0)
            feats_by_label[label].append(F.normalize(vec, dim=0).detach().cpu())

    prototypes: Dict[str, torch.Tensor] = {}
    for label, feats in feats_by_label.items():
        proto = torch.stack(feats, dim=0).mean(dim=0)
        prototypes[label] = F.normalize(proto.float(), dim=0)
    print("Prototype support counts:", {k: len(v) for k, v in feats_by_label.items()})
    return prototypes


def classify_candidate(
    vec: torch.Tensor,
    prototypes: Dict[str, torch.Tensor],
    avoid_labels: Sequence[str],
) -> Tuple[str, float]:
    avoid = set(avoid_labels)
    labels = []
    protos = []
    for label in VALID_LABELS:
        if label not in prototypes or label in avoid:
            continue
        labels.append(label)
        protos.append(prototypes[label])
    if not labels:
        labels = [label for label in VALID_LABELS if label in prototypes]
        protos = [prototypes[label] for label in labels]
    if not labels:
        return "scratch", 0.0
    mat = torch.stack(protos, dim=0).to(vec.device)
    sims = torch.mv(mat, F.normalize(vec.float(), dim=0))
    best = int(torch.argmax(sims).item())
    return labels[best], float(sims[best].item())


def nms_candidates(cands: List[Dict], iou_thr: float, class_aware: bool = True) -> List[Dict]:
    cands = sorted(cands, key=lambda x: float(x["confidence"]), reverse=True)
    kept: List[Dict] = []
    for cand in cands:
        duplicate = False
        for old in kept:
            if class_aware and cand["label"] != old["label"]:
                continue
            if box_iou(cand["bbox"], old["bbox"]) >= float(iou_thr):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


@torch.no_grad()
def anomaly_heatmap(feat: torch.Tensor, memory_t: torch.Tensor, chunk_size: int) -> torch.Tensor:
    c, gh, gw = feat.shape
    rows = feat.permute(1, 2, 0).reshape(-1, c).float()
    rows = F.normalize(rows, dim=1)
    scores = []
    for start in range(0, rows.shape[0], int(chunk_size)):
        part = rows[start : start + int(chunk_size)]
        sim = torch.mm(part, memory_t)
        best_sim = sim.max(dim=1).values
        scores.append((1.0 - best_sim).detach())
    return torch.cat(scores, dim=0).reshape(gh, gw)


def normalize_heat(heat: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, float, float]:
    vals = heat[valid]
    if vals.size == 0:
        return np.zeros_like(heat, dtype=np.float32), 0.0, 1.0
    lo = float(np.percentile(vals, 50.0))
    hi = float(np.percentile(vals, 99.8))
    if hi <= lo + 1e-9:
        hi = float(vals.max() + 1e-6)
    norm = np.clip((heat - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    return norm, lo, hi


def extract_candidates_from_heatmap(
    args,
    image_path: Path,
    feat: torch.Tensor,
    heat_t: torch.Tensor,
    width: int,
    height: int,
    mask: Optional[np.ndarray],
    prototypes: Dict[str, torch.Tensor],
) -> List[Dict]:
    c, gh, gw = feat.shape
    mask_grid = resize_mask_to_grid(mask, (gh, gw))
    heat = heat_t.detach().float().cpu().numpy()
    heat[~mask_grid] = float(np.min(heat[mask_grid])) if mask_grid.any() else 0.0
    norm, raw_lo, raw_hi = normalize_heat(heat, mask_grid)
    valid_vals = heat[mask_grid]
    if valid_vals.size == 0:
        return []

    # Slight smoothing stabilizes connected components without blurring boxes too much.
    smooth = cv2.GaussianBlur(norm, (3, 3), 0)
    raw_smooth = cv2.GaussianBlur(heat.astype(np.float32), (3, 3), 0)
    feature_rows = feat.permute(1, 2, 0).reshape(-1, c).detach().cpu()

    percentiles = [float(x) for x in str(args.percentiles).split(",") if str(x).strip()]
    candidates: List[Dict] = []
    product_area = float(mask.sum()) if mask is not None and mask.any() else float(width * height)
    max_area = max(float(args.max_area), product_area * float(args.max_area_frac))

    for pct in percentiles:
        thr = float(np.percentile(valid_vals, pct))
        binary = ((heat >= thr) & mask_grid).astype(np.uint8)
        if int(args.dilate) > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=int(args.dilate))
            binary = (binary.astype(bool) & mask_grid).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for idx in range(1, num):
            x, y, bw, bh, area_cells = stats[idx].tolist()
            if int(area_cells) < int(args.min_cells):
                continue

            comp = labels == idx
            comp_scores = raw_smooth[comp]
            if comp_scores.size == 0:
                continue
            score_raw = float(comp_scores.max())
            score_norm = float(np.clip((score_raw - raw_lo) / max(raw_hi - raw_lo, 1e-9), 0.0, 1.0))

            x1 = x / float(gw) * float(width)
            y1 = y / float(gh) * float(height)
            x2 = (x + bw) / float(gw) * float(width)
            y2 = (y + bh) / float(gh) * float(height)
            box = normalize_box(
                expand_box([x1, y1, x2, y2], width, height, float(args.box_expand), float(args.box_pad)),
                width,
                height,
            )
            if box is None:
                continue

            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            area = box_w * box_h
            if box_w < float(args.min_box_side) or box_h < float(args.min_box_side):
                continue
            if area < float(args.min_area) or area > max_area:
                continue

            comp_flat = torch.from_numpy(comp.reshape(-1))
            vecs = feature_rows[comp_flat]
            if vecs.numel() == 0:
                continue
            vec = F.normalize(vecs.float(), dim=1).mean(dim=0)
            label, cls_sim = classify_candidate(vec, prototypes, args.avoid_label or [])
            if label in set(args.avoid_label or []):
                continue

            confidence = float(args.score_floor) + score_norm * (float(args.score_ceiling) - float(args.score_floor))
            confidence = confidence * (1.0 + max(-0.5, min(0.5, cls_sim)) * float(args.class_score_weight))
            confidence = max(float(args.score_floor), min(float(args.score_ceiling), confidence))

            candidates.append(
                {
                    "label": label,
                    "bbox": [round(float(v), 3) for v in box],
                    "confidence": round(float(confidence), int(args.score_decimals)),
                    "_raw_score": score_raw,
                    "_pct": pct,
                    "_cls_sim": cls_sim,
                }
            )

    candidates = nms_candidates(candidates, float(args.nms_iou), class_aware=True)
    candidates = nms_candidates(candidates, float(args.all_label_nms_iou), class_aware=False)
    candidates = sorted(candidates, key=lambda x: float(x["confidence"]), reverse=True)
    max_boxes = int(args.max_boxes_per_image)
    if max_boxes > 0:
        candidates = candidates[:max_boxes]
    for cand in candidates:
        cand.pop("_raw_score", None)
        cand.pop("_pct", None)
        cand.pop("_cls_sim", None)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore-style normal-memory anomaly proposals for LBB.")
    parser.add_argument("--normal-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--defect-dir", default="./初赛数据/训练集/负样本")
    parser.add_argument("--test-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--memory-cache", default="./outputs/patchcore_resnet50_memory.pt")
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--normal-patches-per-image", type=int, default=700)
    parser.add_argument("--max-memory-patches", type=int, default=12000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--percentiles", default="99.6,99.2,98.8")
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--box-expand", type=float, default=1.85)
    parser.add_argument("--box-pad", type=float, default=3.0)
    parser.add_argument("--min-box-side", type=float, default=6.0)
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument("--max-area", type=float, default=60000.0)
    parser.add_argument("--max-area-frac", type=float, default=0.080)
    parser.add_argument("--max-boxes-per-image", type=int, default=24)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--all-label-nms-iou", type=float, default=0.60)
    parser.add_argument("--avoid-label", action="append", default=["plain particle"])
    parser.add_argument("--score-floor", type=float, default=0.05)
    parser.add_argument("--score-ceiling", type=float, default=1.0)
    parser.add_argument("--class-score-weight", type=float, default=0.10)
    parser.add_argument("--score-decimals", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process first N test images.")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    torch.backends.cudnn.benchmark = True
    extractor = ResNetPatchFeatures(args.backbone).eval().to(device)
    for p in extractor.parameters():
        p.requires_grad = False

    memory, channel_idx = build_memory_bank(args, extractor, device)
    memory_t = memory.to(device, non_blocking=True).T.contiguous()
    prototypes = build_class_prototypes(args, extractor, device, channel_idx)
    prototypes = {k: v.float().cpu() for k, v in prototypes.items()}

    test_images = find_images(Path(args.test_dir))
    if not test_images:
        raise FileNotFoundError(f"No test images found in {args.test_dir}")
    if int(args.limit) > 0:
        test_images = test_images[: int(args.limit)]

    total = 0
    empty = 0
    label_counts: Counter = Counter()
    for image_path in tqdm(test_images, desc="PatchCore test proposals"):
        feat, width, height = extract_feature_map(
            extractor, image_path, int(args.image_size), device, channel_idx=channel_idx
        )
        mask = load_mask(Path(args.mask_dir), "test", image_path.name, width, height)
        heat = anomaly_heatmap(feat, memory_t, int(args.chunk_size))
        candidates = extract_candidates_from_heatmap(args, image_path, feat, heat, width, height, mask, prototypes)
        total += len(candidates)
        empty += int(len(candidates) == 0)
        label_counts.update(c["label"] for c in candidates)
        payload = {"image_id": image_path.name, "annotations": candidates}
        write_json(out_dir / f"{image_path.stem}.json", payload)

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"Output dir: {out_dir}")
    print(f"Images: {len(test_images)} empty={empty} annotations={total}")
    print(f"Labels: {dict(label_counts)}")


if __name__ == "__main__":
    main()
