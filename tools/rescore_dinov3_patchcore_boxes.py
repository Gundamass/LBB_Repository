import argparse
import importlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from patchcore_anomaly_proposals import (  # noqa: E402
    expand_box,
    grid_box_from_original_box,
    load_mask,
    normalize_box,
    normalize_heat,
    resize_mask_to_grid,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def find_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_map(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in find_images(root)}


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
    for param in model.parameters():
        param.requires_grad = False
    return model


def image_to_tensor(path: Path, image_size: int) -> Tuple[torch.Tensor, int, int]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        img = img.resize((int(image_size), int(image_size)), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), width, height


def choose_channels(total_channels: int, feature_dim: int, seed: int) -> torch.Tensor:
    feature_dim = min(int(feature_dim), int(total_channels))
    if feature_dim == total_channels:
        return torch.arange(total_channels, dtype=torch.long)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(total_channels, size=feature_dim, replace=False))
    return torch.tensor(idx, dtype=torch.long)


@torch.no_grad()
def extract_dino_feature_map(
    model,
    image_path: Path,
    image_size: int,
    device: torch.device,
    channel_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, int]:
    tensor, width, height = image_to_tensor(image_path, image_size)
    x = tensor.unsqueeze(0).to(device, non_blocking=True)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        out = model((x - mean) / std, is_training=True)
    if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
        raise RuntimeError("DINOv3 output does not contain x_norm_patchtokens.")
    patch = out["x_norm_patchtokens"][0].float()
    grid = int(round(float(patch.shape[0]) ** 0.5))
    if grid * grid != patch.shape[0]:
        raise RuntimeError(f"Patch token count is not square: {patch.shape[0]}")
    if channel_idx is not None:
        patch = patch.index_select(1, channel_idx.to(patch.device))
    patch = F.normalize(patch, dim=1)
    feat = patch.reshape(grid, grid, patch.shape[1]).permute(2, 0, 1).contiguous()
    return feat.detach(), width, height


def sample_rows(rows: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if rows.shape[0] <= int(max_rows):
        return rows
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(rows.shape[0], generator=gen)[: int(max_rows)]
    return rows[perm]


def build_memory_bank(args, model, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_path = Path(args.memory_cache).resolve() if str(args.memory_cache).strip() else None
    if cache_path and cache_path.exists() and not args.rebuild_memory:
        cached = torch.load(cache_path, map_location="cpu")
        memory = cached["memory"].float()
        channel_idx = cached["channel_idx"].long()
        print(f"Loaded DINOv3 memory cache: {cache_path} | memory={tuple(memory.shape)}")
        return memory, channel_idx

    normal_images = find_images(Path(args.normal_dir).resolve())
    if not normal_images:
        raise FileNotFoundError(f"No normal images found in {args.normal_dir}")

    first_feat, _, _ = extract_dino_feature_map(model, normal_images[0], int(args.image_size), device)
    channel_idx = choose_channels(first_feat.shape[0], int(args.feature_dim), int(args.seed))

    rng = np.random.default_rng(int(args.seed) + 17)
    all_rows: List[torch.Tensor] = []
    mask_root = Path(args.mask_dir).resolve()
    for image_path in tqdm(normal_images, desc="DINOv3 normal memory"):
        feat, width, height = extract_dino_feature_map(
            model, image_path, int(args.image_size), device, channel_idx=channel_idx
        )
        c, gh, gw = feat.shape
        mask = load_mask(mask_root, "train", image_path.name, width, height)
        mask_grid = resize_mask_to_grid(mask, (gh, gw))
        rows = feat.permute(1, 2, 0).reshape(-1, c).detach().cpu()
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
        raise RuntimeError("DINOv3 normal memory is empty.")

    memory = torch.cat(all_rows, dim=0)
    memory = sample_rows(memory, int(args.max_memory_patches), int(args.seed) + 31)
    memory = F.normalize(memory.float(), dim=1).contiguous()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory": memory.cpu(),
                "channel_idx": channel_idx.cpu(),
                "image_size": int(args.image_size),
                "feature_dim": int(args.feature_dim),
                "normal_patches_per_image": int(args.normal_patches_per_image),
                "max_memory_patches": int(args.max_memory_patches),
            },
            cache_path,
        )
        print(f"Saved DINOv3 memory cache: {cache_path} | memory={tuple(memory.shape)}")
    return memory, channel_idx


@torch.no_grad()
def anomaly_heatmap(feat: torch.Tensor, memory_t: torch.Tensor, chunk_size: int) -> torch.Tensor:
    c, gh, gw = feat.shape
    rows = feat.permute(1, 2, 0).reshape(-1, c).float()
    rows = F.normalize(rows, dim=1)
    scores = []
    for start in range(0, rows.shape[0], int(chunk_size)):
        part = rows[start : start + int(chunk_size)]
        sim = torch.mm(part, memory_t)
        scores.append((1.0 - sim.max(dim=1).values).detach())
    return torch.cat(scores, dim=0).reshape(gh, gw)


def box_anomaly_score(
    norm_heat: np.ndarray,
    raw_heat: np.ndarray,
    valid_grid: np.ndarray,
    box: Iterable[float],
    width: int,
    height: int,
    box_expand: float,
    percentile: float,
) -> Tuple[float, float, float]:
    gh, gw = norm_heat.shape
    base_box = normalize_box(box, width, height)
    if base_box is None:
        return 0.0, 0.0, 0.0
    if abs(float(box_expand) - 1.0) > 1e-6:
        base_box = normalize_box(expand_box(base_box, width, height, float(box_expand), 0.0), width, height)
        if base_box is None:
            return 0.0, 0.0, 0.0

    gx1, gy1, gx2, gy2 = grid_box_from_original_box(base_box, width, height, gw, gh)
    heat_crop = norm_heat[gy1:gy2, gx1:gx2]
    raw_crop = raw_heat[gy1:gy2, gx1:gx2]
    valid_crop = valid_grid[gy1:gy2, gx1:gx2]
    if heat_crop.size == 0:
        return 0.0, 0.0, 0.0

    if valid_crop.any():
        vals = heat_crop[valid_crop]
        raw_vals = raw_crop[valid_crop]
        coverage = float(valid_crop.mean())
    else:
        vals = heat_crop.reshape(-1)
        raw_vals = raw_crop.reshape(-1)
        coverage = 0.0
    if vals.size == 0:
        return 0.0, 0.0, coverage
    pct = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(vals, pct)), float(np.percentile(raw_vals, pct)), coverage


def anomaly_factor(anom: float, low_thr: float, low_factor: float, high_thr: float, high_factor: float) -> float:
    anom = max(0.0, min(1.0, float(anom)))
    low_thr = max(1e-9, float(low_thr))
    high_thr = min(1.0 - 1e-9, float(high_thr))
    if anom < low_thr:
        strength = (low_thr - anom) / low_thr
        return 1.0 - strength * (1.0 - float(low_factor))
    if anom > high_thr and float(high_factor) > 1.0:
        strength = (anom - high_thr) / max(1.0 - high_thr, 1e-9)
        return 1.0 + strength * (float(high_factor) - 1.0)
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 PatchCore-style normal-memory box rescoring for LBB.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--normal-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--repo-dir", default="./third_party/dinov3")
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--weights", default="./dinov3/dinov3_vitl16_from_safetensors-8aa4cbdd.pth")
    parser.add_argument("--memory-cache", default="./outputs/dinov3_patchcore_memory_i384_dim256_24k.pt")
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--box-score-cache", default="")
    parser.add_argument("--rebuild-box-score-cache", action="store_true")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--normal-patches-per-image", type=int, default=350)
    parser.add_argument("--max-memory-patches", type=int, default=24000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--box-expand", type=float, default=1.20)
    parser.add_argument("--box-percentile", type=float, default=85.0)
    parser.add_argument("--min-score", type=float, default=0.00001)
    parser.add_argument("--max-score", type=float, default=0.08)
    parser.add_argument("--low-thr", type=float, default=0.10)
    parser.add_argument("--low-factor", type=float, default=0.70)
    parser.add_argument("--high-thr", type=float, default=0.92)
    parser.add_argument("--high-factor", type=float, default=1.0)
    parser.add_argument("--outside-mask-factor", type=float, default=1.0)
    parser.add_argument("--outside-mask-coverage", type=float, default=0.0)
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Only rescore this label. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Do not rescore this label. Can be passed multiple times.",
    )
    parser.add_argument("--min-score-after", type=float, default=0.0)
    parser.add_argument("--score-ceiling", type=float, default=1.0)
    parser.add_argument("--score-decimals", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    include_labels = {str(x) for x in args.include_label}
    exclude_labels = {str(x) for x in args.exclude_label}

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    pred = Path(args.pred).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    torch.backends.cudnn.benchmark = True
    model = build_dinov3(Path(args.repo_dir), str(args.model_name), Path(args.weights), device)

    memory, channel_idx = build_memory_bank(args, model, device)
    memory_t = memory.to(device, non_blocking=True).T.contiguous()

    images = image_map(Path(args.test_image_dir).resolve())
    mask_root = Path(args.mask_dir).resolve()
    store = Store(pred)
    box_cache_path = Path(args.box_score_cache).resolve() if str(args.box_score_cache).strip() else None
    box_cache: Dict[str, List[Tuple[float, float]]] = {}
    cache_changed = False
    if box_cache_path and box_cache_path.exists() and not bool(args.rebuild_box_score_cache):
        cached = torch.load(box_cache_path, map_location="cpu")
        if isinstance(cached, dict) and isinstance(cached.get("scores"), dict):
            box_cache = cached["scores"]
            print(f"Loaded DINOv3 box anomaly cache: {box_cache_path} | images={len(box_cache)}")

    total = 0
    touched = 0
    lowered = 0
    raised = 0
    outside_mask = 0
    missing_images = 0
    missing_masks = 0
    anom_values: List[float] = []
    label_touched: Counter = Counter()
    label_lowered: Counter = Counter()

    try:
        names = store.names()
        if int(args.limit) > 0:
            names = names[: int(args.limit)]
        for name in tqdm(names, desc="DINOv3 PatchCore box rescore"):
            payload = store.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = images.get(image_name)
            anns = list(payload.get("annotations", []))
            if image_path is None:
                missing_images += 1
                payload["annotations"] = anns
                (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            cached_rows = box_cache.get(name)
            use_cached_rows = cached_rows is not None and len(cached_rows) == len(anns)
            if use_cached_rows:
                norm_heat = raw_heat = valid_grid = None
                width = height = 0
            else:
                feat, width, height = extract_dino_feature_map(
                    model, image_path, int(args.image_size), device, channel_idx=channel_idx
                )
                gh, gw = feat.shape[-2:]
                mask = load_mask(mask_root, "test", image_name, width, height)
                if mask is None:
                    missing_masks += 1
                valid_grid = resize_mask_to_grid(mask, (gh, gw))
                raw_heat_t = anomaly_heatmap(feat, memory_t, int(args.chunk_size))
                raw_heat = raw_heat_t.detach().float().cpu().numpy()
                norm_heat, _, _ = normalize_heat(raw_heat, valid_grid)
                cached_rows = []
                cache_changed = True

            out_anns = []
            for ann_idx, ann in enumerate(anns):
                total += 1
                score = safe_float(ann.get("confidence", 0.0), 0.0)
                out_ann = dict(ann)
                box = ann.get("bbox", [])
                if not isinstance(box, list) or len(box) != 4:
                    if not use_cached_rows:
                        cached_rows.append((float("nan"), float("nan")))
                    out_anns.append(out_ann)
                    continue

                if use_cached_rows:
                    anom, coverage = cached_rows[ann_idx]
                    if not np.isfinite(anom):
                        out_anns.append(out_ann)
                        continue
                else:
                    anom, _, coverage = box_anomaly_score(
                        norm_heat,
                        raw_heat,
                        valid_grid,
                        box,
                        width,
                        height,
                        float(args.box_expand),
                        float(args.box_percentile),
                    )
                    cached_rows.append((float(anom), float(coverage)))

                if score < float(args.min_score) or score > float(args.max_score):
                    out_anns.append(out_ann)
                    continue
                label = str(ann.get("label", ""))
                if (include_labels and label not in include_labels) or label in exclude_labels:
                    out_anns.append(out_ann)
                    continue

                anom_values.append(anom)
                factor = anomaly_factor(
                    anom,
                    float(args.low_thr),
                    float(args.low_factor),
                    float(args.high_thr),
                    float(args.high_factor),
                )
                if coverage <= float(args.outside_mask_coverage):
                    factor *= float(args.outside_mask_factor)
                    outside_mask += 1

                new_score = max(float(args.min_score_after), min(float(args.score_ceiling), score * factor))
                if abs(new_score - score) > 1e-12:
                    touched += 1
                    label_touched[label] += 1
                    if new_score < score:
                        lowered += 1
                        label_lowered[label] += 1
                    elif new_score > score:
                        raised += 1
                    out_ann["confidence"] = round(float(new_score), int(args.score_decimals))
                out_anns.append(out_ann)

            out_anns.sort(key=lambda a: safe_float(a.get("confidence", 0.0), 0.0), reverse=True)
            payload["annotations"] = out_anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if not use_cached_rows and box_cache_path:
                box_cache[name] = cached_rows
    finally:
        store.close()

    if box_cache_path and cache_changed:
        box_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scores": box_cache,
                "pred": str(pred),
                "image_size": int(args.image_size),
                "feature_dim": int(args.feature_dim),
                "box_expand": float(args.box_expand),
                "box_percentile": float(args.box_percentile),
                "memory_cache": str(args.memory_cache),
            },
            box_cache_path,
        )
        print(f"Saved DINOv3 box anomaly cache: {box_cache_path} | images={len(box_cache)}")

    write_zip(out_dir, out_zip)
    print(f"Input: {pred}")
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Annotations total/touched/lowered/raised: {total}/{touched}/{lowered}/{raised}")
    print(f"Outside-mask penalized: {outside_mask}")
    print(f"Missing images/masks: {missing_images}/{missing_masks}")
    print(f"Touched labels: {dict(label_touched)}")
    print(f"Lowered labels: {dict(label_lowered)}")
    if anom_values:
        qs = np.percentile(np.asarray(anom_values, dtype=np.float32), [1, 5, 10, 25, 50, 75, 90, 95, 99])
        print("Box anomaly quantiles 1/5/10/25/50/75/90/95/99:", [round(float(x), 4) for x in qs])


if __name__ == "__main__":
    main()
