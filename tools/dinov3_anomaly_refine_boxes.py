import argparse
import json
import math
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from patchcore_anomaly_proposals import (  # noqa: E402
    box_iou,
    expand_box,
    grid_box_from_original_box,
    load_mask,
    normalize_box,
    normalize_heat,
    resize_mask_to_grid,
)
from rescore_dinov3_patchcore_boxes import (  # noqa: E402
    Store,
    anomaly_heatmap,
    build_dinov3,
    build_memory_bank,
    extract_dino_feature_map,
    image_map,
    safe_float,
    write_zip,
)


VALID_LABELS = {"plain particle", "dirt", "scratch", "collision"}


def ensure_min_side(box: Sequence[float], width: int, height: int, min_side: float) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(float(min_side), x2 - x1)
    bh = max(float(min_side), y2 - y1)
    return normalize_box(
        [
            cx - bw * 0.5,
            cy - bh * 0.5,
            cx + bw * 0.5,
            cy + bh * 0.5,
        ],
        width,
        height,
    )


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def component_score(vals: np.ndarray, cells: int) -> float:
    if vals.size == 0:
        return -1.0
    # Prefer compact peaks, but do not discard slightly larger abrasion-like regions.
    return float(vals.mean() * 0.55 + vals.max() * 0.35 + math.log1p(float(cells)) * 0.02)


def map_grid_box_to_image(
    gx1: int,
    gy1: int,
    gx2: int,
    gy2: int,
    gw: int,
    gh: int,
    width: int,
    height: int,
) -> List[float]:
    return [
        gx1 / float(gw) * float(width),
        gy1 / float(gh) * float(height),
        gx2 / float(gw) * float(width),
        gy2 / float(gh) * float(height),
    ]


def refine_box_from_heatmap(
    norm_heat: np.ndarray,
    valid_grid: np.ndarray,
    source_box: Sequence[float],
    width: int,
    height: int,
    search_expand: float,
    crop_percentile: float,
    abs_thr: float,
    min_cells: int,
    component_pad: int,
    out_expand: float,
    out_pad: float,
    min_side: float,
    min_area: float,
    max_area: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_iou_source: float,
    max_iou_source: float,
) -> Tuple[Optional[List[float]], float, int]:
    gh, gw = norm_heat.shape
    src = normalize_box(source_box, width, height)
    if src is None:
        return None, 0.0, 0

    search = normalize_box(expand_box(src, width, height, search_expand, 0.0), width, height)
    if search is None:
        return None, 0.0, 0
    gx1, gy1, gx2, gy2 = grid_box_from_original_box(search, width, height, gw, gh)
    crop = norm_heat[gy1:gy2, gx1:gx2]
    valid = valid_grid[gy1:gy2, gx1:gx2]
    if crop.size == 0:
        return None, 0.0, 0

    vals = crop[valid] if valid.any() else crop.reshape(-1)
    if vals.size == 0:
        return None, 0.0, 0
    thr = max(float(abs_thr), float(np.percentile(vals, float(crop_percentile))))
    binary = ((crop >= thr) & valid).astype(np.uint8)

    if int(min_cells) <= 1 and binary.sum() == 0:
        y, x = np.unravel_index(int(np.argmax(crop)), crop.shape)
        binary[y, x] = 1

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    best_score = -1.0
    for idx in range(1, num):
        x, y, bw, bh, cells = stats[idx].tolist()
        if int(cells) < int(min_cells):
            continue
        comp = labels == idx
        score = component_score(crop[comp], int(cells))
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh, int(cells))

    if best is None:
        return None, 0.0, 0

    x, y, bw, bh, cells = best
    pad = max(0, int(component_pad))
    cgx1 = max(0, gx1 + x - pad)
    cgy1 = max(0, gy1 + y - pad)
    cgx2 = min(gw, gx1 + x + bw + pad)
    cgy2 = min(gh, gy1 + y + bh + pad)
    refined = map_grid_box_to_image(cgx1, cgy1, cgx2, cgy2, gw, gh, width, height)
    refined = normalize_box(expand_box(refined, width, height, out_expand, out_pad), width, height)
    if refined is None:
        return None, 0.0, cells
    refined = ensure_min_side(refined, width, height, min_side)
    if refined is None:
        return None, 0.0, cells

    area = box_area(refined)
    src_area = max(box_area(src), 1.0)
    if area < float(min_area) or area > float(max_area):
        return None, best_score, cells
    ratio = area / src_area
    if ratio < float(min_area_ratio) or ratio > float(max_area_ratio):
        return None, best_score, cells

    iou = box_iou(refined, src)
    if iou < float(min_iou_source) or iou > float(max_iou_source):
        return None, best_score, cells
    return refined, best_score, cells


def dedup_against(box: Sequence[float], old_boxes: Iterable[Sequence[float]], iou_thr: float) -> bool:
    return any(box_iou(box, old) >= float(iou_thr) for old in old_boxes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append DINOv3 PatchCore anomaly-heatmap refined boxes as low-score tail candidates."
    )
    parser.add_argument("--base", required=True, help="Base competition prediction zip/folder.")
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--normal-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--repo-dir", default="./third_party/dinov3")
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--weights", default="./dinov3/dinov3_vitl16_from_safetensors-8aa4cbdd.pth")
    parser.add_argument("--memory-cache", default="./outputs/dinov3_patchcore_memory_i384_dim256_24k.pt")
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--normal-patches-per-image", type=int, default=350)
    parser.add_argument("--max-memory-patches", type=int, default=24000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--topk-per-image", type=int, default=80)
    parser.add_argument("--source-min-score", type=float, default=0.00001)
    parser.add_argument("--source-max-score", type=float, default=1.0)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--search-expand", type=float, default=1.25)
    parser.add_argument("--crop-percentile", type=float, default=85.0)
    parser.add_argument("--abs-thr", type=float, default=0.20)
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--component-pad", type=int, default=1)
    parser.add_argument("--out-expand", type=float, default=1.70)
    parser.add_argument("--out-pad", type=float, default=2.0)
    parser.add_argument("--min-side", type=float, default=8.0)
    parser.add_argument("--min-area", type=float, default=30.0)
    parser.add_argument("--max-area", type=float, default=45000.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-area-ratio", type=float, default=1.80)
    parser.add_argument("--min-iou-source", type=float, default=0.02)
    parser.add_argument("--max-iou-source", type=float, default=0.98)
    parser.add_argument("--dedup-iou", type=float, default=0.94)
    parser.add_argument("--score-factor", type=float, default=0.00005)
    parser.add_argument("--anomaly-score-factor", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-confidence", type=float, default=0.0002)
    parser.add_argument("--score-decimals", type=int, default=8)
    parser.add_argument("--max-output-per-image", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    include_labels = {str(x) for x in args.include_label}
    exclude_labels = {str(x) for x in args.exclude_label}

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

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
    store = Store(Path(args.base).resolve())

    total_source = 0
    tried = 0
    added = 0
    skipped = Counter()
    added_labels = Counter()

    try:
        names = store.names()
        if int(args.limit) > 0:
            names = names[: int(args.limit)]

        for name in tqdm(names, desc="DINOv3 anomaly refined tail"):
            payload = store.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = images.get(image_name)
            anns = list(payload.get("annotations", []))
            if image_path is None:
                skipped["missing_image"] += 1
                (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            feat, width, height = extract_dino_feature_map(
                model, image_path, int(args.image_size), device, channel_idx=channel_idx
            )
            gh, gw = feat.shape[-2:]
            mask = load_mask(mask_root, "test", image_name, width, height)
            valid_grid = resize_mask_to_grid(mask, (gh, gw))
            raw_heat = anomaly_heatmap(feat, memory_t, int(args.chunk_size)).detach().float().cpu().numpy()
            norm_heat, _, _ = normalize_heat(raw_heat, valid_grid)

            by_label: Dict[str, List[List[float]]] = {}
            for ann in anns:
                label = str(ann.get("label", ""))
                box = ann.get("bbox", [])
                if label and isinstance(box, list) and len(box) == 4:
                    norm = normalize_box(box, width, height)
                    if norm is not None:
                        by_label.setdefault(label, []).append(norm)

            source_rows = []
            for idx, ann in enumerate(anns):
                label = str(ann.get("label", ""))
                score = safe_float(ann.get("confidence", 0.0), 0.0)
                box = ann.get("bbox", [])
                if label not in VALID_LABELS:
                    continue
                if (include_labels and label not in include_labels) or label in exclude_labels:
                    continue
                if score < float(args.source_min_score) or score > float(args.source_max_score):
                    continue
                if not isinstance(box, list) or len(box) != 4:
                    continue
                source_rows.append((score, idx, ann))

            source_rows.sort(key=lambda x: x[0], reverse=True)
            if int(args.topk_per_image) > 0:
                source_rows = source_rows[: int(args.topk_per_image)]
            total_source += len(source_rows)

            new_anns: List[Dict] = []
            for score, _, ann in source_rows:
                tried += 1
                label = str(ann.get("label", ""))
                refined, anom, cells = refine_box_from_heatmap(
                    norm_heat=norm_heat,
                    valid_grid=valid_grid,
                    source_box=ann.get("bbox", []),
                    width=width,
                    height=height,
                    search_expand=float(args.search_expand),
                    crop_percentile=float(args.crop_percentile),
                    abs_thr=float(args.abs_thr),
                    min_cells=int(args.min_cells),
                    component_pad=int(args.component_pad),
                    out_expand=float(args.out_expand),
                    out_pad=float(args.out_pad),
                    min_side=float(args.min_side),
                    min_area=float(args.min_area),
                    max_area=float(args.max_area),
                    min_area_ratio=float(args.min_area_ratio),
                    max_area_ratio=float(args.max_area_ratio),
                    min_iou_source=float(args.min_iou_source),
                    max_iou_source=float(args.max_iou_source),
                )
                if refined is None:
                    skipped["no_refine"] += 1
                    continue
                if dedup_against(refined, by_label.get(label, []), float(args.dedup_iou)):
                    skipped["dedup"] += 1
                    continue

                new_conf = score * float(args.score_factor) + float(anom) * float(args.anomaly_score_factor)
                new_conf = max(float(args.min_confidence), min(float(args.max_confidence), new_conf))
                new_ann = {
                    "label": label,
                    "bbox": [round(float(v), 3) for v in refined],
                    "confidence": round(float(new_conf), int(args.score_decimals)),
                }
                new_anns.append(new_ann)
                by_label.setdefault(label, []).append(refined)
                added += 1
                added_labels[label] += 1

            anns.extend(new_anns)
            anns.sort(key=lambda a: safe_float(a.get("confidence", 0.0), 0.0), reverse=True)
            if int(args.max_output_per_image) > 0:
                anns = anns[: int(args.max_output_per_image)]
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Base: {Path(args.base).resolve()}")
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Sources selected/tried/added: {total_source}/{tried}/{added}")
    print("Skipped:", dict(skipped))
    print("Added labels:", dict(added_labels))


if __name__ == "__main__":
    main()
