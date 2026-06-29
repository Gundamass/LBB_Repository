import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from train_synth_anomaly_segmenter import LABELS, TinyUNet, find_images, load_product_mask  # noqa: E402


def parse_floats(value: str, n: int, name: str) -> List[float]:
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"{name} must contain 1 or {n} values, got: {value!r}")
    return vals


def normalize_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
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
    return normalize_box([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], width, height)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
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


def nms(rows: List[Dict], iou_thr: float, class_aware: bool) -> List[Dict]:
    rows = sorted(rows, key=lambda x: float(x["confidence"]), reverse=True)
    kept: List[Dict] = []
    for row in rows:
        duplicate = False
        for old in kept:
            if class_aware and row["label"] != old["label"]:
                continue
            if bbox_iou(row["bbox"], old["bbox"]) >= float(iou_thr):
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def image_to_tensor(path: Path, image_size: int) -> Tuple[torch.Tensor, int, int, np.ndarray]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        small = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, width, height, arr


@torch.no_grad()
def predict_heatmap(model: torch.nn.Module, image: torch.Tensor, device: torch.device, tta: bool) -> torch.Tensor:
    x = image.unsqueeze(0).to(device, non_blocking=True)
    logits = model(x)
    prob = torch.sigmoid(logits)[0].float().cpu()
    if not tta:
        return prob
    logits_h = model(torch.flip(x, dims=[3]))
    prob_h = torch.flip(torch.sigmoid(logits_h)[0].float().cpu(), dims=[2])
    logits_v = model(torch.flip(x, dims=[2]))
    prob_v = torch.flip(torch.sigmoid(logits_v)[0].float().cpu(), dims=[1])
    return (prob + prob_h + prob_v) / 3.0


def component_proposals(
    heat: np.ndarray,
    valid: np.ndarray,
    label: str,
    width: int,
    height: int,
    image_size: int,
    abs_thr: float,
    percentile: float,
    min_area: float,
    max_area: float,
    min_side: float,
    box_expand: float,
    box_pad: float,
    score_floor: float,
    score_ceiling: float,
    max_components: int,
    score_decimals: int,
) -> List[Dict]:
    if valid.any():
        vals = heat[valid]
    else:
        vals = heat.reshape(-1)
        valid = np.ones_like(heat, dtype=bool)
    if vals.size == 0:
        return []
    smooth = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=0.8)
    thr = max(float(abs_thr), float(np.percentile(vals, float(percentile))))
    binary = ((smooth >= thr) & valid).astype(np.uint8)
    if binary.sum() == 0:
        return []
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = (binary.astype(bool) & valid).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows: List[Dict] = []
    scale_x = float(width) / float(image_size)
    scale_y = float(height) / float(image_size)
    for idx in range(1, num):
        x, y, bw, bh, cells = stats[idx].tolist()
        if int(cells) <= 0:
            continue
        comp = labels == idx
        comp_scores = smooth[comp]
        if comp_scores.size == 0:
            continue
        score = float(np.percentile(comp_scores, 92))
        x1 = x * scale_x
        y1 = y * scale_y
        x2 = (x + bw) * scale_x
        y2 = (y + bh) * scale_y
        box = expand_box([x1, y1, x2, y2], width, height, box_expand, box_pad)
        if box is None:
            continue
        w_box = box[2] - box[0]
        h_box = box[3] - box[1]
        area = w_box * h_box
        if w_box < float(min_side) or h_box < float(min_side):
            continue
        if area < float(min_area) or area > float(max_area):
            continue
        conf = float(score_floor) + max(0.0, min(1.0, score)) * (float(score_ceiling) - float(score_floor))
        rows.append(
            {
                "label": label,
                "bbox": [round(float(v), 3) for v in box],
                "confidence": round(float(conf), int(score_decimals)),
                "_score": score,
            }
        )
    rows.sort(key=lambda r: float(r["_score"]), reverse=True)
    if int(max_components) > 0:
        rows = rows[: int(max_components)]
    for row in rows:
        row.pop("_score", None)
    return rows


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer synthetic anomaly segmenter proposals for LBB.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--image-size", type=int, default=0, help="Override checkpoint image size.")
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--abs-thr", default="0.44,0.40,0.42,0.36")
    parser.add_argument("--percentile", default="99.72,99.65,99.70,99.55")
    parser.add_argument("--min-area", default="24,45,30,70")
    parser.add_argument("--max-area", default="15000,26000,18000,42000")
    parser.add_argument("--min-side", default="5,7,5,8")
    parser.add_argument("--box-expand", default="1.55,1.70,1.60,1.85")
    parser.add_argument("--box-pad", default="2,3,2,5")
    parser.add_argument("--max-components-per-class", type=int, default=12)
    parser.add_argument("--max-boxes-per-image", type=int, default=48)
    parser.add_argument("--score-floor", type=float, default=0.02)
    parser.add_argument("--score-ceiling", type=float, default=1.0)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--all-label-nms-iou", type=float, default=0.55)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-decimals", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location="cpu")
    image_size = int(args.image_size or checkpoint.get("image_size", 512))
    base_channels = int(args.base_channels or checkpoint.get("base_channels", 32))
    labels = list(checkpoint.get("labels", LABELS))
    if labels != LABELS:
        raise ValueError(f"Checkpoint labels {labels} do not match expected {LABELS}")

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = TinyUNet(out_channels=len(LABELS), base=base_channels)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    abs_thr = parse_floats(args.abs_thr, len(LABELS), "abs-thr")
    percentiles = parse_floats(args.percentile, len(LABELS), "percentile")
    min_area = parse_floats(args.min_area, len(LABELS), "min-area")
    max_area = parse_floats(args.max_area, len(LABELS), "max-area")
    min_side = parse_floats(args.min_side, len(LABELS), "min-side")
    box_expand = parse_floats(args.box_expand, len(LABELS), "box-expand")
    box_pad = parse_floats(args.box_pad, len(LABELS), "box-pad")

    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(Path(args.test_image_dir).resolve())
    if int(args.limit) > 0:
        images = images[: int(args.limit)]

    label_counts = Counter()
    total = 0
    for image_path in tqdm(images, desc="infer synth anomaly proposals"):
        tensor, width, height, _ = image_to_tensor(image_path, image_size)
        with Image.open(image_path) as raw_img:
            raw_size = raw_img.size
        mask = load_product_mask(Path(args.mask_dir).resolve(), "test", image_path.name, raw_size)
        mask_small = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mask_small.mean() > 0.10:
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask_small.astype(np.uint8), kernel, iterations=1).astype(bool)
            if eroded.mean() > 0.03:
                mask_small = eroded

        heat = predict_heatmap(model, tensor, device, bool(args.tta)).numpy()
        rows: List[Dict] = []
        for class_idx, label in enumerate(LABELS):
            class_rows = component_proposals(
                heat=heat[class_idx],
                valid=mask_small,
                label=label,
                width=width,
                height=height,
                image_size=image_size,
                abs_thr=abs_thr[class_idx],
                percentile=percentiles[class_idx],
                min_area=min_area[class_idx],
                max_area=max_area[class_idx],
                min_side=min_side[class_idx],
                box_expand=box_expand[class_idx],
                box_pad=box_pad[class_idx],
                score_floor=float(args.score_floor),
                score_ceiling=float(args.score_ceiling),
                max_components=int(args.max_components_per_class),
                score_decimals=int(args.score_decimals),
            )
            rows.extend(class_rows)

        rows = nms(rows, float(args.nms_iou), class_aware=True)
        rows = nms(rows, float(args.all_label_nms_iou), class_aware=False)
        rows.sort(key=lambda r: float(r.get("confidence", 0.0) or 0.0), reverse=True)
        if int(args.max_boxes_per_image) > 0:
            rows = rows[: int(args.max_boxes_per_image)]
        for row in rows:
            label_counts[str(row.get("label"))] += 1
        total += len(rows)
        payload = {"image_id": image_path.name, "annotations": rows}
        (out_dir / image_path.with_suffix(".json").name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    write_zip(out_dir, out_zip)
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Total proposals: {total}")
    print("Label counts:", dict(label_counts))


if __name__ == "__main__":
    main()
