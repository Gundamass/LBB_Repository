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

from infer_synth_anomaly_proposals import image_to_tensor, predict_heatmap  # noqa: E402
from train_synth_anomaly_segmenter import LABELS, TinyUNet, find_images, load_product_mask  # noqa: E402


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


def grid_box(box: Sequence[float], width: int, height: int, image_size: int, expand: float) -> Optional[Tuple[int, int, int, int]]:
    norm = normalize_box(box, width, height)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    if abs(float(expand) - 1.0) > 1e-6:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1) * float(expand)
        bh = (y2 - y1) * float(expand)
        norm = normalize_box([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], width, height)
        if norm is None:
            return None
        x1, y1, x2, y2 = norm
    gx1 = int(np.floor(x1 / max(float(width), 1.0) * image_size))
    gy1 = int(np.floor(y1 / max(float(height), 1.0) * image_size))
    gx2 = int(np.ceil(x2 / max(float(width), 1.0) * image_size))
    gy2 = int(np.ceil(y2 / max(float(height), 1.0) * image_size))
    gx1 = max(0, min(image_size - 1, gx1))
    gy1 = max(0, min(image_size - 1, gy1))
    gx2 = max(gx1 + 1, min(image_size, gx2))
    gy2 = max(gy1 + 1, min(image_size, gy2))
    return gx1, gy1, gx2, gy2


def box_heat_score(
    heat: np.ndarray,
    valid: np.ndarray,
    box: Sequence[float],
    width: int,
    height: int,
    image_size: int,
    expand: float,
    percentile: float,
) -> Tuple[float, float]:
    gb = grid_box(box, width, height, image_size, expand)
    if gb is None:
        return 0.0, 0.0
    gx1, gy1, gx2, gy2 = gb
    crop = heat[gy1:gy2, gx1:gx2]
    mask = valid[gy1:gy2, gx1:gx2]
    if crop.size == 0:
        return 0.0, 0.0
    vals = crop[mask] if mask.any() else crop.reshape(-1)
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.percentile(vals, float(percentile))), float(mask.mean()) if mask.size else 0.0


def factor_from_heat(value: float, low_thr: float, low_factor: float, high_thr: float, high_factor: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    if value < float(low_thr):
        strength = (float(low_thr) - value) / max(float(low_thr), 1e-9)
        return 1.0 - strength * (1.0 - float(low_factor))
    if float(high_factor) > 1.0 and value > float(high_thr):
        strength = (value - float(high_thr)) / max(1.0 - float(high_thr), 1e-9)
        return 1.0 + strength * (float(high_factor) - 1.0)
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore LBB boxes with synthetic anomaly segmentation heatmaps.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.00001)
    parser.add_argument("--max-score", type=float, default=0.08)
    parser.add_argument("--box-expand", type=float, default=1.20)
    parser.add_argument("--box-percentile", type=float, default=90.0)
    parser.add_argument("--low-thr", type=float, default=0.42)
    parser.add_argument("--low-factor", type=float, default=0.82)
    parser.add_argument("--high-thr", type=float, default=0.82)
    parser.add_argument("--high-factor", type=float, default=1.0)
    parser.add_argument("--use-all-class-max", action="store_true")
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--min-mask-coverage", type=float, default=0.0)
    parser.add_argument("--outside-mask-factor", type=float, default=1.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-decimals", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(Path(args.checkpoint).resolve(), map_location="cpu")
    image_size = int(args.image_size or ckpt.get("image_size", 512))
    base_channels = int(args.base_channels or ckpt.get("base_channels", 32))
    model = TinyUNet(out_channels=len(LABELS), base=base_channels)
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model.to(device).eval()

    include_labels = {str(x) for x in args.include_label}
    exclude_labels = {str(x) for x in args.exclude_label}
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    images = {p.name: p for p in find_images(Path(args.test_image_dir).resolve())}

    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = Store(Path(args.pred).resolve())
    total = touched = lowered = raised = 0
    touched_labels = Counter()
    heat_values: List[float] = []
    try:
        names = store.names()
        if int(args.limit) > 0:
            names = names[: int(args.limit)]
        for name in tqdm(names, desc="synthseg box rescore"):
            payload = store.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = images.get(image_name)
            anns = list(payload.get("annotations", []))
            if image_path is None:
                payload["annotations"] = anns
                (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            tensor, width, height, _ = image_to_tensor(image_path, image_size)
            with Image.open(image_path) as raw:
                raw_size = raw.size
            valid = load_product_mask(Path(args.mask_dir).resolve(), "test", image_name, raw_size)
            valid = cv2.resize(valid.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
            heat_t = predict_heatmap(model, tensor, device, bool(args.tta))
            heat_np = heat_t.numpy()
            if bool(args.use_all_class_max):
                heat_all = heat_np.max(axis=0)
            else:
                heat_all = None

            out_anns = []
            for ann in anns:
                total += 1
                out_ann = dict(ann)
                score = safe_float(ann.get("confidence", 0.0), 0.0)
                if score < float(args.min_score) or score > float(args.max_score):
                    out_anns.append(out_ann)
                    continue
                label = str(ann.get("label", ""))
                if label not in label_to_idx:
                    out_anns.append(out_ann)
                    continue
                if (include_labels and label not in include_labels) or label in exclude_labels:
                    out_anns.append(out_ann)
                    continue
                heat = heat_all if heat_all is not None else heat_np[label_to_idx[label]]
                value, coverage = box_heat_score(
                    heat,
                    valid,
                    ann.get("bbox", []),
                    width,
                    height,
                    image_size,
                    float(args.box_expand),
                    float(args.box_percentile),
                )
                heat_values.append(value)
                factor = factor_from_heat(
                    value,
                    float(args.low_thr),
                    float(args.low_factor),
                    float(args.high_thr),
                    float(args.high_factor),
                )
                if coverage < float(args.min_mask_coverage):
                    factor *= float(args.outside_mask_factor)
                new_score = max(0.0, min(1.0, score * factor))
                if abs(new_score - score) > 1e-12:
                    touched += 1
                    touched_labels[label] += 1
                    if new_score < score:
                        lowered += 1
                    else:
                        raised += 1
                    out_ann["confidence"] = round(float(new_score), int(args.score_decimals))
                out_anns.append(out_ann)
            out_anns.sort(key=lambda a: safe_float(a.get("confidence", 0.0), 0.0), reverse=True)
            payload["annotations"] = out_anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Input: {Path(args.pred).resolve()}")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Annotations total/touched/lowered/raised: {total}/{touched}/{lowered}/{raised}")
    print("Touched labels:", dict(touched_labels))
    if heat_values:
        qs = np.percentile(np.asarray(heat_values, dtype=np.float32), [1, 5, 10, 25, 50, 75, 90, 95, 99])
        print("Box heat quantiles 1/5/10/25/50/75/90/95/99:", [round(float(x), 4) for x in qs])


if __name__ == "__main__":
    main()
