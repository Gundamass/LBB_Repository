import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image

try:
    from tools.convert_labelme_to_yolo import (
        Sample,
        build_samples,
        class_mapping,
        image_size,
        labelme_box,
        load_yaml,
        resolve_path,
    )
except Exception:
    from convert_labelme_to_yolo import (
        Sample,
        build_samples,
        class_mapping,
        image_size,
        labelme_box,
        load_yaml,
        resolve_path,
    )


Box = Tuple[float, float, float, float]
Window = Tuple[int, int, int, int]


def clamp_window(cx: float, cy: float, tile_size: int, width: int, height: int) -> Window:
    if width <= tile_size:
        x1, x2 = 0, width
    else:
        x1 = int(round(cx - tile_size * 0.5))
        x1 = max(0, min(width - tile_size, x1))
        x2 = x1 + tile_size

    if height <= tile_size:
        y1, y2 = 0, height
    else:
        y1 = int(round(cy - tile_size * 0.5))
        y1 = max(0, min(height - tile_size, y1))
        y2 = y1 + tile_size

    return int(x1), int(y1), int(x2), int(y2)


def grid_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(int(x) for x in starts))


def grid_windows(width: int, height: int, tile_size: int, overlap: float) -> List[Window]:
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = grid_starts(width, tile_size, stride)
    ys = grid_starts(height, tile_size, stride)
    windows: List[Window] = []
    for y in ys:
        for x in xs:
            windows.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))
    return windows


def parse_labelme_boxes(
    sample: Sample,
    label_to_yolo: Dict[str, int],
    min_box_size: float,
) -> Tuple[List[Dict], Tuple[int, int]]:
    if sample.ann_path is None or not sample.ann_path.exists():
        with Image.open(sample.image_path) as im:
            return [], (int(im.width), int(im.height))

    payload = json.loads(sample.ann_path.read_text(encoding="utf-8"))
    width, height = image_size(sample.image_path, payload)
    boxes: List[Dict] = []

    for shape in payload.get("shapes", []):
        label = shape.get("label")
        if label not in label_to_yolo:
            continue
        box = labelme_box(shape.get("points", []))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(width), float(x1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y1 = max(0.0, min(float(height), float(y1)))
        y2 = max(0.0, min(float(height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue
        boxes.append({"label": str(label), "cls": int(label_to_yolo[label]), "box": (x1, y1, x2, y2)})

    # Extra synthetic/real-paste datasets are stored in competition format.
    for ann in payload.get("annotations", []):
        label = ann.get("label")
        if label not in label_to_yolo:
            continue
        bbox = ann.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue
        boxes.append({"label": str(label), "cls": int(label_to_yolo[label]), "box": (x1, y1, x2, y2)})

    return boxes, (int(width), int(height))


def box_center(box: Box) -> Tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def positive_windows(
    boxes: List[Dict],
    width: int,
    height: int,
    tile_size: int,
    base_jitters: int,
    rare_extra_jitters: Dict[str, int],
    rng: random.Random,
) -> List[Window]:
    windows: List[Window] = []
    for item in boxes:
        cx, cy = box_center(item["box"])
        label = item["label"]
        n_jitters = max(0, int(base_jitters)) + max(0, int(rare_extra_jitters.get(label, 0)))
        windows.append(clamp_window(cx, cy, tile_size, width, height))
        for _ in range(n_jitters):
            # Jitter around the object so tiny defects appear at varied positions
            # while staying inside a local high-resolution crop.
            dx = rng.uniform(-0.25, 0.25) * tile_size
            dy = rng.uniform(-0.25, 0.25) * tile_size
            windows.append(clamp_window(cx + dx, cy + dy, tile_size, width, height))
    return windows


def dedupe_windows(windows: Iterable[Window]) -> List[Window]:
    seen = set()
    out: List[Window] = []
    for w in windows:
        key = tuple(int(v) for v in w)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def clip_box_to_window(box: Box, window: Window) -> Optional[Box]:
    x1, y1, x2, y2 = box
    wx1, wy1, wx2, wy2 = window
    ix1 = max(float(wx1), float(x1))
    iy1 = max(float(wy1), float(y1))
    ix2 = min(float(wx2), float(x2))
    iy2 = min(float(wy2), float(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def labels_for_window(
    boxes: List[Dict],
    window: Window,
    min_box_size: float,
    min_visible: float,
) -> Tuple[List[str], Dict[str, int]]:
    wx1, wy1, wx2, wy2 = window
    tile_w = float(wx2 - wx1)
    tile_h = float(wy2 - wy1)
    lines: List[str] = []
    counts: Dict[str, int] = {}

    for item in boxes:
        box = item["box"]
        clipped = clip_box_to_window(box, window)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_box_size or bh < min_box_size:
            continue

        orig_area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        visible = (bw * bh) / orig_area
        if visible < min_visible:
            continue

        lx1, ly1 = x1 - wx1, y1 - wy1
        lx2, ly2 = x2 - wx1, y2 - wy1
        cx = (lx1 + lx2) * 0.5 / tile_w
        cy = (ly1 + ly2) * 0.5 / tile_h
        nw = (lx2 - lx1) / tile_w
        nh = (ly2 - ly1) / tile_h
        lines.append(f"{item['cls']} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
        counts[item["label"]] = counts.get(item["label"], 0) + 1

    return lines, counts


def sample_key(sample: Sample) -> str:
    digest = hashlib.sha1(str(sample.image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{sample.image_path.stem}_{digest}"


def write_split(
    samples: List[Sample],
    split: str,
    output_dir: Path,
    label_to_yolo: Dict[str, int],
    tile_size: int,
    overlap: float,
    min_box_size: float,
    min_visible: float,
    positive_jitters: int,
    rare_extra_jitters: Dict[str, int],
    jpeg_quality: int,
    seed: int,
    extra_grid: bool,
    extra_positive_jitters: Optional[int],
) -> Dict:
    img_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_images": len(samples),
        "tiles": 0,
        "positive_tiles": 0,
        "empty_tiles": 0,
        "boxes": 0,
        "boxes_per_class": {},
    }

    for idx, sample in enumerate(samples):
        rng = random.Random(seed + idx * 1009 + (0 if split == "train" else 1000003))
        boxes, (width, height) = parse_labelme_boxes(sample, label_to_yolo, min_box_size)
        is_extra = sample.split_tag not in {"clean", "defect"}
        use_grid = (not is_extra) or bool(extra_grid) or split != "train"
        windows = grid_windows(width, height, tile_size, overlap) if use_grid else []
        if boxes and split == "train":
            sample_jitters = int(positive_jitters)
            if is_extra and extra_positive_jitters is not None:
                sample_jitters = int(extra_positive_jitters)
            windows.extend(
                positive_windows(
                    boxes,
                    width,
                    height,
                    tile_size,
                    sample_jitters,
                    rare_extra_jitters,
                    rng,
                )
            )
        windows = dedupe_windows(windows)
        prefix = sample_key(sample)

        with Image.open(sample.image_path) as im:
            im = im.convert("RGB")
            for wi, window in enumerate(windows):
                lines, counts = labels_for_window(boxes, window, min_box_size, min_visible)
                wx1, wy1, wx2, wy2 = window
                tile_name = f"{prefix}_x{wx1}_y{wy1}_x{wx2}_y{wy2}_{wi:03d}.jpg"
                tile_path = img_dir / tile_name
                label_path = label_dir / f"{Path(tile_name).stem}.txt"

                crop = im.crop(window)
                crop.save(tile_path, quality=jpeg_quality)
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                stats["tiles"] += 1
                stats["boxes"] += len(lines)
                if lines:
                    stats["positive_tiles"] += 1
                else:
                    stats["empty_tiles"] += 1
                for label, count in counts.items():
                    stats["boxes_per_class"][label] = stats["boxes_per_class"].get(label, 0) + count

    return stats


def parse_rare_jitters(values: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=N for --rare-extra-jitter, got: {value}")
        label, n = value.split("=", 1)
        out[label.strip()] = int(n)
    return out


def convert_tile_dataset(
    config_path: str,
    output_dir: Optional[str],
    tile_size: int,
    overlap: float,
    min_visible: float,
    positive_jitters: int,
    rare_extra_jitters: Dict[str, int],
    force: bool,
    jpeg_quality: int,
) -> Path:
    cfg_path = Path(config_path).resolve()
    base = cfg_path.parent
    cfg = load_yaml(cfg_path)
    tile_cfg = cfg.get("tile", {})
    if not rare_extra_jitters:
        rare_extra_jitters = {str(k): int(v) for k, v in tile_cfg.get("rare_extra_jitters", {}).items()}
    out_dir = Path(output_dir) if output_dir else resolve_path(base, cfg.get("yolo", {}).get("data_dir", "./yolo_tile_data"))
    out_dir = out_dir.resolve()

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples = build_samples(cfg, base)
    label_to_yolo, names = class_mapping(cfg)
    min_box_size = float(cfg["data"].get("min_box_size", 2.0))
    seed = int(cfg.get("seed", 42))
    extra_grid = bool(tile_cfg.get("extra_grid", False))
    extra_positive_jitters = tile_cfg.get("extra_positive_jitters", None)
    if extra_positive_jitters is not None:
        extra_positive_jitters = int(extra_positive_jitters)

    stats = {
        "tile_size": int(tile_size),
        "overlap": float(overlap),
        "min_visible": float(min_visible),
        "positive_jitters": int(positive_jitters),
        "rare_extra_jitters": rare_extra_jitters,
        "extra_grid": extra_grid,
        "extra_positive_jitters": extra_positive_jitters,
        "train": write_split(
            train_samples,
            "train",
            out_dir,
            label_to_yolo,
            tile_size,
            overlap,
            min_box_size,
            min_visible,
            positive_jitters,
            rare_extra_jitters,
            jpeg_quality,
            seed,
            extra_grid,
            extra_positive_jitters,
        ),
        "val": write_split(
            val_samples,
            "val",
            out_dir,
            label_to_yolo,
            tile_size,
            overlap,
            min_box_size,
            min_visible,
            positive_jitters,
            rare_extra_jitters,
            jpeg_quality,
            seed,
            True,
            None,
        ),
    }

    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml_path = out_dir / "lbb.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.tile_yolo26.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--min-visible", type=float, default=0.25)
    parser.add_argument("--positive-jitters", type=int, default=2)
    parser.add_argument(
        "--rare-extra-jitter",
        action="append",
        default=[],
        help="Extra centered jitter crops for rare classes, e.g. 'collision=6'. Can be repeated.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    yaml_path = convert_tile_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        tile_size=int(args.tile_size),
        overlap=float(args.overlap),
        min_visible=float(args.min_visible),
        positive_jitters=int(args.positive_jitters),
        rare_extra_jitters=parse_rare_jitters(args.rare_extra_jitter),
        force=bool(args.force),
        jpeg_quality=int(args.jpeg_quality),
    )
    print(f"Tile YOLO data yaml: {yaml_path}")
    print(f"Stats: {yaml_path.parent / 'stats.json'}")


if __name__ == "__main__":
    main()
