import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    image_path: Path
    ann_path: Optional[Path]
    split_tag: str


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def labelme_box(points: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def image_size(image_path: Path, payload: Optional[Dict] = None) -> Tuple[int, int]:
    if payload is not None:
        w = payload.get("imageWidth")
        h = payload.get("imageHeight")
        if w and h:
            return int(w), int(h)
    with Image.open(image_path) as im:
        return int(im.width), int(im.height)


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def build_samples(cfg: Dict, base: Path) -> Tuple[List[Sample], List[Sample]]:
    data_cfg = cfg["data"]
    pos_dir = resolve_path(base, data_cfg["train_pos_dir"])
    neg_dir = resolve_path(base, data_cfg["train_neg_dir"])
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    seed = int(cfg.get("seed", 42))

    all_samples: List[Sample] = []
    for image_path in iter_images(pos_dir):
        all_samples.append(Sample(image_path=image_path, ann_path=None, split_tag="clean"))
    for image_path in iter_images(neg_dir):
        ann_path = image_path.with_suffix(".json")
        all_samples.append(
            Sample(
                image_path=image_path,
                ann_path=ann_path if ann_path.exists() else None,
                split_tag="defect",
            )
        )

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_ratio))
    train_samples = all_samples[n_val:]
    val_samples = all_samples[:n_val]

    for extra_cfg in cfg.get("data", {}).get("extra_train_sets", []):
        image_dir = resolve_path(base, extra_cfg["image_dir"])
        ann_dir = resolve_path(base, extra_cfg["ann_dir"])
        split_tag = str(extra_cfg.get("split_tag", "extra"))
        repeat = max(1, int(extra_cfg.get("repeat", 1)))
        require_ann = bool(extra_cfg.get("require_ann", True))
        for image_path in iter_images(image_dir):
            ann_path = ann_dir / f"{image_path.stem}.json"
            if require_ann and not ann_path.exists():
                continue
            sample = Sample(
                image_path=image_path,
                ann_path=ann_path if ann_path.exists() else None,
                split_tag=split_tag,
            )
            for _ in range(repeat):
                train_samples.append(sample)

    return train_samples, val_samples


def class_mapping(cfg: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_yolo: Dict[str, int] = {}
    names: Dict[int, str] = {}
    for label, cls_id in cfg["classes"].items():
        if label == "background":
            continue
        yolo_id = int(cls_id) - 1
        label_to_yolo[str(label)] = yolo_id
        names[yolo_id] = str(label)
    return label_to_yolo, names


def parse_labelme(
    sample: Sample,
    label_to_yolo: Dict[str, int],
    min_box_size: float,
) -> Tuple[List[str], Dict[str, int]]:
    if sample.ann_path is None or not sample.ann_path.exists():
        return [], {}

    payload = json.loads(sample.ann_path.read_text(encoding="utf-8"))
    width, height = image_size(sample.image_path, payload)
    lines: List[str] = []
    counts: Dict[str, int] = {}

    for shape in payload.get("shapes", []):
        label = shape.get("label")
        if label not in label_to_yolo:
            continue
        box = labelme_box(shape.get("points", []))
        if box is None:
            continue

        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_box_size or bh < min_box_size:
            continue

        cx = (x1 + x2) * 0.5 / float(width)
        cy = (y1 + y2) * 0.5 / float(height)
        nw = bw / float(width)
        nh = bh / float(height)
        cls = label_to_yolo[label]
        lines.append(f"{cls} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
        counts[label] = counts.get(label, 0) + 1

    # Offline real/synthetic copy-paste datasets use competition-format
    # annotations instead of LabelMe shapes.
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
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_box_size or bh < min_box_size:
            continue

        cx = (x1 + x2) * 0.5 / float(width)
        cy = (y1 + y2) * 0.5 / float(height)
        nw = bw / float(width)
        nh = bh / float(height)
        cls = label_to_yolo[label]
        lines.append(f"{cls} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
        counts[label] = counts.get(label, 0) + 1

    return lines, counts


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_split(
    samples: List[Sample],
    split: str,
    output_dir: Path,
    label_to_yolo: Dict[str, int],
    min_box_size: float,
    copy_images: bool,
) -> Dict:
    stats = {
        "images": 0,
        "clean_images": 0,
        "defect_images": 0,
        "boxes": 0,
        "boxes_per_class": {},
    }
    img_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        dst_img = img_dir / sample.image_path.name
        dst_label = label_dir / f"{sample.image_path.stem}.txt"
        link_or_copy(sample.image_path, dst_img, copy_images=copy_images)

        lines, counts = parse_labelme(sample, label_to_yolo, min_box_size)
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats["images"] += 1
        if lines:
            stats["defect_images"] += 1
        else:
            stats["clean_images"] += 1
        stats["boxes"] += len(lines)
        for label, count in counts.items():
            stats["boxes_per_class"][label] = stats["boxes_per_class"].get(label, 0) + count

    return stats


def convert_dataset(config_path: str, output_dir: Optional[str] = None, force: bool = False, copy_images: bool = False) -> Path:
    cfg_path = Path(config_path).resolve()
    base = cfg_path.parent
    cfg = load_yaml(cfg_path)
    yolo_cfg = cfg.get("yolo", {})
    out_dir = Path(output_dir) if output_dir else resolve_path(base, yolo_cfg.get("data_dir", "./yolo_data"))
    out_dir = out_dir.resolve()

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples = build_samples(cfg, base)
    label_to_yolo, names = class_mapping(cfg)
    min_box_size = float(cfg["data"].get("min_box_size", 2.0))

    stats = {
        "train": write_split(train_samples, "train", out_dir, label_to_yolo, min_box_size, copy_images),
        "val": write_split(val_samples, "val", out_dir, label_to_yolo, min_box_size, copy_images),
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
    parser.add_argument("--config", default="./config.yolo26.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    args = parser.parse_args()

    yaml_path = convert_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        force=bool(args.force),
        copy_images=bool(args.copy_images),
    )
    print(f"YOLO data yaml: {yaml_path}")
    print(f"Stats: {yaml_path.parent / 'stats.json'}")


if __name__ == "__main__":
    main()
