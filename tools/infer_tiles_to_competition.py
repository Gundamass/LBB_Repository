import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from PIL import Image

try:
    from tools.make_tile_yolo_dataset import grid_windows
except Exception:
    from make_tile_yolo_dataset import grid_windows


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
Window = Tuple[int, int, int, int]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def iter_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def id_to_label(cfg: Dict[str, Any]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for label, cls_id in cfg["classes"].items():
        if label == "background":
            continue
        mapping[int(cls_id) - 1] = str(label)
    return mapping


def bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def classwise_nms(annotations: List[Dict], iou_thr: float, topk: int) -> List[Dict]:
    kept: List[Dict] = []
    labels = sorted({a.get("label") for a in annotations})
    for label in labels:
        rows = [a for a in annotations if a.get("label") == label]
        rows.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
        label_kept: List[Dict] = []
        for ann in rows:
            if all(bbox_iou(ann["bbox"], old["bbox"]) < iou_thr for old in label_kept):
                label_kept.append(ann)
        kept.extend(label_kept)
    kept.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
    if topk > 0:
        kept = kept[:topk]
    return kept


def zip_submission(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def prepare_tiles(
    test_images: Iterable[Path],
    tile_dir: Path,
    tile_size: int,
    overlap: float,
    force: bool,
    jpeg_quality: int,
) -> Dict[str, Dict[str, Any]]:
    if tile_dir.exists() and force:
        shutil.rmtree(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)

    meta_path = tile_dir.parent / "metadata.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    metadata: Dict[str, Dict[str, Any]] = {}
    for image_path in test_images:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            width, height = int(im.width), int(im.height)
            windows = grid_windows(width, height, tile_size, overlap)
            for idx, window in enumerate(windows):
                x1, y1, x2, y2 = window
                tile_name = f"{image_path.stem}__x{x1}_y{y1}_x{x2}_y{y2}_{idx:03d}.jpg"
                out_path = tile_dir / tile_name
                if force or not out_path.exists():
                    crop = im.crop(window)
                    crop.save(out_path, quality=jpeg_quality)
                metadata[tile_name] = {
                    "image_name": image_path.name,
                    "image_stem": image_path.stem,
                    "width": width,
                    "height": height,
                    "window": [int(x1), int(y1), int(x2), int(y2)],
                }

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.tile_yolo26.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--global-nms-iou", type=float, default=None)
    parser.add_argument("--topk-final", type=int, default=None)
    parser.add_argument("--zip-name", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base = cfg_path.parent
    cfg = load_yaml(cfg_path)
    yolo_cfg = cfg.get("yolo", {})
    tile_cfg = cfg.get("tile", {})
    infer_cfg = cfg.get("inference", {})

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required: python -m pip install -U ultralytics") from exc

    test_dir = resolve_path(base, cfg["data"]["test_dir"])
    output_dir = resolve_path(base, infer_cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = args.zip_name or infer_cfg.get("zip_name", "tile_yolo_submission.zip")
    if not zip_name.endswith(".zip"):
        zip_name += ".zip"
    folder_name = zip_name[:-4]
    submit_dir = output_dir / folder_name
    if submit_dir.exists():
        shutil.rmtree(submit_dir)
    submit_dir.mkdir(parents=True, exist_ok=True)

    tile_size = int(args.tile_size or tile_cfg.get("size", 640))
    overlap = float(args.overlap if args.overlap is not None else tile_cfg.get("overlap", 0.35))
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / "_tile_cache" / folder_name
    cache_root = cache_root.resolve()
    tile_dir = cache_root / "images"

    test_images = iter_images(test_dir)
    metadata = prepare_tiles(
        test_images=test_images,
        tile_dir=tile_dir,
        tile_size=tile_size,
        overlap=overlap,
        force=bool(args.force_cache),
        jpeg_quality=int(args.jpeg_quality),
    )

    labels = id_to_label(cfg)
    by_image: Dict[str, List[Dict]] = {p.name: [] for p in test_images}
    model = YOLO(str(args.weights))

    pred_kwargs = {
        "source": str(tile_dir),
        "imgsz": int(args.imgsz or yolo_cfg.get("imgsz", 1024)),
        "conf": float(args.conf if args.conf is not None else yolo_cfg.get("conf", 0.001)),
        "iou": float(args.iou if args.iou is not None else yolo_cfg.get("iou", 0.5)),
        "max_det": int(args.max_det or yolo_cfg.get("max_det", 300)),
        "stream": True,
        "verbose": False,
        "save": False,
    }
    if args.device is not None or yolo_cfg.get("device") is not None:
        pred_kwargs["device"] = str(args.device if args.device is not None else yolo_cfg.get("device"))
    if args.batch is not None or yolo_cfg.get("predict_batch") is not None:
        pred_kwargs["batch"] = int(args.batch or yolo_cfg.get("predict_batch"))

    for result in model.predict(**pred_kwargs):
        tile_name = Path(result.path).name
        meta = metadata.get(tile_name)
        if meta is None:
            continue
        wx1, wy1, wx2, wy2 = [float(v) for v in meta["window"]]
        width = float(meta["width"])
        height = float(meta["height"])
        image_name = str(meta["image_name"])

        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        confs = result.boxes.conf.detach().cpu().tolist()
        clss = result.boxes.cls.detach().cpu().tolist()
        for box, score, cls_id in zip(boxes, confs, clss):
            label = labels.get(int(cls_id))
            if label is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            gx1 = max(0.0, min(width, x1 + wx1))
            gy1 = max(0.0, min(height, y1 + wy1))
            gx2 = max(0.0, min(width, x2 + wx1))
            gy2 = max(0.0, min(height, y2 + wy1))
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            by_image.setdefault(image_name, []).append(
                {
                    "label": label,
                    "bbox": [round(gx1, 3), round(gy1, 3), round(gx2, 3), round(gy2, 3)],
                    "confidence": round(float(score), 6),
                }
            )

    global_nms_iou = float(
        args.global_nms_iou if args.global_nms_iou is not None else tile_cfg.get("global_nms_iou", 0.5)
    )
    topk_final = int(args.topk_final if args.topk_final is not None else tile_cfg.get("topk_final", 500))

    for image_path in test_images:
        anns = by_image.get(image_path.name, [])
        anns = classwise_nms(anns, iou_thr=global_nms_iou, topk=topk_final)
        payload = {"image_id": image_path.name, "annotations": anns}
        (submit_dir / f"{image_path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    zip_path = output_dir / zip_name
    zip_submission(submit_dir, zip_path)
    print(f"Tile cache: {cache_root}")
    print(f"Prediction folder: {submit_dir}")
    print(f"Submission zip: {zip_path}")
    print(f"JSON count: {len(list(submit_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
