import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def iter_test_images(test_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in sorted(test_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def id_to_label(cfg: Dict[str, Any]) -> Dict[int, str]:
    mapping = {}
    for label, cls_id in cfg["classes"].items():
        if label == "background":
            continue
        mapping[int(cls_id) - 1] = str(label)
    return mapping


def zip_submission(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yolo26.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--zip-name", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base = cfg_path.parent
    cfg = load_yaml(cfg_path)
    yolo_cfg = cfg.get("yolo", {})
    infer_cfg = cfg.get("inference", {})

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not installed in this environment. "
            "Install it with: python -m pip install -U ultralytics"
        ) from exc

    test_dir = resolve_path(base, cfg["data"]["test_dir"])
    output_dir = resolve_path(base, infer_cfg.get("output_dir", "./outputs"))
    zip_name = args.zip_name or infer_cfg.get("zip_name", "yolo26_submission.zip")
    if not zip_name.endswith(".zip"):
        zip_name += ".zip"
    folder_name = zip_name[:-4]
    submit_dir = output_dir / folder_name
    if submit_dir.exists():
        shutil.rmtree(submit_dir)
    submit_dir.mkdir(parents=True, exist_ok=True)

    labels = id_to_label(cfg)
    model = YOLO(str(args.weights))

    pred_kwargs = {
        "source": str(test_dir),
        "imgsz": int(args.imgsz or yolo_cfg.get("imgsz", 1536)),
        "conf": float(args.conf if args.conf is not None else yolo_cfg.get("conf", 0.001)),
        "iou": float(args.iou if args.iou is not None else yolo_cfg.get("iou", 0.5)),
        "max_det": int(args.max_det or yolo_cfg.get("max_det", 500)),
        "stream": True,
        "verbose": False,
        "save": False,
    }
    if args.device is not None or yolo_cfg.get("device") is not None:
        pred_kwargs["device"] = str(args.device if args.device is not None else yolo_cfg.get("device"))
    if args.batch is not None or yolo_cfg.get("predict_batch") is not None:
        pred_kwargs["batch"] = int(args.batch or yolo_cfg.get("predict_batch"))
    if args.augment or bool(yolo_cfg.get("augment", False)):
        pred_kwargs["augment"] = True

    seen = set()
    for result in model.predict(**pred_kwargs):
        image_path = Path(result.path)
        seen.add(image_path.name)
        annos = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            clss = result.boxes.cls.detach().cpu().tolist()
            rows = sorted(zip(boxes, confs, clss), key=lambda x: float(x[1]), reverse=True)
            for box, score, cls_id in rows:
                label = labels.get(int(cls_id))
                if label is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                if x2 <= x1 or y2 <= y1:
                    continue
                annos.append(
                    {
                        "label": label,
                        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                        "confidence": round(float(score), 6),
                    }
                )

        payload = {"image_id": image_path.name, "annotations": annos}
        (submit_dir / f"{image_path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Ensure the submission has one JSON per test image even if predict skipped an image.
    for image_path in iter_test_images(test_dir):
        out_path = submit_dir / f"{image_path.stem}.json"
        if not out_path.exists():
            payload = {"image_id": image_path.name, "annotations": []}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = output_dir / zip_name
    zip_submission(submit_dir, zip_path)
    print(f"Prediction folder: {submit_dir}")
    print(f"Submission zip: {zip_path}")
    print(f"JSON count: {len(list(submit_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
