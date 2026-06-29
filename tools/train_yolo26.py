import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from tools.convert_labelme_to_yolo import convert_dataset
except Exception:
    from convert_labelme_to_yolo import convert_dataset

try:
    from tools.make_tile_yolo_dataset import convert_tile_dataset
except Exception:
    try:
        from make_tile_yolo_dataset import convert_tile_dataset
    except Exception:
        convert_tile_dataset = None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yolo26.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--save-period", type=int, default=None)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base = cfg_path.parent
    cfg = load_yaml(cfg_path)
    yolo_cfg = cfg.get("yolo", {})

    data_dir = resolve_path(base, yolo_cfg.get("data_dir", "./yolo_data"))
    data_yaml = data_dir / "lbb.yaml"
    if args.force_convert or not data_yaml.exists():
        if "tile" in cfg:
            if convert_tile_dataset is None:
                raise RuntimeError("Tile config detected, but tools/make_tile_yolo_dataset.py could not be imported.")
            tile_cfg = cfg.get("tile", {})
            data_yaml = convert_tile_dataset(
                config_path=str(cfg_path),
                output_dir=str(data_dir),
                tile_size=int(tile_cfg.get("size", 640)),
                overlap=float(tile_cfg.get("overlap", 0.35)),
                min_visible=float(tile_cfg.get("min_visible", 0.25)),
                positive_jitters=int(tile_cfg.get("positive_jitters", 2)),
                rare_extra_jitters={str(k): int(v) for k, v in tile_cfg.get("rare_extra_jitters", {}).items()},
                force=bool(args.force_convert),
                jpeg_quality=int(tile_cfg.get("jpeg_quality", 95)),
            )
        else:
            data_yaml = convert_dataset(
                config_path=str(cfg_path),
                output_dir=str(data_dir),
                force=bool(args.force_convert),
                copy_images=bool(args.copy_images),
            )

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not installed in this environment. "
            "Install it with: python -m pip install -U ultralytics"
        ) from exc

    model_name = args.model or yolo_cfg.get("model", "yolo26s.pt")
    model = YOLO(model_name)

    train_kwargs = {
        "data": str(data_yaml),
        "imgsz": int(args.imgsz or yolo_cfg.get("imgsz", 1536)),
        "epochs": int(args.epochs or yolo_cfg.get("epochs", 200)),
        "batch": int(args.batch or yolo_cfg.get("batch", 8)),
        "patience": int(yolo_cfg.get("patience", 40)),
        "workers": int(yolo_cfg.get("workers", 8)),
        "project": str(resolve_path(base, args.project or yolo_cfg.get("project", "./runs/yolo26"))),
        "name": str(args.name or yolo_cfg.get("name", "yolo26s_img1536")),
        "exist_ok": bool(yolo_cfg.get("exist_ok", True)),
        "plots": bool(yolo_cfg.get("plots", True)),
        "cache": yolo_cfg.get("cache", False),
        "seed": int(cfg.get("seed", 42)),
    }
    save_period = args.save_period if args.save_period is not None else yolo_cfg.get("save_period", None)
    if save_period is not None:
        train_kwargs["save_period"] = int(save_period)
    device = args.device if args.device is not None else yolo_cfg.get("device", None)
    if device is not None:
        train_kwargs["device"] = str(device)

    results = model.train(**train_kwargs)
    print(results)


if __name__ == "__main__":
    main()
