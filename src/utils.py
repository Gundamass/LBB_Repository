import json
import logging
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass
class HardwareInfo:
    device: str
    gpu_count: int
    gpu_names: List[str]
    total_vram_gb: float


def load_config(config_path: str) -> Dict:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = path.parent
    path_keys = [
        ("data", "root_dir"),
        ("data", "train_pos_dir"),
        ("data", "train_neg_dir"),
        ("data", "test_dir"),
        ("inference", "output_dir"),
        ("system", "checkpoints_dir"),
        ("system", "logs_dir"),
        ("mask", "train_dir"),
        ("mask", "test_dir"),
        ("sam2", "repo_dir"),
        ("sam2", "checkpoint"),
    ]

    for sec, key in path_keys:
        val = cfg.get(sec, {}).get(key)
        if val is None:
            continue
        p = Path(val)
        if not p.is_absolute():
            cfg[sec][key] = str((base / p).resolve())

    gen_val = cfg.get("mask", {}).get("generator", {}).get("output_dir")
    if gen_val is not None:
        p = Path(gen_val)
        if not p.is_absolute():
            cfg["mask"]["generator"]["output_dir"] = str((base / p).resolve())

    return cfg


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def setup_logger(logs_dir: str, name: str = "lbb") -> logging.Logger:
    ensure_dirs([logs_dir])
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(logs_dir) / f"{name}_{ts}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logger initialized: %s", log_path)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def detect_hardware() -> HardwareInfo:
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            total = 0.0
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                total += props.total_memory / (1024**3)
            return HardwareInfo(
                device="cuda",
                gpu_count=n,
                gpu_names=names,
                total_vram_gb=round(total, 2),
            )
    except Exception:
        pass

    return HardwareInfo(device="cpu", gpu_count=0, gpu_names=[], total_vram_gb=0.0)


def auto_tune_loader_params(cfg: Dict, hw: HardwareInfo) -> Tuple[int, int]:
    cpu_count = os.cpu_count() or 4

    batch = int(cfg["training"].get("batch_size", 4))
    workers = int(cfg["training"].get("num_workers", 4))

    if hw.device == "cpu":
        batch = min(batch, 2)
        workers = min(max(cpu_count // 2, 1), 4)
    else:
        if hw.total_vram_gb < 8:
            batch = min(batch, 2)
        elif hw.total_vram_gb < 20:
            batch = min(batch, 4)
        else:
            batch = min(batch, 8)
        workers = min(max(cpu_count - 2, 2), 12)

    return batch, workers


def save_json(path: str, payload: Dict, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding=encoding, newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def zip_submission(folder_path: str, zip_path: str) -> str:
    folder = Path(folder_path)
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in folder.rglob("*.json"):
            zf.write(fp, arcname=str(fp.relative_to(folder.parent)))

    return str(target)


def copy_best_as_latest(best_ckpt: str, latest_ckpt: str) -> None:
    src = Path(best_ckpt)
    dst = Path(latest_ckpt)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default
