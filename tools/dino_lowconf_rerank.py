import argparse
import importlib
import json
import math
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


VALID_LABELS = ["plain particle", "dirt", "scratch", "collision"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class PredictionStore:
    def __init__(self, path: Path):
        self.path = path
        self.zip: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, str] = {}
        if path.suffix.lower() == ".zip":
            self.zip = zipfile.ZipFile(path, "r")
            for member in self.zip.namelist():
                if member.endswith(".json"):
                    self.members[Path(member).name] = member

    def names(self) -> List[str]:
        if self.zip is not None:
            return sorted(self.members)
        return sorted(p.name for p in self.path.glob("*.json"))

    def read(self, name: str) -> Dict:
        if self.zip is not None:
            return json.loads(self.zip.read(self.members[name]).decode("utf-8"))
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


def find_images(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def normalize_box(box: Iterable[float], width: int, height: int) -> Optional[List[float]]:
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


def expand_box(box: List[float], width: int, height: int, factor: float) -> List[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * factor
    bh = (y2 - y1) * factor
    return [
        max(0.0, cx - bw * 0.5),
        max(0.0, cy - bh * 0.5),
        min(float(width - 1), cx + bw * 0.5),
        min(float(height - 1), cy + bh * 0.5),
    ]


def load_support_boxes(support_dirs: Iterable[Path]) -> List[Tuple[Path, str, List[float]]]:
    rows: List[Tuple[Path, str, List[float]]] = []
    for root in support_dirs:
        if not root.exists():
            continue
        for ann_path in root.rglob("*.json"):
            image_path = ann_path.with_suffix(".jpg")
            if not image_path.exists():
                continue
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for shape in data.get("shapes", []):
                label = shape.get("label")
                pts = shape.get("points", [])
                if label not in VALID_LABELS or len(pts) < 2:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                rows.append((image_path, label, [min(xs), min(ys), max(xs), max(ys)]))
            for ann in data.get("annotations", []):
                label = ann.get("label")
                bbox = ann.get("bbox", [])
                if label in VALID_LABELS and len(bbox) == 4:
                    rows.append((image_path, label, [float(v) for v in bbox]))
    return rows


def build_dinov3(repo_dir: Path, model_name: str, weights: Path, device: torch.device):
    repo = str(repo_dir.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    backbones = importlib.import_module("dinov3.hub.backbones")
    if not hasattr(backbones, model_name):
        raise ValueError(f"DINOv3 model not found: {model_name}")
    ctor = getattr(backbones, model_name)
    model = ctor(pretrained=True, weights=str(weights), check_hash=False)
    model.eval().to(device)
    return model


def crop_to_tensor(image: Image.Image, box: List[float], input_size: int, expand: float) -> Optional[torch.Tensor]:
    width, height = image.size
    box = normalize_box(box, width, height)
    if box is None:
        return None
    box = expand_box(box, width, height, expand)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    patch = image.crop(tuple(box)).resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def embed_patches(model, patches: List[torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
    if not patches:
        return torch.empty((0, 0), dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    feats = []
    for start in range(0, len(patches), batch_size):
        x = torch.stack(patches[start : start + batch_size]).to(device, non_blocking=True)
        x = (x - mean) / std
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(x, is_training=True)
        feat = out["x_norm_clstoken"] if isinstance(out, dict) else out
        feats.append(F.normalize(feat.float(), dim=-1).detach().cpu())
    return torch.cat(feats, dim=0)


def build_prototypes(model, support_rows: List[Tuple[Path, str, List[float]]], args, device: torch.device) -> Dict[str, torch.Tensor]:
    by_image: Dict[Path, List[Tuple[str, List[float]]]] = defaultdict(list)
    for image_path, label, box in support_rows:
        by_image[image_path].append((label, box))

    feats_by_label: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for image_path, items in tqdm(sorted(by_image.items()), desc="DINO support prototypes"):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            patches = []
            labels = []
            for label, box in items:
                patch = crop_to_tensor(image, box, int(args.input_size), float(args.crop_expand))
                if patch is not None:
                    patches.append(patch)
                    labels.append(label)
        feats = embed_patches(model, patches, device, int(args.batch_size))
        for feat, label in zip(feats, labels):
            feats_by_label[label].append(feat)

    prototypes: Dict[str, torch.Tensor] = {}
    for label, feats in feats_by_label.items():
        stacked = torch.stack(feats)
        prototypes[label] = F.normalize(stacked.mean(dim=0), dim=0)
    return prototypes


def sim_to_score(sim: float, sim_min: float, sim_max: float) -> float:
    if sim_max <= sim_min + 1e-9:
        return 0.5
    return max(0.0, min(1.0, (sim - sim_min) / (sim_max - sim_min)))


def apply_dino_scores(payload: Dict, sims: Dict[int, float], args) -> Dict:
    anns = list(payload.get("annotations", []))
    for idx, sim in sims.items():
        if idx < 0 or idx >= len(anns):
            continue
        try:
            old_score = float(anns[idx].get("confidence", 0.0) or 0.0)
        except Exception:
            old_score = 0.0
        old_norm = (old_score - float(args.min_score)) / max(float(args.low_score_max) - float(args.min_score), 1e-12)
        old_norm = max(0.0, min(1.0, old_norm))
        dino_norm = sim_to_score(float(sim), float(args.sim_min), float(args.sim_max))
        fused = (1.0 - float(args.dino_alpha)) * old_norm + float(args.dino_alpha) * dino_norm
        new_score = float(args.min_score) + fused * (float(args.low_score_ceiling) - float(args.min_score))
        anns[idx]["confidence"] = round(max(0.0, min(float(args.low_score_ceiling), new_score)), 8)

    anns.sort(key=lambda a: float(a.get("confidence", 0.0) or 0.0), reverse=True)
    if int(args.top_per_image) > 0:
        anns = anns[: int(args.top_per_image)]
    out = dict(payload)
    out["annotations"] = anns
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--support-dirs", nargs="*", default=["./初赛数据/训练集/负样本"])
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--repo-dir", default="./third_party/dinov3")
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--weights", default="./dinov3/dinov3_vitl16_from_safetensors-8aa4cbdd.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-expand", type=float, default=1.35)
    parser.add_argument("--min-score", type=float, default=0.00005)
    parser.add_argument("--low-score-max", type=float, default=0.00015)
    parser.add_argument("--low-score-ceiling", type=float, default=0.00015)
    parser.add_argument("--dino-alpha", type=float, default=0.60)
    parser.add_argument("--sim-min", type=float, default=0.45)
    parser.add_argument("--sim-max", type=float, default=0.80)
    parser.add_argument("--top-per-image", type=int, default=400)
    parser.add_argument("--max-candidates-per-image", type=int, default=0)
    args = parser.parse_args()

    pred_path = Path(args.pred).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    image_dir = Path(args.test_image_dir).resolve()
    support_dirs = [Path(p).resolve() for p in args.support_dirs]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = build_dinov3(Path(args.repo_dir), str(args.model_name), Path(args.weights), device)

    support_rows = load_support_boxes(support_dirs)
    prototypes = build_prototypes(model, support_rows, args, device)
    print("Support boxes:", len(support_rows))
    for label in VALID_LABELS:
        print(f"  prototype[{label}]:", "yes" if label in prototypes else "missing")
    if not prototypes:
        raise RuntimeError("No DINO prototypes were built.")

    images = find_images(image_dir)
    store = PredictionStore(pred_path)
    total_before = 0
    total_after = 0
    total_candidates = 0
    sim_values: List[float] = []
    missing_images = 0

    try:
        for name in tqdm(store.names(), desc="DINO low-conf rerank"):
            payload = store.read(name)
            anns = list(payload.get("annotations", []))
            total_before += len(anns)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = images.get(image_name)
            sims: Dict[int, float] = {}
            if image_path is None:
                missing_images += 1
            else:
                candidate_indices = []
                for idx, ann in enumerate(anns):
                    label = ann.get("label")
                    if label not in prototypes:
                        continue
                    try:
                        score = float(ann.get("confidence", 0.0) or 0.0)
                    except Exception:
                        score = 0.0
                    if score < float(args.min_score) or score > float(args.low_score_max):
                        continue
                    if len(ann.get("bbox", [])) != 4:
                        continue
                    candidate_indices.append(idx)

                if int(args.max_candidates_per_image) > 0 and len(candidate_indices) > int(args.max_candidates_per_image):
                    candidate_indices.sort(
                        key=lambda i: float(anns[i].get("confidence", 0.0) or 0.0),
                        reverse=True,
                    )
                    candidate_indices = candidate_indices[: int(args.max_candidates_per_image)]

                if candidate_indices:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        patches = []
                        kept = []
                        for idx in candidate_indices:
                            patch = crop_to_tensor(
                                image,
                                anns[idx].get("bbox", []),
                                int(args.input_size),
                                float(args.crop_expand),
                            )
                            if patch is not None:
                                patches.append(patch)
                                kept.append(idx)
                    feats = embed_patches(model, patches, device, int(args.batch_size))
                    for feat, idx in zip(feats, kept):
                        label = anns[idx].get("label")
                        sim = float((feat * prototypes[label]).sum().item())
                        sims[idx] = sim
                        sim_values.append(sim)
                    total_candidates += len(sims)

            payload = dict(payload)
            payload["annotations"] = anns
            out_payload = apply_dino_scores(payload, sims, args)
            total_after += len(out_payload.get("annotations", []))
            (out_dir / name).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Input: {pred_path}")
    print(f"Output folder: {out_dir}")
    print(f"Output zip: {out_zip}")
    print(f"Missing images: {missing_images}")
    print(f"Annotations before/after: {total_before}/{total_after}")
    print(f"DINO candidates reranked: {total_candidates}")
    if sim_values:
        arr = np.asarray(sim_values, dtype=np.float32)
        print(
            "DINO sim stats:",
            f"min={arr.min():.4f}",
            f"p50={np.percentile(arr, 50):.4f}",
            f"p90={np.percentile(arr, 90):.4f}",
            f"max={arr.max():.4f}",
        )


if __name__ == "__main__":
    main()
