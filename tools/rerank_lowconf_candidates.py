import argparse
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


VALID_LABELS = {"plain particle", "dirt", "scratch", "collision"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class PredictionStore:
    def __init__(self, path: Path):
        self.path = path
        self.zip: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, str] = {}
        if path.suffix.lower() == ".zip":
            self.zip = zipfile.ZipFile(path, "r")
            for name in self.zip.namelist():
                if name.endswith(".json"):
                    self.members[Path(name).name] = name

    def names(self) -> List[str]:
        if self.zip is not None:
            return sorted(self.members)
        return sorted(p.name for p in self.path.glob("*.json"))

    def read(self, name: str) -> Dict:
        if self.zip is not None:
            member = self.members[name]
            return json.loads(self.zip.read(member).decode("utf-8"))
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


def find_images(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def zip_submission(folder: Path, zip_path: Path) -> None:
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


def integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def rect_sum(ii: np.ndarray, box: List[float]) -> float:
    h = ii.shape[0] - 1
    w = ii.shape[1] - 1
    x1 = int(max(0, min(w, math.floor(box[0]))))
    y1 = int(max(0, min(h, math.floor(box[1]))))
    x2 = int(max(0, min(w, math.ceil(box[2]))))
    y2 = int(max(0, min(h, math.ceil(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])


def rect_area(box: List[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


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


def robust01(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi <= lo + 1e-9:
        return [0.5] * len(values)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out.tolist()


def load_labelme_areas(paths: Iterable[Path]) -> Dict[str, List[float]]:
    areas: Dict[str, List[float]] = defaultdict(list)
    for root in paths:
        if not root.exists():
            continue
        for ann_path in root.rglob("*.json"):
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label not in VALID_LABELS:
                    continue
                pts = shape.get("points", [])
                if len(pts) < 2:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
                if area > 0:
                    areas[label].append(area)
            for ann in data.get("annotations", []):
                label = ann.get("label")
                if label not in VALID_LABELS:
                    continue
                box = ann.get("bbox", [])
                if len(box) != 4:
                    continue
                area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
                if area > 0:
                    areas[label].append(area)
    return areas


def build_area_priors(paths: Iterable[Path]) -> Dict[str, Tuple[float, float]]:
    areas = load_labelme_areas(paths)
    priors: Dict[str, Tuple[float, float]] = {}
    for label, vals in areas.items():
        if len(vals) < 4:
            continue
        logs = np.log(np.asarray(vals, dtype=np.float32) + 1.0)
        med = float(np.median(logs))
        mad = float(np.median(np.abs(logs - med)))
        priors[label] = (med, max(mad * 1.4826, 0.35))
    return priors


def area_prior_score(label: str, area: float, priors: Dict[str, Tuple[float, float]], scale: float) -> float:
    prior = priors.get(label)
    if prior is None or area <= 0:
        return 0.5
    med, sigma = prior
    z = (math.log(area + 1.0) - med) / max(sigma * scale, 1e-6)
    return float(math.exp(-0.5 * z * z))


def highpass_map(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    blur = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + 4.0 * gray
    ) / 8.0
    return np.abs(gray - blur).astype(np.float32)


def visual_features_for_image(image_path: Path, anns: List[Dict], args, priors: Dict[str, Tuple[float, float]]) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        arr = np.asarray(image, dtype=np.float32) / 255.0

    gray = arr.mean(axis=2).astype(np.float32)
    hp = highpass_map(gray)
    gray_ii = integral_image(gray)
    gray2_ii = integral_image(gray * gray)
    hp_ii = integral_image(hp)

    raw_hf: List[float] = []
    raw_std: List[float] = []
    raw_delta: List[float] = []
    raw_small: List[float] = []
    raw_prior: List[float] = []
    targets: List[int] = []

    for idx, ann in enumerate(anns):
        try:
            score = float(ann.get("confidence", 0.0))
        except Exception:
            score = 0.0
        if score > float(args.low_score_max):
            continue

        label = ann.get("label")
        if label not in VALID_LABELS:
            continue
        box = normalize_box(ann.get("bbox", []), width=width, height=height)
        if box is None:
            continue

        area = max(1.0, rect_area(box))
        mean = rect_sum(gray_ii, box) / area
        mean2 = rect_sum(gray2_ii, box) / area
        std = math.sqrt(max(0.0, mean2 - mean * mean))
        hf = rect_sum(hp_ii, box) / area

        outer = expand_box(box, width, height, float(args.context_expand))
        outer_area = max(1.0, rect_area(outer))
        ring_area = max(1.0, outer_area - area)
        outer_sum = rect_sum(gray_ii, outer)
        inner_sum = rect_sum(gray_ii, box)
        ring_mean = (outer_sum - inner_sum) / ring_area
        delta = abs(mean - ring_mean)

        # Defect boxes are usually small relative to the whole product image.
        # This is only a soft rank feature; scratches are allowed to be larger.
        img_area = float(width * height)
        small = 1.0 / (1.0 + area / max(float(args.small_area_scale) * img_area, 1.0))
        prior = area_prior_score(label, area, priors, float(args.area_prior_scale))

        raw_hf.append(float(hf))
        raw_std.append(float(std))
        raw_delta.append(float(delta))
        raw_small.append(float(small))
        raw_prior.append(float(prior))
        targets.append(idx)

    if not targets:
        return

    hf01 = robust01(raw_hf)
    std01 = robust01(raw_std)
    delta01 = robust01(raw_delta)
    small01 = robust01(raw_small)

    for local_i, ann_idx in enumerate(targets):
        q = (
            float(args.hf_weight) * hf01[local_i]
            + float(args.std_weight) * std01[local_i]
            + float(args.context_weight) * delta01[local_i]
            + float(args.small_weight) * small01[local_i]
            + float(args.area_prior_weight) * raw_prior[local_i]
        )
        denom = (
            float(args.hf_weight)
            + float(args.std_weight)
            + float(args.context_weight)
            + float(args.small_weight)
            + float(args.area_prior_weight)
        )
        q = q / max(denom, 1e-9)
        anns[ann_idx]["_visual_quality"] = max(0.0, min(1.0, q))


def rerank_payload(payload: Dict, args) -> Dict:
    anns = list(payload.get("annotations", []))
    low_indices = []
    low_scores = []
    for i, ann in enumerate(anns):
        try:
            score = float(ann.get("confidence", 0.0))
        except Exception:
            score = 0.0
        if score <= float(args.low_score_max):
            low_indices.append(i)
            low_scores.append(max(0.0, score))

    if low_indices:
        max_low = max(max(low_scores), float(args.low_score_floor))
        for idx, old_score in zip(low_indices, low_scores):
            q = float(anns[idx].pop("_visual_quality", 0.0))
            old_norm = min(1.0, old_score / max(max_low, 1e-12))
            fused = (1.0 - float(args.visual_alpha)) * old_norm + float(args.visual_alpha) * q
            new_score = float(args.low_score_floor) + fused * (float(args.low_score_ceiling) - float(args.low_score_floor))
            # Keep this branch below normal detector confidence. We only reorder
            # the low-score recall pool rather than challenging high-confidence detections.
            anns[idx]["confidence"] = round(max(0.0, min(float(args.low_score_ceiling), new_score)), 8)

    for ann in anns:
        ann.pop("_visual_quality", None)

    anns.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
    if int(args.top_per_image) > 0:
        anns = anns[: int(args.top_per_image)]
    out = dict(payload)
    out["annotations"] = anns
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Input prediction folder or zip")
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--support-dirs", nargs="*", default=["./初赛数据/训练集/负样本"])
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--top-per-image", type=int, default=400)
    parser.add_argument("--low-score-max", type=float, default=0.00015)
    parser.add_argument("--low-score-floor", type=float, default=0.0)
    parser.add_argument("--low-score-ceiling", type=float, default=0.00015)
    parser.add_argument("--visual-alpha", type=float, default=0.70)
    parser.add_argument("--context-expand", type=float, default=2.5)
    parser.add_argument("--small-area-scale", type=float, default=0.02)
    parser.add_argument("--area-prior-scale", type=float, default=2.5)
    parser.add_argument("--hf-weight", type=float, default=0.35)
    parser.add_argument("--std-weight", type=float, default=0.25)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--small-weight", type=float, default=0.05)
    parser.add_argument("--area-prior-weight", type=float, default=0.10)
    args = parser.parse_args()

    pred_path = Path(args.pred).resolve()
    image_dir = Path(args.test_image_dir).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Test image dir not found: {image_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priors = build_area_priors([Path(p).resolve() for p in args.support_dirs])
    print("Area priors:")
    for label, (med, sigma) in sorted(priors.items()):
        print(f"  {label}: log_area_med={med:.3f} sigma={sigma:.3f}")

    images = find_images(image_dir)
    store = PredictionStore(pred_path)
    total_before = 0
    total_after = 0
    touched = 0
    missing_images = 0
    try:
        for name in tqdm(store.names(), desc="rerank low-conf candidates"):
            payload = store.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            anns = list(payload.get("annotations", []))
            total_before += len(anns)
            image_path = images.get(image_name)
            if image_path is None:
                missing_images += 1
            else:
                before_touched = sum(
                    1
                    for ann in anns
                    if float(ann.get("confidence", 0.0) or 0.0) <= float(args.low_score_max)
                )
                visual_features_for_image(image_path, anns, args, priors)
                touched += before_touched
            payload = dict(payload)
            payload["annotations"] = anns
            out_payload = rerank_payload(payload, args)
            total_after += len(out_payload.get("annotations", []))
            (out_dir / name).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    zip_submission(out_dir, out_zip)
    print(f"Input: {pred_path}")
    print(f"Output folder: {out_dir}")
    print(f"Output zip: {out_zip}")
    print(f"Missing images: {missing_images}")
    print(f"Annotations before/after: {total_before}/{total_after}")
    print(f"Low-score annotations touched: {touched}")


if __name__ == "__main__":
    main()
