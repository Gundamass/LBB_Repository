import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from src.masking import load_mask_for_image, should_keep_box_by_mask
from src.utils import load_config, save_json, zip_submission


VALID_LABELS = {"plain particle", "dirt", "scratch", "collision"}


class JsonPredictionReader:
    def __init__(self, pred_path: Path):
        self.pred_path = pred_path
        self.is_zip = pred_path.suffix.lower() == ".zip"
        self._zip: Optional[zipfile.ZipFile] = None
        self._members: Dict[str, str] = {}
        if self.is_zip:
            self._zip = zipfile.ZipFile(pred_path, "r")
            for member in self._zip.namelist():
                if member.endswith(".json"):
                    self._members[Path(member).name] = member

    def names(self) -> List[str]:
        if self.is_zip:
            return sorted(self._members.keys())
        return sorted(p.name for p in self.pred_path.glob("*.json"))

    def read(self, name: str) -> Dict:
        if self.is_zip:
            if self._zip is None:
                raise RuntimeError("Prediction zip reader already closed.")
            member = self._members[name]
            return json.loads(self._zip.read(member).decode("utf-8"))
        return json.loads((self.pred_path / name).read_text(encoding="utf-8"))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _image_size(test_image_dir: Path, image_name: str) -> Optional[Tuple[int, int]]:
    image_path = test_image_dir / image_name
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return image.size


def _normalize_box(bbox) -> Optional[List[float]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _box_features(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = w / max(h, 1e-6)
    return area, aspect


def _is_square_proposal_tile(box: List[float], args) -> bool:
    area, aspect = _box_features(box)
    return (
        float(args.square_area_min) <= area <= float(args.square_area_max)
        and float(args.square_aspect_min) <= aspect <= float(args.square_aspect_max)
    )


def _calibrate_annotations(payload: Dict, mask, args) -> Tuple[Dict, Dict[str, int]]:
    calibrated = []
    stats = {
        "invalid": 0,
        "mask_rescored": 0,
        "classic_rescored": 0,
        "square_rescored": 0,
        "label_dropped": 0,
        "area_dropped": 0,
    }

    for ann in payload.get("annotations", []):
        label = ann.get("label")
        if label not in VALID_LABELS:
            stats["label_dropped"] += 1
            continue

        box = _normalize_box(ann.get("bbox", []))
        if box is None:
            stats["invalid"] += 1
            continue

        area, _ = _box_features(box)
        if area < float(args.min_area) or area > float(args.max_area):
            stats["area_dropped"] += 1
            continue

        out_ann = dict(ann)
        out_ann["bbox"] = [round(v, 3) for v in box]

        try:
            score = float(out_ann.get("confidence", 1.0))
        except Exception:
            score = 1.0

        if mask is not None:
            keep, _, _ = should_keep_box_by_mask(
                mask,
                box,
                min_coverage=float(args.mask_min_coverage),
                keep_rule=str(args.mask_keep_rule),
            )
            if not keep:
                score *= float(args.mask_factor)
                stats["mask_rescored"] += 1

        # The hybrid classical proposal branch creates many near-0.55 tiled boxes.
        # Penalizing their score improves ranking without removing possible recall.
        if score <= float(args.classic_score_max):
            score *= float(args.classic_factor)
            stats["classic_rescored"] += 1
            if _is_square_proposal_tile(box, args):
                score *= float(args.square_factor)
                stats["square_rescored"] += 1

        out_ann["confidence"] = round(max(0.0, min(1.0, score)), 6)
        calibrated.append(out_ann)

    calibrated.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
    if args.top_per_image is not None and int(args.top_per_image) > 0:
        calibrated = calibrated[: int(args.top_per_image)]

    out_payload = dict(payload)
    out_payload["annotations"] = calibrated
    return out_payload, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.sam2_mask.yaml")
    parser.add_argument("--pred", type=str, required=True, help="Prediction folder or zip")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--zip-name", type=str, default=None)
    parser.add_argument("--no-zip", action="store_true")

    parser.add_argument("--mask-factor", type=float, default=0.50)
    parser.add_argument("--mask-min-coverage", type=float, default=0.01)
    parser.add_argument("--mask-keep-rule", type=str, default="coverage")

    parser.add_argument("--classic-score-max", type=float, default=0.550001)
    parser.add_argument("--classic-factor", type=float, default=0.75)

    parser.add_argument("--square-factor", type=float, default=0.15)
    parser.add_argument("--square-area-min", type=float, default=900.0)
    parser.add_argument("--square-area-max", type=float, default=3500.0)
    parser.add_argument("--square-aspect-min", type=float, default=0.85)
    parser.add_argument("--square-aspect-max", type=float, default=1.18)

    parser.add_argument("--top-per-image", type=int, default=30)
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--max-area", type=float, default=1e18)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = Path(args.config).resolve().parent

    pred_path = Path(args.pred)
    if not pred_path.is_absolute():
        pred_path = (base / pred_path).resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction path not found: {pred_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (base / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(cfg["mask"]["test_dir"])
    test_image_dir = Path(cfg["data"]["test_dir"])
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if not test_image_dir.exists():
        raise FileNotFoundError(f"Test image dir not found: {test_image_dir}")

    reader = JsonPredictionReader(pred_path)
    total_before = 0
    total_after = 0
    missing_masks = 0
    missing_images = 0
    totals = {
        "invalid": 0,
        "mask_rescored": 0,
        "classic_rescored": 0,
        "square_rescored": 0,
        "label_dropped": 0,
        "area_dropped": 0,
    }

    try:
        for name in tqdm(reader.names(), desc="calibrated postprocess"):
            payload = reader.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            size = _image_size(test_image_dir, image_name)
            if size is None:
                missing_images += 1

            mask = load_mask_for_image(mask_dir, image_name, expected_size=size)
            if mask is None:
                missing_masks += 1

            before = len(payload.get("annotations", []))
            out_payload, stats = _calibrate_annotations(payload, mask, args)
            after = len(out_payload.get("annotations", []))

            total_before += before
            total_after += after
            for k, v in stats.items():
                totals[k] += int(v)

            save_json(out_dir / name, out_payload, encoding=cfg["inference"].get("json_encoding", "utf-8"))
    finally:
        reader.close()

    print(f"Prediction input: {pred_path}")
    print(f"Output dir: {out_dir}")
    print(f"Missing masks: {missing_masks}")
    print(f"Missing images: {missing_images}")
    print(f"Annotations before/after: {total_before}/{total_after}")
    for k, v in totals.items():
        print(f"{k}: {v}")

    if not args.no_zip:
        zip_name = args.zip_name or f"{out_dir.name}.zip"
        zip_path = out_dir.parent / zip_name
        zip_submission(str(out_dir), str(zip_path))
        print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
