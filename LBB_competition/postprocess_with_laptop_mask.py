import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image
from tqdm import tqdm

from src.masking import load_mask_for_image, should_keep_box_by_mask
from src.utils import load_config, save_json, zip_submission


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


def _image_size(test_image_dir: Path, image_name: str):
    image_path = test_image_dir / image_name
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return image.size


def _filter_payload(payload: Dict, mask, cfg: Dict):
    infer_cfg = cfg.get("mask", {}).get("inference", {})
    min_coverage = float(infer_cfg.get("min_box_coverage", 0.10))
    keep_rule = str(infer_cfg.get("keep_rule", "center_or_coverage"))
    mode = str(infer_cfg.get("mode", "filter")).lower()
    background_score_factor = float(infer_cfg.get("background_score_factor", 0.15))
    min_score_after_rescore = float(infer_cfg.get("min_score_after_rescore", 0.0))

    kept = []
    removed = 0
    rescored = 0
    for ann in payload.get("annotations", []):
        bbox = ann.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            removed += 1
            continue

        keep, coverage, center_inside = should_keep_box_by_mask(
            mask,
            bbox,
            min_coverage=min_coverage,
            keep_rule=keep_rule,
        )

        if mode == "rescore":
            out_ann = dict(ann)
            if not keep:
                old_score = float(out_ann.get("confidence", 1.0))
                out_ann["confidence"] = round(old_score * background_score_factor, 6)
                rescored += 1
            if float(out_ann.get("confidence", 1.0)) >= min_score_after_rescore:
                kept.append(out_ann)
            else:
                removed += 1
            continue

        if keep:
            kept.append(ann)
        else:
            removed += 1

    payload = dict(payload)
    payload["annotations"] = kept
    return payload, removed, rescored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.mask.yaml")
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Existing prediction folder or zip. Defaults to the config zip output.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--zip-name", type=str, default=None)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_path = Path(args.config).resolve()
    base = cfg_path.parent

    if args.pred is None:
        zip_name = cfg["inference"]["zip_name"]
        pred_path = Path(cfg["inference"]["output_dir"]) / zip_name
        if not pred_path.exists():
            pred_path = Path(cfg["inference"]["output_dir"]) / Path(zip_name).with_suffix("").name
    else:
        pred_path = Path(args.pred)
        if not pred_path.is_absolute():
            pred_path = (base / pred_path).resolve()

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction path not found: {pred_path}")

    mask_dir = Path(cfg["mask"]["test_dir"])
    test_image_dir = Path(cfg["data"]["test_dir"])
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if not test_image_dir.exists():
        raise FileNotFoundError(f"Test image dir not found: {test_image_dir}")

    if args.out_dir is None:
        out_dir = Path(cfg["inference"]["output_dir"]) / "mask_filtered"
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (base / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = JsonPredictionReader(pred_path)
    missing_masks = 0
    missing_images = 0
    total_before = 0
    total_after = 0
    total_removed = 0
    total_rescored = 0

    try:
        for name in tqdm(reader.names(), desc="mask postprocess"):
            payload = reader.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            size = _image_size(test_image_dir, image_name)
            if size is None:
                missing_images += 1
                size = None

            mask = load_mask_for_image(mask_dir, image_name, expected_size=size)
            before = len(payload.get("annotations", []))
            if mask is None:
                missing_masks += 1
                filtered = payload
                removed = 0
                rescored = 0
            else:
                filtered, removed, rescored = _filter_payload(payload, mask, cfg)

            after = len(filtered.get("annotations", []))
            total_before += before
            total_after += after
            total_removed += removed
            total_rescored += rescored
            save_json(out_dir / name, filtered, encoding=cfg["inference"].get("json_encoding", "utf-8"))
    finally:
        reader.close()

    print(f"Prediction input: {pred_path}")
    print(f"Output dir: {out_dir}")
    print(f"Missing masks: {missing_masks}")
    print(f"Missing images: {missing_images}")
    print(f"Annotations before/after: {total_before}/{total_after}")
    print(f"Removed: {total_removed}")
    print(f"Rescored: {total_rescored}")

    if not args.no_zip:
        zip_name = args.zip_name or f"{out_dir.name}.zip"
        zip_path = out_dir.parent / zip_name
        zip_submission(str(out_dir), str(zip_path))
        print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
