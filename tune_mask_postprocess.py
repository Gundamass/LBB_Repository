import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from evaluate_local_map import evaluate_map50, parse_annotations
from postprocess_with_laptop_mask import JsonPredictionReader
from src.masking import load_mask_for_image, should_keep_box_by_mask
from src.utils import load_config, save_json, zip_submission


DEFAULT_CLASSES = ["plain particle", "dirt", "scratch", "collision"]


def _image_size(test_image_dir: Path, image_name: str) -> Optional[Tuple[int, int]]:
    image_path = test_image_dir / image_name
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return image.size


def _load_gt_records(gt_dir: Path, label_to_id: Dict[str, int]) -> Tuple[List[Path], List[Dict]]:
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        raise RuntimeError(f"No GT json files found in: {gt_dir}")

    records: List[Dict] = []
    for gt_file in gt_files:
        payload = json.loads(gt_file.read_text(encoding="utf-8"))
        boxes, labels, _ = parse_annotations(payload, label_to_id, score_thr=0.0)
        image_id = str(payload.get("image_id") or gt_file.with_suffix(".jpg").name)
        records.append(
            {
                "json_name": Path(image_id).with_suffix(".json").name,
                "image_name": Path(image_id).with_suffix(".jpg").name,
                "boxes": boxes,
                "labels": labels,
            }
        )
    return gt_files, records


def _load_pred_payloads(pred_path: Path, json_names: List[str]) -> Dict[str, Dict]:
    reader = JsonPredictionReader(pred_path)
    payloads: Dict[str, Dict] = {}
    try:
        for name in json_names:
            payloads[name] = reader.read(name)
    finally:
        reader.close()
    return payloads


def _load_masks(mask_dir: Path, test_image_dir: Path, image_names: List[str]) -> Dict[str, object]:
    masks: Dict[str, object] = {}
    size_cache: Dict[str, Optional[Tuple[int, int]]] = {}
    for image_name in tqdm(image_names, desc="load masks"):
        size_cache[image_name] = _image_size(test_image_dir, image_name)
        masks[image_name] = load_mask_for_image(
            mask_dir,
            image_name,
            expected_size=size_cache[image_name],
        )
    return masks


def _apply_strategy(payload: Dict, mask, strategy: Dict) -> Dict:
    mode = strategy["mode"]
    min_coverage = float(strategy["min_box_coverage"])
    keep_rule = strategy["keep_rule"]
    background_score_factor = float(strategy.get("background_score_factor", 0.15))
    min_score_after_rescore = float(strategy.get("min_score_after_rescore", 0.0))
    min_score_after_filter = float(strategy.get("min_score_after_filter", 0.0))

    anns = []
    for ann in payload.get("annotations", []):
        bbox = ann.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        if mask is None:
            keep = True
        else:
            keep, _, _ = should_keep_box_by_mask(
                mask,
                bbox,
                min_coverage=min_coverage,
                keep_rule=keep_rule,
            )

        score = float(ann.get("confidence", 1.0))
        if mode == "none":
            out_ann = ann
        elif mode == "filter":
            if not keep or score < min_score_after_filter:
                continue
            out_ann = ann
        elif mode == "rescore":
            out_ann = dict(ann)
            if not keep:
                score *= background_score_factor
                out_ann["confidence"] = round(score, 6)
            if score < min_score_after_rescore:
                continue
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        anns.append(out_ann)

    out = dict(payload)
    out["annotations"] = anns
    return out


def _evaluate_strategy(
    gt_records: List[Dict],
    pred_payloads: Dict[str, Dict],
    masks: Dict[str, object],
    label_to_id: Dict[str, int],
    strategy: Dict,
) -> Tuple[Dict, int]:
    pred_records = []
    gt_eval_records = []
    total_anns = 0

    for gt in gt_records:
        payload = pred_payloads.get(gt["json_name"], {"annotations": []})
        filtered = _apply_strategy(payload, masks.get(gt["image_name"]), strategy)
        boxes, labels, scores = parse_annotations(filtered, label_to_id, score_thr=0.0)
        total_anns += len(boxes)
        pred_records.append({"boxes": boxes, "labels": labels, "scores": scores})
        gt_eval_records.append({"boxes": gt["boxes"], "labels": gt["labels"]})

    metrics = evaluate_map50(
        pred_records,
        gt_eval_records,
        num_classes=len(label_to_id) + 1,
    )
    return metrics, total_anns


def _strategy_grid() -> List[Dict]:
    strategies = [
        {
            "name": "baseline_none",
            "mode": "none",
            "keep_rule": "center_or_coverage",
            "min_box_coverage": 0.10,
        }
    ]

    keep_rules = ["center_or_coverage", "center", "coverage"]
    coverages = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]

    for keep_rule in keep_rules:
        for cov in coverages:
            strategies.append(
                {
                    "name": f"filter_{keep_rule}_cov{cov:g}",
                    "mode": "filter",
                    "keep_rule": keep_rule,
                    "min_box_coverage": cov,
                }
            )

    for keep_rule in keep_rules:
        for cov in coverages:
            for factor in [0.05, 0.10, 0.15, 0.25, 0.50, 0.75]:
                strategies.append(
                    {
                        "name": f"rescore_{keep_rule}_cov{cov:g}_bg{factor:g}",
                        "mode": "rescore",
                        "keep_rule": keep_rule,
                        "min_box_coverage": cov,
                        "background_score_factor": factor,
                        "min_score_after_rescore": 0.0,
                    }
                )

    return strategies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.sam2_mask.yaml")
    parser.add_argument("--pred", type=str, required=True, help="Prediction dir or zip")
    parser.add_argument("--gt-dir", type=str, default="./手标数据")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--write-best", action="store_true")
    parser.add_argument("--out-dir", type=str, default="./outputs/mask_tuned_best")
    parser.add_argument("--zip-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = Path(args.config).resolve().parent
    pred_path = Path(args.pred)
    if not pred_path.is_absolute():
        pred_path = (base / pred_path).resolve()
    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_absolute():
        gt_dir = (base / gt_dir).resolve()

    classes = DEFAULT_CLASSES
    label_to_id = {name: i + 1 for i, name in enumerate(classes)}
    id_to_label = {v: k for k, v in label_to_id.items()}

    _, gt_records = _load_gt_records(gt_dir, label_to_id)
    json_names = [r["json_name"] for r in gt_records]
    image_names = [r["image_name"] for r in gt_records]

    pred_payloads = _load_pred_payloads(pred_path, json_names)
    masks = _load_masks(
        Path(cfg["mask"]["test_dir"]),
        Path(cfg["data"]["test_dir"]),
        image_names,
    )

    results = []
    for strategy in tqdm(_strategy_grid(), desc="scan strategies"):
        metrics, total_anns = _evaluate_strategy(
            gt_records,
            pred_payloads,
            masks,
            label_to_id,
            strategy,
        )
        row = {
            "strategy": strategy,
            "mAP50": metrics["mAP50"],
            "ap_per_class": metrics["ap_per_class"],
            "annotations": total_anns,
        }
        results.append(row)

    results.sort(key=lambda r: r["mAP50"], reverse=True)
    print(f"Scanned strategies: {len(results)}")
    for i, row in enumerate(results[: max(args.top_k, 1)], start=1):
        strategy = row["strategy"]
        ap_bits = []
        for cls_id, ap in row["ap_per_class"].items():
            ap_bits.append(f"{id_to_label.get(int(cls_id), cls_id)}={float(ap):.6f}")
        print(
            f"{i:02d}. mAP50={row['mAP50']:.6f} anns={row['annotations']} "
            f"name={strategy['name']} | " + ", ".join(ap_bits)
        )

    if args.write_best:
        best = results[0]
        strategy = best["strategy"]
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (base / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for gt in tqdm(gt_records, desc="write best"):
            payload = pred_payloads.get(gt["json_name"], {"annotations": []})
            filtered = _apply_strategy(payload, masks.get(gt["image_name"]), strategy)
            save_json(out_dir / gt["json_name"], filtered, encoding=cfg["inference"].get("json_encoding", "utf-8"))

        zip_name = args.zip_name or f"{out_dir.name}.zip"
        zip_path = out_dir.parent / zip_name
        zip_submission(str(out_dir), str(zip_path))
        print(f"Best strategy: {strategy}")
        print(f"Best mAP50: {best['mAP50']:.6f}")
        print(f"Output dir: {out_dir}")
        print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
