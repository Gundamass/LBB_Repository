#!/usr/bin/env python3
"""Duplicate likely cross-label boxes as collision proposals.

The current ensemble often localizes collision defects with scratch/dirt boxes.
This postprocessor keeps the original annotations and appends low-score
collision copies of selected source-label boxes, so it can improve collision
recall without destroying the source-class ranking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_box(box: Sequence[float]) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: Sequence[float]) -> float:
    norm = normalize_box(box)
    if norm is None:
        return 0.0
    return max(0.0, norm[2] - norm[0]) * max(0.0, norm[3] - norm[1])


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_box(a)
    bb = normalize_box(b)
    if aa is None or bb is None:
        return 0.0
    x1 = max(aa[0], bb[0])
    y1 = max(aa[1], bb[1])
    x2 = min(aa[2], bb[2])
    y2 = min(aa[3], bb[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = box_area(aa)
    area_b = box_area(bb)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


class PredictionStore:
    def __init__(self, pred_path: Path):
        self.pred_path = pred_path
        self.zf: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, str] = {}
        if pred_path.suffix.lower() == ".zip":
            self.zf = zipfile.ZipFile(pred_path, "r")
            self.members = {Path(n).name: n for n in self.zf.namelist() if n.endswith(".json")}

    def names(self) -> List[str]:
        if self.zf is not None:
            return sorted(self.members)
        return sorted(p.name for p in self.pred_path.glob("*.json"))

    def read(self, name: str) -> Dict:
        if self.zf is not None:
            return json.loads(self.zf.read(self.members[name]).decode("utf-8"))
        return json.loads((self.pred_path / name).read_text(encoding="utf-8"))

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def parse_labels(values: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                labels.append(part)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Input prediction zip or JSON dir.")
    parser.add_argument("--out-zip", required=True, help="Output zip path.")
    parser.add_argument("--source-label", action="append", default=[])
    parser.add_argument("--target-label", default="collision")
    parser.add_argument("--topk-per-image", type=int, default=60)
    parser.add_argument("--max-added-per-image", type=int, default=60)
    parser.add_argument("--min-source-score", type=float, default=0.0)
    parser.add_argument("--max-source-score", type=float, default=1.0)
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--max-area", type=float, default=40000.0)
    parser.add_argument("--min-aspect", type=float, default=0.0)
    parser.add_argument("--max-aspect", type=float, default=999999.0)
    parser.add_argument("--score-factor", type=float, default=0.0005)
    parser.add_argument("--score-cap", type=float, default=0.0008)
    parser.add_argument("--score-floor", type=float, default=0.0)
    parser.add_argument("--rank-decay", type=float, default=0.0, help="Optional multiplicative decay per rank.")
    parser.add_argument("--dedup-iou", type=float, default=0.70, help="Skip if target box already overlaps this much.")
    parser.add_argument("--self-dedup-iou", type=float, default=0.95, help="Deduplicate newly appended boxes.")
    parser.add_argument("--score-decimals", type=int, default=8)
    args = parser.parse_args()

    source_labels = set(parse_labels(args.source_label) or ["scratch", "dirt"])
    pred_path = Path(args.pred).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = PredictionStore(pred_path)
    total_added = 0
    added_by_source = Counter()
    skipped_existing = 0
    skipped_self = 0
    json_count = 0

    try:
        for name in store.names():
            payload = store.read(name)
            anns = [dict(x) for x in payload.get("annotations", []) if isinstance(x, dict)]
            target_boxes = [
                normalize_box(a.get("bbox", []))
                for a in anns
                if str(a.get("label", "")) == str(args.target_label)
            ]
            target_boxes = [b for b in target_boxes if b is not None]

            candidates = []
            for ann in anns:
                label = str(ann.get("label", ""))
                if label not in source_labels:
                    continue
                score = safe_float(ann.get("confidence", 0.0), 0.0)
                if score < float(args.min_source_score) or score > float(args.max_source_score):
                    continue
                bbox = normalize_box(ann.get("bbox", []))
                if bbox is None:
                    continue
                area = box_area(bbox)
                if area < float(args.min_area) or area > float(args.max_area):
                    continue
                width = max(0.0, bbox[2] - bbox[0])
                height = max(0.0, bbox[3] - bbox[1])
                aspect = width / max(height, 1e-9)
                if aspect < float(args.min_aspect) or aspect > float(args.max_aspect):
                    continue
                candidates.append((score, label, bbox, ann))

            candidates.sort(key=lambda x: x[0], reverse=True)
            appended: List[List[float]] = []
            for rank, (score, label, bbox, ann) in enumerate(candidates[: max(0, int(args.topk_per_image))]):
                if len(appended) >= int(args.max_added_per_image):
                    break
                if any(bbox_iou(bbox, existing) >= float(args.dedup_iou) for existing in target_boxes):
                    skipped_existing += 1
                    continue
                if any(bbox_iou(bbox, existing) >= float(args.self_dedup_iou) for existing in appended):
                    skipped_self += 1
                    continue

                decay = 1.0
                if float(args.rank_decay) > 0:
                    decay = max(0.0, 1.0 - float(args.rank_decay) * float(rank))
                new_score = score * float(args.score_factor) * decay
                new_score = min(float(args.score_cap), max(float(args.score_floor), new_score))
                if new_score <= 0:
                    continue
                new_ann = dict(ann)
                new_ann["label"] = str(args.target_label)
                new_ann["bbox"] = [round(float(v), 4) for v in bbox]
                new_ann["confidence"] = round(float(new_score), int(args.score_decimals))
                anns.append(new_ann)
                appended.append(bbox)
                target_boxes.append(bbox)
                total_added += 1
                added_by_source[label] += 1

            anns.sort(key=lambda a: safe_float(a.get("confidence", 0.0), 0.0), reverse=True)
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            json_count += 1
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Input: {pred_path}")
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {json_count}")
    print(f"Added: {total_added}")
    print(f"Added by source: {dict(added_by_source)}")
    print(f"Skipped existing target/self: {skipped_existing}/{skipped_self}")


if __name__ == "__main__":
    main()
