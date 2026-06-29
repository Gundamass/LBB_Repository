#!/usr/bin/env python3
"""Append branch-only rescue boxes as an ultra-low-score tail.

This differs from a naive ensemble: an extra branch box is appended only when
the current base prediction does not already contain a same-label neighbor.
That preserves single-branch detections that may have been suppressed by TTA or
checkpoint ensemble NMS, while avoiding most duplicate clutter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


VALID_LABELS = {"plain particle", "dirt", "scratch", "collision"}


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
    b = normalize_box(box)
    if b is None:
        return 0.0
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_box(a)
    bb = normalize_box(b)
    if aa is None or bb is None:
        return 0.0
    x1 = max(aa[0], bb[0])
    y1 = max(aa[1], bb[1])
    x2 = min(aa[2], bb[2])
    y2 = min(aa[3], bb[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(aa) + box_area(bb) - inter
    return float(inter / union) if union > 0 else 0.0


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.zf: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, str] = {}
        if path.suffix.lower() == ".zip":
            self.zf = zipfile.ZipFile(path, "r")
            self.members = {Path(n).name: n for n in self.zf.namelist() if n.endswith(".json")}

    def names(self) -> List[str]:
        if self.zf is not None:
            return sorted(self.members)
        return sorted(p.name for p in self.path.glob("*.json"))

    def read(self, name: str) -> Optional[Dict]:
        try:
            if self.zf is not None:
                member = self.members.get(name)
                if member is None:
                    return None
                return json.loads(self.zf.read(member).decode("utf-8"))
            p = self.path / name
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def parse_label_set(values: Iterable[str], default: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--min-source-score", type=float, default=0.0)
    parser.add_argument("--max-source-score", type=float, default=1.0)
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--max-area", type=float, default=1e18)
    parser.add_argument("--min-aspect", type=float, default=0.0)
    parser.add_argument("--max-aspect", type=float, default=1e18)
    parser.add_argument("--topk-per-image", type=int, default=0, help="Optional global cap after score sorting; 0 disables.")
    parser.add_argument("--topk-per-label", type=int, default=20)
    parser.add_argument("--skip-base-iou", type=float, default=0.30)
    parser.add_argument("--self-dedup-iou", type=float, default=0.70)
    parser.add_argument("--score-factor", type=float, default=0.0001)
    parser.add_argument("--score-cap", type=float, default=0.0002)
    parser.add_argument("--score-floor", type=float, default=0.0)
    parser.add_argument("--score-decimals", type=int, default=8)
    args = parser.parse_args()

    labels = set(parse_label_set(args.include_label, ["plain particle", "scratch", "collision"]))
    unknown = labels - VALID_LABELS
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    base = Store(Path(args.base).resolve())
    extra = Store(Path(args.extra).resolve())
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped_base = 0
    skipped_self = 0
    added_by_label = Counter()
    considered_by_label = Counter()

    try:
        for name in base.names():
            payload = base.read(name) or {"image_id": Path(name).with_suffix(".jpg").name, "annotations": []}
            anns = [dict(a) for a in payload.get("annotations", []) if isinstance(a, dict)]

            base_boxes: Dict[str, List[List[float]]] = defaultdict(list)
            for ann in anns:
                label = str(ann.get("label", ""))
                box = normalize_box(ann.get("bbox", []))
                if label and box is not None:
                    base_boxes[label].append(box)

            extra_payload = extra.read(name)
            candidates = []
            if extra_payload is not None:
                for ann in extra_payload.get("annotations", []):
                    if not isinstance(ann, dict):
                        continue
                    label = str(ann.get("label", ""))
                    if label not in labels:
                        continue
                    score = safe_float(ann.get("confidence", 0.0), 0.0)
                    if score < float(args.min_source_score) or score > float(args.max_source_score):
                        continue
                    box = normalize_box(ann.get("bbox", []))
                    if box is None:
                        continue
                    area = box_area(box)
                    if area < float(args.min_area) or area > float(args.max_area):
                        continue
                    w = max(0.0, box[2] - box[0])
                    h = max(0.0, box[3] - box[1])
                    aspect = w / max(h, 1e-9)
                    if aspect < float(args.min_aspect) or aspect > float(args.max_aspect):
                        continue
                    candidates.append((score, label, box, ann))
                    considered_by_label[label] += 1

            candidates.sort(key=lambda x: x[0], reverse=True)
            if int(args.topk_per_image) > 0:
                candidates = candidates[: int(args.topk_per_image)]

            used_per_label = Counter()
            appended: Dict[str, List[List[float]]] = defaultdict(list)
            for score, label, box, ann in candidates:
                if used_per_label[label] >= int(args.topk_per_label):
                    continue
                if any(bbox_iou(box, old) >= float(args.skip_base_iou) for old in base_boxes.get(label, [])):
                    skipped_base += 1
                    continue
                if any(bbox_iou(box, old) >= float(args.self_dedup_iou) for old in appended.get(label, [])):
                    skipped_self += 1
                    continue
                new_score = score * float(args.score_factor)
                new_score = min(float(args.score_cap), max(float(args.score_floor), new_score))
                if new_score <= 0:
                    continue
                new_ann = dict(ann)
                new_ann["label"] = label
                new_ann["bbox"] = [round(float(v), 4) for v in box]
                new_ann["confidence"] = round(float(new_score), int(args.score_decimals))
                anns.append(new_ann)
                base_boxes[label].append(box)
                appended[label].append(box)
                used_per_label[label] += 1
                added += 1
                added_by_label[label] += 1

            anns.sort(key=lambda a: safe_float(a.get("confidence", 0.0), 0.0), reverse=True)
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        base.close()
        extra.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"Added: {added}")
    print(f"Added by label: {dict(added_by_label)}")
    print(f"Considered by label: {dict(considered_by_label)}")
    print(f"Skipped base/self: {skipped_base}/{skipped_self}")


if __name__ == "__main__":
    main()
