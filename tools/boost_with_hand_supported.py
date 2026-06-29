import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


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


def valid_box(box: List[float]) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Current best competition-format zip/folder.")
    parser.add_argument("--hand", required=True, help="Hand annotation competition-format folder/zip.")
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--boost-score", type=float, default=0.85)
    parser.add_argument("--iou-bonus", type=float, default=0.10)
    parser.add_argument("--orig-score-bonus", type=float, default=0.02)
    parser.add_argument("--max-score", type=float, default=0.999)
    parser.add_argument("--score-decimals", type=int, default=8)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--exclude-label", action="append", default=[])
    args = parser.parse_args()

    include_labels = set(args.include_label)
    exclude_labels = set(args.exclude_label)
    base = Store(Path(args.base).resolve())
    hand = Store(Path(args.hand).resolve())
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boosted = 0
    skipped_no_match = 0
    skipped_bad_hand = 0
    boosted_labels = Counter()

    try:
        for name in base.names():
            payload = base.read(name) or {"image_id": Path(name).with_suffix(".jpg").name, "annotations": []}
            anns = list(payload.get("annotations", []))
            hand_payload = hand.read(name)

            if hand_payload:
                by_label: Dict[str, List[Tuple[int, Dict]]] = {}
                for idx, ann in enumerate(anns):
                    label = str(ann.get("label", ""))
                    box = ann.get("bbox", [])
                    if label and valid_box(box):
                        by_label.setdefault(label, []).append((idx, ann))

                # One boosted duplicate per hand box. We keep the model box geometry,
                # because earlier snapping to hand coordinates hurt official score.
                for h_ann in hand_payload.get("annotations", []):
                    label = str(h_ann.get("label", ""))
                    if include_labels and label not in include_labels:
                        continue
                    if label in exclude_labels:
                        continue
                    h_box = h_ann.get("bbox", [])
                    if not label or not valid_box(h_box):
                        skipped_bad_hand += 1
                        continue

                    best_iou = 0.0
                    best_ann: Optional[Dict] = None
                    for _, cand in by_label.get(label, []):
                        iou = bbox_iou(h_box, cand.get("bbox", []))
                        if iou > best_iou:
                            best_iou = iou
                            best_ann = cand

                    if best_ann is None or best_iou < float(args.match_iou):
                        skipped_no_match += 1
                        continue

                    orig_score = float(best_ann.get("confidence", 0.0) or 0.0)
                    new_score = (
                        float(args.boost_score)
                        + float(args.iou_bonus) * best_iou
                        + float(args.orig_score_bonus) * min(max(orig_score, 0.0), 1.0)
                    )
                    new_ann = dict(best_ann)
                    new_ann["confidence"] = round(min(float(args.max_score), new_score), int(args.score_decimals))
                    anns.append(new_ann)
                    boosted += 1
                    boosted_labels[label] += 1

            anns.sort(key=lambda a: float(a.get("confidence", 0.0) or 0.0), reverse=True)
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        base.close()
        hand.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Boosted/skipped_no_match/skipped_bad_hand: {boosted}/{skipped_no_match}/{skipped_bad_hand}")
    print("Boosted labels:", dict(boosted_labels))


if __name__ == "__main__":
    main()
