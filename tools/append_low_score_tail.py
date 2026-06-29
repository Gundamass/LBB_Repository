import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--topk-per-image", type=int, default=50)
    parser.add_argument("--score-factor", type=float, default=0.00005)
    parser.add_argument("--extra-score-thr", type=float, default=0.0)
    parser.add_argument("--dedup-iou", type=float, default=0.95)
    parser.add_argument("--score-decimals", type=int, default=8)
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Only append this label. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Do not append this label. Can be passed multiple times.",
    )
    args = parser.parse_args()
    include_labels = {str(x) for x in args.include_label}
    exclude_labels = {str(x) for x in args.exclude_label}

    base = Store(Path(args.base).resolve())
    extra = Store(Path(args.extra).resolve())
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped_dup = 0
    added_labels = Counter()
    try:
        for name in base.names():
            payload = base.read(name) or {"image_id": Path(name).with_suffix(".jpg").name, "annotations": []}
            anns = list(payload.get("annotations", []))
            by_label: Dict[str, List[List[float]]] = {}
            for ann in anns:
                label = ann.get("label")
                box = ann.get("bbox", [])
                if label and isinstance(box, list) and len(box) == 4:
                    by_label.setdefault(str(label), []).append([float(v) for v in box])

            extra_payload = extra.read(name)
            if extra_payload:
                rows = [
                    a
                    for a in extra_payload.get("annotations", [])
                    if float(a.get("confidence", 0.0) or 0.0) >= float(args.extra_score_thr)
                    and (not include_labels or str(a.get("label")) in include_labels)
                    and str(a.get("label")) not in exclude_labels
                ]
                rows.sort(key=lambda a: float(a.get("confidence", 0.0) or 0.0), reverse=True)
                for ann in rows[: int(args.topk_per_image)]:
                    label = ann.get("label")
                    box = ann.get("bbox", [])
                    if not label or not isinstance(box, list) or len(box) != 4:
                        continue
                    fbox = [float(v) for v in box]
                    if any(bbox_iou(fbox, old) >= float(args.dedup_iou) for old in by_label.get(str(label), [])):
                        skipped_dup += 1
                        continue
                    new_ann = dict(ann)
                    new_ann["confidence"] = round(
                        float(new_ann.get("confidence", 0.0) or 0.0) * float(args.score_factor),
                        int(args.score_decimals),
                    )
                    anns.append(new_ann)
                    by_label.setdefault(str(label), []).append(fbox)
                    added += 1
                    added_labels[str(label)] += 1

            anns.sort(key=lambda a: float(a.get("confidence", 0.0) or 0.0), reverse=True)
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        base.close()
        extra.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Added/skipped_dup: {added}/{skipped_dup}")
    print("Added labels:", dict(added_labels))


if __name__ == "__main__":
    main()
