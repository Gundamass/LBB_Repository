import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


class Store:
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

    def read(self, name: str) -> Optional[Dict]:
        try:
            if self.zip is not None:
                member = self.members.get(name)
                if member is None:
                    return None
                return json.loads(self.zip.read(member).decode("utf-8"))
            p = self.path / name
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


def classwise_nms(annotations: List[Dict], iou_thr: float) -> List[Dict]:
    if iou_thr <= 0:
        return annotations
    kept: List[Dict] = []
    for label in sorted({a.get("label") for a in annotations}):
        rows = [a for a in annotations if a.get("label") == label]
        rows.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
        label_kept: List[Dict] = []
        for ann in rows:
            if all(bbox_iou(ann["bbox"], old["bbox"]) < iou_thr for old in label_kept):
                label_kept.append(ann)
        kept.extend(label_kept)
    kept.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
    return kept


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(folder.glob("*.json")):
            z.write(fp, arcname=f"{folder.name}/{fp.name}")


def fuse(
    base_pred: Path,
    extra_preds: Iterable[Path],
    output_zip: Path,
    topk_per_image: int,
    score_factor: float,
    extra_score_thr: float,
    nms_iou: float,
    score_decimals: int,
) -> None:
    base_store = Store(base_pred)
    extra_stores = [Store(p) for p in extra_preds]
    out_folder = output_zip.with_suffix("")
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    try:
        for name in base_store.names():
            payload = base_store.read(name) or {"image_id": name.replace(".json", ".jpg"), "annotations": []}
            merged = list(payload.get("annotations", []))

            for store in extra_stores:
                extra = store.read(name)
                if not extra:
                    continue
                anns = list(extra.get("annotations", []))
                anns = [a for a in anns if float(a.get("confidence", 0.0)) >= extra_score_thr]
                anns.sort(key=lambda a: float(a.get("confidence", 0.0)), reverse=True)
                for ann in anns[:topk_per_image]:
                    new_ann = dict(ann)
                    new_ann["confidence"] = round(
                        float(new_ann.get("confidence", 0.0)) * score_factor,
                        int(score_decimals),
                    )
                    merged.append(new_ann)

            if nms_iou > 0:
                merged = classwise_nms(merged, iou_thr=nms_iou)
            payload["annotations"] = merged
            (out_folder / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        base_store.close()
        for store in extra_stores:
            store.close()

    write_zip(out_folder, output_zip)
    print(f"Fused folder: {out_folder}")
    print(f"Fused zip: {output_zip}")
    print(f"JSON count: {len(list(out_folder.glob('*.json')))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", nargs="+", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--topk-per-image", type=int, default=20)
    parser.add_argument("--score-factor", type=float, default=0.5)
    parser.add_argument("--extra-score-thr", type=float, default=0.0)
    parser.add_argument("--nms-iou", type=float, default=-1.0)
    parser.add_argument("--score-decimals", type=int, default=6)
    args = parser.parse_args()

    fuse(
        base_pred=Path(args.base).resolve(),
        extra_preds=[Path(p).resolve() for p in args.extra],
        output_zip=Path(args.output_zip).resolve(),
        topk_per_image=int(args.topk_per_image),
        score_factor=float(args.score_factor),
        extra_score_thr=float(args.extra_score_thr),
        nms_iou=float(args.nms_iou),
        score_decimals=int(args.score_decimals),
    )


if __name__ == "__main__":
    main()
