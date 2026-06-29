import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set


VALID_LABELS = {"plain particle", "dirt", "scratch", "collision"}


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

    def read(self, name: str) -> Dict:
        if self.zf is not None:
            return json.loads(self.zf.read(self.members[name]).decode("utf-8"))
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def labels_from_payload(payload: Dict) -> Set[str]:
    labels: Set[str] = set()
    for ann in payload.get("annotations", []):
        label = ann.get("label")
        if label in VALID_LABELS:
            labels.add(str(label))
    for shape in payload.get("shapes", []):
        label = shape.get("label")
        if label in VALID_LABELS:
            labels.add(str(label))
    return labels


def load_manual_priors(manual_dir: Path) -> Dict[str, Set[str]]:
    priors: Dict[str, Set[str]] = {}
    for path in sorted(manual_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        image_id = str(payload.get("image_id") or path.with_suffix(".jpg").name)
        priors[Path(image_id).with_suffix(".json").name] = labels_from_payload(payload)
    return priors


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--manual-dir", default="./手标数据")
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--empty-mode", choices=["keep", "drop", "downweight"], default="drop")
    parser.add_argument("--empty-factor", type=float, default=0.0)
    parser.add_argument(
        "--class-presence-filter",
        action="store_true",
        help="For manually non-empty images, keep only classes present in manual labels.",
    )
    parser.add_argument("--absent-class-factor", type=float, default=0.0)
    args = parser.parse_args()

    pred = Path(args.pred).resolve()
    manual_dir = Path(args.manual_dir).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")

    priors = load_manual_priors(manual_dir)
    if not priors:
        raise RuntimeError(f"No manual JSON priors found in {manual_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = Store(pred)
    stats = Counter()
    label_removed = Counter()
    try:
        for name in store.names():
            payload = store.read(name)
            anns = list(payload.get("annotations", []))
            prior_labels = priors.get(name)
            if prior_labels is not None and len(prior_labels) == 0:
                stats["manual_empty_images"] += 1
                stats["manual_empty_annotations_before"] += len(anns)
                if args.empty_mode == "drop":
                    label_removed.update(a.get("label") for a in anns)
                    anns = []
                elif args.empty_mode == "downweight":
                    factor = float(args.empty_factor)
                    for ann in anns:
                        ann["confidence"] = round(float(ann.get("confidence", 0.0) or 0.0) * factor, 8)
                stats["manual_empty_annotations_after"] += len(anns)
            elif prior_labels is not None and args.class_presence_filter:
                stats["manual_nonempty_images"] += 1
                kept = []
                for ann in anns:
                    label = ann.get("label")
                    if label in prior_labels:
                        kept.append(ann)
                    else:
                        label_removed[label] += 1
                        factor = float(args.absent_class_factor)
                        if factor > 0:
                            ann = dict(ann)
                            ann["confidence"] = round(float(ann.get("confidence", 0.0) or 0.0) * factor, 8)
                            kept.append(ann)
                stats["class_filter_before"] += len(anns)
                stats["class_filter_after"] += len(kept)
                anns = kept

            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print("Stats:", dict(stats))
    print("Removed/downweighted labels:", dict(label_removed))


if __name__ == "__main__":
    main()
