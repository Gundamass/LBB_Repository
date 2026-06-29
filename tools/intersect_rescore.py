import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


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


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def score(ann: Dict) -> float:
    try:
        return float(ann.get("confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def box_key(ann: Dict) -> Optional[Tuple]:
    label = ann.get("label")
    box = ann.get("bbox", [])
    if not label or not isinstance(box, list) or len(box) != 4:
        return None
    return (str(label),) + tuple(round(float(v), 3) for v in box)


def map_scores(payload: Dict) -> Dict[Tuple, float]:
    out: Dict[Tuple, float] = {}
    for ann in payload.get("annotations", []):
        key = box_key(ann)
        if key is not None:
            out[key] = score(ann)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a rescore only when two rescored zips agree to lower a box.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--mode", choices=["a", "b", "min", "mean"], default="min")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--score-decimals", type=int, default=8)
    args = parser.parse_args()

    base = Store(Path(args.base).resolve())
    store_a = Store(Path(args.a).resolve())
    store_b = Store(Path(args.b).resolve())
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    changed = 0
    total = 0
    try:
        for name in tqdm(base.names(), desc="intersect rescore"):
            payload = base.read(name)
            map_a = map_scores(store_a.read(name))
            map_b = map_scores(store_b.read(name))
            out_anns = []
            for ann in payload.get("annotations", []):
                total += 1
                out_ann = dict(ann)
                key = box_key(ann)
                base_score = score(ann)
                if key is not None and key in map_a and key in map_b:
                    sa = map_a[key]
                    sb = map_b[key]
                    if sa < base_score - float(args.epsilon) and sb < base_score - float(args.epsilon):
                        if args.mode == "a":
                            new_score = sa
                        elif args.mode == "b":
                            new_score = sb
                        elif args.mode == "mean":
                            new_score = (sa + sb) * 0.5
                        else:
                            new_score = min(sa, sb)
                        out_ann["confidence"] = round(float(new_score), int(args.score_decimals))
                        changed += 1
                out_anns.append(out_ann)
            out_anns.sort(key=score, reverse=True)
            payload["annotations"] = out_anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        base.close()
        store_a.close()
        store_b.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"Annotations changed/total: {changed}/{total}")


if __name__ == "__main__":
    main()
