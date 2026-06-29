import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def image_index(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def scale_box(box: Iterable[float], factor: float, width: int, height: int) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * float(factor)
    bh = (y2 - y1) * float(factor)
    nx1 = max(0.0, cx - bw * 0.5)
    ny1 = max(0.0, cy - bh * 0.5)
    nx2 = min(float(width - 1), cx + bw * 0.5)
    ny2 = min(float(height - 1), cy + bh * 0.5)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [round(nx1, 3), round(ny1, 3), round(nx2, 3), round(ny2, 3)]


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(folder.glob("*.json")):
            zf.write(fp, arcname=f"{folder.name}/{fp.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--factor", type=float, default=1.5)
    parser.add_argument(
        "--score-min",
        type=float,
        default=0.0,
        help="Only scale boxes with confidence >= score-min; lower-score boxes stay unchanged.",
    )
    args = parser.parse_args()

    pred = Path(args.pred).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    images = image_index(Path(args.test_image_dir).resolve())
    size_cache: Dict[str, Tuple[int, int]] = {}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = Store(pred)
    scaled = 0
    unchanged = 0
    missing_images = 0
    try:
        for name in store.names():
            payload = store.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            img_path = images.get(image_name)
            if img_path is None:
                missing_images += 1
                width = height = 10**9
            else:
                if image_name not in size_cache:
                    size_cache[image_name] = image_size(img_path)
                width, height = size_cache[image_name]

            anns = []
            for ann in payload.get("annotations", []):
                new_ann = dict(ann)
                conf = float(new_ann.get("confidence", 0.0) or 0.0)
                if conf >= float(args.score_min):
                    new_box = scale_box(new_ann.get("bbox", []), float(args.factor), width, height)
                    if new_box is not None:
                        new_ann["bbox"] = new_box
                        scaled += 1
                    else:
                        unchanged += 1
                else:
                    unchanged += 1
                anns.append(new_ann)
            payload["annotations"] = anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Scaled/unchanged/missing_images: {scaled}/{unchanged}/{missing_images}")


if __name__ == "__main__":
    main()
