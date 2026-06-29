import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


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


def image_map(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def load_mask(mask_dir: Optional[Path], image_name: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_dir is None:
        return None
    path = mask_dir / f"{Path(image_name).stem}.png"
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            if img.size != size:
                img = img.resize(size, Image.NEAREST)
            return np.asarray(img, dtype=np.uint8) > 127
    except Exception:
        return None


def integral(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def rect_sum(ii: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])


def norm_box(box, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_coverage(mask: Optional[np.ndarray], x1: int, y1: int, x2: int, y2: int) -> float:
    if mask is None:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    return float(crop.mean()) if crop.size else 0.0


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
    parser.add_argument("--mask-dir", default="./masks/sam2_laptop_object/test")
    parser.add_argument("--disable-mask", action="store_true")
    parser.add_argument("--white-mean", type=float, default=0.90)
    parser.add_argument("--max-std", type=float, default=0.055)
    parser.add_argument("--max-mask-coverage", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.00001)
    parser.add_argument("--max-score", type=float, default=1.0)
    parser.add_argument("--score-factor", type=float, default=0.15)
    parser.add_argument("--min-score-after", type=float, default=0.0)
    parser.add_argument("--score-decimals", type=int, default=8)
    args = parser.parse_args()

    pred = Path(args.pred).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = image_map(Path(args.test_image_dir).resolve())
    mask_dir = None if args.disable_mask else Path(args.mask_dir).resolve()
    store = Store(pred)

    total = 0
    rescored = 0
    missing_images = 0
    missing_masks = 0
    try:
        for name in tqdm(store.names(), desc="white-background rescore"):
            payload = store.read(name)
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = images.get(image_name)
            anns = list(payload.get("annotations", []))
            if image_path is None:
                missing_images += 1
                payload["annotations"] = anns
                (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                arr = np.asarray(img, dtype=np.float32) / 255.0
            gray = arr.mean(axis=2)
            ii = integral(gray)
            ii2 = integral(gray * gray)
            mask = load_mask(mask_dir, image_name, (width, height))
            if mask_dir is not None and mask is None:
                missing_masks += 1

            out_anns = []
            for ann in anns:
                total += 1
                box = norm_box(ann.get("bbox", []), width, height)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                area = max(1, (x2 - x1) * (y2 - y1))
                mean = rect_sum(ii, x1, y1, x2, y2) / area
                mean2 = rect_sum(ii2, x1, y1, x2, y2) / area
                std = max(0.0, mean2 - mean * mean) ** 0.5
                coverage = mask_coverage(mask, x1, y1, x2, y2)
                score = float(ann.get("confidence", 0.0) or 0.0)

                out_ann = dict(ann)
                is_white_bg = (
                    score >= float(args.min_score)
                    and score <= float(args.max_score)
                    and mean >= float(args.white_mean)
                    and std <= float(args.max_std)
                    and (mask is None or coverage <= float(args.max_mask_coverage))
                )
                if is_white_bg:
                    score = max(float(args.min_score_after), score * float(args.score_factor))
                    out_ann["confidence"] = round(score, int(args.score_decimals))
                    rescored += 1
                out_anns.append(out_ann)

            out_anns.sort(key=lambda a: float(a.get("confidence", 0.0) or 0.0), reverse=True)
            payload["annotations"] = out_anns
            (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Input: {pred}")
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Annotations/rescored: {total}/{rescored}")
    print(f"Missing images/masks: {missing_images}/{missing_masks}")


if __name__ == "__main__":
    main()
