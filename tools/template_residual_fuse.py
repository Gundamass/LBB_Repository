import argparse
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VALID_LABELS = {"plain particle", "dirt", "scratch", "collision"}


def parse_key(path: Path, with_view: bool) -> Tuple[str, ...]:
    parts = path.stem.split("_")
    return tuple(parts[:3] + parts[-2:]) if with_view else tuple(parts[-2:])


def image_paths(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


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


def normalize_box(box: Iterable[float], width: int, height: int) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def gray_image(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.mean(axis=2)


def robust_norm(arr: np.ndarray) -> np.ndarray:
    med = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    scale = max(float(q3 - q1), 0.05)
    return (arr - med) / scale


def build_groups(pos_dir: Path) -> Tuple[Dict[Tuple[str, ...], List[Path]], Dict[Tuple[str, ...], List[Path]]]:
    by_view: Dict[Tuple[str, ...], List[Path]] = defaultdict(list)
    by_rc: Dict[Tuple[str, ...], List[Path]] = defaultdict(list)
    for path in image_paths(pos_dir):
        by_view[parse_key(path, with_view=True)].append(path)
        by_rc[parse_key(path, with_view=False)].append(path)
    return by_view, by_rc


def make_template(refs: List[Path], size: Tuple[int, int]) -> np.ndarray:
    hps = []
    for ref in refs:
        gray = gray_image(ref, size=size)
        norm = robust_norm(gray)
        low = ndimage.gaussian_filter(norm, sigma=7.0)
        hps.append(norm - low)
    return np.median(np.stack(hps, axis=0), axis=0).astype(np.float32)


def choose_refs(
    image_name: str,
    by_view: Dict[Tuple[str, ...], List[Path]],
    by_rc: Dict[Tuple[str, ...], List[Path]],
    min_refs: int,
) -> List[Path]:
    stem = Path(image_name).stem
    fake = Path(stem)
    refs = by_view.get(parse_key(fake, with_view=True), [])
    if len(refs) >= min_refs:
        return refs
    return by_rc.get(parse_key(fake, with_view=False), refs)


def component_boxes(
    residual: np.ndarray,
    gray: np.ndarray,
    topk: int,
    min_area: int,
    max_area: int,
) -> List[Tuple[List[float], float]]:
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med))) + 1e-6
    thr = max(float(np.percentile(residual, 99.82)), med + 6.0 * mad)
    mask = residual > thr
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, num = ndimage.label(mask)

    rows: List[Tuple[List[float], float]] = []
    h, w = residual.shape
    for idx in range(1, num + 1):
        ys, xs = np.where(labels == idx)
        area = int(xs.size)
        if area < min_area or area > max_area:
            continue
        x1 = max(0, int(xs.min()) - 2)
        y1 = max(0, int(ys.min()) - 2)
        x2 = min(w - 1, int(xs.max()) + 3)
        y2 = min(h - 1, int(ys.max()) + 3)
        bw = x2 - x1
        bh = y2 - y1
        if bw < 3 or bh < 3:
            continue
        if bw > 180 or bh > 180 or bw * bh > max_area * 6:
            continue
        # The white fixture/background is explicitly not a defect.
        patch_gray = gray[y1:y2, x1:x2]
        if patch_gray.size and float(np.median(patch_gray)) > 0.92:
            continue
        score = float(np.percentile(residual[y1:y2, x1:x2], 95))
        rows.append(([float(x1), float(y1), float(x2), float(y2)], score))

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:topk]


def label_for_box(box: List[float], anns: List[Dict]) -> str:
    best_score = -1.0
    best_label = ""
    for ann in anns:
        label = ann.get("label")
        if label not in VALID_LABELS:
            continue
        other = ann.get("bbox", [])
        if len(other) != 4:
            continue
        ov = bbox_iou(box, [float(v) for v in other])
        if ov <= 0.01:
            continue
        conf = float(ann.get("confidence", 0.0) or 0.0)
        score = (0.05 + ov) * (1.0 + math.log1p(max(conf, 0.0) * 1000.0))
        if score > best_score:
            best_score = score
            best_label = str(label)
    if best_label:
        return best_label

    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    ratio = max(bw / bh, bh / bw)
    return "scratch" if ratio >= 3.0 else "plain particle"


class Store:
    def __init__(self, path: Path):
        self.zf = zipfile.ZipFile(path, "r")
        self.members = {Path(n).name: n for n in self.zf.namelist() if n.endswith(".json")}

    def names(self) -> List[str]:
        return sorted(self.members)

    def read(self, name: str) -> Dict:
        return json.loads(self.zf.read(self.members[name]).decode("utf-8"))

    def close(self) -> None:
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
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--test-image-dir", default="./初赛数据/测试集/image")
    parser.add_argument("--train-pos-dir", default="./初赛数据/训练集/正样本")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--min-refs", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=3500)
    parser.add_argument("--score-ceiling", type=float, default=0.000005)
    parser.add_argument("--dedup-iou", type=float, default=0.95)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing output JSON files and only process missing images.",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_zip = Path(args.out_zip).resolve()
    out_dir = out_zip.with_suffix("")
    test_images = {p.name: p for p in image_paths(Path(args.test_image_dir).resolve())}
    by_view, by_rc = build_groups(Path(args.train_pos_dir).resolve())
    template_cache: Dict[Tuple[Tuple[str, ...], Tuple[int, int]], np.ndarray] = {}

    if out_dir.exists() and not args.resume:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = Store(base)
    added = 0
    skipped_dup = 0
    skipped_no_refs = 0
    try:
        for name in store.names():
            out_file = out_dir / name
            if args.resume and out_file.exists():
                continue
            payload = store.read(name)
            anns = list(payload.get("annotations", []))
            image_name = str(payload.get("image_id") or Path(name).with_suffix(".jpg").name)
            image_path = test_images.get(image_name)
            if image_path is None:
                payload["annotations"] = anns
                out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            with Image.open(image_path) as im:
                size = im.size
            refs = choose_refs(image_name, by_view, by_rc, int(args.min_refs))
            if len(refs) < int(args.min_refs):
                skipped_no_refs += 1
                payload["annotations"] = anns
                out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            group_key = parse_key(Path(image_name), with_view=True)
            if group_key not in by_view or len(by_view[group_key]) < int(args.min_refs):
                group_key = parse_key(Path(image_name), with_view=False)
            cache_key = (group_key, size)
            if cache_key not in template_cache:
                template_cache[cache_key] = make_template(refs, size=size)

            gray = gray_image(image_path)
            norm = robust_norm(gray)
            hp = norm - ndimage.gaussian_filter(norm, sigma=7.0)
            residual = np.abs(hp - template_cache[cache_key])
            residual = ndimage.gaussian_filter(residual, sigma=0.8)
            proposals = component_boxes(
                residual=residual,
                gray=gray,
                topk=int(args.topk),
                min_area=int(args.min_area),
                max_area=int(args.max_area),
            )

            max_prop_score = max([s for _, s in proposals], default=1.0)
            for box, prop_score in proposals:
                label = label_for_box(box, anns)
                dup = False
                for old in anns:
                    if old.get("label") != label:
                        continue
                    old_box = old.get("bbox", [])
                    if len(old_box) == 4 and bbox_iou(box, [float(v) for v in old_box]) >= float(args.dedup_iou):
                        dup = True
                        break
                if dup:
                    skipped_dup += 1
                    continue
                conf = float(args.score_ceiling) * max(0.15, min(1.0, prop_score / max(max_prop_score, 1e-6)))
                anns.append(
                    {
                        "label": label,
                        "bbox": [round(float(v), 3) for v in box],
                        "confidence": round(conf, 8),
                    }
                )
                added += 1

            anns.sort(key=lambda a: float(a.get("confidence", 0.0) or 0.0), reverse=True)
            payload["annotations"] = anns
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()

    write_zip(out_dir, out_zip)
    print(f"Output zip: {out_zip}")
    print(f"JSON count: {len(list(out_dir.glob('*.json')))}")
    print(f"Templates cached: {len(template_cache)}")
    print(f"Added proposals: {added}")
    print(f"Skipped duplicate/no refs: {skipped_dup}/{skipped_no_refs}")


if __name__ == "__main__":
    main()
