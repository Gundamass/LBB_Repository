import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_CLASSES = ["plain particle", "dirt", "scratch", "collision"]


def bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def evaluate_map50(pred_records: List[Dict], gt_records: List[Dict], num_classes: int) -> Dict:
    per_cls_ap = {}
    eps = 1e-9

    for cls in range(1, num_classes):
        gts = {}
        npos = 0
        for i, gt in enumerate(gt_records):
            boxes = []
            labels = gt["labels"]
            for b, l in zip(gt["boxes"], labels):
                if int(l) == cls:
                    boxes.append(list(map(float, b)))
            gts[i] = {"boxes": boxes, "used": [False] * len(boxes)}
            npos += len(boxes)

        preds = []
        for i, pred in enumerate(pred_records):
            for b, l, s in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if int(l) == cls:
                    preds.append((float(s), i, list(map(float, b))))

        preds.sort(key=lambda x: x[0], reverse=True)
        tp = [0.0] * len(preds)
        fp = [0.0] * len(preds)

        for pi, (_, img_idx, pbox) in enumerate(preds):
            gt_img = gts[img_idx]
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(gt_img["boxes"]):
                iou = bbox_iou_xyxy(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.5 and best_j >= 0:
                if not gt_img["used"][best_j]:
                    tp[pi] = 1.0
                    gt_img["used"][best_j] = True
                else:
                    fp[pi] = 1.0
            else:
                fp[pi] = 1.0

        if len(preds) == 0:
            per_cls_ap[cls] = 0.0
            continue

        tp_cum, fp_cum = [], []
        t, f = 0.0, 0.0
        for ti, fi in zip(tp, fp):
            t += ti
            f += fi
            tp_cum.append(t)
            fp_cum.append(f)

        rec = [x / max(float(npos), eps) for x in tp_cum]
        prec = [tp_cum[i] / max(tp_cum[i] + fp_cum[i], eps) for i in range(len(tp_cum))]

        mrec = [0.0] + rec + [1.0]
        mpre = [0.0] + prec + [0.0]
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        per_cls_ap[cls] = float(ap)

    mAP = sum(per_cls_ap.values()) / max(len(per_cls_ap), 1)
    return {"mAP50": float(mAP), "ap_per_class": per_cls_ap}


def parse_annotations(
    payload: Dict,
    label_to_id: Dict[str, int],
    score_thr: float = 0.0,
) -> Tuple[List[List[float]], List[int], List[float]]:
    boxes: List[List[float]] = []
    labels: List[int] = []
    scores: List[float] = []
    anns = payload.get("annotations", [])
    for ann in anns:
        label = ann.get("label")
        if label not in label_to_id:
            continue
        bbox = ann.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        # Manual annotation tools can record drag direction, so normalize boxes
        # before IoU matching instead of dropping reversed coordinates.
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(ann.get("confidence", 1.0))
        if score < score_thr:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(int(label_to_id[label]))
        scores.append(score)
    return boxes, labels, scores


class PredictionStore:
    def __init__(self, pred_path: Path):
        self.pred_path = pred_path
        self.is_zip = pred_path.suffix.lower() == ".zip"
        self._zip: Optional[zipfile.ZipFile] = None
        self._name_to_member: Dict[str, str] = {}

        if self.is_zip:
            self._zip = zipfile.ZipFile(pred_path, "r")
            for member in self._zip.namelist():
                if not member.endswith(".json"):
                    continue
                self._name_to_member[Path(member).name] = member

    def read_by_json_name(self, json_name: str) -> Optional[Dict]:
        if self.is_zip:
            member = self._name_to_member.get(json_name)
            if member is None or self._zip is None:
                return None
            try:
                raw = self._zip.read(member).decode("utf-8")
                return json.loads(raw)
            except Exception:
                return None

        p = self.pred_path / json_name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def close(self):
        if self._zip is not None:
            self._zip.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Prediction dir or zip")
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="./初赛数据/手标数据/annotition",
        help="Ground-truth JSON directory (competition format)",
    )
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names",
    )
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    label_to_id = {name: i + 1 for i, name in enumerate(classes)}
    id_to_label = {v: k for k, v in label_to_id.items()}

    gt_dir = Path(args.gt_dir).resolve()
    pred_path = Path(args.pred).resolve()
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction path not found: {pred_path}")

    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        raise RuntimeError(f"No GT json files found in: {gt_dir}")

    store = PredictionStore(pred_path)
    missing_pred = 0
    pred_records: List[Dict] = []
    gt_records: List[Dict] = []

    try:
        for gt_file in gt_files:
            gt_payload = json.loads(gt_file.read_text(encoding="utf-8"))
            gt_boxes, gt_labels, _ = parse_annotations(gt_payload, label_to_id, score_thr=0.0)

            image_id = str(gt_payload.get("image_id", "")).strip()
            if image_id:
                pred_json_name = Path(image_id).with_suffix(".json").name
            else:
                pred_json_name = gt_file.name

            pred_payload = store.read_by_json_name(pred_json_name)
            if pred_payload is None:
                missing_pred += 1
                pred_boxes, pred_labels, pred_scores = [], [], []
            else:
                pred_boxes, pred_labels, pred_scores = parse_annotations(
                    pred_payload,
                    label_to_id,
                    score_thr=float(args.score_thr),
                )

            gt_records.append({"boxes": gt_boxes, "labels": gt_labels})
            pred_records.append({"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores})
    finally:
        store.close()

    metrics = evaluate_map50(pred_records, gt_records, num_classes=len(classes) + 1)
    print(f"GT images: {len(gt_files)}")
    print(f"Missing prediction json: {missing_pred}")
    print(f"mAP@0.5: {metrics['mAP50']:.6f}")
    for cls_id, ap in metrics["ap_per_class"].items():
        label = id_to_label.get(int(cls_id), str(cls_id))
        print(f"  AP@0.5 [{label}]: {float(ap):.6f}")


if __name__ == "__main__":
    main()
