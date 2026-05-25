from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms


class PatchEmbedder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        return z


class HybridDefectModel(nn.Module):
    def __init__(self, cfg: Dict, logger=None):
        super().__init__()
        self.cfg = cfg
        self.logger = logger

        self.class_to_id = {k: int(v) for k, v in cfg["classes"].items() if k != "background"}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        self.num_classes = len(cfg["classes"])

        self.detector = self._build_detector(cfg)

        self.enable_classical_proposals = bool(cfg["hybrid"].get("enable_classical_proposals", True))
        self.enable_metric_rescore = bool(cfg["hybrid"].get("enable_metric_rescore", True))
        self.metric_weight = float(cfg["hybrid"].get("metric_weight", 0.25))
        self.temperature = float(cfg["hybrid"].get("metric_temperature", 0.2))

        self.embedder = PatchEmbedder(emb_dim=128)

        # index 0 is background and is not used for prototype matching
        self.register_buffer("prototype_bank", torch.zeros(self.num_classes, 128))
        self.register_buffer("prototype_count", torch.zeros(self.num_classes))

    def _build_detector(self, cfg: Dict):
        name = cfg["model"]["detector_name"]
        pretrained = bool(cfg["model"].get("pretrained", True))
        score_thr = float(cfg["model"].get("score_threshold", 0.05))
        nms_thr = float(cfg["model"].get("nms_iou_threshold", 0.5))
        det_per_img = int(cfg["model"].get("detections_per_img", 300))
        det_kwargs = self._collect_detector_kwargs(cfg["model"])

        if name == "fasterrcnn_resnet50_fpn_v2":
            if pretrained:
                weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            else:
                weights = None

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                **det_kwargs,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

            model.roi_heads.score_thresh = score_thr
            model.roi_heads.nms_thresh = nms_thr
            model.roi_heads.detections_per_img = det_per_img
            return model

        if name == "retinanet_resnet50_fpn_v2":
            if pretrained:
                # For custom num_classes we cannot directly use COCO full-head weights.
                # Use ImageNet backbone init instead.
                weights = None
                weights_backbone = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
                weights_backbone = None

            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=weights_backbone,
                num_classes=self.num_classes,
                **det_kwargs,
            )
            if hasattr(model, "score_thresh"):
                model.score_thresh = score_thr
            if hasattr(model, "nms_thresh"):
                model.nms_thresh = nms_thr
            if hasattr(model, "detections_per_img"):
                model.detections_per_img = det_per_img
            return model

        raise ValueError(f"Unsupported detector: {name}")

    def _build_anchor_generator(self, model_cfg: Dict) -> Optional[AnchorGenerator]:
        sizes_cfg = model_cfg.get("anchor_sizes")
        ratios_cfg = model_cfg.get("anchor_aspect_ratios")
        if sizes_cfg is None and ratios_cfg is None:
            return None

        if sizes_cfg is None:
            sizes_cfg = [[32], [64], [128], [256], [512]]
        sizes = tuple(tuple(int(max(1, s)) for s in level) for level in sizes_cfg)

        if ratios_cfg is None:
            ratios = ((0.5, 1.0, 2.0),) * len(sizes)
        else:
            ratios = tuple(tuple(float(r) for r in level) for level in ratios_cfg)
            if len(ratios) == 1 and len(sizes) > 1:
                ratios = ratios * len(sizes)
            if len(ratios) != len(sizes):
                raise ValueError(
                    f"anchor_aspect_ratios length ({len(ratios)}) must match "
                    f"anchor_sizes length ({len(sizes)}), or provide a single ratio list."
                )

        return AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

    def _collect_detector_kwargs(self, model_cfg: Dict) -> Dict:
        kwargs = {}

        int_keys = [
            "min_size",
            "max_size",
            "rpn_pre_nms_top_n_train",
            "rpn_pre_nms_top_n_test",
            "rpn_post_nms_top_n_train",
            "rpn_post_nms_top_n_test",
            "rpn_batch_size_per_image",
            "box_batch_size_per_image",
            "box_detections_per_img",
        ]
        float_keys = [
            "rpn_nms_thresh",
            "rpn_fg_iou_thresh",
            "rpn_bg_iou_thresh",
            "rpn_positive_fraction",
            "box_positive_fraction",
            "box_fg_iou_thresh",
            "box_bg_iou_thresh",
            "box_score_thresh",
            "box_nms_thresh",
        ]

        for k in int_keys:
            if k in model_cfg and model_cfg[k] is not None:
                kwargs[k] = int(model_cfg[k])
        for k in float_keys:
            if k in model_cfg and model_cfg[k] is not None:
                kwargs[k] = float(model_cfg[k])

        anchor_gen = self._build_anchor_generator(model_cfg)
        if anchor_gen is not None:
            kwargs["rpn_anchor_generator"] = anchor_gen

        return kwargs

    def _crop_box(self, image: torch.Tensor, box: torch.Tensor, out_size: int = 64) -> Optional[torch.Tensor]:
        h, w = image.shape[1], image.shape[2]
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(1, min(w, x2)))
        y2 = int(max(1, min(h, y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        patch = image[:, y1:y2, x1:x2]
        if patch.numel() == 0:
            return None

        patch = patch.unsqueeze(0)
        patch = F.interpolate(patch, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return patch

    @torch.no_grad()
    def update_prototypes(self, images: List[torch.Tensor], targets: List[Dict]) -> None:
        self.embedder.eval()
        device = next(self.parameters()).device

        for img, tgt in zip(images, targets):
            boxes = tgt["boxes"]
            labels = tgt["labels"]
            for box, label in zip(boxes, labels):
                cls = int(label.item())
                if cls <= 0:
                    continue
                patch = self._crop_box(img.to(device), box.to(device))
                if patch is None:
                    continue
                emb = self.embedder(patch).squeeze(0)
                old = self.prototype_bank[cls]
                cnt = self.prototype_count[cls].item()
                new_proto = (old * cnt + emb) / (cnt + 1.0)
                self.prototype_bank[cls] = F.normalize(new_proto, dim=0)
                self.prototype_count[cls] = cnt + 1.0

    def compute_center_loss(self, images: List[torch.Tensor], targets: List[Dict]) -> torch.Tensor:
        """Center-style metric loss to stabilize few-shot embeddings.

        Uses current prototype bank as class centers and penalizes (1 - cosine).
        """
        device = next(self.parameters()).device
        losses: List[torch.Tensor] = []
        self.embedder.train()

        for img, tgt in zip(images, targets):
            boxes = tgt["boxes"]
            labels = tgt["labels"]
            for box, label in zip(boxes, labels):
                cls = int(label.item())
                if cls <= 0:
                    continue
                if self.prototype_count[cls].item() < 1:
                    continue

                patch = self._crop_box(img.to(device), box.to(device))
                if patch is None:
                    continue
                emb = self.embedder(patch).squeeze(0)
                proto = self.prototype_bank[cls].detach()
                cos = torch.sum(emb * proto)
                losses.append(1.0 - cos)

        if not losses:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(losses).mean()

    def _classical_proposals(self, image: torch.Tensor, max_k: int = 80) -> torch.Tensor:
        arr = image.detach().cpu().numpy()
        gray = arr.mean(axis=0)

        # Light high-pass response approximating top-hat behavior.
        blur = gray.copy()
        blur = (
            np.pad(blur, 1, mode="reflect")[:-2, 1:-1]
            + np.pad(blur, 1, mode="reflect")[2:, 1:-1]
            + np.pad(blur, 1, mode="reflect")[1:-1, :-2]
            + np.pad(blur, 1, mode="reflect")[1:-1, 2:]
            + 4.0 * blur
        ) / 8.0
        hp = np.abs(gray - blur)

        h, w = hp.shape
        k = min(max_k, h * w)
        flat_idx = np.argpartition(hp.reshape(-1), -k)[-k:]

        side = max(6, int(min(h, w) * 0.02))
        boxes = []
        for idx in flat_idx:
            y = idx // w
            x = idx % w
            x1 = max(0, x - side)
            y1 = max(0, y - side)
            x2 = min(w - 1, x + side)
            y2 = min(h - 1, y + side)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32, device=image.device)

        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=image.device)
        return boxes_t

    @torch.no_grad()
    def _rescore_with_prototypes(self, image: torch.Tensor, pred: Dict) -> Dict:
        if pred["boxes"].numel() == 0:
            return pred

        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        new_scores = scores.clone()

        for i in range(boxes.shape[0]):
            cls = int(labels[i].item())
            if cls <= 0:
                continue
            if self.prototype_count[cls].item() < 1:
                continue

            patch = self._crop_box(image, boxes[i])
            if patch is None:
                continue
            emb = self.embedder(patch.to(image.device)).squeeze(0)
            proto = self.prototype_bank[cls]
            sim = torch.sum(emb * proto)
            sim = torch.clamp((sim + 1.0) * 0.5, 0.0, 1.0)
            score = scores[i]
            fused = (1.0 - self.metric_weight) * score + self.metric_weight * sim
            new_scores[i] = fused

        keep = nms(boxes, new_scores, iou_threshold=0.5)
        pred["boxes"] = boxes[keep]
        pred["labels"] = labels[keep]
        pred["scores"] = new_scores[keep]
        return pred

    @torch.no_grad()
    def _merge_proposals(self, image: torch.Tensor, pred: Dict) -> Dict:
        if not self.enable_classical_proposals:
            return pred

        prop_boxes = self._classical_proposals(
            image,
            max_k=int(self.cfg["hybrid"].get("proposal_max_per_image", 80)),
        )
        if prop_boxes.numel() == 0:
            return pred

        det_boxes = pred["boxes"]
        det_labels = pred["labels"]
        det_scores = pred["scores"]

        extra_boxes = []
        extra_labels = []
        extra_scores = []

        # Label each proposal by nearest prototype if available.
        valid_cls = [c for c in range(1, self.num_classes) if self.prototype_count[c].item() > 0]
        if not valid_cls:
            return pred

        for box in prop_boxes:
            patch = self._crop_box(image, box)
            if patch is None:
                continue
            emb = self.embedder(patch.to(image.device)).squeeze(0)

            sims = []
            for cls in valid_cls:
                sim = torch.sum(emb * self.prototype_bank[cls])
                sims.append(sim)
            sims_t = torch.stack(sims)
            best_idx = int(torch.argmax(sims_t).item())
            best_cls = valid_cls[best_idx]
            best_sim = torch.clamp((sims_t[best_idx] + 1.0) * 0.5, 0.0, 1.0)

            if best_sim.item() < 0.55:
                continue

            extra_boxes.append(box)
            extra_labels.append(best_cls)
            extra_scores.append(best_sim * 0.4)

        if not extra_boxes:
            return pred

        ext_boxes = torch.stack(extra_boxes)
        ext_labels = torch.tensor(extra_labels, dtype=torch.int64, device=image.device)
        ext_scores = torch.stack(extra_scores).to(image.device)

        all_boxes = torch.cat([det_boxes, ext_boxes], dim=0)
        all_labels = torch.cat([det_labels, ext_labels], dim=0)
        all_scores = torch.cat([det_scores, ext_scores], dim=0)

        keep = nms(all_boxes, all_scores, iou_threshold=0.5)

        pred["boxes"] = all_boxes[keep]
        pred["labels"] = all_labels[keep]
        pred["scores"] = all_scores[keep]
        return pred

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        if self.training:
            return self.detector(images, targets)

        preds = self.detector(images)
        out = []
        for img, pred in zip(images, preds):
            pred = self._merge_proposals(img, pred)
            if self.enable_metric_rescore:
                pred = self._rescore_with_prototypes(img, pred)
            out.append(pred)
        return out


def build_model(cfg: Dict, logger=None) -> HybridDefectModel:
    model = HybridDefectModel(cfg=cfg, logger=logger)
    if logger is not None:
        logger.info(
            "Model built: detector=%s classes=%d hybrid(proposal=%s metric=%s)",
            cfg["model"]["detector_name"],
            model.num_classes,
            model.enable_classical_proposals,
            model.enable_metric_rescore,
        )
    return model
