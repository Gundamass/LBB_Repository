import importlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
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


class DINOv3RoIEmbedder(nn.Module):
    """RoI embedder based on DINOv3 class-token features."""

    def __init__(
        self,
        repo_dir: str,
        model_name: str = "dinov3_vits16",
        weights: Optional[str] = None,
        pretrained: bool = True,
        check_hash: bool = False,
        input_size: int = 224,
        out_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.repo_dir = str(Path(repo_dir).expanduser().resolve())
        self.model_name = str(model_name)
        self.input_size = int(input_size)
        self.freeze_backbone = bool(freeze_backbone)

        repo_root = str(Path(self.repo_dir).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `dinov3.hub.backbones` for DINOv3 metric embedder."
            ) from exc

        if not hasattr(backbones, self.model_name):
            raise ValueError(
                f"DINOv3 metric model `{self.model_name}` not found in dinov3.hub.backbones."
            )
        backbone_ctor = getattr(backbones, self.model_name)

        ctor_kwargs = {"pretrained": bool(pretrained)}
        if weights is not None and str(weights).strip():
            ctor_kwargs["pretrained"] = True
            ctor_kwargs["weights"] = str(weights)
        if "convnext" not in self.model_name and check_hash:
            ctor_kwargs["check_hash"] = True

        try:
            self.backbone = backbone_ctor(**ctor_kwargs)
        except Exception as exc:
            hint = (
                "Failed to load DINOv3 metric backbone. "
                "If network download is blocked, set "
                "`hybrid.metric_embedder.dinov3.weights` to a local .pth checkpoint path."
            )
            raise RuntimeError(hint) from exc
        self.backbone.eval()
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32)
            raw = self._forward_backbone(dummy)
            raw_dim = int(raw.shape[-1])

        self.raw_dim = raw_dim
        self.out_dim = int(out_dim)
        if self.out_dim != self.raw_dim:
            self.proj = nn.Linear(self.raw_dim, self.out_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def _extract_cls_token(self, out) -> torch.Tensor:
        if isinstance(out, dict):
            if "x_norm_clstoken" not in out:
                raise RuntimeError("DINOv3 output dict missing `x_norm_clstoken`.")
            feat = out["x_norm_clstoken"]
        elif isinstance(out, torch.Tensor):
            feat = out
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], dict):
            feat = out[0].get("x_norm_clstoken", None)
            if feat is None:
                raise RuntimeError("DINOv3 output list item missing `x_norm_clstoken`.")
        else:
            raise RuntimeError(f"Unsupported DINOv3 output type: {type(out)}")

        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.dim() != 2:
            raise RuntimeError(f"Expected cls token shape [B, D], got {tuple(feat.shape)}")
        return feat

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(x, is_training=True)
        else:
            out = self.backbone(x, is_training=True)
        feat = self._extract_cls_token(out)
        return F.normalize(feat, dim=-1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"Expected patches [B,3,H,W], got {tuple(patches.shape)}")

        x = patches.float().clamp(0.0, 1.0)
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        x = (x - self.pixel_mean) / self.pixel_std
        feat = self._forward_backbone(x)
        feat = self.proj(feat)
        return F.normalize(feat, dim=-1)


class DinoV3ConvNeXtFPNBackbone(nn.Module):
    """DINOv3 ConvNeXt backbone wrapped by an FPN for Faster R-CNN."""

    def __init__(
        self,
        body: nn.Module,
        out_indices: Sequence[int],
        fpn_out_channels: int = 256,
        norm_intermediate: bool = False,
    ):
        super().__init__()
        self.body = body
        self.out_indices = [int(i) for i in out_indices]
        self.norm_intermediate = bool(norm_intermediate)

        if not hasattr(self.body, "embed_dims"):
            raise ValueError("DINOv3 ConvNeXt backbone must expose `embed_dims`.")

        embed_dims = list(self.body.embed_dims)
        for i in self.out_indices:
            if i < 0 or i >= len(embed_dims):
                raise ValueError(
                    f"Invalid `out_indices` value {i}. "
                    f"Expected range [0, {len(embed_dims) - 1}]."
                )

        in_channels_list = [int(embed_dims[i]) for i in self.out_indices]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=int(fpn_out_channels),
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = int(fpn_out_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.body.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=self.norm_intermediate,
        )
        feat_dict = OrderedDict((str(i), feat) for i, feat in enumerate(features))
        return self.fpn(feat_dict)


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
        metric_cfg = cfg["hybrid"].get("metric_embedder", {})
        metric_name = str(metric_cfg.get("name", "cnn")).lower()
        self.metric_embedder_name = metric_name

        if metric_name == "dinov3":
            dino_cfg = metric_cfg.get("dinov3", {})
            dino_repo = dino_cfg.get("repo_dir", cfg["model"].get("dinov3", {}).get("repo_dir", "./third_party/dinov3"))
            dino_model_name = dino_cfg.get("model_name", "dinov3_vits16")
            dino_weights = dino_cfg.get("weights", None)
            dino_pretrained = bool(dino_cfg.get("pretrained", True))
            dino_check_hash = bool(dino_cfg.get("check_hash", False))
            dino_input_size = int(metric_cfg.get("input_size", 224))
            dino_out_dim = int(metric_cfg.get("out_dim", 128))
            dino_freeze = bool(dino_cfg.get("freeze_backbone", True))

            self.embedder = DINOv3RoIEmbedder(
                repo_dir=dino_repo,
                model_name=dino_model_name,
                weights=dino_weights,
                pretrained=dino_pretrained,
                check_hash=dino_check_hash,
                input_size=dino_input_size,
                out_dim=dino_out_dim,
                freeze_backbone=dino_freeze,
            )
        else:
            emb_dim = int(metric_cfg.get("out_dim", 128))
            self.embedder = PatchEmbedder(emb_dim=emb_dim)

        default_trainable = (self.metric_embedder_name == "cnn")
        self.metric_embedder_trainable = bool(metric_cfg.get("trainable", default_trainable))
        if not self.metric_embedder_trainable:
            self.embedder.eval()
            for p in self.embedder.parameters():
                p.requires_grad = False

        self.metric_crop_size = int(metric_cfg.get("crop_size", getattr(self.embedder, "input_size", 64)))
        metric_dim = int(getattr(self.embedder, "out_dim", 128))

        # index 0 is background and is not used for prototype matching
        self.register_buffer("prototype_bank", torch.zeros(self.num_classes, metric_dim))
        self.register_buffer("prototype_count", torch.zeros(self.num_classes))

    def _build_detector(self, cfg: Dict):
        name = cfg["model"]["detector_name"]
        pretrained = bool(cfg["model"].get("pretrained", True))
        score_thr = float(cfg["model"].get("score_threshold", 0.05))
        nms_thr = float(cfg["model"].get("nms_iou_threshold", 0.5))
        det_per_img = int(cfg["model"].get("detections_per_img", 300))
        det_kwargs = self._collect_detector_kwargs(cfg["model"], include_anchor=False)

        if name == "fasterrcnn_dinov3_convnext_tiny":
            return self._build_dinov3_fasterrcnn(cfg, backbone_name="dinov3_convnext_tiny")
        if name == "fasterrcnn_dinov3_convnext_small":
            return self._build_dinov3_fasterrcnn(cfg, backbone_name="dinov3_convnext_small")
        if name == "fasterrcnn_dinov3_convnext_base":
            return self._build_dinov3_fasterrcnn(cfg, backbone_name="dinov3_convnext_base")
        if name == "fasterrcnn_dinov3_convnext_large":
            return self._build_dinov3_fasterrcnn(cfg, backbone_name="dinov3_convnext_large")

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

        if name == "fcos_resnet50_fpn":
            return self._build_fcos_resnet50_fpn(cfg)

        if name == "retinanet_resnet50_fpn_v2":
            retinanet_kwargs = self._collect_one_stage_detector_kwargs(cfg["model"])
            weights = None
            weights_backbone = self._default_resnet50_weights() if pretrained else None

            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=weights_backbone,
                num_classes=self.num_classes,
                **retinanet_kwargs,
            )
            pretrained_source = str(cfg["model"].get("pretrained_source", "imagenet")).lower()
            if pretrained and pretrained_source == "coco":
                self._try_load_matching_weights(
                    model,
                    torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
                    context="RetinaNet COCO partial init",
                )
            if hasattr(model, "score_thresh"):
                model.score_thresh = score_thr
            if hasattr(model, "nms_thresh"):
                model.nms_thresh = nms_thr
            if hasattr(model, "detections_per_img"):
                model.detections_per_img = det_per_img
            return model

        raise ValueError(f"Unsupported detector: {name}")

    def _default_resnet50_weights(self):
        weights_enum = torchvision.models.ResNet50_Weights
        return getattr(weights_enum, "IMAGENET1K_V2", weights_enum.IMAGENET1K_V1)

    def _try_load_matching_weights(self, model: nn.Module, weights, context: str) -> None:
        """Load only tensors whose names and shapes match the current detector.

        This lets us reuse COCO-trained backbone/FPN/regression weights while keeping
        our small custom classification head shape.
        """
        try:
            source_state = weights.get_state_dict(progress=True, check_hash=True)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("%s skipped: failed to load weights: %s", context, exc)
            return

        target_state = model.state_dict()
        compatible = {
            k: v
            for k, v in source_state.items()
            if k in target_state and tuple(target_state[k].shape) == tuple(v.shape)
        }
        skipped = len(source_state) - len(compatible)
        model.load_state_dict(compatible, strict=False)

        if self.logger is not None:
            self.logger.info(
                "%s loaded %d tensors, skipped %d shape-mismatched tensors",
                context,
                len(compatible),
                skipped,
            )

    def _collect_one_stage_detector_kwargs(self, model_cfg: Dict) -> Dict:
        kwargs = {}

        int_map = {
            "min_size": "min_size",
            "max_size": "max_size",
            "topk_candidates": "topk_candidates",
            "detections_per_img": "detections_per_img",
        }

        for src, dst in int_map.items():
            if src in model_cfg and model_cfg[src] is not None:
                kwargs[dst] = int(model_cfg[src])

        if "score_threshold" in model_cfg:
            kwargs["score_thresh"] = float(model_cfg["score_threshold"])
        elif "score_thresh" in model_cfg:
            kwargs["score_thresh"] = float(model_cfg["score_thresh"])

        if "nms_iou_threshold" in model_cfg:
            kwargs["nms_thresh"] = float(model_cfg["nms_iou_threshold"])
        elif "nms_thresh" in model_cfg:
            kwargs["nms_thresh"] = float(model_cfg["nms_thresh"])

        return kwargs

    def _build_fcos_resnet50_fpn(self, cfg: Dict):
        model_cfg = cfg["model"]
        pretrained = bool(model_cfg.get("pretrained", True))
        pretrained_source = str(model_cfg.get("pretrained_source", "coco")).lower()
        kwargs = self._collect_one_stage_detector_kwargs(model_cfg)
        if "center_sampling_radius" in model_cfg and model_cfg["center_sampling_radius"] is not None:
            kwargs["center_sampling_radius"] = float(model_cfg["center_sampling_radius"])

        weights = None
        # Start from ImageNet backbone weights, then optionally overlay matching
        # COCO detector tensors. If COCO weights are unavailable, this still
        # avoids fully-random initialization on few-shot data.
        weights_backbone = self._default_resnet50_weights() if pretrained else None

        model = torchvision.models.detection.fcos_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=self.num_classes,
            **kwargs,
        )

        if pretrained and pretrained_source == "coco":
            self._try_load_matching_weights(
                model,
                torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT,
                context="FCOS COCO partial init",
            )

        return model

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

    def _collect_detector_kwargs(self, model_cfg: Dict, include_anchor: bool = False) -> Dict:
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

        if include_anchor:
            anchor_gen = self._build_anchor_generator(model_cfg)
            if anchor_gen is not None:
                kwargs["rpn_anchor_generator"] = anchor_gen

        return kwargs

    def _import_dinov3_backbones(self, repo_dir: str):
        repo_path = Path(repo_dir).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv3 repo not found: {repo_path}. "
                "Please clone https://github.com/facebookresearch/dinov3 "
                "or set `model.dinov3.repo_dir` correctly."
            )

        repo_root = str(repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `dinov3.hub.backbones`. "
                "Please ensure DINOv3 repo and dependencies are available."
            ) from exc
        return backbones

    def _build_dinov3_fasterrcnn(self, cfg: Dict, backbone_name: str):
        model_cfg = cfg["model"]
        dino_cfg = model_cfg.get("dinov3", {})

        repo_dir = dino_cfg.get("repo_dir", "./third_party/dinov3")
        backbones = self._import_dinov3_backbones(repo_dir)

        if not hasattr(backbones, backbone_name):
            raise ValueError(
                f"Backbone `{backbone_name}` not found in DINOv3 hub backbones."
            )

        backbone_ctor = getattr(backbones, backbone_name)
        use_pretrained = bool(model_cfg.get("pretrained", True))
        weights_spec = dino_cfg.get("weights", None)
        check_hash = bool(dino_cfg.get("check_hash", False))

        try:
            if use_pretrained and weights_spec:
                dino_body = backbone_ctor(
                    pretrained=True,
                    weights=str(weights_spec),
                    check_hash=check_hash,
                )
            else:
                dino_body = backbone_ctor(pretrained=False)
        except Exception as exc:
            hint = (
                "If you want pretrained DINOv3 weights, download the checkpoint URL/path "
                "from the official DINOv3 access page and set `model.dinov3.weights`."
            )
            raise RuntimeError(f"Failed to build DINOv3 backbone: {exc}. {hint}") from exc

        out_indices = dino_cfg.get("out_indices", [0, 1, 2, 3])
        fpn_out_channels = int(dino_cfg.get("fpn_out_channels", 256))
        norm_intermediate = bool(dino_cfg.get("norm_intermediate", False))
        backbone = DinoV3ConvNeXtFPNBackbone(
            body=dino_body,
            out_indices=out_indices,
            fpn_out_channels=fpn_out_channels,
            norm_intermediate=norm_intermediate,
        )

        det_kwargs = self._collect_detector_kwargs(model_cfg, include_anchor=False)

        # Build anchors for FPN feature levels (including last pooled level).
        anchor_sizes_cfg = model_cfg.get("anchor_sizes", [[8], [16], [32], [64], [128]])
        anchor_sizes = tuple(tuple(int(max(1, x)) for x in level) for level in anchor_sizes_cfg)

        aspect_ratios_cfg = model_cfg.get("anchor_aspect_ratios", [[0.5, 1.0, 2.0]])
        aspect_ratios = tuple(tuple(float(r) for r in level) for level in aspect_ratios_cfg)
        if len(aspect_ratios) == 1:
            aspect_ratios = aspect_ratios * len(anchor_sizes)
        if len(aspect_ratios) != len(anchor_sizes):
            raise ValueError(
                f"anchor_aspect_ratios length ({len(aspect_ratios)}) must match "
                f"anchor_sizes length ({len(anchor_sizes)}), or provide one list to broadcast."
            )

        rpn_anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

        model = FasterRCNN(
            backbone=backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            **det_kwargs,
        )
        return model

    def _crop_box(self, image: torch.Tensor, box: torch.Tensor, out_size: Optional[int] = None) -> Optional[torch.Tensor]:
        if out_size is None:
            out_size = self.metric_crop_size
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

    def _embed_patch(self, patch: torch.Tensor, allow_grad: bool = False) -> torch.Tensor:
        if allow_grad and self.metric_embedder_trainable:
            emb = self.embedder(patch)
        else:
            with torch.no_grad():
                emb = self.embedder(patch)
        return emb.squeeze(0)

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
                emb = self._embed_patch(patch, allow_grad=False)
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
        if not self.metric_embedder_trainable:
            return torch.zeros((), dtype=torch.float32, device=device)

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
                emb = self._embed_patch(patch, allow_grad=True)
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
            emb = self._embed_patch(patch.to(image.device), allow_grad=False)
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
            emb = self._embed_patch(patch.to(image.device), allow_grad=False)

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
            "Model built: detector=%s classes=%d hybrid(proposal=%s metric=%s embedder=%s trainable=%s)",
            cfg["model"]["detector_name"],
            model.num_classes,
            model.enable_classical_proposals,
            model.enable_metric_rescore,
            model.metric_embedder_name,
            model.metric_embedder_trainable,
        )
    return model
