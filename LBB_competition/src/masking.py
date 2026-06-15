from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - handled at runtime with a clearer error
    ndi = None


@dataclass
class MaskStats:
    foreground_ratio: float
    threshold: float
    fallback: bool
    reason: str


def _require_scipy() -> None:
    if ndi is None:
        raise RuntimeError(
            "scipy is required for mask generation. Install it with: "
            "conda install scipy -y"
        )


def _border_values(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    pad = max(4, min(40, min(h, w) // 20))
    values = [
        gray[:pad, :].reshape(-1),
        gray[-pad:, :].reshape(-1),
        gray[:, :pad].reshape(-1),
        gray[:, -pad:].reshape(-1),
    ]
    return np.concatenate(values, axis=0)


def _estimate_background_threshold(
    gray: np.ndarray,
    white_threshold: float,
    min_background_threshold: float,
    adaptive_margin: float,
) -> float:
    border = _border_values(gray)
    border_hi = float(np.percentile(border, 85))
    adaptive = border_hi - float(adaptive_margin)
    return float(min(float(white_threshold), max(float(min_background_threshold), adaptive)))


def segment_laptop_mask(image: Image.Image, params: Optional[Dict] = None) -> Tuple[np.ndarray, MaskStats]:
    """Segment the non-background product area from a white-background inspection image.

    This is intentionally conservative: if the estimated foreground is implausibly
    small or huge, it returns a full-image mask so inference behavior does not
    silently collapse.
    """
    _require_scipy()
    params = params or {}

    white_threshold = float(params.get("white_threshold", 245.0))
    min_background_threshold = float(params.get("min_background_threshold", 220.0))
    adaptive_margin = float(params.get("adaptive_margin", 8.0))
    chroma_threshold = float(params.get("chroma_threshold", 18.0))
    edge_threshold = float(params.get("edge_threshold", 18.0))
    min_component_area_ratio = float(params.get("min_component_area_ratio", 0.00025))
    min_foreground_ratio = float(params.get("min_foreground_ratio", 0.01))
    max_foreground_ratio = float(params.get("max_foreground_ratio", 0.98))
    close_iterations = int(params.get("close_iterations", 2))
    dilate_iterations = int(params.get("dilate_iterations", 8))
    fallback_to_full = bool(params.get("fallback_to_full", True))

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)
    chroma = arr.max(axis=2) - arr.min(axis=2)

    threshold = _estimate_background_threshold(
        gray,
        white_threshold=white_threshold,
        min_background_threshold=min_background_threshold,
        adaptive_margin=adaptive_margin,
    )

    bright = gray >= threshold
    low_chroma = chroma <= chroma_threshold

    if edge_threshold >= 0:
        smooth = ndi.gaussian_filter(gray, sigma=1.0)
        gx = ndi.sobel(smooth, axis=1)
        gy = ndi.sobel(smooth, axis=0)
        grad = np.hypot(gx, gy)
        low_edge = grad <= edge_threshold
        very_bright = gray >= min(252.0, threshold + 8.0)
        bg_candidate = bright & low_chroma & (low_edge | very_bright)
    else:
        bg_candidate = bright & low_chroma

    structure = np.ones((3, 3), dtype=bool)
    labels, _ = ndi.label(bg_candidate, structure=structure)

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels > 0]

    if border_labels.size == 0:
        foreground = np.ones((h, w), dtype=bool)
    else:
        background = np.isin(labels, border_labels)
        foreground = ~background

    if close_iterations > 0:
        foreground = ndi.binary_closing(
            foreground,
            structure=structure,
            iterations=close_iterations,
        )

    foreground = ndi.binary_fill_holes(foreground)

    min_area = max(16, int(round(float(h * w) * min_component_area_ratio)))
    fg_labels, num_fg = ndi.label(foreground, structure=structure)
    if num_fg > 0 and min_area > 0:
        counts = np.bincount(fg_labels.reshape(-1))
        keep = counts >= min_area
        keep[0] = False
        foreground = keep[fg_labels]

    foreground = ndi.binary_fill_holes(foreground)

    if dilate_iterations > 0:
        foreground = ndi.binary_dilation(
            foreground,
            structure=structure,
            iterations=dilate_iterations,
        )

    foreground = foreground.astype(bool)
    ratio = float(foreground.mean())
    fallback = False
    reason = ""
    if fallback_to_full and ratio < min_foreground_ratio:
        foreground = np.ones((h, w), dtype=bool)
        fallback = True
        reason = f"foreground ratio {ratio:.6f} < {min_foreground_ratio:.6f}"
        ratio = 1.0
    elif fallback_to_full and ratio > max_foreground_ratio:
        foreground = np.ones((h, w), dtype=bool)
        fallback = True
        reason = f"foreground ratio {ratio:.6f} > {max_foreground_ratio:.6f}"
        ratio = 1.0

    return foreground, MaskStats(
        foreground_ratio=ratio,
        threshold=threshold,
        fallback=fallback,
        reason=reason,
    )


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(path)


def load_mask_for_image(
    masks_dir: Path,
    image_name: str,
    expected_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    mask_path = Path(masks_dir) / (Path(image_name).stem + ".png")
    if not mask_path.exists():
        return None

    mask_img = Image.open(mask_path).convert("L")
    if expected_size is not None and mask_img.size != expected_size:
        mask_img = mask_img.resize(expected_size, resample=Image.NEAREST)
    return np.asarray(mask_img, dtype=np.uint8) > 127


def bbox_mask_coverage(mask: np.ndarray, box) -> Tuple[float, bool]:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    ix1 = int(np.floor(max(0.0, min(float(w), x1))))
    iy1 = int(np.floor(max(0.0, min(float(h), y1))))
    ix2 = int(np.ceil(max(0.0, min(float(w), x2))))
    iy2 = int(np.ceil(max(0.0, min(float(h), y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, False

    crop = mask[iy1:iy2, ix1:ix2]
    coverage = float(crop.mean()) if crop.size else 0.0

    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    center_inside = 0 <= cx < w and 0 <= cy < h and bool(mask[cy, cx])
    return coverage, center_inside


def should_keep_box_by_mask(
    mask: np.ndarray,
    box,
    min_coverage: float,
    keep_rule: str = "center_or_coverage",
) -> Tuple[bool, float, bool]:
    coverage, center_inside = bbox_mask_coverage(mask, box)
    rule = str(keep_rule or "center_or_coverage").lower()

    if rule == "center":
        keep = center_inside
    elif rule == "coverage":
        keep = coverage >= float(min_coverage)
    elif rule == "center_and_coverage":
        keep = center_inside and coverage >= float(min_coverage)
    else:
        keep = center_inside or coverage >= float(min_coverage)

    return bool(keep), coverage, center_inside


def clean_binary_mask(mask: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
    _require_scipy()
    params = params or {}
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape[:2]

    close_iterations = int(params.get("close_iterations", 1))
    dilate_iterations = int(params.get("dilate_iterations", 4))
    keep_largest_components = int(params.get("keep_largest_components", 1))
    min_component_area_ratio = float(params.get("min_component_area_ratio", 0.00025))

    structure = np.ones((3, 3), dtype=bool)

    if close_iterations > 0:
        mask = ndi.binary_closing(mask, structure=structure, iterations=close_iterations)
    mask = ndi.binary_fill_holes(mask)

    labels, num = ndi.label(mask, structure=structure)
    if num > 0:
        counts = np.bincount(labels.reshape(-1))
        min_area = max(16, int(round(float(h * w) * min_component_area_ratio)))
        comp_ids = [i for i in range(1, len(counts)) if counts[i] >= min_area]
        comp_ids.sort(key=lambda i: counts[i], reverse=True)
        if keep_largest_components > 0:
            comp_ids = comp_ids[:keep_largest_components]
        if comp_ids:
            mask = np.isin(labels, comp_ids)
        else:
            mask = np.zeros_like(mask, dtype=bool)

    mask = ndi.binary_fill_holes(mask)
    if dilate_iterations > 0:
        mask = ndi.binary_dilation(mask, structure=structure, iterations=dilate_iterations)
    return mask.astype(bool)


def mask_to_box(mask: np.ndarray, padding: int = 0) -> Optional[np.ndarray]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - int(padding))
    y1 = max(0, int(ys.min()) - int(padding))
    x2 = min(w - 1, int(xs.max()) + int(padding))
    y2 = min(h - 1, int(ys.max()) + int(padding))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def sample_points_from_mask(
    mask: np.ndarray,
    count: int,
    margin: int = 0,
) -> np.ndarray:
    _require_scipy()
    mask = np.asarray(mask).astype(bool)
    if margin > 0:
        structure = np.ones((3, 3), dtype=bool)
        mask = ndi.binary_erosion(mask, structure=structure, iterations=int(margin))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if xs.size <= count:
        idx = np.arange(xs.size)
    else:
        # Deterministic spatial spread: sort by y/x and take evenly spaced samples.
        order = np.lexsort((xs, ys))
        idx = order[np.linspace(0, len(order) - 1, num=count).round().astype(int)]

    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def sample_background_points(mask: np.ndarray, count: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape[:2]
    bg = ~mask
    ys, xs = np.where(bg)
    pts = []

    corners = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w // 2, 0),
        (w // 2, h - 1),
        (0, h // 2),
        (w - 1, h // 2),
    ]
    for x, y in corners:
        if len(pts) >= count:
            break
        if 0 <= x < w and 0 <= y < h and bg[y, x]:
            pts.append((x, y))

    remaining = count - len(pts)
    if remaining > 0 and xs.size > 0:
        order = np.lexsort((xs, ys))
        take = order[np.linspace(0, len(order) - 1, num=min(remaining, len(order))).round().astype(int)]
        pts.extend((int(xs[i]), int(ys[i])) for i in take)

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts[:count], dtype=np.float32)
