import argparse
import copy
import itertools
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import detect_hardware, ensure_dirs, load_config, seed_everything, setup_logger
from train import strip_targets_for_model, validate


def trial_space(detectors: List[str]) -> List[Dict]:
    lrs = [1e-4, 2e-4, 4e-4]
    wds = [1e-4, 5e-4]
    center_ws = [0.0, 0.03, 0.06]
    metric_ws = [0.15, 0.25, 0.35]
    score_thrs = [0.03, 0.05, 0.08]
    proposal_ks = [60, 80, 120]

    combos = []
    for d, lr, wd, cw, mw, st, pk in itertools.product(
        detectors,
        lrs,
        wds,
        center_ws,
        metric_ws,
        score_thrs,
        proposal_ks,
    ):
        combos.append(
            {
                "model.detector_name": d,
                "training.lr": lr,
                "training.weight_decay": wd,
                "training.center_loss_weight": cw,
                "hybrid.metric_weight": mw,
                "model.score_threshold": st,
                "hybrid.proposal_max_per_image": pk,
            }
        )
    return combos


def set_nested(cfg: Dict, key: str, value):
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict, overrides: Dict) -> Dict:
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        set_nested(out, k, v)
    return out


def run_trial(cfg: Dict, trial_id: int, hw, logger) -> Dict:
    device = torch.device(hw.device)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    train_loader, val_loader, _, _ = build_dataloaders(cfg, hw, logger=None)
    model = build_model(cfg, logger=None).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"]["epochs"]),
        eta_min=float(cfg["training"]["lr_scheduler"].get("min_lr", 1e-6)),
    )

    amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    center_w = float(cfg["training"].get("center_loss_weight", 0.0))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 5.0))

    best_map = -1.0
    best_epoch = -1

    epochs = int(cfg["training"]["epochs"])
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"trial{trial_id} ep{epoch}/{epochs}", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            model_targets = strip_targets_for_model(targets, device=device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                loss_dict = model(images, model_targets)
                det_loss = sum(loss_dict.values())
                if center_w > 0:
                    center_loss = model.compute_center_loss(images, model_targets)
                    loss = det_loss + center_w * center_loss
                else:
                    center_loss = torch.zeros((), device=device)
                    loss = det_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                model.update_prototypes(images, model_targets)

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", det=f"{det_loss.item():.4f}", ctr=f"{center_loss.item():.4f}")

        scheduler.step()
        train_loss = running / max(1, len(train_loader))
        metrics = validate(model, val_loader, device, amp_enabled, logger)
        val_map = float(metrics["mAP50"])

        logger.info(
            "trial=%d epoch=%d train_loss=%.6f val_mAP50=%.6f lr=%.8f",
            trial_id,
            epoch,
            train_loss,
            val_map,
            optimizer.param_groups[0]["lr"],
        )

        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch

    elapsed = time.time() - t0

    result = {
        "trial_id": trial_id,
        "best_map50": best_map,
        "best_epoch": best_epoch,
        "elapsed_sec": round(elapsed, 2),
        "detector": cfg["model"]["detector_name"],
        "lr": cfg["training"]["lr"],
        "weight_decay": cfg["training"]["weight_decay"],
        "center_loss_weight": cfg["training"].get("center_loss_weight", 0.0),
        "metric_weight": cfg["hybrid"].get("metric_weight", 0.25),
        "proposal_max_per_image": cfg["hybrid"].get("proposal_max_per_image", 80),
        "score_threshold": cfg["model"].get("score_threshold", 0.05),
    }

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main(config_path: str, n_trials: int, epochs: int, out_dir: str, seed: int, detectors: List[str]):
    base_cfg = load_config(config_path)

    ensure_dirs([out_dir, base_cfg["system"]["logs_dir"]])
    logger = setup_logger(base_cfg["system"]["logs_dir"], name="autotune")

    hw = detect_hardware()
    if hw.device != "cuda":
        raise RuntimeError("Autotune requires CUDA. Please run in GPU environment.")

    logger.info("Autotune hardware: %s | GPUs=%d | names=%s", hw.device, hw.gpu_count, hw.gpu_names)

    space = trial_space(detectors=detectors)
    rng = random.Random(seed)
    rng.shuffle(space)
    picked = space[:n_trials]

    logger.info("Picked %d trials from %d candidates", len(picked), len(space))

    results = []
    for i, overrides in enumerate(picked, 1):
        cfg = apply_overrides(base_cfg, overrides)
        cfg["training"]["epochs"] = int(epochs)
        cfg["training"]["early_stopping_patience"] = max(2, min(4, int(epochs)))
        cfg["training"]["batch_size"] = int(min(base_cfg["training"].get("batch_size", 4), 4))
        cfg["seed"] = int(base_cfg.get("seed", 42))

        logger.info("===== Trial %d/%d =====", i, len(picked))
        logger.info("Overrides: %s", json.dumps(overrides, ensure_ascii=False))

        result = run_trial(cfg, i, hw, logger)
        results.append(result)
        logger.info("Trial %d result: %s", i, result)

        with open(Path(out_dir) / "autotune_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    results.sort(key=lambda x: x["best_map50"], reverse=True)
    best = results[0]

    best_overrides = picked[best["trial_id"] - 1]
    best_cfg = apply_overrides(base_cfg, best_overrides)

    best_cfg_path = Path(out_dir) / "config.autotuned.yaml"
    import yaml

    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    summary_path = Path(out_dir) / "autotune_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "all_results": results}, f, ensure_ascii=False, indent=2)

    logger.info("Autotune finished. Best=%s", best)
    logger.info("Best config saved: %s", best_cfg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="./outputs/autotune")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--detectors",
        type=str,
        default="fasterrcnn_resnet50_fpn_v2,retinanet_resnet50_fpn_v2",
        help="Comma-separated detector names.",
    )
    args = parser.parse_args()

    detectors = [x.strip() for x in args.detectors.split(",") if x.strip()]
    main(args.config, args.n_trials, args.epochs, args.out_dir, args.seed, detectors)
