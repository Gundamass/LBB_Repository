# LBB 少样本外观缺陷检测方案 C

这个仓库的项目根目录就是 GitHub 仓库根目录，不再额外套一层 `LBB_competition/`。也就是说，克隆仓库后可以直接在仓库根目录运行 `train.py`、`inference.py` 和各类配置文件。

## 文件说明

- `config.yaml`: 基础方案 C 配置。
- `config.no_classic.thr001_nms0445.ensemble_seed123.yaml`: 当前已知线上最优配置，线上 mAP@0.5 为 `0.283265`。
- `config.no_classic.thr001_nms0455.ensemble_seed123.yaml`: 保留的相邻 NMS 配置，线上 mAP@0.5 为 `0.283147`。
- `train.py`: 训练、验证、早停和 checkpoint 保存入口。
- `inference.py`: TTA 推理、ensemble、JSON 导出和提交 ZIP 打包入口。
- `evaluate_local_map.py`: 使用本地标注数据评估 mAP@0.5。
- `src/dataset.py`: 数据解析、增强和 dataloader 实现。
- `src/model.py`: 检测器、传统候选框分支和 metric rescore 实现。
- `src/masking.py`: 笔记本区域 mask 后处理相关工具。
- `src/utils.py`: 配置加载、日志、硬件检测和辅助函数。
- `checkpoints/seed123/best_model.pt`: seed-123 模型权重，已通过 Git LFS 提交。

## 环境准备

建议使用已有的 `lbb` conda 环境：

```bash
conda activate lbb
```

如需重新安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 数据目录

训练和推理默认读取仓库根目录下的 `初赛数据/`：

```text
初赛数据/
├── 训练集/
│   ├── 正样本/
│   └── 负样本/
└── 测试集/
    └── image/
```

数据集、手标数据、日志、输出 ZIP、SAM/DINO 本地权重等都不会提交到 Git。

## 基础训练

在仓库根目录执行：

```bash
conda activate lbb
python3 train.py --config ./config.yaml
```

训练完成后默认会保存：

```text
./checkpoints/best_model.pt
./checkpoints/latest.pt
```

## 基础推理

```bash
conda activate lbb
python3 inference.py --config ./config.yaml --ckpt ./checkpoints/best_model.pt
```

提交 ZIP 会生成在：

```text
./outputs/
```

## 复现当前最优线上分数

当前已知最优线上提交：

- ZIP 文件名：`少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123.zip`
- 线上 mAP@0.5：`0.283265`
- 最优配置：`config.no_classic.thr001_nms0445.ensemble_seed123.yaml`
- 关键设置：关闭 classical proposal 分支，置信度阈值 `0.01`，按类别 NMS IoU `0.445`，两个 checkpoint ensemble，TTA 使用尺度 `[1.0, 1.1, 0.9]` 加水平翻转。

先进入仓库根目录：

```bash
cd /home/heqing
conda activate lbb
```

如果是从 GitHub 新克隆的仓库，请先拉取 LFS 权重：

```bash
git lfs pull
```

本仓库已提交 seed-123 权重：

```text
./checkpoints/seed123/best_model.pt
```

注意：`/home/heqing/LBB_competition/checkpoints/seed123/latest.pt` 和 `best_model.pt` 的 SHA256 完全一致，所以本次只提交 `best_model.pt`，不重复提交 `latest.pt`。如果本地代码需要 `latest.pt`，可以这样恢复：

```bash
cp ./checkpoints/seed123/best_model.pt ./checkpoints/seed123/latest.pt
```

当前最优线上分数使用两个模型 ensemble，因此还需要准备 seed-42 权重：

```bash
python3 train.py --config ./config.no_classic.thr001_nms0445.ensemble_seed123.yaml
mkdir -p ./checkpoints/seed42
cp ./checkpoints/best_model.pt ./checkpoints/seed42/best_model.pt
```

然后使用 seed-42 和 seed-123 两个 checkpoint 做 ensemble 推理：

```bash
python3 inference.py \
  --config ./config.no_classic.thr001_nms0445.ensemble_seed123.yaml \
  --ckpt ./checkpoints/seed42/best_model.pt,./checkpoints/seed123/best_model.pt
```

最终生成的提交文件：

```text
./outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123.zip
```

如果需要重新训练 seed-123，而不是使用仓库里的 LFS 权重，可以运行：

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

base_path = Path("config.no_classic.thr001_nms0445.ensemble_seed123.yaml")
cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
cfg["seed"] = 123
cfg["system"]["checkpoints_dir"] = "./checkpoints/seed123"

out_path = Path("config.no_classic.thr001_nms0445.seed123.train.yaml")
out_path.write_text(
    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
print(out_path)
PY

python3 train.py --config ./config.no_classic.thr001_nms0445.seed123.train.yaml
```

## 本地 mAP 评估

如果本地有手标测试集标注，可以用：

```bash
python3 evaluate_local_map.py \
  --pred_dir ./outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123 \
  --gt_dir ./手标数据
```

具体 `--gt_dir` 以本地手标数据实际目录为准。

## 备注

- 训练时会把多边形标注转换成检测框。
- 训练包含强增强：mixup、copy-paste、颜色增强、cutout 和翻转。
- 当前最优提交来自推理侧优化：关闭 classical proposal、低阈值召回、TTA、双 checkpoint ensemble 和更细的 NMS 网格。
- 验证指标是 mAP@0.5，代码里的自定义实现按比赛说明实现。
