# LBB 比赛方案 C 流程

## 文件说明
- `config.yaml`: 统一配置文件，包含数据路径、训练参数、推理参数和混合分支开关
- `config.dinov3.yaml`: DINOv3-ConvNeXt + FasterRCNN 训练配置
- `dataset.py`: 数据集模块入口转发
- `model.py`: 模型模块入口转发
- `train.py`: 训练、验证、早停和 checkpoint 保存
- `inference.py`: TTA 推理、JSON 导出和提交 ZIP 打包
- `src/dataset.py`: 数据解析、增强和 dataloader 实现
- `src/model.py`: 检测器、传统候选框分支和 metric rescore 实现
- `src/utils.py`: 日志、配置加载、硬件检测和辅助函数

## 安装依赖
```bash
python3 -m pip install --user -r requirements.txt
```

## 训练
```bash
python3 train.py --config ./config.yaml
```

### 使用 DINOv3 训练
1. 克隆 DINOv3 仓库，只需要执行一次：
```bash
git clone --depth 1 https://github.com/facebookresearch/dinov3.git ./third_party/dinov3
```
2. 可选但推荐：把官方 DINOv3 权重路径写入 `config.dinov3.yaml` 的 `model.dinov3.weights`。
3. 启动训练：
```bash
python3 train.py --config ./config.dinov3.yaml
```

## 推理并生成提交 ZIP
```bash
python3 inference.py --config ./config.yaml --ckpt ./checkpoints/best_model.pt
```

输出 ZIP 会生成在 `outputs/` 目录下。

## 复现当前最优线上分数
当前已知最优线上提交：

- ZIP 文件名：`少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123.zip`
- 线上 mAP@0.5：`0.283265`
- 关键推理设置：关闭 classical proposal 分支，置信度阈值 `0.01`，按类别 NMS IoU `0.445`，两个 checkpoint ensemble，TTA 使用尺度 `[1.0, 1.1, 0.9]` 加水平翻转。

下面所有命令都在 `LBB_competition/` 目录下执行：

```bash
cd /home/heqing/LBB_competition
conda activate lbb
```

训练 seed-42 模型：

```bash
python3 train.py --config ./config.no_classic.thr001_nms0445.ensemble_seed123.yaml
```

保存 seed-42 的最优 checkpoint，避免后续训练 seed-123 时覆盖：

```bash
mkdir -p ./checkpoints/seed42
cp ./checkpoints/best_model.pt ./checkpoints/seed42/best_model.pt
```

训练第二个 seed-123 模型。这里保持架构和训练策略不变，只修改随机种子和 checkpoint 保存目录：

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

使用两个 checkpoint ensemble 推理，生成当前最优提交 ZIP：

```bash
python3 inference.py \
  --config ./config.no_classic.thr001_nms0445.ensemble_seed123.yaml \
  --ckpt ./checkpoints/seed42/best_model.pt,./checkpoints/seed123/best_model.pt
```

最终生成的提交文件是：

```text
./outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_no_classic_thr001_nms0445_ensemble_seed123.zip
```

作为参考，仓库里也保留了 `nms=0.455` 的配置，因为当时明确要求保存；不过它的线上分数略低，是 `0.283147`。当前已知最优配置是 `config.no_classic.thr001_nms0445.ensemble_seed123.yaml`。

## 备注
- 训练时会把多边形标注转换成检测框。
- 训练包含强增强：mixup、copy-paste、颜色增强、cutout 和翻转。
- Hybrid 分支包含：
  - 传统高频候选框 proposal
  - 基于类别原型的 metric re-scoring
- 验证指标是 mAP@0.5，代码里的自定义实现与比赛规则对齐。
