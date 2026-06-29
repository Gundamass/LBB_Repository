# 当前方案记录

更新时间：2026-06-29

当前项目实际工作目录：

```bash
/home/heqing/LBB_competition
```

当前线上最好分数：

```text
0.290255
```

对应提交文件：

```text
outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_dinorank_plus_fcos_collision_rescue_tpl4_iou03_fac00005.zip
```

这份记录只描述当前有效做法和可复现模块。数据集、手标数据、输出 ZIP、牛客登录态、第三方仓库、本地权重和大模型权重不进入 Git。

## 总体思路

当前方案不是单一检测器直接出结果，而是一个以旧方案 C 检测结果为主干的少样本工业缺陷检测后处理体系：

1. 主干检测器给出基础候选框。
2. 使用背景/异常度重排降低明显背景框影响。
3. 使用 DINOv3 PatchCore 对缺陷候选做正常纹理异常度重排。
4. 利用少样本类别混淆规律，把部分 `scratch` 框以极低分复制为 `collision`，提升 `collision` 召回。
5. 使用 DINOv3 RoI prototype margin 对 `scratch -> collision` 复制框做软排序，而不是硬过滤。
6. 使用 FCOS 单分支的 `collision` 独占命中作为 ultra-low-score rescue tail 追加，保留单模型能切中的漏检框。

关键经验：

- 简单堆模型、WBF 或大量 topK 追加容易带入 FP，线上提升很小甚至下降。
- 当前更有效的是“保留主干排序 + 超低分补召回”。
- DINOv3 更适合做 soft rank/rescore，不适合硬删除候选。
- FCOS 整体不是最好，但有少量 `collision` 独占命中，适合作为低分 rescue 分支。

## 当前最佳链路

当前最好提交是在以下链路基础上得到的：

```text
base DINOv3 PatchCore rescore
  -> scratch 复制为 collision 低分尾框
  -> DINOv3 prototype margin 软排序
  -> FCOS collision rescue tail
```

线上关键分数演进：

```text
0.290222  probe_best_plus_dinov3_patchcore_i384_low005_f080.zip
0.290252  probe_best_crosslabel_collision_scratch_top150_fac0002_area250_20000.zip
0.290254  probe_best_crosslabel_dino_rank_collision_scratch_top150_mneg035_c000_a5_fac0002.zip
0.290255  probe_best_dinorank_plus_fcos_collision_rescue_tpl4_iou03_fac00005.zip
```

分数记录文件在本地：

```text
outputs/submission_scores.csv
```

该 CSV 是实验记录，不提交到 Git。

## 模块和文件

### 训练和基础推理

相关文件：

```text
train.py
inference.py
src/model.py
src/dataset.py
src/utils.py
src/masking.py
config.yaml
config.fcos.yaml
config.fasterrcnn.legacy.yaml
```

作用：

- `train.py`：训练、验证、checkpoint 保存，支持从 `training.init_checkpoint` 初始化。
- `inference.py`：测试集推理、TTA、多 checkpoint ensemble、mask 后处理、提交 JSON/ZIP 导出。
- `src/model.py`：检测器构建，保留 Faster R-CNN、FCOS、RetinaNet、DINOv3 Faster R-CNN 和混合 proposal/rescore 逻辑。
- `config.yaml`：当前默认训练配置，已切到 FCOS ResNet50-FPN。
- `config.fcos.yaml`：FCOS 对照配置。
- `config.fasterrcnn.legacy.yaml`：旧 Faster R-CNN baseline 配置。

训练示例：

```bash
cd /home/heqing/LBB_competition
conda activate lbb
python3 train.py --config ./config.yaml
```

推理示例：

```bash
cd /home/heqing/LBB_competition
conda activate lbb
python3 inference.py --config ./config.yaml --ckpt ./checkpoints/fcos/best_model.pt
```

### 本地评估

相关文件：

```text
evaluate_local_map.py
tools/diagnose_branch_unique_hits.py
```

作用：

- `evaluate_local_map.py`：按比赛 mAP@0.5 逻辑在手标测试子集上评估 zip 或 json 目录。
- `tools/diagnose_branch_unique_hits.py`：从 GT 视角统计当前 best 漏掉但某个分支命中的目标，用来寻找单分支 rescue 机会。

本地 mAP 示例：

```bash
python3 evaluate_local_map.py \
  --pred ./outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_dinorank_plus_fcos_collision_rescue_tpl4_iou03_fac00005.zip \
  --gt-dir ./手标数据 \
  --score-thr 0.0
```

独占命中诊断示例：

```bash
python3 tools/diagnose_branch_unique_hits.py \
  --base ./outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_crosslabel_dino_rank_collision_scratch_top150_mneg035_c000_a5_fac0002.zip \
  --branch 'fcos=./outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_fcos_resnet50_fpn.zip' \
  --branch 'yolo_tile_e16=./outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_yolo26s_tile640_img1024_realpastev23_posonly_e16_conf0005_top700.zip' \
  --out-json ./outputs/branch_unique_hits_best_a5.json
```

已观察到的手标集诊断：

```text
GT objects: 1009
当前 best hit: 839
多分支 oracle hit: 883
FCOS 对当前 best 的独占命中: 28
FCOS 独占 collision: 14
```

这说明用户提出的“某个模型原框能切中，但融合/主结果可能没保留”的假设成立。

### 低分尾框融合

相关文件：

```text
tools/fuse_submissions.py
tools/append_low_score_tail.py
tools/append_branch_rescue_tail.py
```

作用：

- `fuse_submissions.py`：把额外提交中的 topK 框按低分追加到 base，可选 class-wise NMS。
- `append_low_score_tail.py`：从单个 extra 结果中追加低分 tail，支持类别包含/排除与去重。
- `append_branch_rescue_tail.py`：当前关键工具。只有当 base 中没有同类别近邻框时，才追加额外分支框，用于保留单模型独占命中。

当前最佳最后一步命令：

```bash
python3 tools/append_branch_rescue_tail.py \
  --base './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_crosslabel_dino_rank_collision_scratch_top150_mneg035_c000_a5_fac0002.zip' \
  --extra './outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_fcos_resnet50_fpn.zip' \
  --out-zip './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_dinorank_plus_fcos_collision_rescue_tpl4_iou03_fac00005.zip' \
  --include-label collision \
  --topk-per-label 4 \
  --skip-base-iou 0.30 \
  --self-dedup-iou 0.70 \
  --min-source-score 0.30 \
  --score-factor 0.00005 \
  --score-cap 0.00002
```

官方验证结果：

```text
0.290255
```

### 类别混淆补召回

相关文件：

```text
tools/duplicate_cross_label_to_collision.py
tools/duplicate_cross_label_to_collision_dino.py
```

作用：

- `duplicate_cross_label_to_collision.py`：把 `scratch` 或 `dirt` 候选低分复制为 `collision`，提升 `collision` recall。
- `duplicate_cross_label_to_collision_dino.py`：用 DINOv3 RoI prototype margin 对 `scratch -> collision` 候选做筛选或软排序。

重要结论：

- 硬过滤 DINO margin 后线上 `0.290246`，低于 best。
- 保留大部分候选、只用 DINO margin 做软排序后线上 `0.290254`，高于朴素 cross-label。

DINO 软排序命令：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/duplicate_cross_label_to_collision_dino.py \
  --pred './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_plus_dinov3_patchcore_i384_low005_f080.zip' \
  --out-zip './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_crosslabel_dino_rank_collision_scratch_top150_mneg035_c000_a5_fac0002.zip' \
  --topk-per-image 150 \
  --max-added-per-image 150 \
  --min-area 250 \
  --max-area 20000 \
  --margin-thr -0.35 \
  --score-margin-center 0.0 \
  --margin-score-alpha 5.0 \
  --score-factor 0.0002 \
  --score-cap 0.0005 \
  --dedup-iou 0.70 \
  --self-dedup-iou 0.92 \
  --batch-size 56 \
  --device cuda
```

### DINOv3 / PatchCore 异常重排

相关文件：

```text
tools/rescore_dinov3_patchcore_boxes.py
tools/rescore_patchcore_boxes.py
tools/patchcore_anomaly_proposals.py
tools/rescore_white_background.py
tools/dinov3_anomaly_refine_boxes.py
```

作用：

- `rescore_white_background.py`：白背景区域降权，早期把线上从 `0.290116` 推到 `0.290152`。
- `rescore_patchcore_boxes.py`：普通特征 PatchCore 异常重排。
- `rescore_dinov3_patchcore_boxes.py`：DINOv3 特征 PatchCore 正常纹理 memory bank，对 box anomaly score 做重排。
- `patchcore_anomaly_proposals.py`：PatchCore proposal/heatmap 工具函数。
- `dinov3_anomaly_refine_boxes.py`：DINOv3 heatmap/refine 实验工具。

关键线上结果：

```text
0.290193  PatchCore box rescore
0.290222  DINOv3 PatchCore box rescore
```

注意：DINOv3 第三方仓库和权重不提交 Git，默认本地路径为：

```text
third_party/dinov3
dinov3/dinov3_vitl16_from_safetensors-8aa4cbdd.pth
```

### YOLO26 / tile / 合成增强实验

相关文件：

```text
tools/train_yolo26.py
tools/infer_yolo_to_competition.py
tools/make_tile_yolo_dataset.py
tools/infer_tiles_to_competition.py
tools/generate_synthetic_defects.py
tools/generate_realpaste_defects.py
config.yolo26.yaml
config.yolo26s.realpastev23.yaml
config.tile_yolo26.yaml
config.tile_yolo26s.realpastev23.yaml
```

作用：

- YOLO26 整图和 tile 训练/推理。
- real-paste 和 synthetic defect 数据增强。
- tile 分支主要用于补低分 recall，单独提交效果较弱，但作为 ultra-low tail 有微小正收益。

代表线上结果：

```text
0.290103  YOLO26s tile realpastev23 低分 tail
0.290116  YOLO26s tile continued 低分 tail
```

### SynthSeg / anomaly segmentation 实验

相关文件：

```text
tools/train_synth_anomaly_segmenter.py
tools/infer_synth_anomaly_proposals.py
tools/rescore_synthseg_boxes.py
```

作用：

- 训练合成异常分割器。
- 生成 anomaly proposals。
- 对已有框按异常分割响应重排。

结论：

```text
synthseg rescore 线上约 0.290205-0.290212，低于 DINO/PatchCore 和 cross-label 方向。
```

当前不作为主线继续投入。

### 牛客提交 Agent

相关文件：

```text
tools/nowcoder_submit_agent.py
tools/export_nowcoder_state.py
NOWCODER_LOGIN_EXPORT.md
requirements.submit.txt
```

作用：

- 自动/半自动提交 zip。
- 支持本地浏览器导出 Playwright 登录态后放回服务器。
- 不保存账号密码。

敏感文件绝对不能提交：

```text
outputs/.nowcoder_submitter/state.json
outputs/.nowcoder_submitter/*.html
outputs/.nowcoder_submitter/*.png
```

## 当前方案复现顺序

假设已有以下中间文件：

```text
outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_plus_dinov3_patchcore_i384_low005_f080.zip
outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_fcos_resnet50_fpn.zip
third_party/dinov3
dinov3/dinov3_vitl16_from_safetensors-8aa4cbdd.pth
```

复现当前最佳后两步：

```bash
cd /home/heqing/LBB_competition
conda activate lbb

CUDA_VISIBLE_DEVICES=2 python tools/duplicate_cross_label_to_collision_dino.py \
  --pred './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_plus_dinov3_patchcore_i384_low005_f080.zip' \
  --out-zip './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_crosslabel_dino_rank_collision_scratch_top150_mneg035_c000_a5_fac0002.zip' \
  --topk-per-image 150 \
  --max-added-per-image 150 \
  --min-area 250 \
  --max-area 20000 \
  --margin-thr -0.35 \
  --score-margin-center 0.0 \
  --margin-score-alpha 5.0 \
  --score-factor 0.0002 \
  --score-cap 0.0005 \
  --dedup-iou 0.70 \
  --self-dedup-iou 0.92 \
  --batch-size 56 \
  --device cuda

python tools/append_branch_rescue_tail.py \
  --base './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_crosslabel_dino_rank_collision_scratch_top150_mneg035_c000_a5_fac0002.zip' \
  --extra './outputs/少样本条件下电子产品外观缺陷检测_默认团队_方案C_fcos_resnet50_fpn.zip' \
  --out-zip './outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_dinorank_plus_fcos_collision_rescue_tpl4_iou03_fac00005.zip' \
  --include-label collision \
  --topk-per-label 4 \
  --skip-base-iou 0.30 \
  --self-dedup-iou 0.70 \
  --min-source-score 0.30 \
  --score-factor 0.00005 \
  --score-cap 0.00002
```

最终提交：

```text
outputs/少样本条件下电子产品外观缺陷检测_默认团队_probe_best_dinorank_plus_fcos_collision_rescue_tpl4_iou03_fac00005.zip
```

## Git 提交范围

应该提交：

```text
CURRENT_SOLUTION.md
README.md
PIPELINE.md
requirements.txt
requirements.submit.txt
config*.yaml
train.py
inference.py
evaluate_local_map.py
src/*.py
tools/*.py
NOWCODER_LOGIN_EXPORT.md
```

不应该提交：

```text
outputs/
logs/
runs/
初赛数据/
手标数据/
masks/
third_party/
dinov3/
sam2_checkpoints/
*.pt
*.pth
*.safetensors
outputs/.nowcoder_submitter/state.json
```

例外：历史上已经通过 Git LFS 跟踪了：

```text
checkpoints/seed123/best_model.pt
```

后续如无必要，不再新增大权重进入 Git。
