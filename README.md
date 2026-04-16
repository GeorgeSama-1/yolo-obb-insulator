# 绝缘子 OBB 实验项目

这个项目用于支撑 3 条实验线：

- `Stage 0`：绝缘子 OBB 检测与计数
- `Stage 1`：两阶段缺陷识别，先检测再做 patch 分类
- `Stage 2`：一阶段 `normal / abnormal` OBB 检测

当前推荐的推进顺序是：

1. 先固定并复现 `Stage 0` 基线
2. 再整理并训练 `Stage 2` 一阶段缺陷 OBB
3. 最后补 `Stage 1` patch 分类实验

## 1. 目录规范

从现在开始，推荐使用这套目录命名：

```text
incoming_batches/                         # 新进入项目、尚未预标注的图片
prelabel_batches/stage0_insulator_obb/   # Stage 0 预标注批次
cvat_exports/                            # CVAT 导入导出 XML、ZIP、修标结果
data_sources/stage2_defect_obb/          # Stage 2 一阶段缺陷标注来源
datasets_pool_stage0_insulator_obb/      # Stage 0 总训练池
datasets_pool_stage2_defect_obb/         # Stage 2 总训练池
data/processed/stage0_insulator_obb/     # Stage 0 标准训练集
data/processed/stage0_insulator_obb_aug20/
data/processed/stage1_patch_classifier/  # Stage 1 patch 分类数据
data/processed/stage2_defect_obb/        # Stage 2 标准训练集
data/processed/stage2_defect_obb_aug20/
data/processed/stage2_defect_obb_abn_boost/
data/processed/stage2_defect_obb_abn_light_aug/
```

已经完成的目录迁移：

- `labels_1` -> `data_sources/stage2_defect_obb/batch_001`
- `labels_2` -> `data_sources/stage2_defect_obb/batch_002`
- `images_merged` -> `data_sources/stage2_defect_obb/merged_train_source`
- `new_images` -> `prelabel_batches/stage0_insulator_obb/batch_001`
- `new_images_add1` -> `prelabel_batches/stage0_insulator_obb/batch_002`

更详细的规则见：

- [naming-and-layout.md](/mnt/f/yolo-obb/docs/project/naming-and-layout.md)
- [experiment-plan.md](/mnt/f/yolo-obb/docs/project/experiment-plan.md)

## 2. 环境依赖

建议 Python 3.10 及以上。

```bash
pip install -r requirements.txt
```

## 3. Stage 0：绝缘子 OBB 基线

### 3.1 总训练池

Stage 0 默认总训练池：

```text
datasets_pool_stage0_insulator_obb/
```

目录格式是平铺的：

```text
datasets_pool_stage0_insulator_obb/
├─ a.jpg
├─ a.txt
├─ b.jpg
└─ b.txt
```

### 3.2 数据准备

生成切分：

```bash
python scripts/data/make_split.py
```

默认会生成：

```text
data/splits/stage0_insulator_obb_split.json
```

整理成标准训练结构：

```bash
python scripts/data/prepare_yolo_obb_dataset.py \
  --split-json data/splits/stage0_insulator_obb_split.json
```

默认输出：

```text
data/processed/stage0_insulator_obb/
```

增强：

```bash
python scripts/data/augment_yolo_obb.py \
  --input data/processed/stage0_insulator_obb \
  --output data/processed/stage0_insulator_obb_aug20 \
  --target-per-image 20 \
  --augment-splits train \
  --seed 123 \
  --preview \
  --preview-limit 50
```

校验：

```bash
python scripts/data/validate_yolo_obb_dataset.py \
  --dataset data/processed/stage0_insulator_obb_aug20
```

### 3.3 训练

配置文件：

- [train_insulator.yaml](/mnt/f/yolo-obb/configs/stage0_obb/train_insulator.yaml)

当前默认配置会读取：

```text
data/processed/stage0_insulator_obb_aug20/dataset.yaml
```

训练命令：

```bash
python scripts/train/train_stage0_obb.py \
  --config configs/stage0_obb/train_insulator.yaml
```

如果你在服务器上指定 GPU，例如物理卡 `4`：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train/train_stage0_obb.py \
  --config configs/stage0_obb/train_insulator.yaml
```

## 4. Stage 0：新图预标注与 CVAT 修标

预标注批次建议放在：

```text
prelabel_batches/stage0_insulator_obb/
```

例如：

```text
prelabel_batches/stage0_insulator_obb/batch_001/
├─ images/train/
└─ labels/train/
```

如果你要把这批预标注导入 CVAT：

```bash
python3 scripts/data/convert_yolo_obb_to_cvat_xml.py \
  --source prelabel_batches/stage0_insulator_obb/batch_001 \
  --output cvat_exports/batch_001_stage0_insulator_obb.xml \
  --class-name insulator \
  --split train
```

在 CVAT 中导入时选择：

- `CVAT for images 1.1`

## 5. Stage 2：一阶段缺陷 OBB

### 5.1 标注来源

一阶段 `normal / abnormal` 的历史来源放在：

```text
data_sources/stage2_defect_obb/
├─ batch_001/
├─ batch_002/
└─ merged_train_source/
```

`merged_train_source/` 是当前整理好的合并训练源。

### 5.2 建立 Stage 2 总训练池

把一阶段数据并入总训练池：

```bash
python3 scripts/data/merge_into_dataset_pool.py \
  --source data_sources/stage2_defect_obb/merged_train_source \
  --pool datasets_pool_stage2_defect_obb \
  --split train
```

### 5.3 数据准备

生成切分：

```bash
python scripts/data/make_split.py \
  --source datasets_pool_stage2_defect_obb \
  --output data/splits/stage2_defect_obb_split.json
```

整理成标准训练结构：

```bash
python scripts/data/prepare_yolo_obb_dataset.py \
  --source datasets_pool_stage2_defect_obb \
  --split-json data/splits/stage2_defect_obb_split.json \
  --output data/processed/stage2_defect_obb \
  --class-name normal \
  --class-name abnormal
```

增强：

```bash
python scripts/data/augment_yolo_obb.py \
  --input data/processed/stage2_defect_obb \
  --output data/processed/stage2_defect_obb_aug20 \
  --target-per-image 20 \
  --augment-splits train \
  --seed 123 \
  --preview \
  --preview-limit 50
```

校验：

```bash
python scripts/data/validate_yolo_obb_dataset.py \
  --dataset data/processed/stage2_defect_obb_aug20
```

### 5.4 Stage 2 类别比例与 abnormal 对照实验

如果你要做 `normal / abnormal` 类别比例统计，以及两套 abnormal 对照实验，可以直接基于准备好的 Stage 2 数据集生成：

- 类别统计：
  - `reports/metrics/stage2_class_balance.json`
  - `reports/tables/stage2_class_balance.md`
- abnormal 采样加强版：
  - `data/processed/stage2_defect_obb_abn_boost/`
- abnormal 轻增强对照版：
  - `data/processed/stage2_defect_obb_abn_light_aug/`

命令：

```bash
python scripts/data/prepare_stage2_balance_experiments.py \
  --input data/processed/stage2_defect_obb_aug20 \
  --abnormal-target-per-image 3 \
  --seed 123
```

默认会同时完成三件事：

1. 统计 `normal / abnormal` 数量比例
2. 生成只提高 abnormal 图片采样概率的数据集
3. 生成只对 abnormal 做轻量旋转、缩放、亮度扰动的数据集

如果你想分别训练这两套对照数据，只需要把一阶段配置里的 `data` 改到对应目录：

```yaml
data: data/processed/stage2_defect_obb_abn_boost/dataset.yaml
```

或者：

```yaml
data: data/processed/stage2_defect_obb_abn_light_aug/dataset.yaml
```

### 5.5 训练

配置文件：

- [train_defect_obb.yaml](/mnt/f/yolo-obb/configs/stage2_one_stage/train_defect_obb.yaml)

当前默认配置会读取：

```text
data/processed/stage2_defect_obb_aug20/dataset.yaml
```

训练命令：

```bash
python scripts/train/train_stage2_obb.py \
  --config configs/stage2_one_stage/train_defect_obb.yaml
```

如果你在服务器上指定 GPU，例如物理卡 `4`：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train/train_stage2_obb.py \
  --config configs/stage2_one_stage/train_defect_obb.yaml
```

## 6. Stage 1：两阶段 patch 分类

Stage 1 当前默认分类数据目录已经同步到：

```text
data/processed/stage1_patch_classifier
```

配置文件：

- [classifier.yaml](/mnt/f/yolo-obb/configs/stage1_two_stage/classifier.yaml)

这条线建议在 `Stage 0` 和 `Stage 2` 都稳定之后再正式展开。

## 7. 运行测试

如果你修改了路径、脚本或配置，建议至少运行：

```bash
pytest -q tests/common/test_default_paths.py tests/common/test_paths.py
```

## 8. 你接下来最推荐的顺序

你已经明确计划：

1. 先做 `Stage 0`
2. 再做 `Stage 2`

推荐执行顺序就是：

1. 固定服务器上的 `Stage 0` 最优参数
2. 复现一次 `Stage 0` 基线实验
3. 输出一份 `Stage 0` 阶段性结果
4. 用 `merged_train_source` 建立 `Stage 2` 总训练池
5. 训练 `Stage 2` 一阶段 `normal / abnormal` OBB
6. 输出第二份阶段性结果

这两轮结果出来后，再决定要不要补 `Stage 1` 两阶段 patch 分类实验。
