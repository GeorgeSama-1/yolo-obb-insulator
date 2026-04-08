# 绝缘子 OBB 项目

这是一个面向绝缘子检测、计数和后续缺陷识别实验的项目骨架，当前已经优先打通了 `Stage 0`：

- 将 `LabelMe OBB` 标注转换为 `YOLO OBB` 训练格式
- 对训练集做离线增强
- 检查增强后标签是否合法
- 可视化抽检增强后的 OBB 框
- 训练 `YOLO11 OBB` 单类绝缘子检测模型
- 训练后批量可视化预测结果

后续还预留了两条扩展路线：

- `Stage 1`：先检测绝缘子，再裁片做 `normal / abnormal` 分类
- `Stage 2`：直接训练 `normal / abnormal` 的 OBB 检测模型

## 1. 当前项目结构

```text
yolo-obb/
├─ configs/
├─ data/
├─ datasets/
├─ docs/
├─ reports/
├─ runs/
├─ scripts/
├─ src/
└─ tests/
```

关键目录说明：

- `datasets/`
  - 当前原始数据目录
  - 存放 `LabelMe json + 图片`
- `data/raw/`
  - 规范化后的原始数据副本
- `data/processed/`
  - 转换后和增强后的 `YOLO OBB` 数据集
- `configs/stage0_obb/train_insulator.yaml`
  - Stage 0 训练配置
- `scripts/data/`
  - 数据转换、增强、校验脚本
- `scripts/train/`
  - 训练入口脚本
- `scripts/infer/`
  - 推理与可视化脚本
- `reports/figures/`
  - 可视化检查图输出目录

## 2. 环境依赖

建议 Python 3.10 及以上。

安装依赖：

```bash
pip install -r requirements.txt
```

当前依赖包括：

- `ultralytics`
- `opencv-python`
- `Pillow`
- `PyYAML`
- `pytest`

## 3. 数据格式说明

### 3.1 原始数据

当前原始数据是 `LabelMe OBB` 风格：

- 图片文件：如 `23.JPG`
- 标注文件：如 `23.json`
- 每个目标是 `4` 个点的 `polygon`

### 3.2 训练数据

训练使用 `YOLO OBB` 格式，标签示例：

```text
0 0.229036 0.521354 0.247786 0.513368 0.263151 0.575868 0.245703 0.583507
```

含义是：

- 第 1 列：类别 id
- 后 8 列：4 个点的归一化坐标 `x1 y1 x2 y2 x3 y3 x4 y4`

## 4. 推荐使用流程

如果你要在服务器上训练，建议严格按下面流程执行。

### 第一步：生成 train/val 划分

```bash
python scripts/data/make_split.py \
  --source datasets \
  --output data/splits/debug_split.json
```

输出：

- `data/splits/debug_split.json`

### 第二步：检查原始 LabelMe OBB 标注

```bash
python scripts/data/validate_labelme_obb.py \
  --source datasets \
  --label insulator
```

如果正常，会输出：

```bash
Validation passed
```

### 第三步：转换成 YOLO OBB 数据集

```bash
python scripts/data/convert_to_yolo_obb.py \
  --source datasets \
  --split-json data/splits/debug_split.json \
  --output data/processed/yolo_obb_insulator
```

输出结果目录：

- `data/processed/yolo_obb_insulator/images/train`
- `data/processed/yolo_obb_insulator/images/val`
- `data/processed/yolo_obb_insulator/labels/train`
- `data/processed/yolo_obb_insulator/labels/val`
- `data/processed/yolo_obb_insulator/dataset.yaml`

### 第四步：对训练集做离线增强

比如每张训练图扩成 `20` 张：

```bash
python scripts/data/augment_yolo_obb.py \
  --input data/processed/yolo_obb_insulator \
  --output data/processed/yolo_obb_insulator_aug20 \
  --target-per-image 20 \
  --augment-splits train \
  --seed 123
```

说明：

- `--target-per-image 20`
  - 表示每张原图最终保留 `20` 张
  - 包括原图本身
- `--augment-splits train`
  - 只增强训练集
  - 验证集不增强，原样复制

当前脚本会做：

- 水平翻转
- 垂直翻转
- 90 / 180 / 270 度旋转
- 亮度扰动
- 对比度扰动
- 颜色扰动

### 第五步：检查增强后标签是否合法

```bash
python scripts/data/validate_yolo_obb_dataset.py \
  --dataset data/processed/yolo_obb_insulator_aug20
```

这个脚本会检查：

- 图片和标签是否一一对应
- 每行是否为 `9` 列
- 类别 id 是否合法
- 坐标是否都在 `[0, 1]`

如果正常，会输出：

```bash
YOLO OBB dataset validation passed
```

### 第六步：可视化抽检增强后的 OBB 标签

```bash
python scripts/infer/visualize_yolo_obb_dataset.py \
  --dataset data/processed/yolo_obb_insulator_aug20 \
  --split train \
  --output reports/figures/aug_check \
  --limit 50
```

建议重点检查：

- 框是否跟着目标一起旋转/翻转
- 框是否明显偏移
- 点顺序是否异常
- 是否有框跑出目标区域

输出目录：

- `reports/figures/aug_check`

## 5. 开始训练 Stage 0

训练前先修改配置文件：

- [configs/stage0_obb/train_insulator.yaml](/mnt/f/yolo-obb/configs/stage0_obb/train_insulator.yaml)

推荐把 `data` 改成增强后的数据集：

```yaml
model: yolo11n-obb.pt
data: data/processed/yolo_obb_insulator_aug20/dataset.yaml
epochs: 100
imgsz: 1024
batch: 4
project: runs/stage0_obb
name: insulator
device: 0
```

然后开始训练：

```bash
python scripts/train/train_stage0_obb.py \
  --config configs/stage0_obb/train_insulator.yaml
```

训练结果通常会输出到：

```bash
runs/stage0_obb/insulator/
```

常见重要文件：

- `weights/best.pt`
- `weights/last.pt`

## 6. 训练后批量可视化预测结果

训练完成后，可以把预测结果批量画出来：

```bash
python scripts/infer/visualize_stage0_predictions.py \
  --weights runs/stage0_obb/insulator/weights/best.pt \
  --source datasets \
  --output reports/figures/predictions \
  --dataset-yaml data/processed/yolo_obb_insulator_aug20/dataset.yaml \
  --imgsz 1024 \
  --conf 0.25
```

输出目录：

- `reports/figures/predictions`

这个步骤主要用来看：

- 框是否贴合目标
- 方向是否正确
- 有没有明显漏检
- 有没有明显误检

## 7. 计数评估脚本

目前有一个简单的计数评估入口：

```bash
python scripts/eval/eval_stage0_obb.py \
  --ground-truth 70 27 70 \
  --predicted 69 28 70 \
  --output reports/metrics/stage0_count_metrics.json
```

输出指标包括：

- `mae`
- `exact_accuracy`

## 8. 当前最推荐的训练前检查清单

在服务器上开始训练前，建议至少确认这 3 件事都做了：

1. `validate_yolo_obb_dataset.py` 已通过  
2. `visualize_yolo_obb_dataset.py` 已人工抽检过一批增强图  
3. `train_insulator.yaml` 中的 `data` 已指向增强后的数据集  

## 9. 其他已预留能力

### Stage 1：两阶段路线

相关文件：

- `src/stage1_two_stage/crops.py`
- `configs/stage1_two_stage/classifier.yaml`
- `scripts/data/generate_stage1_crops.py`
- `scripts/train/train_stage1_classifier.py`

目标：

- 先检测绝缘子
- 再裁剪 patch
- 再训练 `normal / abnormal` 分类模型

### Stage 2：一阶段路线

相关文件：

- `src/stage2_one_stage/dataset.py`
- `src/stage2_one_stage/train.py`
- `configs/stage2_one_stage/train_defect_obb.yaml`
- `scripts/data/prepare_stage2_dataset.py`
- `scripts/train/train_stage2_obb.py`

目标：

- 直接训练 `normal / abnormal` 的 OBB 模型

## 10. 运行测试

如果你修改了脚本或数据处理逻辑，可以运行：

```bash
pytest -q
```

当前本地验证结果是：

- `22 passed`

## 11. 一套最常用命令

如果你只想快速照着执行，最常用的一套命令如下：

```bash
python scripts/data/make_split.py \
  --source datasets \
  --output data/splits/debug_split.json

python scripts/data/validate_labelme_obb.py \
  --source datasets \
  --label insulator

python scripts/data/convert_to_yolo_obb.py \
  --source datasets \
  --split-json data/splits/debug_split.json \
  --output data/processed/yolo_obb_insulator

python scripts/data/augment_yolo_obb.py \
  --input data/processed/yolo_obb_insulator \
  --output data/processed/yolo_obb_insulator_aug20 \
  --target-per-image 20 \
  --augment-splits train \
  --seed 123

python scripts/data/validate_yolo_obb_dataset.py \
  --dataset data/processed/yolo_obb_insulator_aug20

python scripts/infer/visualize_yolo_obb_dataset.py \
  --dataset data/processed/yolo_obb_insulator_aug20 \
  --split train \
  --output reports/figures/aug_check \
  --limit 50

python scripts/train/train_stage0_obb.py \
  --config configs/stage0_obb/train_insulator.yaml

python scripts/infer/visualize_stage0_predictions.py \
  --weights runs/stage0_obb/insulator/weights/best.pt \
  --source datasets \
  --output reports/figures/predictions \
  --dataset-yaml data/processed/yolo_obb_insulator_aug20/dataset.yaml \
  --imgsz 1024 \
  --conf 0.25
```
