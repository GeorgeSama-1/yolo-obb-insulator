# 项目命名与目录规范

这份规范的目标不是立即重命名现有数据，而是为后续新增实验、数据迭代和脚本复用提供一套统一规则。

## 1. 设计原则

- 目录名优先表达“资产角色”，而不是临时处理步骤
- 目录名优先表达“任务语义”，例如 `stage0_insulator_obb`、`stage2_defect_obb`
- 原始数据、预标注数据、CVAT 导出数据、训练池数据彼此分层
- 新实验优先使用这套命名；旧目录可以逐步迁移，不强制一次改完

## 2. 推荐目录角色

### 2.1 新图入口

新进入项目、尚未预标注的图片统一放在：

```text
incoming_batches/
```

如果需要分批，可以继续往下建子目录：

```text
incoming_batches/batch_001/
incoming_batches/batch_002/
```

### 2.2 模型预标注结果

模型对新图推理并导出 `YOLO OBB` 标签后，统一放在：

```text
prelabel_batches/
```

推荐结构：

```text
prelabel_batches/batch_001/
├─ images/train/
└─ labels/train/
```

### 2.3 CVAT 导出结果

用于导入或导出 CVAT 的 XML、ZIP、修标结果，统一放在：

```text
cvat_exports/
```

例如：

```text
cvat_exports/batch_001_annotations.xml
cvat_exports/batch_001_corrected/
```

### 2.4 总训练池

适合长期迭代的数据总池统一命名为：

```text
datasets_pool_<task_name>/
```

例如：

```text
datasets_pool_stage0_insulator_obb/
datasets_pool_stage2_defect_obb/
```

这样既保留“总池”的语义，也能区分不同实验任务。

### 2.5 处理后训练集

处理后的标准训练目录放在：

```text
data/processed/<stage_name>/
```

推荐名称：

```text
data/processed/stage0_insulator_obb/
data/processed/stage0_insulator_obb_aug20/
data/processed/stage2_defect_obb/
data/processed/stage2_defect_obb_aug20/
```

## 3. 任务命名建议

推荐把阶段和任务语义一起写进名字：

- `stage0_insulator_obb`
- `stage1_patch_classifier`
- `stage2_defect_obb`

这样命名的好处是：

- 一眼能看出属于哪个阶段
- 一眼能看出训练目标
- 不会出现 `labels_1`、`labels_2` 这类后期难以追溯语义的目录

## 4. 当前目录迁移结果

本项目当前已经完成一轮目录整理，旧目录与新目录的对应关系如下：

- `labels_1` -> `data_sources/stage2_defect_obb/batch_001`
- `labels_2` -> `data_sources/stage2_defect_obb/batch_002`
- `images_merged` -> `data_sources/stage2_defect_obb/merged_train_source`
- `new_images` -> `prelabel_batches/stage0_insulator_obb/batch_001`
- `new_images_add1` -> `prelabel_batches/stage0_insulator_obb/batch_002`

这样整理后的含义是：

- `data_sources/stage2_defect_obb/...`
  表示一阶段 `normal / abnormal` OBB 的标注来源
- `merged_train_source`
  表示已经合并好的当前训练源
- `prelabel_batches/stage0_insulator_obb/...`
  表示阶段 0 绝缘子检测模型的预标注批次

后续新增数据时，建议直接沿用这些新目录，而不要再继续新增 `labels_x`、`new_images_x` 这类低语义名字。

## 5. 复用规则

新增脚本、配置、实验时，优先复用以下命名：

- 阶段配置：`configs/stage0_...`、`configs/stage1_...`、`configs/stage2_...`
- 阶段运行输出：`runs/stage0_...`、`runs/stage1_...`、`runs/stage2_...`
- 处理后数据：`data/processed/<stage_name>`

如果一个目录名无法回答“这是什么任务的数据”或“它处于哪个阶段”，通常说明它还不够稳定。
