# 阶段性实验计划表

这份计划表用于支撑两件事：

- 生成阶段性实验报告
- 让后续实验目录保持易读、可复现

## 1. 当前实验主线

### 实验线 A：Stage 0 绝缘子 OBB 基线

目标：

- 固定一套已经在服务器上验证过的参数
- 作为后续两阶段和一阶段的统一基线

输入数据：

- `datasets_pool_stage0_insulator_obb/`

输出目录建议：

- `data/processed/stage0_insulator_obb/`
- `data/processed/stage0_insulator_obb_aug20/`
- `runs/stage0_obb/exp01_stage0_baseline/`

报告中需要记录：

- 模型参数
- 数据量
- 增强策略
- 检测效果
- 计数效果
- 可视化结果

### 实验线 B：两阶段缺陷识别

目标：

- 利用 Stage 0 检测框裁出绝缘子 patch
- 对 patch 做 `normal / abnormal` 分类

输入数据：

- 阶段 0 检测结果
- 后续整理出的 patch 标注数据

输出目录建议：

- `data/processed/stage1_patch_classifier/`
- `runs/stage1_two_stage/exp02_patch_classifier/`

报告中需要记录：

- patch 生成方式
- 分类数据量
- 分类模型设置
- 分类准确率
- abnormal 召回率

### 实验线 C：一阶段缺陷 OBB

目标：

- 直接在原图上训练 `normal / abnormal` OBB 模型

输入数据：

- `data_sources/stage2_defect_obb/merged_train_source/`

输出目录建议：

- `datasets_pool_stage2_defect_obb/`
- `data/processed/stage2_defect_obb/`
- `data/processed/stage2_defect_obb_aug20/`
- `data/processed/stage2_defect_obb_abn_boost/`
- `data/processed/stage2_defect_obb_abn_light_aug/`
- `runs/stage2_one_stage/exp03_stage2_defect_obb/`

报告中需要记录：

- 类别分布
- 增强策略
- 训练参数
- 检测效果
- abnormal 检测表现
- abnormal/normal 比例
- abnormal boost 与 abnormal 轻增强对照结果

## 2. 推荐执行顺序

1. 先固定 `Stage 0` 服务器最优参数
2. 重跑 `Stage 0` 基线，产出统一基线结果
3. 基于当前目录规范整理一阶段数据并训练 `Stage 2`
4. 基于 Stage 2 生成 abnormal boost 与 abnormal 轻增强对照实验
5. 最后补两阶段 patch 分类实验

## 3. 数据目录职责

### 3.1 预标注批次

- `prelabel_batches/stage0_insulator_obb/batch_001`
- `prelabel_batches/stage0_insulator_obb/batch_002`

用途：

- 存放阶段 0 模型对新图的预标注结果
- 后续转成 CVAT XML 做人工修标

### 3.2 一阶段标注来源

- `data_sources/stage2_defect_obb/batch_001`
- `data_sources/stage2_defect_obb/batch_002`
- `data_sources/stage2_defect_obb/merged_train_source`

用途：

- 存放一阶段 `normal / abnormal` OBB 的历史来源和当前合并训练源

### 3.3 总训练池

建议后续长期维护：

- `datasets_pool_stage0_insulator_obb/`
- `datasets_pool_stage2_defect_obb/`

用途：

- 所有确认可训练的数据统一并入总池
- 每次实验都从总池重新切分与增强

## 4. 阶段性报告建议结构

每次阶段性报告建议固定这几个章节：

1. 本阶段目标
2. 数据来源与目录整理
3. 实验设置
4. 结果与可视化
5. 当前结论
6. 下一步计划

## 5. 当前最优先任务

当前建议优先完成：

1. 复现并固定 `Stage 0` 基线实验
2. 使用 `merged_train_source` 整理 `Stage 2` 一阶段训练集
3. 输出一份基线 + 一阶段的阶段性实验报告
