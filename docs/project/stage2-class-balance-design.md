# Stage2 Class Balance Experiment Design

**Date:** 2026-04-16

**Scope:** Stage 2 one-stage `normal/abnormal` OBB dataset only

## Goal

Add a reproducible Stage 2 data preparation path that:

- reports the `normal/abnormal` class balance clearly
- produces a dataset variant with higher sampling probability for images containing `abnormal`
- produces a control dataset variant where only `abnormal` images receive light augmentation

The design is optimized for experiment reporting and reuse, not just one-off training.

## Current Context

The repository already has:

- a standard Stage 2 dataset flow based on YOLO OBB datasets
- an offline augmentation pipeline in `src/data_tools/augment_yolo_obb.py`
- Stage 2 label normalization built around `normal` and `abnormal`

What it does not have yet:

- a class-balance report for Stage 2
- class-aware dataset resampling
- class-aware augmentation policies targeted at `abnormal`

## Recommended Approach

Use a dedicated Stage 2 data-preparation entrypoint instead of overloading the generic augmentation script with too many task-specific branches.

### Why

- Keeps the generic OBB augmentation utility reusable across Stage 0 and Stage 2
- Makes the Stage 2 experiment outputs easy to explain in a report
- Lets us generate multiple clearly named dataset variants from the same source

## Outputs

### 1. Class balance report

Write machine-readable and human-readable summaries, for example:

- `reports/metrics/stage2_class_balance.json`
- `reports/tables/stage2_class_balance.md`

The report should include:

- total `normal` instances
- total `abnormal` instances
- total images containing `normal`
- total images containing `abnormal`
- class ratio

### 2. Abnormal boost dataset

Generate a dataset variant such as:

- `data/processed/stage2_defect_obb_abn_boost`

Policy:

- only affect `train`
- copy `val` unchanged
- images containing `abnormal` receive more replicas than purely `normal` images
- use existing safe OBB transforms already present in the augmentation pipeline

### 3. Abnormal light augmentation control dataset

Generate a dataset variant such as:

- `data/processed/stage2_defect_obb_abn_light_aug`

Policy:

- only affect `train`
- only images containing `abnormal` get extra augmented copies
- augmentation should stay light:
  - small rotation
  - light scale jitter
  - light brightness perturbation

This control experiment is intended to preserve defect morphology as much as possible.

## Proposed Implementation Layout

### New logic

- `src/stage2_one_stage/balance.py`
  - Stage 2 class counting
  - per-image label inspection
  - abnormal image detection
  - variant dataset generation helpers

### New CLI

- `scripts/data/prepare_stage2_balance_experiments.py`
  - generate reports
  - generate abnormal boost dataset
  - generate abnormal light augmentation dataset

### Existing logic reused

- `src/data_tools/augment_yolo_obb.py`
  - existing geometric transform logic
  - existing label parsing/formatting
  - existing preview output support

## Experiment Variants

### Exp-01: Baseline

Dataset:

- `data/processed/stage2_defect_obb_aug20`

Purpose:

- current Stage 2 baseline

### Exp-02: Abnormal Boost

Dataset:

- `data/processed/stage2_defect_obb_abn_boost`

Purpose:

- isolate the effect of increasing abnormal-image sampling probability

### Exp-03: Abnormal Light Aug

Dataset:

- `data/processed/stage2_defect_obb_abn_light_aug`

Purpose:

- test whether gentle defect-focused augmentation improves abnormal recall without distorting defect cues

## Validation Strategy

- unit tests for class counting and abnormal image detection
- unit tests for generated dataset sizes and split preservation
- unit tests ensuring only abnormal-containing images receive light-augmented copies
- keep existing YOLO OBB validation script in the loop for generated datasets

## Success Criteria

This work is successful when:

- Stage 2 class statistics can be generated on demand
- the two new dataset variants are reproducible from the same source dataset
- generated labels remain valid YOLO OBB labels
- experiment directories are clearly named and report-ready
