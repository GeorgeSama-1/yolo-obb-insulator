# Stage1 YOLO-CLS Patch Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage 1 patch dataset preparation from Stage 2 OBB data, train a YOLO-cls classifier, and add a Stage 2 abnormal boost x6 helper.

**Architecture:** Keep Stage 1 data preparation separate from Stage 2 detection logic. A dedicated patch dataset module reads a prepared Stage 2 YOLO OBB dataset, crops per-object patches, oversamples only `abnormal` train patches to approach `2:1`, and writes a standard image-classification directory. Stage 1 training uses Ultralytics classification models through a proper training entrypoint.

**Tech Stack:** Python, Pillow, PyYAML, Ultralytics, pytest

---

## Chunk 1: Stage 1 patch dataset generation

### Task 1: Add failing tests for patch dataset generation

**Files:**
- Create: `tests/stage1_two_stage/test_patch_dataset.py`
- Create: `src/stage1_two_stage/patch_dataset.py`

- [ ] **Step 1: Write the failing test for generating a Stage 1 classifier dataset from Stage 2 OBB data**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal patch dataset generation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Add failing tests for abnormal-only oversampling

**Files:**
- Modify: `tests/stage1_two_stage/test_patch_dataset.py`
- Modify: `src/stage1_two_stage/patch_dataset.py`

- [ ] **Step 1: Write the failing test for `2:1` balancing without downsampling `normal`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal abnormal oversampling**
- [ ] **Step 4: Run test to verify it passes**

### Task 3: Add Stage 1 patch dataset CLI

**Files:**
- Create: `scripts/data/prepare_stage1_patch_classifier.py`
- Modify: `src/common/defaults.py`

- [ ] **Step 1: Add CLI defaults and wiring**
- [ ] **Step 2: Verify the script can generate a dataset on a tiny smoke sample**

## Chunk 2: Stage 1 YOLO-cls training entrypoint

### Task 4: Add failing tests for Stage 1 train args

**Files:**
- Modify: `tests/stage1_two_stage/test_classifier_config.py`
- Modify: `src/stage1_two_stage/classifier_train.py`

- [ ] **Step 1: Write the failing test for Stage 1 training arg construction**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement Stage 1 train arg builder with extra arg passthrough**
- [ ] **Step 4: Run test to verify it passes**

### Task 5: Turn Stage 1 train script into a real YOLO-cls entrypoint

**Files:**
- Modify: `scripts/train/train_stage1_classifier.py`
- Modify: `configs/stage1_two_stage/classifier.yaml`

- [ ] **Step 1: Hook the config into Ultralytics classification training**
- [ ] **Step 2: Set Stage 1 defaults to `yolo11m-cls.pt` and `imgsz=384`**
- [ ] **Step 3: Verify config loading and script behavior**

## Chunk 3: Stage 2 abnormal boost x6 helper

### Task 6: Add a dedicated x6 helper script

**Files:**
- Create: `scripts/data/prepare_stage2_abn_boost_x6.py`
- Modify: `src/common/defaults.py`

- [ ] **Step 1: Add the x6 output default**
- [ ] **Step 2: Add the helper script that generates `stage2_defect_obb_abn_boost_x6`**
- [ ] **Step 3: Verify the helper runs on a tiny sample**

## Chunk 4: Docs and verification

### Task 7: Update usage docs

**Files:**
- Modify: `README.md`
- Modify: `docs/project/experiment-plan.md`

- [ ] **Step 1: Document Stage 1 patch generation from Stage 2 data**
- [ ] **Step 2: Document Stage 1 YOLO-cls training**
- [ ] **Step 3: Document Stage 2 x6 helper usage**

### Task 8: Run focused verification

**Files:**
- Test: `tests/stage1_two_stage/test_patch_dataset.py`
- Test: `tests/stage1_two_stage/test_classifier_config.py`
- Test: `tests/stage1_two_stage/test_crops.py`

- [ ] **Step 1: Run focused Stage 1 tests**
- [ ] **Step 2: Run tiny smoke checks for the new CLIs**
- [ ] **Step 3: Commit**
