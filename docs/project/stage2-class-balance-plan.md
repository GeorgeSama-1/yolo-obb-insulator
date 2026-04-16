# Stage2 Class Balance Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage 2 class balance reporting plus two reproducible class-aware dataset variants for `abnormal`-focused experiments.

**Architecture:** Keep the generic YOLO OBB augmentation utility reusable, and add Stage 2-specific balancing logic in a dedicated module and CLI. The new flow reads a prepared Stage 2 YOLO OBB dataset, counts class balance, detects images containing `abnormal`, and emits report files plus two variant datasets.

**Tech Stack:** Python, PyYAML, Pillow, pytest

---

## Chunk 1: Stage 2 class balance reporting

### Task 1: Add failing tests for Stage 2 class balance stats

**Files:**
- Create: `tests/stage2_one_stage/test_balance.py`
- Create: `src/stage2_one_stage/balance.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation for class counting**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Add report rendering helpers

**Files:**
- Modify: `src/stage2_one_stage/balance.py`
- Test: `tests/stage2_one_stage/test_balance.py`

- [ ] **Step 1: Write the failing test for JSON/Markdown report output**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal report writers**
- [ ] **Step 4: Run test to verify it passes**

## Chunk 2: Abnormal-focused dataset variants

### Task 3: Add abnormal-image detection and boost dataset generation

**Files:**
- Modify: `src/stage2_one_stage/balance.py`
- Test: `tests/stage2_one_stage/test_balance.py`

- [ ] **Step 1: Write the failing test for abnormal image detection**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal abnormal-image detection**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Write the failing test for abnormal boost dataset generation**
- [ ] **Step 6: Run test to verify it fails**
- [ ] **Step 7: Implement minimal abnormal boost generation**
- [ ] **Step 8: Run test to verify it passes**

### Task 4: Add abnormal light augmentation dataset generation

**Files:**
- Modify: `src/stage2_one_stage/balance.py`
- Test: `tests/stage2_one_stage/test_balance.py`

- [ ] **Step 1: Write the failing test for abnormal-only light augmentation**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal light augmentation generation**
- [ ] **Step 4: Run test to verify it passes**

## Chunk 3: CLI and docs

### Task 5: Add Stage 2 balance experiment CLI

**Files:**
- Create: `scripts/data/prepare_stage2_balance_experiments.py`
- Modify: `src/common/defaults.py`
- Test: `tests/stage2_one_stage/test_balance.py`

- [ ] **Step 1: Write the failing test for CLI-facing defaults or entry behavior**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal CLI wiring**
- [ ] **Step 4: Run test to verify it passes**

### Task 6: Document usage

**Files:**
- Modify: `README.md`
- Modify: `docs/project/experiment-plan.md`

- [ ] **Step 1: Document how to generate Stage 2 class balance reports**
- [ ] **Step 2: Document how to generate abnormal boost and light-aug variants**
- [ ] **Step 3: Document how to train against each variant**

## Chunk 4: Verification and wrap-up

### Task 7: Run focused verification

**Files:**
- Test: `tests/stage2_one_stage/test_balance.py`
- Test: `tests/data_tools/test_validate_yolo_obb.py`

- [ ] **Step 1: Run focused pytest for new Stage 2 balance coverage**
- [ ] **Step 2: Run any existing related tests touched by the implementation**
- [ ] **Step 3: Verify generated dataset labels still pass YOLO OBB validation**
- [ ] **Step 4: Commit**
