# Insulator OBB Project Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified YOLO11-based repository that first validates single-class insulator OBB detection and counting, while reserving clean extension points for later two-stage and one-stage defect recognition.

**Architecture:** The repository will separate immutable raw data, processed training datasets, experiment-specific modules, and reproducible script entry points. Stage 0 will be fully implemented first, while Stage 1 and Stage 2 will receive scaffolded modules, configs, and interfaces so later work can extend the same platform without structural churn.

**Tech Stack:** Python, Ultralytics YOLO11 OBB, OpenCV/Pillow, PyYAML, pytest

---

## Chunk 1: Repository Skeleton And Shared Infrastructure

### Task 1: Create the top-level project layout

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `configs/stage0_obb/.gitkeep`
- Create: `configs/stage1_two_stage/.gitkeep`
- Create: `configs/stage2_one_stage/.gitkeep`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `data/splits/.gitkeep`
- Create: `src/common/__init__.py`
- Create: `src/data_tools/__init__.py`
- Create: `src/stage0_obb/__init__.py`
- Create: `src/stage1_two_stage/__init__.py`
- Create: `src/stage2_one_stage/__init__.py`
- Create: `scripts/data/.gitkeep`
- Create: `scripts/train/.gitkeep`
- Create: `scripts/eval/.gitkeep`
- Create: `scripts/infer/.gitkeep`
- Create: `reports/figures/.gitkeep`
- Create: `reports/metrics/.gitkeep`
- Create: `reports/comparisons/.gitkeep`

- [ ] **Step 1: Create the failing smoke test**

```python
from pathlib import Path

def test_project_layout_exists():
    assert Path("src/common").exists()
    assert Path("src/stage0_obb").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q`
Expected: FAIL because the project layout does not exist yet

- [ ] **Step 3: Create the minimal layout and package markers**

Create the directories and placeholder files listed above. Keep files ASCII-only and minimal.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q`
Expected: PASS for the smoke test

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt configs data src scripts reports
git commit -m "chore: scaffold insulator obb project layout"
```

### Task 2: Add shared settings and path utilities

**Files:**
- Create: `src/common/paths.py`
- Create: `src/common/io.py`
- Test: `tests/common/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
from src.common.paths import project_root, data_dir

def test_paths_resolve():
    assert project_root().exists()
    assert data_dir().name == "data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/common/test_paths.py -v`
Expected: FAIL with import or attribute errors

- [ ] **Step 3: Write minimal implementation**

Implement helper functions that resolve:
- project root
- `data/`
- `reports/`
- `runs/`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/common/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/common/paths.py src/common/io.py tests/common/test_paths.py
git commit -m "feat: add shared project path helpers"
```

## Chunk 2: Raw Data Ingestion And Validation

### Task 3: Add raw dataset ingestion for the current LabelMe data

**Files:**
- Create: `src/data_tools/ingest.py`
- Create: `scripts/data/ingest_labelme_dataset.py`
- Test: `tests/data_tools/test_ingest.py`

- [ ] **Step 1: Write the failing test**

```python
from src.data_tools.ingest import find_labelme_pairs

def test_find_labelme_pairs_returns_image_annotation_pairs():
    pairs = find_labelme_pairs("datasets")
    assert len(pairs) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_tools/test_ingest.py -v`
Expected: FAIL because the ingestion module does not exist yet

- [ ] **Step 3: Write minimal implementation**

Implement:
- pair discovery for `.JPG` and `.json`
- validation that each image has one matching annotation
- a script entry point to copy or stage the source files into `data/raw/labelme_obb/`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_tools/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_tools/ingest.py scripts/data/ingest_labelme_dataset.py tests/data_tools/test_ingest.py
git commit -m "feat: add labelme dataset ingestion"
```

### Task 4: Add annotation validation for OBB structure

**Files:**
- Create: `src/data_tools/validate.py`
- Create: `scripts/data/validate_labelme_obb.py`
- Test: `tests/data_tools/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.data_tools.validate import validate_labelme_shape

def test_validate_labelme_shape_accepts_four_point_polygon():
    shape = {"label": "insulator", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    assert validate_labelme_shape(shape) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_tools/test_validate.py -v`
Expected: FAIL because validation logic does not exist

- [ ] **Step 3: Write minimal implementation**

Implement validation for:
- shape type is polygon
- exactly four points
- known labels
- image metadata is present

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_tools/test_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_tools/validate.py scripts/data/validate_labelme_obb.py tests/data_tools/test_validate.py
git commit -m "feat: add labelme obb validation"
```

## Chunk 3: Label Conversion And Split Metadata

### Task 5: Convert LabelMe polygons into YOLO11 OBB labels

**Files:**
- Create: `src/data_tools/convert_labelme_to_yolo_obb.py`
- Create: `scripts/data/convert_to_yolo_obb.py`
- Test: `tests/data_tools/test_convert_labelme_to_yolo_obb.py`

- [ ] **Step 1: Write the failing test**

```python
from src.data_tools.convert_labelme_to_yolo_obb import convert_shape_to_yolo_obb_line

def test_convert_shape_to_yolo_obb_line_emits_class_and_eight_coordinates():
    shape = {"label": "insulator", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    line = convert_shape_to_yolo_obb_line(shape, image_width=20, image_height=20, class_to_id={"insulator": 0})
    assert len(line.split()) == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_tools/test_convert_labelme_to_yolo_obb.py -v`
Expected: FAIL because conversion is not implemented

- [ ] **Step 3: Write minimal implementation**

Implement:
- class-to-id mapping
- normalized four-point coordinate export
- train and val image and label directory creation
- YAML dataset file export for Ultralytics

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_tools/test_convert_labelme_to_yolo_obb.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_tools/convert_labelme_to_yolo_obb.py scripts/data/convert_to_yolo_obb.py tests/data_tools/test_convert_labelme_to_yolo_obb.py
git commit -m "feat: convert labelme polygons to yolo obb labels"
```

### Task 6: Add split generation and manifest writing

**Files:**
- Create: `src/data_tools/splits.py`
- Create: `scripts/data/make_split.py`
- Test: `tests/data_tools/test_splits.py`

- [ ] **Step 1: Write the failing test**

```python
from src.data_tools.splits import make_debug_split

def test_make_debug_split_contains_train_and_val_keys():
    split = make_debug_split(["a", "b", "c", "d"])
    assert "train" in split and "val" in split
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_tools/test_splits.py -v`
Expected: FAIL because split generation is missing

- [ ] **Step 3: Write minimal implementation**

Implement:
- debug split generation
- optional deterministic split by seed
- manifest writing to `data/splits/`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_tools/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_tools/splits.py scripts/data/make_split.py tests/data_tools/test_splits.py
git commit -m "feat: add reproducible dataset splits"
```

## Chunk 4: Stage 0 Training And Evaluation

### Task 7: Add Stage 0 configuration and training wrapper

**Files:**
- Create: `configs/stage0_obb/train_insulator.yaml`
- Create: `src/stage0_obb/config.py`
- Create: `src/stage0_obb/train.py`
- Create: `scripts/train/train_stage0_obb.py`
- Test: `tests/stage0_obb/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage0_obb.config import load_stage0_config

def test_load_stage0_config_reads_model_name(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: yolo11n-obb.pt\n", encoding="utf-8")
    data = load_stage0_config(cfg)
    assert data["model"] == "yolo11n-obb.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage0_obb/test_config.py -v`
Expected: FAIL because the config module does not exist

- [ ] **Step 3: Write minimal implementation**

Implement:
- config loading
- default train arguments
- train wrapper calling Ultralytics YOLO OBB train API
- run directory handling

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage0_obb/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/stage0_obb/train_insulator.yaml src/stage0_obb/config.py src/stage0_obb/train.py scripts/train/train_stage0_obb.py tests/stage0_obb/test_config.py
git commit -m "feat: add stage0 obb training wrapper"
```

### Task 8: Add Stage 0 prediction export and visualization

**Files:**
- Create: `src/stage0_obb/predict.py`
- Create: `src/stage0_obb/visualize.py`
- Create: `scripts/infer/infer_stage0_obb.py`
- Test: `tests/stage0_obb/test_visualize.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage0_obb.visualize import color_for_class

def test_color_for_class_returns_rgb_triplet():
    color = color_for_class("insulator")
    assert len(color) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage0_obb/test_visualize.py -v`
Expected: FAIL because visualization code is absent

- [ ] **Step 3: Write minimal implementation**

Implement:
- prediction wrapper
- overlay drawing for OBB quadrilaterals
- saved result images in `reports/figures/`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage0_obb/test_visualize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stage0_obb/predict.py src/stage0_obb/visualize.py scripts/infer/infer_stage0_obb.py tests/stage0_obb/test_visualize.py
git commit -m "feat: add stage0 obb inference visualization"
```

### Task 9: Add detection and count metrics

**Files:**
- Create: `src/stage0_obb/metrics.py`
- Create: `scripts/eval/eval_stage0_obb.py`
- Test: `tests/stage0_obb/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage0_obb.metrics import compute_count_metrics

def test_compute_count_metrics_reports_mae_and_exact_accuracy():
    metrics = compute_count_metrics([3, 4, 5], [3, 5, 4])
    assert "mae" in metrics
    assert "exact_accuracy" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage0_obb/test_metrics.py -v`
Expected: FAIL because metrics code does not exist

- [ ] **Step 3: Write minimal implementation**

Implement:
- per-image count extraction
- MAE and exact-count accuracy
- report writing to `reports/metrics/`
- pass-through loading of Ultralytics validation metrics when available

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage0_obb/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stage0_obb/metrics.py scripts/eval/eval_stage0_obb.py tests/stage0_obb/test_metrics.py
git commit -m "feat: add stage0 detection and count metrics"
```

## Chunk 5: Stage 1 Scaffolding

### Task 10: Add patch crop generation pipeline

**Files:**
- Create: `src/stage1_two_stage/crops.py`
- Create: `scripts/data/generate_stage1_crops.py`
- Test: `tests/stage1_two_stage/test_crops.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage1_two_stage.crops import padded_crop_bounds

def test_padded_crop_bounds_expands_box_within_image():
    bounds = padded_crop_bounds((10, 10, 20, 20), image_size=(100, 100), padding=4)
    assert bounds[0] >= 0
    assert bounds[1] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage1_two_stage/test_crops.py -v`
Expected: FAIL because crop code does not exist

- [ ] **Step 3: Write minimal implementation**

Implement:
- OBB-to-crop conversion helpers
- image patch export
- crop manifest JSON or CSV

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage1_two_stage/test_crops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stage1_two_stage/crops.py scripts/data/generate_stage1_crops.py tests/stage1_two_stage/test_crops.py
git commit -m "feat: add stage1 insulator crop generation"
```

### Task 11: Add classifier scaffolding

**Files:**
- Create: `configs/stage1_two_stage/classifier.yaml`
- Create: `src/stage1_two_stage/classifier_train.py`
- Create: `src/stage1_two_stage/classifier_infer.py`
- Create: `scripts/train/train_stage1_classifier.py`
- Test: `tests/stage1_two_stage/test_classifier_config.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage1_two_stage.classifier_train import normalize_label

def test_normalize_label_accepts_normal_and_abnormal():
    assert normalize_label("normal") == "normal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage1_two_stage/test_classifier_config.py -v`
Expected: FAIL because classifier scaffolding is missing

- [ ] **Step 3: Write minimal implementation**

Implement:
- config loading
- label normalization
- training placeholder entry point
- inference placeholder entry point

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage1_two_stage/test_classifier_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/stage1_two_stage/classifier.yaml src/stage1_two_stage/classifier_train.py src/stage1_two_stage/classifier_infer.py scripts/train/train_stage1_classifier.py tests/stage1_two_stage/test_classifier_config.py
git commit -m "feat: scaffold stage1 classifier pipeline"
```

## Chunk 6: Stage 2 Scaffolding

### Task 12: Add one-stage defect OBB dataset support

**Files:**
- Create: `configs/stage2_one_stage/train_defect_obb.yaml`
- Create: `src/stage2_one_stage/dataset.py`
- Create: `scripts/data/prepare_stage2_dataset.py`
- Test: `tests/stage2_one_stage/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage2_one_stage.dataset import normalize_stage2_label

def test_normalize_stage2_label_accepts_expected_classes():
    assert normalize_stage2_label("abnormal") == "abnormal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_one_stage/test_dataset.py -v`
Expected: FAIL because stage2 dataset code does not exist

- [ ] **Step 3: Write minimal implementation**

Implement:
- class normalization for `normal` and `abnormal`
- dataset directory conventions
- scaffolded preparation script

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_one_stage/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/stage2_one_stage/train_defect_obb.yaml src/stage2_one_stage/dataset.py scripts/data/prepare_stage2_dataset.py tests/stage2_one_stage/test_dataset.py
git commit -m "feat: scaffold stage2 defect obb dataset support"
```

### Task 13: Add one-stage training wrapper scaffold

**Files:**
- Create: `src/stage2_one_stage/train.py`
- Create: `scripts/train/train_stage2_obb.py`
- Test: `tests/stage2_one_stage/test_train_config.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage2_one_stage.train import stage2_model_name

def test_stage2_model_name_reads_from_config_dict():
    assert stage2_model_name({"model": "yolo11n-obb.pt"}) == "yolo11n-obb.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage2_one_stage/test_train_config.py -v`
Expected: FAIL because training scaffold does not exist

- [ ] **Step 3: Write minimal implementation**

Implement:
- config read helpers
- training wrapper scaffold matching Stage 0 style

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage2_one_stage/test_train_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stage2_one_stage/train.py scripts/train/train_stage2_obb.py tests/stage2_one_stage/test_train_config.py
git commit -m "feat: scaffold stage2 obb training wrapper"
```

## Chunk 7: Documentation And Reproducibility

### Task 14: Document repository usage and experiment flow

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/specs/2026-04-07-insulator-obb-project-design.md`
- Create: `docs/superpowers/plans/2026-04-07-insulator-obb-project.md`

- [ ] **Step 1: Write the failing documentation checklist**

Create a checklist in `README.md` with unchecked items for:
- data ingestion
- conversion
- stage0 training
- evaluation

- [ ] **Step 2: Review the documentation gap**

Run: `rg -n "stage0|stage1|stage2|YOLO11" README.md docs/superpowers`
Expected: Missing or incomplete guidance before documentation work

- [ ] **Step 3: Write the documentation**

Document:
- project purpose
- directory overview
- Stage 0 workflow
- future Stage 1 and Stage 2 workflows
- expected commands

- [ ] **Step 4: Verify the documentation content**

Run: `rg -n "stage0|stage1|stage2|YOLO11" README.md docs/superpowers`
Expected: Matching lines for all key workflow sections

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-04-07-insulator-obb-project-design.md docs/superpowers/plans/2026-04-07-insulator-obb-project.md
git commit -m "docs: add insulator obb project design and plan"
```

Plan complete and saved to `docs/superpowers/plans/2026-04-07-insulator-obb-project.md`. Ready to execute?
