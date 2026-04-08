# Insulator OBB Project Design

**Date:** 2026-04-07

**Goal**

Build a single experiment repository for insulator detection, counting, and defect recognition that supports:

- Stage 0: OBB feasibility validation with one-class `insulator` labels
- Stage 1: two-stage pipeline with OBB detection plus patch classification
- Stage 2: one-stage OBB detection with `normal` and `abnormal` classes

## 1. Problem Framing

The project has one long-term business goal and one short-term technical goal.

Long-term goal:
- Detect each insulator in aerial images
- Count the number of insulators in each image
- Judge whether each insulator is normal or defective
- Compare the effectiveness of a two-stage route against a one-stage route

Short-term goal:
- Use the current OBB-only `insulator` dataset to validate whether YOLO11 OBB can localize insulators accurately enough and count them reliably enough to justify later defect experiments

This means the first deliverable is not a full defect system. It is a reliable OBB experiment platform with a strong Stage 0 baseline.

## 2. Recommended Technical Strategy

Use a single repository with shared tooling and three experiment tracks.

### Why this strategy

- The user ultimately wants to compare two-stage and one-stage methods, so separate ad hoc projects would duplicate data processing, evaluation, and inference code.
- The current dataset only supports one-class OBB detection, so Stage 0 must exist as an isolated validation track.
- Shared infrastructure will make later relabeling and ablation work much easier.

### Tracks

1. **Stage 0: OBB feasibility**
   - Input: current LabelMe-style OBB annotations for `insulator`
   - Train: YOLO11 OBB single-class detector
   - Evaluate: OBB detection quality and counting quality
   - Output: model weights, metrics, visualizations, count reports

2. **Stage 1: Two-stage defect recognition**
   - Step A: detect insulators with Stage 0 detector
   - Step B: crop detected insulator patches
   - Step C: relabel cropped patches as `normal` or `abnormal`
   - Step D: train a patch classifier
   - Output: per-insulator location plus defect class

3. **Stage 2: One-stage defect OBB**
   - Relabel original OBB dataset directly with `normal` and `abnormal`
   - Train YOLO11 OBB as a two-class detector
   - Output: per-insulator location plus defect class in one pass

## 3. Repository Layout

```text
yolo-obb/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ stage0_obb/
│  ├─ stage1_two_stage/
│  └─ stage2_one_stage/
├─ data/
│  ├─ raw/
│  │  ├─ labelme_obb/
│  │  ├─ cropped_patches/
│  │  └─ relabeled_one_stage/
│  ├─ processed/
│  │  ├─ yolo_obb_insulator/
│  │  ├─ classifier_patches/
│  │  └─ yolo_obb_defect/
│  └─ splits/
├─ src/
│  ├─ common/
│  ├─ data_tools/
│  ├─ stage0_obb/
│  ├─ stage1_two_stage/
│  └─ stage2_one_stage/
├─ scripts/
│  ├─ data/
│  ├─ train/
│  ├─ eval/
│  └─ infer/
├─ runs/
├─ reports/
│  ├─ figures/
│  ├─ metrics/
│  └─ comparisons/
└─ docs/
   └─ superpowers/
      ├─ specs/
      └─ plans/
```

## 4. Data Strategy

### 4.1 Raw data

The current raw dataset is LabelMe JSON with four-point polygons and matching JPG images. This is valid source data, but not the training format expected by YOLO11 OBB.

The repository should treat the current `datasets/` folder as temporary source input and move or copy it into:

- `data/raw/labelme_obb/images/`
- `data/raw/labelme_obb/annotations/`

Raw data should remain immutable after ingestion.

### 4.2 Processed data

Three processed datasets are needed.

1. `data/processed/yolo_obb_insulator/`
   - For Stage 0
   - YOLO11 OBB single-class labels

2. `data/processed/classifier_patches/`
   - For Stage 1
   - Cropped patch images and classification labels

3. `data/processed/yolo_obb_defect/`
   - For Stage 2
   - YOLO11 OBB two-class labels

### 4.3 Splits

Because the current dataset is very small, Stage 0 should support two evaluation modes.

- `debug_split`
  - Small-sample development mode
  - Strong augmentation
  - Used to check whether training and visualization behave correctly

- `formal_split`
  - A repeatable train/val/test split definition file
  - Used later when more data exists

All split files should live in `data/splits/` and be stored explicitly for reproducibility.

## 5. Stage 0 Data Flow

Stage 0 is the first concrete experiment path.

### Flow

1. Ingest raw LabelMe OBB files
2. Validate annotation structure
3. Convert polygons into YOLO11 OBB label format
4. Build train/val split metadata
5. Train YOLO11 OBB single-class detector
6. Run validation inference
7. Compute detection metrics
8. Compute count metrics
9. Export visualizations and count comparison tables

### Outputs

- Trained weights
- Per-image prediction text or JSON exports
- Visual overlays with predicted OBBs
- Aggregate metrics for detection and counting

## 6. Stage 1 Data Flow

Stage 1 depends on a usable Stage 0 detector.

### Flow

1. Use the Stage 0 detector to infer insulator OBBs
2. Crop image patches around each predicted or ground-truth insulator
3. Save patch metadata linking each crop to the original image and OBB
4. Manually relabel cropped patches as `normal` or `abnormal`
5. Train a classification model
6. Combine detection results with classifier outputs during inference

### Expected advantages

- Easier defect annotation because users label small patches rather than full images
- Classifier can focus on local texture and damage cues

### Expected risks

- Error propagation from detection to classification
- Patch quality depends on crop geometry and padding rules

## 7. Stage 2 Data Flow

Stage 2 is a direct end-to-end defect OBB route.

### Flow

1. Relabel original OBB boxes as `normal` or `abnormal`
2. Convert relabeled annotations into YOLO11 OBB format
3. Train a two-class YOLO11 OBB detector
4. Evaluate location quality, class quality, and counting quality

### Expected advantages

- Single model and simpler inference pipeline
- No error handoff between detector and classifier

### Expected risks

- More difficult annotation on full-resolution images
- Defect cues may be too small for direct one-stage learning

## 8. Module Responsibilities

### `src/common`

Shared utilities:
- path helpers
- YAML and JSON loading
- run directory naming
- lightweight logging
- image drawing
- metric report writing

### `src/data_tools`

Shared data tooling:
- LabelMe dataset ingestion
- annotation validation
- LabelMe to YOLO OBB conversion
- split generation
- crop generation
- dataset manifest export

### `src/stage0_obb`

Stage 0 only:
- stage config parsing
- YOLO11 OBB train wrapper
- prediction export
- OBB metric summarization
- count metric summarization

### `src/stage1_two_stage`

Stage 1 only:
- crop pipeline
- crop metadata handling
- patch classifier training wrapper
- combined detection plus classification inference

### `src/stage2_one_stage`

Stage 2 only:
- relabeled OBB dataset handling
- two-class OBB train wrapper
- combined localization and defect evaluation

## 9. Evaluation Design

The project needs unified evaluation so later comparisons are fair.

### 9.1 Detection metrics

For Stage 0 and Stage 2:
- mAP50
- mAP50-95
- precision
- recall

### 9.2 Counting metrics

For Stage 0 and Stage 2:
- per-image ground-truth count
- per-image predicted count
- absolute count error
- mean absolute error
- exact-count accuracy

### 9.3 Defect metrics

For Stage 1 and Stage 2:
- classification accuracy
- precision and recall for `abnormal`
- F1 for `abnormal`

### 9.4 Comparison outputs

Save comparison-ready artifacts in:
- `reports/metrics/`
- `reports/figures/`
- `reports/comparisons/`

## 10. Augmentation Strategy

The user explicitly wants to rely on augmentation for the current small dataset. This is acceptable for Stage 0 feasibility work, but must be treated as a technical validation rather than a generalization claim.

Recommended augmentation scope for Stage 0:
- flips
- rotation
- scale jitter
- brightness and contrast changes
- mild blur or noise

Rules:
- keep augmentation configuration versioned
- store it in config files
- separate training-time augmentation from offline data conversion
- do not present Stage 0 results as final production performance

## 11. Inference Interface

The repository should expose simple script entry points.

Examples:
- convert raw LabelMe OBB to YOLO OBB
- train Stage 0 detector
- evaluate Stage 0 detector
- crop Stage 1 patches
- train Stage 1 classifier
- train Stage 2 OBB detector
- run image inference and save overlays

This keeps experiments reproducible and reduces manual notebook-style drift.

## 12. Error Handling

The data tools should fail early and clearly on:

- missing image or annotation pairs
- malformed polygons
- unsupported point counts
- unknown class names
- empty split files

The training wrappers should also record:
- config used
- dataset path used
- weights path used
- command or API arguments used

## 13. Testing Strategy

Testing should focus on the repository logic rather than Ultralytics internals.

Priority tests:
- dataset pairing
- LabelMe parsing
- polygon conversion
- split generation
- count metric calculation
- crop generation geometry
- report writing

Integration tests can be lightweight and use tiny fixtures.

## 14. Delivery Order

Recommended implementation order:

1. Build data ingestion and conversion
2. Build Stage 0 training and evaluation
3. Build Stage 0 count reporting and visualization
4. Build Stage 1 crop pipeline
5. Build Stage 1 classifier scaffolding
6. Build Stage 2 relabel pipeline and training scaffold
7. Build unified comparison outputs

## 15. Success Criteria

### Stage 0 success

The repository is successful for Stage 0 when it can:
- convert the current LabelMe OBB dataset into YOLO11 OBB format
- train a one-class YOLO11 OBB detector
- generate OBB visualization outputs
- report both detection and counting metrics

### Full-project success

The full project is successful when it can:
- run two-stage defect recognition
- run one-stage defect recognition
- export comparable metrics across both routes
- support future dataset growth without structural rework

## 16. Final Recommendation

Proceed with a unified repository centered on Stage 0 OBB feasibility first. Treat Stage 1 and Stage 2 as reserved but planned extensions, not speculative afterthoughts. This gives the fastest path to technical evidence while preserving the final comparison goal.
