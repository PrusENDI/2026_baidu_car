# Fixed-Track CV + Manual Long Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible 200-epoch D4-initialized fixed-track trainer over 2822 CV rows and 646 crossroad manual rows, with `70/15/15` grouped sampling, staged unfreezing, resumable checkpoints, behavior evaluation, and candidate export.

**Architecture:** Keep the existing short `train_action` entry point unchanged for compatibility. Add focused modules for deterministic grouped sampling, staged optimizer/checkpoint state, and action-sequence evaluation, then compose them in a dedicated long-training CLI. Every fifth epoch writes a resumable checkpoint and evaluates the crossroad sequence, CV validation, vehicle post-processing, and three independent manual reference sets.

**Tech Stack:** Python 3.10, PaddlePaddle 3.3.1, NumPy, Pillow, pytest, JSON/JSONL, PowerShell/OpenSSH, RTX 4080 SUPER.

---

## File map

- Create `lane_training/grouped_sampling.py`: partition rows and build reproducible `45/10/9` / `45/9/10` batches.
- Create `lane_training/staged_action.py`: define the three epoch stages, freeze layers, build Paddle optimizer groups, and save/load complete training state.
- Create `lane_training/action_sequence_metrics.py`: summarize crossroad sequence behavior and replay the real vehicle speed/curvature controller.
- Create `tools/lane_training/train_fixed_track_long.py`: dedicated 200-epoch orchestration CLI.
- Create `tools/lane_training/evaluate_fixed_track_long.py`: evaluate one checkpoint or a checkpoint directory against all agreed sets.
- Create `tools/lane_training/select_fixed_track_candidates.py`: create the three-role shortlist without declaring a deployed winner.
- Modify `tools/lane_training/export_action.py`: expose testable export helpers and attach training/selection provenance.
- Create focused tests matching each new module and CLI.
- Modify `docs/lane-cnn-training-handoff.md` and `docs/cv-lane-development-status-2026-08-15.md` only after verified training results exist.

### Task 0: Preserve the verified 20-epoch joint-training baseline

**Files:**
- Existing modified: `lane_training/action_loss.py`
- Existing modified: `lane_training/action_training.py`
- Existing modified: `lane_training/dataset.py`
- Existing modified: `tests/lane_training/test_action_loss.py`
- Existing modified: `tests/lane_training/test_dataset.py`
- Existing modified: `docs/lane-cnn-training-handoff.md`
- Existing modified: `docs/cv-lane-development-status-2026-08-15.md`

- [ ] **Step 1: Re-run the already established baseline checks**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest `
  tests/lane_training/test_action_loss.py `
  tests/lane_training/test_dataset.py `
  tests/lane_training/test_manifest.py -q
git diff --check
```

Expected: 16 focused tests pass and whitespace validation is silent.

- [ ] **Step 2: Review the exact baseline patch**

```powershell
git diff -- `
  lane_training/action_loss.py lane_training/action_training.py lane_training/dataset.py `
  tests/lane_training/test_action_loss.py tests/lane_training/test_dataset.py `
  docs/lane-cnn-training-handoff.md docs/cv-lane-development-status-2026-08-15.md
```

Expected: only the already verified `kappa/5.0` loss, action flip control, fixed-track flip disablement, related tests, and recorded 20-epoch results appear. Stop if unrelated user edits are present in any listed file.

- [ ] **Step 3: Commit code and documentation separately**

```powershell
git add lane_training/action_loss.py lane_training/action_training.py lane_training/dataset.py tests/lane_training/test_action_loss.py tests/lane_training/test_dataset.py
git commit -m "feat: align joint action training semantics"
git add docs/lane-cnn-training-handoff.md docs/cv-lane-development-status-2026-08-15.md
git commit -m "docs: record first joint action training"
```

Expected: no `.venv-lane`, `artifacts`, unrelated untracked tests/tools, or old untracked plan files enter either commit.

### Task 1: Configurable staged photometric augmentation

**Files:**
- Modify: `lane_training/dataset.py:19-110`
- Modify: `tests/lane_training/test_dataset.py`

- [ ] **Step 1: Write failing augmentation-profile tests**

```python
from lane_training.dataset import (
    CLEAN_ACTION_AUGMENTATION,
    MILD_FIXED_TRACK_ACTION_AUGMENTATION,
    ActionAugmentationConfig,
)


def test_fixed_track_augmentation_profiles_match_approved_ranges():
    assert CLEAN_ACTION_AUGMENTATION.application_probability == 0.0
    mild = MILD_FIXED_TRACK_ACTION_AUGMENTATION
    assert mild.application_probability == pytest.approx(0.30)
    assert mild.brightness_range == pytest.approx((0.95, 1.05))
    assert mild.contrast_range == pytest.approx((0.95, 1.05))
    assert mild.saturation_range == pytest.approx((0.97, 1.03))
    assert mild.hue_delta == 2
    assert mild.blur_probability == 0.0
    assert mild.noise_probability == 0.0
    assert mild.horizontal_flip_probability == 0.0


def test_action_augmentation_config_rejects_invalid_probability():
    with pytest.raises(ValueError, match="application_probability"):
        ActionAugmentationConfig(application_probability=1.1)
```

- [ ] **Step 2: Run the tests and confirm the missing symbols**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_dataset.py -q
```

Expected: collection fails because the new augmentation types/constants do not exist.

- [ ] **Step 3: Implement the immutable augmentation configuration**

Add this validated frozen dataclass and three constants to `dataset.py`:

```python
@dataclass(frozen=True)
class ActionAugmentationConfig:
    application_probability: float = 1.0
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.85, 1.15)
    hue_delta: int = 8
    blur_probability: float = 0.15
    noise_probability: float = 0.15
    horizontal_flip_probability: float = 0.5


LEGACY_ACTION_AUGMENTATION = ActionAugmentationConfig()
CLEAN_ACTION_AUGMENTATION = ActionAugmentationConfig(
    application_probability=0.0,
    brightness_range=(1.0, 1.0),
    contrast_range=(1.0, 1.0),
    saturation_range=(1.0, 1.0),
    hue_delta=0,
    blur_probability=0.0,
    noise_probability=0.0,
    horizontal_flip_probability=0.0,
)
MILD_FIXED_TRACK_ACTION_AUGMENTATION = ActionAugmentationConfig(
    application_probability=0.30,
    brightness_range=(0.95, 1.05),
    contrast_range=(0.95, 1.05),
    saturation_range=(0.97, 1.03),
    hue_delta=2,
    blur_probability=0.0,
    noise_probability=0.0,
    horizontal_flip_probability=0.0,
)
```

Validate all probabilities in `[0,1]`, ordered positive scalar ranges, and nonnegative `hue_delta` in `__post_init__`.

- [ ] **Step 4: Route `augment_rgb` and `LaneDataset` through the profile**

Add `config: ActionAugmentationConfig = LEGACY_ACTION_AUGMENTATION` and keyword `return_applied: bool = False` to `augment_rgb`. Consume one gate draw first; if it is outside `application_probability`, keep the image/label unchanged and set `applied=False`. Otherwise sample only within the config ranges, and use config blur/noise/flip probabilities. Preserve the existing two-value `(image, label)` return by default; return `(image, label, applied)` only when `return_applied=True`, which is how `LaneDataset` counts hits.

Add `action_augmentation_config: ActionAugmentationConfig | None = None` to `LaneDataset`. When absent, create a legacy config whose flip probability comes from the existing `horizontal_flip_probability` argument, preserving all old callers. Add `set_action_augmentation_config(config)` and `augmentation_applied_count`; reset the counter in `set_epoch` and increment it only after the gate succeeds.

- [ ] **Step 5: Test clean inputs, mild gating, and backward compatibility**

Add tests proving:

1. `CLEAN_ACTION_AUGMENTATION` returns identical image, target, and mask across epochs;
2. a forced `application_probability=1.0` mild profile changes pixels but never changes target/mask;
3. mild profile never calls blur, noise, or horizontal flip (monkeypatch those operations to fail if called);
4. legacy `horizontal_flip_probability=1.0` still flips curvature exactly as before;
5. `augmentation_applied_count` is zero for clean and equals accessed samples for a forced profile.

- [ ] **Step 6: Run dataset tests and commit**

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_dataset.py -q
git add lane_training/dataset.py tests/lane_training/test_dataset.py
git commit -m "feat: add staged action augmentation profiles"
```

Expected: all previous dataset behavior remains compatible and the clean/mild profiles pass.

### Task 2: Deterministic three-group sampling

**Files:**
- Create: `lane_training/grouped_sampling.py`
- Create: `tests/lane_training/test_grouped_sampling.py`

- [ ] **Step 1: Write failing partition tests**

```python
from lane_training.grouped_sampling import partition_action_rows


def test_partition_action_rows_uses_source_mask_and_active_threshold():
    rows = [
        {"source": "cv", "target_mask": [1.0, 1.0], "kappa_action": 0.0},
        {"source": "manual", "target_mask": [0.0, 1.0], "kappa_action": 0.05},
        {"source": "manual", "target_mask": [0.0, 1.0], "kappa_action": -0.049},
    ]
    groups = partition_action_rows(rows, active_threshold=0.05)
    assert groups == {"cv": [0], "manual_active": [1], "manual_context": [2]}


def test_partition_action_rows_rejects_wrong_manual_mask():
    rows = [{"source": "manual", "target_mask": [1.0, 1.0], "kappa_action": 1.0}]
    try:
        partition_action_rows(rows)
    except ValueError as error:
        assert "manual target_mask" in str(error)
    else:
        raise AssertionError("wrong manual mask was accepted")
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_grouped_sampling.py -q
```

Expected: collection fails with `ModuleNotFoundError: lane_training.grouped_sampling`.

- [ ] **Step 3: Implement row partitioning**

Create the module with this public contract:

```python
from __future__ import annotations

from collections.abc import Sequence

GROUPS = ("cv", "manual_active", "manual_context")


def partition_action_rows(
    rows: Sequence[dict], *, active_threshold: float = 0.05,
) -> dict[str, list[int]]:
    if active_threshold <= 0:
        raise ValueError("active_threshold must be positive")
    groups = {name: [] for name in GROUPS}
    for index, row in enumerate(rows):
        source = row.get("source")
        mask = [float(value) for value in row.get("target_mask", [1.0, 1.0])]
        if source == "cv":
            if mask != [1.0, 1.0]:
                raise ValueError("cv target_mask must be [1, 1]")
            groups["cv"].append(index)
        elif source == "manual":
            if mask != [0.0, 1.0]:
                raise ValueError("manual target_mask must be [0, 1]")
            name = (
                "manual_active"
                if abs(float(row["kappa_action"])) >= active_threshold
                else "manual_context"
            )
            groups[name].append(index)
        else:
            raise ValueError(f"unsupported training source: {source!r}")
    empty = [name for name, indices in groups.items() if not indices]
    if empty:
        raise ValueError(f"empty training groups: {', '.join(empty)}")
    return groups
```

- [ ] **Step 4: Add and verify deterministic batch-schedule tests**

Append tests that call:

```python
from lane_training.grouped_sampling import build_epoch_batches


def test_epoch_batches_cover_cv_and_alternate_manual_counts():
    groups = {
        "cv": list(range(2822)),
        "manual_active": list(range(2822, 2896)),
        "manual_context": list(range(2896, 3468)),
    }
    batches, report = build_epoch_batches(groups, batch_size=64, seed=20260817, epoch=1)
    assert len(batches) == 63
    assert all(len(batch) == 64 for batch in batches)
    assert report["draw_counts"] == {
        "cv": 2835, "manual_active": 599, "manual_context": 598,
    }
    assert set(groups["cv"]).issubset({index for batch in batches for index in batch})
    assert batches == build_epoch_batches(
        groups, batch_size=64, seed=20260817, epoch=1,
    )[0]
    assert batches != build_epoch_batches(
        groups, batch_size=64, seed=20260817, epoch=2,
    )[0]
```

Implement `build_epoch_batches` so odd batches draw `45/10/9`, even batches draw `45/9/10`, `ceil(len(cv)/45)` batches are produced, each group has its own `numpy.random.default_rng(seed + epoch * 1009 + group_offset)`, and a depleted group is reshuffled before cycling. Return a JSON-safe report containing source row counts, draw counts, draw ratios, seed, epoch, and batch count.

Run:

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_grouped_sampling.py -q
```

Expected: all grouped-sampling tests pass.

- [ ] **Step 5: Commit the sampler**

```powershell
git add lane_training/grouped_sampling.py tests/lane_training/test_grouped_sampling.py
git commit -m "feat: add fixed-track grouped sampler"
```

### Task 3: Three-stage freezing and learning rates

**Files:**
- Create: `lane_training/staged_action.py`
- Create: `tests/lane_training/test_staged_action.py`
- Read: `lane_training/model.py:5-20`

- [ ] **Step 1: Write failing stage-boundary tests**

```python
import pytest

from lane_training.staged_action import stage_for_epoch


@pytest.mark.parametrize(
    ("epoch", "name", "head_lr", "backbone_lr", "augmentation_profile"),
    [
        (1, "head", 1e-5, None, "clean"),
        (20, "head", 1e-5, None, "clean"),
        (21, "rear", 5e-6, 1e-6, "mild_photometric"),
        (160, "rear", 5e-6, 1e-6, "mild_photometric"),
        (161, "full", 2e-6, 2e-7, "clean"),
        (200, "full", 2e-6, 2e-7, "clean"),
    ],
)
def test_stage_for_epoch(epoch, name, head_lr, backbone_lr, augmentation_profile):
    stage = stage_for_epoch(epoch)
    assert stage.name == name
    assert stage.head_learning_rate == pytest.approx(head_lr)
    assert stage.backbone_learning_rate == pytest.approx(backbone_lr) if backbone_lr else stage.backbone_learning_rate is None
    assert stage.augmentation_profile == augmentation_profile
```

Run the test and expect a missing-module collection failure.

- [ ] **Step 2: Implement immutable stage definitions**

Create `TrainingStage` as a frozen dataclass and implement `stage_for_epoch(epoch)` with inclusive ranges `1–20`, `21–160`, and `161–200`. Reject epochs outside `1..200` with `ValueError`.

```python
@dataclass(frozen=True)
class TrainingStage:
    name: str
    start_epoch: int
    end_epoch: int
    head_learning_rate: float
    backbone_learning_rate: float | None
    trainable_convolution_indices: tuple[int, ...]
    augmentation_profile: str


STAGES = (
    TrainingStage("head", 1, 20, 1e-5, None, (), "clean"),
    TrainingStage("rear", 21, 160, 5e-6, 1e-6, (8, 11), "mild_photometric"),
    TrainingStage("full", 161, 200, 2e-6, 2e-7, (0, 2, 4, 6, 8, 11), "clean"),
)
```

- [ ] **Step 3: Write parameter-freezing tests**

```python
from lane_training.model import CnnModel
from lane_training.staged_action import configure_stage, optimizer_parameter_groups


def _trainable_layer_indices(model):
    return {
        index for index, layer in enumerate(model.features)
        if list(layer.parameters()) and any(not p.stop_gradient for p in layer.parameters())
    }


def test_configure_head_stage_trains_only_linear_head():
    model = CnnModel()
    stage = configure_stage(model, epoch=1)
    assert _trainable_layer_indices(model) == {15, 17, 20}
    groups = optimizer_parameter_groups(model, stage)
    assert len(groups) == 1
    assert groups[0]["learning_rate"] == 1.0


def test_configure_rear_and_full_stages_use_layered_rates():
    model = CnnModel()
    rear = configure_stage(model, epoch=21)
    assert _trainable_layer_indices(model) == {8, 11, 15, 17, 20}
    assert [g["learning_rate"] for g in optimizer_parameter_groups(model, rear)] == [0.2, 1.0]
    full = configure_stage(model, epoch=161)
    assert _trainable_layer_indices(model) == {0, 2, 4, 6, 8, 11, 15, 17, 20}
    assert [g["learning_rate"] for g in optimizer_parameter_groups(model, full)] == [0.1, 1.0]
```

Implement `configure_stage` by first setting every parameter to `stop_gradient=True`, then enabling convolution layers listed by the stage and linear head indices `(15, 17, 20)`. Implement `optimizer_parameter_groups` with non-overlapping backbone and head lists; reject an empty group and duplicate parameter IDs. The optimizer base learning rate is always the stage head rate, so group rates are multipliers `backbone/head` and `1.0`.

- [ ] **Step 4: Verify and commit staged optimization**

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_staged_action.py -q
git add lane_training/staged_action.py tests/lane_training/test_staged_action.py
git commit -m "feat: add staged action optimization"
```

Expected: all stage tests pass and only the two listed files enter the commit.

### Task 4: Complete resumable checkpoint state

**Files:**
- Modify: `lane_training/staged_action.py`
- Modify: `tests/lane_training/test_staged_action.py`

- [ ] **Step 1: Write failing checkpoint round-trip tests**

```python
import paddle

from lane_training.model import CnnModel
from lane_training.staged_action import (
    build_optimizer, load_training_checkpoint, save_training_checkpoint,
)


def test_training_checkpoint_restores_epoch_stage_optimizer_and_sampler(tmp_path):
    model = CnnModel()
    optimizer, stage = build_optimizer(model, epoch=21)
    checkpoint = save_training_checkpoint(
        tmp_path,
        epoch=21,
        model=model,
        optimizer=optimizer,
        stage=stage,
        sampler_state={"base_seed": 20260817, "next_epoch": 22},
        history=[{"epoch": 21, "train_loss": 0.2}],
    )
    restored = CnnModel()
    result = load_training_checkpoint(checkpoint, restored)
    assert result.last_completed_epoch == 21
    assert result.stage.name == "rear"
    assert result.sampler_state == {"base_seed": 20260817, "next_epoch": 22}
    assert result.history[-1]["epoch"] == 21
    assert result.optimizer is not None
    for name, value in model.state_dict().items():
        assert paddle.allclose(value, restored.state_dict()[name])
```

- [ ] **Step 2: Implement atomic checkpoint directories**

Use this layout:

```text
checkpoints/epoch_0021/
  model.pdparams
  optimizer.pdopt
  state.json
```

`state.json` must contain schema version `1`, last completed epoch, stage name, augmentation profile, sampler state, history, and SHA256 values for both Paddle files. Write into sibling `epoch_0021.incomplete`, validate hashes, then rename it to `epoch_0021`; reject an existing final directory. `load_training_checkpoint` validates schema, hashes, model keys, stage name and augmentation profile for the saved epoch, configures the model for the saved stage, rebuilds the optimizer, and loads optimizer state.

- [ ] **Step 3: Add corruption and stage-transition tests**

Test that changing one byte in `model.pdparams` produces `ValueError("checkpoint SHA256 mismatch")`. Test that resuming epoch 20 returns `next_epoch=21` and the trainer creates a fresh rear-stage optimizer rather than loading the head-stage optimizer into different parameter groups. Test that resuming epoch 21 restores the saved rear optimizer.

- [ ] **Step 4: Run focused tests and commit**

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_staged_action.py -q
git add lane_training/staged_action.py tests/lane_training/test_staged_action.py
git commit -m "feat: add resumable staged checkpoints"
```

Expected: round-trip, corruption, and transition tests pass.

### Task 5: Fixed-track long-training CLI

**Files:**
- Create: `tools/lane_training/train_fixed_track_long.py`
- Create: `tests/lane_training/test_train_fixed_track_long.py`
- Read: `lane_training/action_training.py:11-62`
- Read: `lane_training/dataset.py:85-175`

- [ ] **Step 1: Write failing CLI-default and audit tests**

```python
import pytest

from tools.lane_training.train_fixed_track_long import audit_training_rows, parse_args


def test_long_cli_uses_approved_defaults():
    args = parse_args([
        "--manifest", "joint.jsonl",
        "--initial-checkpoint", "d4.pdparams",
        "--output", "long_v1",
    ])
    assert args.epochs == 200
    assert args.batch_size == 64
    assert args.kappa_weight == pytest.approx(2.0)
    assert args.checkpoint_interval == 5
    assert args.seed == 20260817
    assert args.expected_cv == 2822
    assert args.expected_manual_active == 74
    assert args.expected_manual_context == 572


def test_audit_requires_exact_fixed_track_counts():
    groups = {"cv": list(range(2)), "manual_active": [2], "manual_context": [3]}
    with pytest.raises(ValueError, match="expected training group counts"):
        audit_training_rows(groups, expected={
            "cv": 2822, "manual_active": 74, "manual_context": 572,
        })
```

- [ ] **Step 2: Implement CLI parsing and startup audit**

The initial CLI accepts `--manifest`, `--initial-checkpoint`, `--output`, optional `--resume`, `--device`, the approved defaults above, and `--max-epochs` for smoke. Task 7 adds repeatable evaluation-set arguments after the evaluator exists. It must reject simultaneous `--initial-checkpoint` and `--resume`, an existing non-resume output, wrong group counts, non-finite labels, wrong masks, and a D4 checkpoint whose keys do not exactly match `CnnModel.state_dict()`.

- [ ] **Step 3: Write a failing one-epoch CPU integration test**

Create four 320x240 images in `tmp_path`, a manifest with two CV, one manual-active, one manual-context row, and call `train()` with expected counts overridden to `2/1/1`, `max_epochs=1`, batch size 4, and a saved `CnnModel` checkpoint. Assert:

```python
assert (output / "checkpoints/epoch_0001/model.pdparams").is_file()
assert (output / "checkpoints/epoch_0001/optimizer.pdopt").is_file()
assert (output / "checkpoints/epoch_0001/state.json").is_file()
report = json.loads((output / "training_report.json").read_text())
assert report["data_counts"] == {"cv": 2, "manual_active": 1, "manual_context": 1}
assert report["history"][0]["stage"] == "head"
assert report["history"][0]["valid_speed"] > 0
assert report["history"][0]["valid_kappa"] > 0
```

- [ ] **Step 4: Implement the training loop**

For each epoch:

1. call `configure_stage` and create a new optimizer only at epoch 1 or a stage boundary;
2. create `LaneDataset` with `training=True`, `return_target_mask=True`, and `horizontal_flip_probability=0.0`; at each epoch set `CLEAN_ACTION_AUGMENTATION` for stage A/C or `MILD_FIXED_TRACK_ACTION_AUGMENTATION` for stage B, then call `set_epoch(epoch)`;
3. obtain grouped batch indices from `build_epoch_batches`;
4. run `masked_smooth_l1_loss(..., kappa_weight=2.0)`;
5. reject non-finite predictions, total loss, speed loss, or kappa loss;
6. accumulate actual group draws, valid-mask counts, augmentation profile, and `augmentation_applied_count`;
7. save a complete checkpoint every fifth epoch and at the requested final/smoke epoch;
8. rewrite `training_report.json` atomically after every epoch.

Set Python `random`, NumPy, and Paddle seeds from `--seed` before model or optimizer creation. Filter the manifest to `split=train` before partitioning. Add boundary tests asserting augmentation hits are zero at epochs 20 and 161, while epoch 21 reports profile `mild_photometric` and a deterministic nonzero hit count. The report must record D4 SHA256, manifest SHA256, output semantics, all CLI settings, stage, augmentation profile/hits, parameter-group rates, group draws/ratios, valid counts, total/speed/kappa losses, elapsed seconds, and checkpoint path. Do not write or update `best.pdparams` because selection is behavioral and occurs later.

- [ ] **Step 5: Verify exact resume behavior**

Add an integration test that trains one epoch, resumes the saved checkpoint to epoch 2, and asserts the final history contains epochs `[1, 2]` exactly once. Add a boundary unit test using a synthetic epoch-20 checkpoint and assert epoch 21 reports stage `rear`, learning rates `5e-6/1e-6`, and augmentation profile `mild_photometric`. Add a synthetic epoch-160 resume test asserting epoch 161 returns to profile `clean`.

- [ ] **Step 6: Run tests and commit the trainer**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_train_fixed_track_long.py tests/lane_training/test_action_loss.py tests/lane_training/test_dataset.py -q
git add tools/lane_training/train_fixed_track_long.py tests/lane_training/test_train_fixed_track_long.py
git commit -m "feat: add fixed-track long trainer"
```

Expected: focused trainer, loss, and dataset tests pass.

### Task 6: Sequence and vehicle-postprocessing metrics

**Files:**
- Create: `lane_training/action_sequence_metrics.py`
- Create: `tests/lane_training/test_action_sequence_metrics.py`
- Read: `smartcar/whalesbot/tools/curvature_control.py:39-90`

- [ ] **Step 1: Write failing sequence-metric tests**

```python
import numpy as np
import pytest

from lane_training.action_sequence_metrics import summarize_action_sequence


def test_sequence_summary_measures_direction_false_steer_impulse_and_jitter():
    target = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
    prediction = np.array([0.06, 0.0, 1.0, 1.0, 1.0, 0.0])
    report = summarize_action_sequence(target, prediction, active_threshold=0.05)
    assert report["active_count"] == 3
    assert report["context_count"] == 3
    assert report["direction_accuracy"] == pytest.approx(1.0)
    assert report["active_recall"] == pytest.approx(1.0)
    assert report["false_steer_rate"] == pytest.approx(1 / 3)
    assert report["impulse_ratio"] == pytest.approx(0.75)
    assert report["turn_count"] == 1
    assert report["median_onset_offset_frames"] == pytest.approx(0.0)
```

- [ ] **Step 2: Implement finite sequence metrics**

Implement contiguous active-run discovery and return: row count, active/context counts, kappa MAE, direction accuracy, active recall, false-steer rate, sequence correlation, target/prediction mean absolute kappa, impulse ratio, turn count, median onset/exit offsets, peak absolute kappa, and p95 absolute adjacent-frame delta. Accept a parallel `sequence_keys` list, split runs at key changes, and derive each default key from `row["session_id"]` or the parent of `image_path`; the boundary between the two manual sessions must never become one artificial turn. Return `None`, not JSON NaN, when correlation or turn timing is undefined.

- [ ] **Step 3: Write failing controller-replay tests**

```python
from lane_training.action_sequence_metrics import replay_vehicle_commands


def test_vehicle_replay_uses_speed_demand_real_dt_and_wz_clip():
    rows = [
        {"effective_dt_s": 0.05},
        {"effective_dt_s": 0.05},
    ]
    predictions = np.array([[0.0, 0.0], [1.0, 10.0]], dtype=np.float32)
    report = replay_vehicle_commands(rows, predictions)
    assert report["actual_vx"][0] == pytest.approx(0.30)
    assert report["actual_vx"][1] == pytest.approx(0.30 - 0.194 * 0.05)
    assert report["wz"][1] == pytest.approx(1.50)
    assert report["angular_clip_rate"] == pytest.approx(0.5)
```

Implement replay with the production `CurvatureSpeedController` defaults, `speed_demand=prediction[0]`, `curvature=prediction[1]`, and each row's `effective_dt_s` or `0.05`. Reset the controller whenever the derived sequence key changes. Return JSON-safe sequences plus min/mean/max vx, min/mean/max absolute wz, angular clip rate, p95 speed delta, p95 wz delta, and the number of rows that used fallback `dt`.

- [ ] **Step 4: Verify and commit metrics**

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_action_sequence_metrics.py -q
git add lane_training/action_sequence_metrics.py tests/lane_training/test_action_sequence_metrics.py
git commit -m "feat: add action sequence evaluation"
```

Expected: sequence and controller-replay tests pass.

### Task 7: Per-checkpoint evaluation suite

**Files:**
- Create: `tools/lane_training/evaluate_fixed_track_long.py`
- Create: `tests/lane_training/test_evaluate_fixed_track_long.py`
- Modify: `lane_training/action_training.py:30-38`
- Modify: `tools/lane_training/train_fixed_track_long.py`
- Modify: `tests/lane_training/test_train_fixed_track_long.py`

- [ ] **Step 1: Expose prediction arrays with a failing regression test**

Add `predict_action(model, rows, batch_size=64) -> tuple[np.ndarray, np.ndarray, np.ndarray]` to `action_training.py`. Refactor `evaluate_model` to call it without changing the existing result keys or values. Test a small fake model and assert prediction, target, and mask shapes are `(N, 2)` and the old `evaluate_model` result remains unchanged.

- [ ] **Step 2: Write failing evaluation-suite tests**

Build tiny manifests named `cross_train`, `cv_validation`, `lap002`, `official_train_reference`, and `official_eval`. Save a deterministic `CnnModel`; call `evaluate_checkpoint`; assert the report contains:

```python
assert set(report["sets"]) == {
    "cross_train", "cv_validation", "lap002",
    "official_train_reference", "official_eval",
}
assert report["sets"]["cross_train"]["sequence"]["active_count"] > 0
assert report["sets"]["cross_train"]["speed_mae"] is None
assert "vehicle_replay" in report["sets"]["cv_validation"]
assert "vehicle_replay" not in report["sets"]["lap002"]
```

- [ ] **Step 3: Implement checkpoint and directory evaluation**

The CLI accepts repeatable `--set name=manifest:selector`, `--checkpoint` or `--checkpoint-dir`, `--output`, `--device`, and `--batch-size`. Define selectors `train_manual` as `source=manual and split=train`, `validation_cv` as `source=cv and split in {val, validation}`, and ordinary split names as exact split matching. Independent manual manifests use selector `validation`.

For every set, record mask-aware MAE, whole-sequence metrics, and per-sequence metrics keyed by session ID or image parent. Run vehicle replay only where speed mask has valid rows. For a checkpoint directory, evaluate sorted `epoch_*` checkpoints and atomically write one JSON file per epoch plus `evaluation_index.json`. Reject missing rows, duplicate set names, non-finite predictions, and checkpoint/model-key mismatch.

- [ ] **Step 4: Integrate evaluation at every saved fifth epoch**

Add repeatable `--eval-set name=manifest:selector` and `--evaluation-output` to the long trainer. At every fifth-epoch checkpoint, call the shared `evaluate_checkpoint` function after saving and append the report path to that epoch's training history. A failed evaluation must stop the trainer while leaving the complete training checkpoint recoverable. Add a two-epoch smoke test with `checkpoint_interval=1` and assert evaluation reports exist for epochs 1 and 2.

- [ ] **Step 5: Run focused tests and commit**

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_evaluate_fixed_track_long.py tests/lane_training/test_action_sequence_metrics.py -q
git add lane_training/action_training.py tools/lane_training/evaluate_fixed_track_long.py tests/lane_training/test_evaluate_fixed_track_long.py tools/lane_training/train_fixed_track_long.py tests/lane_training/test_train_fixed_track_long.py
git commit -m "feat: evaluate fixed-track long checkpoints"
```

Expected: evaluation reports are finite, JSON-safe, and split manual speed from CV speed.

### Task 8: Three-role candidate shortlist

**Files:**
- Create: `tools/lane_training/select_fixed_track_candidates.py`
- Create: `tests/lane_training/test_select_fixed_track_candidates.py`

- [ ] **Step 1: Write failing role-selection tests**

```python
from tools.lane_training.select_fixed_track_candidates import shortlist_candidates


def test_shortlist_keeps_cross_cv_and_balanced_roles_without_final_selection():
    reports = [
        report(10, direction=0.95, false_steer=0.10, cross_mae=0.30, speed_mae=0.20, cv_kappa=0.40),
        report(100, direction=0.99, false_steer=0.08, cross_mae=0.20, speed_mae=0.15, cv_kappa=0.35),
        report(180, direction=0.97, false_steer=0.09, cross_mae=0.22, speed_mae=0.05, cv_kappa=0.18),
    ]
    result = shortlist_candidates(reports)
    assert result["roles"]["crossroad_best"]["epoch"] == 100
    assert result["roles"]["cv_preservation_best"]["epoch"] == 180
    assert result["selected"] is False
    assert result["vehicle_test_performed"] is False
    assert 1 <= len(result["candidates"]) <= 3
```

Define the local `report` test helper with the exact nested keys produced by Task 7.

- [ ] **Step 2: Implement deterministic role ranking**

Reject reports with missing or non-finite required metrics. Rank `crossroad_best` lexicographically by highest direction accuracy, highest active recall, lowest false-steer rate, lowest cross kappa MAE, then lowest epoch. Rank `cv_preservation_best` by lowest speed MAE, lowest CV kappa MAE, then lowest epoch. Rank `balanced` by the lowest worst normalized rank across cross direction, active recall, cross false steer, cross MAE, CV speed MAE, and CV kappa MAE; break ties by lower mean rank and lower epoch. Merge roles that resolve to the same epoch, preserve all role names, and emit a `candidates` list with one item per unique epoch containing its roles and checkpoint path.

The CLI reads `evaluation_index.json`, writes `candidate_shortlist.json`, and always records `selected=false`, `vehicle_test_performed=false`, and the priority order from the approved design.

- [ ] **Step 3: Verify and commit shortlist logic**

```powershell
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_select_fixed_track_candidates.py -q
git add tools/lane_training/select_fixed_track_candidates.py tests/lane_training/test_select_fixed_track_candidates.py
git commit -m "feat: shortlist fixed-track checkpoints"
```

Expected: role ranking and role merging tests pass.

### Task 9: Provenance-rich candidate export

**Files:**
- Modify: `tools/lane_training/export_action.py`
- Create: `tests/lane_training/test_export_action.py`

- [ ] **Step 1: Write a failing metadata test**

```python
from tools.lane_training.export_action import build_deployment_metadata


def test_export_metadata_includes_long_training_provenance():
    metadata = build_deployment_metadata(
        checkpoint_sha256="abc",
        difference=1e-7,
        tested_batch_size=8,
        model_filename="cnn_lane.json",
        training={"manifest_sha256": "def", "initial_checkpoint_sha256": "ghi"},
        selection={"epoch": 100, "roles": ["crossroad_best", "balanced"]},
    )
    assert metadata["output_semantics"] == ["speed_demand", "kappa_action"]
    assert metadata["training"]["manifest_sha256"] == "def"
    assert metadata["selection"]["epoch"] == 100
    assert metadata["selected"] is False
    assert metadata["vehicle_test_performed"] is False
```

- [ ] **Step 2: Refactor export into testable helpers**

Keep the current single-checkpoint CLI compatible, then add optional `--training-report`, `--selection-report`, and `--selection-role`. Move hashing, metadata creation, static conversion, consistency checking, and tar creation into functions. Add `export_shortlist(selection_report, checkpoint_dir, output_root, manifest, training_report, device)` to export every unique item in `selection_report["candidates"]` once, naming its directory `epoch_NNNN_<joined_roles>`. When provenance arguments are present, verify checkpoint epoch and SHA256 agree with both reports. Continue rejecting an existing output directory/archive and dynamic/static difference above `1e-5`.

- [ ] **Step 3: Add an end-to-end CPU export test**

Save two `CnnModel` checkpoint directories, create a two-row manifest, training report, and shortlist report whose three roles collapse to two unique candidates, run `export_shortlist`, and assert exactly two export directories and two `.tgz` files exist. Read both `deployment.json` files and assert checkpoint SHA256, roles, epoch, D4 SHA256, manifest SHA256, and dynamic/static difference match the inputs.

- [ ] **Step 4: Verify and commit export changes**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest tests/lane_training/test_export_action.py -q
git add tools/lane_training/export_action.py tests/lane_training/test_export_action.py
git commit -m "feat: export fixed-track candidate provenance"
```

Expected: export metadata and CPU round-trip tests pass.

### Task 10: Full local verification and remote GPU smoke

**Files:**
- Verify: all files from Tasks 1–9
- Remote project: `/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project`
- Remote manifest: `/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl`
- Remote D4: `/root/autodl-tmp/lane-cnn/project/artifacts/lane_hard_turn_d4_full/best.pdparams`

- [ ] **Step 1: Run the complete relevant local lane-training suite**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest `
  tests/lane_training/test_manifest.py `
  tests/lane_training/test_dataset.py `
  tests/lane_training/test_action_loss.py `
  tests/lane_training/test_grouped_sampling.py `
  tests/lane_training/test_staged_action.py `
  tests/lane_training/test_train_fixed_track_long.py `
  tests/lane_training/test_action_sequence_metrics.py `
  tests/lane_training/test_evaluate_fixed_track_long.py `
  tests/lane_training/test_select_fixed_track_candidates.py `
  tests/lane_training/test_export_action.py -q
git diff --check
```

Expected: all manifest/action/long-training tests pass; only the known Paddle GPU-to-CPU fallback warning is allowed; `git diff --check` is silent. Unrelated untracked legacy experiment tests are outside this run and must not be edited or deleted.

- [ ] **Step 2: Audit the remote inputs before syncing code**

```powershell
ssh -i 'C:\Users\fjcy\.ssh\autodl_bjb2_28297' -p 43447 root@connect.bjb1.seetacloud.com "sha256sum /root/autodl-tmp/lane-cnn/project/artifacts/lane_hard_turn_d4_full/best.pdparams /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

Expected: D4 SHA256 is `3398fc6b2d2be73d1ec7eed6f54f51f0be6e2b6c5814f255485d23442167cb49`; the GPU is RTX 4080 SUPER with 32 GB. Record the manifest SHA256 in the run report rather than accepting an unrecorded value.

- [ ] **Step 3: Sync only source and tests to the isolated remote project**

Use these explicit `scp` commands, preserving the existing manifest and datasets:

```powershell
$repo='C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning'
$sshTarget='root@connect.bjb1.seetacloud.com'
scp -i 'C:\Users\fjcy\.ssh\autodl_bjb2_28297' -P 43447 `
  "$repo\lane_training\dataset.py" `
  "$repo\lane_training\action_loss.py" `
  "$repo\lane_training\action_training.py" `
  "$repo\lane_training\grouped_sampling.py" `
  "$repo\lane_training\staged_action.py" `
  "$repo\lane_training\action_sequence_metrics.py" `
  "${sshTarget}:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project/lane_training/"
scp -i 'C:\Users\fjcy\.ssh\autodl_bjb2_28297' -P 43447 `
  "$repo\tools\lane_training\train_fixed_track_long.py" `
  "$repo\tools\lane_training\evaluate_fixed_track_long.py" `
  "$repo\tools\lane_training\select_fixed_track_candidates.py" `
  "$repo\tools\lane_training\export_action.py" `
  "${sshTarget}:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project/tools/lane_training/"
scp -i 'C:\Users\fjcy\.ssh\autodl_bjb2_28297' -P 43447 `
  "$repo\tests\lane_training\test_dataset.py" `
  "$repo\tests\lane_training\test_grouped_sampling.py" `
  "$repo\tests\lane_training\test_staged_action.py" `
  "$repo\tests\lane_training\test_train_fixed_track_long.py" `
  "$repo\tests\lane_training\test_action_sequence_metrics.py" `
  "$repo\tests\lane_training\test_evaluate_fixed_track_long.py" `
  "$repo\tests\lane_training\test_select_fixed_track_candidates.py" `
  "$repo\tests\lane_training\test_export_action.py" `
  "${sshTarget}:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project/tests/lane_training/"
```

Do not copy local `.venv-lane`, `artifacts`, `.git`, or unrelated untracked scripts. On the remote host run:

```bash
cd /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project
CUDA_VISIBLE_DEVICES=-1 /root/autodl-tmp/envs/lane-d5/bin/python -m pytest \
  tests/lane_training/test_grouped_sampling.py \
  tests/lane_training/test_staged_action.py \
  tests/lane_training/test_train_fixed_track_long.py \
  tests/lane_training/test_action_sequence_metrics.py \
  tests/lane_training/test_evaluate_fixed_track_long.py \
  tests/lane_training/test_select_fixed_track_candidates.py \
  tests/lane_training/test_export_action.py -q
```

Expected: every focused remote test passes.

- [ ] **Step 4: Run a clean two-epoch GPU smoke**

```bash
cd /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project
FLAGS_enable_pir_api=0 /root/autodl-tmp/envs/lane-d5/bin/python \
  tools/lane_training/train_fixed_track_long.py \
  --manifest /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl \
  --initial-checkpoint /root/autodl-tmp/lane-cnn/project/artifacts/lane_hard_turn_d4_full/best.pdparams \
  --output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1_smoke \
  --device gpu --max-epochs 2
```

Expected: finite total/speed/kappa losses; data counts `2822/74/572`; both epochs report stage `head`; actual draw ratios are close to `70/15/15`; epoch 2 checkpoint reloads successfully in a separate process.

- [ ] **Step 5: Commit verification-facing adjustments only if smoke exposed them**

For each smoke defect, first add a failing local regression test, make the smallest fix, rerun the focused and full suites, then commit only the exact test and source files with a message naming the defect. Do not edit the remote copy without applying the same tracked local change.

### Task 11: Formal 200-epoch run, evaluation, and documentation

**Files:**
- Remote output: `/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1`
- Modify after results: `docs/lane-cnn-training-handoff.md`
- Modify after results: `docs/cv-lane-development-status-2026-08-15.md`
- Download artifacts to: `artifacts/cross_manual_long_20260817_remote/`

- [ ] **Step 1: Start the formal run in a persistent remote log**

```bash
cd /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/project
nohup env FLAGS_enable_pir_api=0 /root/autodl-tmp/envs/lane-d5/bin/python \
  tools/lane_training/train_fixed_track_long.py \
  --manifest /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl \
  --initial-checkpoint /root/autodl-tmp/lane-cnn/project/artifacts/lane_hard_turn_d4_full/best.pdparams \
  --output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1 \
  --device gpu --epochs 200 \
  --eval-set cross_train=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl:train_manual \
  --eval-set cv_validation=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl:validation_cv \
  --eval-set lap002=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/lap002_validation.jsonl:validation \
  --eval-set official_train_reference=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/official_train_reference_validation.jsonl:validation \
  --eval-set official_eval=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/official_eval_validation.jsonl:validation \
  --evaluation-output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/evaluations \
  > /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1.log 2>&1 &
```

Expected: one training process, a growing log, GPU utilization, and checkpoints at epochs divisible by five. Do not launch a second process when reconnecting.

- [ ] **Step 2: Monitor stage boundaries and resumability**

At epochs 5, 20, 21, 160, 161, and 200, read `training_report.json`, verify finite losses, exact source counts, nonzero valid masks, correct stage, augmentation profile/hit count, correct learning rates, checkpoint hashes, and available disk:

```bash
tail -80 /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1.log
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
df -h /root/autodl-tmp
/root/autodl-tmp/envs/lane-d5/bin/python -m json.tool \
  /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/training_report.json
```

If the process exits, resolve only complete checkpoint directories and resume without `--initial-checkpoint`:

```bash
resume_checkpoint=$(find /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/checkpoints \
  -maxdepth 1 -type d -name 'epoch_[0-9][0-9][0-9][0-9]' | sort | tail -1)
test -n "$resume_checkpoint"
FLAGS_enable_pir_api=0 /root/autodl-tmp/envs/lane-d5/bin/python \
  tools/lane_training/train_fixed_track_long.py \
  --manifest /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl \
  --resume "$resume_checkpoint" \
  --output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1 \
  --device gpu --epochs 200 \
  --eval-set cross_train=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl:train_manual \
  --eval-set cv_validation=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl:validation_cv \
  --eval-set lap002=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/lap002_validation.jsonl:validation \
  --eval-set official_train_reference=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/official_train_reference_validation.jsonl:validation \
  --eval-set official_eval=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/official_eval_validation.jsonl:validation \
  --evaluation-output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/evaluations
```

Never resume from `.incomplete`.

- [ ] **Step 3: Evaluate every saved checkpoint**

```bash
FLAGS_enable_pir_api=0 /root/autodl-tmp/envs/lane-d5/bin/python \
  tools/lane_training/evaluate_fixed_track_long.py \
  --checkpoint-dir /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/checkpoints \
  --set cross_train=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl:train_manual \
  --set cv_validation=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl:validation_cv \
  --set lap002=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/lap002_validation.jsonl:validation \
  --set official_train_reference=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/official_train_reference_validation.jsonl:validation \
  --set official_eval=/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/official_eval_validation.jsonl:validation \
  --output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/evaluations \
  --device gpu
```

Expected: one finite JSON report for each five-epoch checkpoint, with speed metrics only on `cv_validation` and vehicle replay only where speed labels are valid.

- [ ] **Step 4: Generate and manually review the three-role shortlist**

```bash
/root/autodl-tmp/envs/lane-d5/bin/python \
  tools/lane_training/select_fixed_track_candidates.py \
  --evaluation-index /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/evaluations/evaluation_index.json \
  --output /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/candidate_shortlist.json
```

Expected: one to three unique epochs covering `crossroad_best`, `cv_preservation_best`, and `balanced`; `selected=false`; `vehicle_test_performed=false`. Review the full crossroad sequences and vehicle replay before accepting the shortlist.

- [ ] **Step 5: Export each unique shortlist candidate**

```bash
FLAGS_enable_pir_api=0 /root/autodl-tmp/envs/lane-d5/bin/python \
  tools/lane_training/export_action.py \
  --selection-report /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/candidate_shortlist.json \
  --checkpoint-dir /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/checkpoints \
  --manifest /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/manifests/joint_train_v2_cv2822_manual646.jsonl \
  --training-report /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/training_report.json \
  --output-root /root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/exports \
  --device gpu
```

Expected: one archive per unique `candidates` entry, not one per role. Verify every dynamic/static difference is at most `1e-5`, then record archive SHA256.

- [ ] **Step 6: Download reports and candidate archives without deploying**

```powershell
$repo='C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning'
$localArtifacts="$repo\artifacts\cross_manual_long_20260817_remote"
New-Item -ItemType Directory -Path $localArtifacts
scp -i 'C:\Users\fjcy\.ssh\autodl_bjb2_28297' -P 43447 `
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/training_report.json `
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/candidate_shortlist.json `
  $localArtifacts
scp -i 'C:\Users\fjcy\.ssh\autodl_bjb2_28297' -P 43447 -r `
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/evaluations `
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/lane-cnn/run-20260817-cross-manual/outputs/long_v1/exports `
  $localArtifacts
Get-ChildItem -LiteralPath $localArtifacts -File -Recurse | Get-FileHash -Algorithm SHA256 | Sort-Object Path
```

Before copying, require that `$localArtifacts` does not exist; do not overwrite an earlier download. Generate the corresponding remote SHA256 listing with `find ... -type f -exec sha256sum {} + | sort` and compare every relative path/hash pair. Do not change `config_car.yml`, do not copy a model to Orin, and do not delete remote checkpoints.

- [ ] **Step 7: Update progress documents with measured results**

Add the exact run configuration, D4/manifest hashes, stage history, per-role checkpoint metrics, dynamic/static differences, archive hashes, and remaining low-speed vehicle-test requirement to both progress documents. Explicitly state that independent manual sets were references rather than optimization gates and that no candidate is selected or deployed.

- [ ] **Step 8: Run final verification and commit the measured report**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv-lane\Scripts\python.exe' -m pytest `
  tests/lane_training/test_manifest.py `
  tests/lane_training/test_dataset.py `
  tests/lane_training/test_action_loss.py `
  tests/lane_training/test_grouped_sampling.py `
  tests/lane_training/test_staged_action.py `
  tests/lane_training/test_train_fixed_track_long.py `
  tests/lane_training/test_action_sequence_metrics.py `
  tests/lane_training/test_evaluate_fixed_track_long.py `
  tests/lane_training/test_select_fixed_track_candidates.py `
  tests/lane_training/test_export_action.py -q
git diff --check
git add docs/lane-cnn-training-handoff.md docs/cv-lane-development-status-2026-08-15.md
git commit -m "docs: record fixed-track long training results"
```

Expected: the full suite passes, whitespace validation is silent, and the documentation commit contains only the two progress documents. Model artifacts remain local untracked files unless the user separately requests artifact versioning.
