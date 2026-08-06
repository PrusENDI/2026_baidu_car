# Crop Harvesting Random Slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `crop_harvesting` find the first ball while slowly lane-tracing, then scan up to eight fixed-width slots, skip empty slots, retry a slot when its ball remains after placement, and stop immediately after a configurable number of successfully cleared balls.

**Architecture:** Keep the existing arm motion sequence and task pipeline intact, but turn the harvest loop into a slot state machine. A slot is counted only after post-placement visual verification shows that the ball is gone; the vehicle advances by exactly one slot spacing after every unresolved slot, including empty slots. The first-ball search is a separate low-speed lane-tracing phase and is not counted as one of the fixed slot moves.

**Tech Stack:** Python, existing `MyCar` lane/vision APIs, pytest-style unit tests with fake car/arm objects.

---

### Task 1: Define configurable harvest parameters and preserve the public call path

**Files:**
- Modify: `car_task_function.py:884-966`
- Modify: `car_start_2026.py:64-66` only if the configured count must be passed explicitly

- [ ] **Step 1: Add explicit parameters without breaking existing callers**

Change the signature to accept the independently adjustable values while retaining defaults for the current main program:

```python
def crop_harvesting(
    debug=False,
    target_ball_count=8,
    max_slots=8,
    slot_step=0.04,
    max_grasp_attempts=2,
    search_speed=0.08,
    search_timeout=15.0,
):
```

Validate that `target_ball_count` and `max_slots` are positive integers, `target_ball_count <= max_slots`, `slot_step > 0`, and both speed/timeout are positive. Keep `crop_harvesting()` valid for the existing call in `car_start_2026.py`; a later caller can pass a different count.

- [ ] **Step 2: Keep all tunables in one visible block**

If the project prefers constants over a long signature, introduce a `CROP_HARVESTING_CONFIG` dictionary next to the other task constants and have function arguments override it. Do not change `sort_and_store`’s existing interface in this task.

- [ ] **Step 3: Run the existing tests before behavior changes**

Run:

```powershell
pytest -q
```

Expected: existing tests pass; this establishes the baseline.

### Task 2: Add vision helpers that only accept ball classes

**Files:**
- Modify: `car_task_function.py` inside or immediately above `crop_harvesting`
- Test: `tests/test_crop_harvesting.py`

- [ ] **Step 1: Write tests for ball filtering and stable presence**

Use a fake `my_car.get_detection_results()` that returns the project’s detection shape `[cls_id, det_id, label, score, x_c, y_c, w, h]`. Verify that `storage` and unrelated labels are ignored and only `ball_blue`/`ball_yellow` are accepted.

```python
def test_only_ball_labels_are_candidates():
    detections = [
        [1, 1, "storage", 0.9, 0.0, 0.0, 0.2, 0.2],
        [2, 2, "ball_blue", 0.9, 0.0, 0.0, 0.2, 0.2],
    ]
    assert _ball_detections(detections) == [detections[1]]
```

- [ ] **Step 2: Implement `_ball_detections`**

Filter `my_car.get_detection_results()` by `item[2] in ("ball_blue", "ball_yellow")`; never let the nearest arbitrary detection become a harvest target. Return the nearest/first ball according to the existing detection ordering.

- [ ] **Step 3: Implement stable presence checks**

Add a helper that samples the current slot for a small, configurable number of frames (for example three), returning a ball detection only when the same slot has a ball in the required consecutive samples. Return `None` for an empty/unreliable slot. This helper is used both before grabbing and after placing so a single stale frame cannot increment the count.

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
pytest -q tests/test_crop_harvesting.py -k "ball or stable"
```

Expected: PASS.

### Task 3: Implement the low-speed first-ball search phase

**Files:**
- Modify: `car_task_function.py:894-898`
- Test: `tests/test_crop_harvesting.py`

- [ ] **Step 1: Write the first-ball search test**

Stub the lane controller and detection stream so no ball is visible for several iterations, then a ball appears. Assert that the vehicle uses low-speed lane tracing, stops at the first stable ball, and does not apply the fixed `slot_step` before processing that ball.

- [ ] **Step 2: Implement the search callback using the existing lane API**

After the base `task_distance` move, call the existing lane-tracing primitive with an end callback that polls `_ball_detections()`. Stop only after a stable ball is observed or `search_timeout` expires. Do not use `lane_dis_offset(slot_step)` for this phase; the first ball can be between nominal slot centers.

- [ ] **Step 3: Handle no-first-ball safely**

If the low-speed search times out, perform normal arm/storage cleanup and return `None` (or raise the project’s chosen task error) without issuing a grasp command. Record that zero balls were collected.

- [ ] **Step 4: Run the focused search tests**

Run:

```powershell
pytest -q tests/test_crop_harvesting.py -k first_ball
```

Expected: PASS.

### Task 4: Replace the fixed eight-grasp loop with the slot state machine

**Files:**
- Modify: `car_task_function.py:899-958`
- Test: `tests/test_crop_harvesting.py`

- [ ] **Step 1: Write tests for empty gaps and configurable stopping**

Cover these exact sequences:

```text
ball, empty, ball, empty, empty, ball, ...
```

Assert that empty slots issue no `grasp(True)`, every checked slot advances exactly `slot_step`, and the loop stops as soon as `picked_count == target_ball_count` even when fewer than eight slots were needed.

- [ ] **Step 2: Process the first found slot without an extra move**

Use the stable detection returned by the first-ball search as the current slot result. Set `first_label` only when the first real ball is found, not based on `i == 0`.

- [ ] **Step 3: Scan subsequent slots at fixed spacing**

For each later slot, call `my_car.lane_dis_offset(speed=search_speed, dis_hold=slot_step)` exactly once before checking that slot. An empty slot is still a checked slot and still consumes one spacing interval.

- [ ] **Step 4: Skip empty slots without arm motion**

When the stable ball helper returns `None`, record the slot as empty, do not call `adjust_arm_position`, `grasp`, or the placement sequence, then continue to the next slot.

- [ ] **Step 5: Add a bounded per-slot retry loop**

For a detected ball, run the existing pick/place arm sequence. Restore `arm="LEFT"` and hand `"DOWN"`, wait for settling, and re-check the same physical slot. Count the ball only when the post-placement check confirms the ball has disappeared. If it remains, retry the complete pick/place sequence up to `max_grasp_attempts` (default two total attempts). After the limit, log a failed slot and advance rather than looping forever.

- [ ] **Step 6: Stop on the configured successful count**

Break the slot loop immediately after a verified successful clear reaches `target_ball_count`. Do not perform the next `slot_step` move after this break; leave the vehicle at the final processed slot.

- [ ] **Step 7: Report incomplete collection after all slots**

If all `max_slots` slots have been checked while `picked_count < target_ball_count`, print/log the collected count and missing count. Keep the existing cleanup and return `first_label` so the caller can decide how to handle the incomplete task.

- [ ] **Step 8: Run the state-machine tests**

Run:

```powershell
pytest -q tests/test_crop_harvesting.py -k "empty or retry or target_count or slot"
```

Expected: PASS.

### Task 5: Preserve cleanup and integrate the configurable count

**Files:**
- Modify: `car_task_function.py:960-966`
- Modify: `car_start_2026.py:64-66` if the operational count is configured at the top-level
- Test: `tests/test_crop_harvesting.py`

- [ ] **Step 1: Write cleanup tests**

Assert that both early target-count completion and timeout/no-ball completion still lower the storage rack, raise the arm to the safe `y` position, and perform the debug return only when `debug=True`.

- [ ] **Step 2: Put cleanup in one guaranteed path**

Use a `try/finally` around the search/scan operations, or an equivalent single cleanup block reached by every `break`/failure path. Do not return directly from inside the slot loop before cleanup.

- [ ] **Step 3: Pass the configured operational count**

If the runtime configuration is maintained in `car_start_2026.py`, define one clearly named variable such as `CROP_HARVEST_TARGET_COUNT` and call:

```python
first_label = crop_harvesting(
    target_ball_count=CROP_HARVEST_TARGET_COUNT
)
```

Otherwise leave the backward-compatible default and document the function argument in `README.md`.

- [ ] **Step 4: Run the complete test suite**

Run:

```powershell
pytest -q
python -m compileall car_task_function.py car_start_2026.py
```

Expected: all tests pass and compilation exits with code 0.

### Task 6: Perform a hardware-safe review of the motion order

**Files:**
- Review: `car_task_function.py:884-966`
- Review: `car_wrap_2026.py:1741-2010`

- [ ] **Step 1: Verify target selection**

Confirm every harvest detection is filtered to `ball_blue`/`ball_yellow`; no `storage` or unrelated object can trigger an arm movement.

- [ ] **Step 2: Verify slot accounting**

Confirm the first-ball search is outside the fixed-slot count, each later slot advances exactly once, and no advance occurs after the target count is reached.

- [ ] **Step 3: Verify retry safety**

Confirm a persistent ball cannot cause an unbounded loop and a failed slot does not increment `picked_count`.

- [ ] **Step 4: Verify downstream behavior**

Confirm `crop_harvesting` still returns `first_label` and the next main-program statement remains `sort_and_store(first_label)`.
