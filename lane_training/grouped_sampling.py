from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


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


def build_epoch_batches(
    groups: dict[str, list[int]], *, batch_size: int = 64,
    seed: int = 20260817, epoch: int,
) -> tuple[list[list[int]], dict]:
    if batch_size < 3:
        raise ValueError("batch_size must leave room for all three groups")
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if set(groups) != set(GROUPS) or any(not groups[name] for name in GROUPS):
        raise ValueError("all fixed-track training groups must be non-empty")

    offsets = {"cv": 11, "manual_active": 23, "manual_context": 37}
    rngs = {
        name: np.random.default_rng(seed + epoch * 1009 + offsets[name])
        for name in GROUPS
    }
    pools = {
        name: rngs[name].permutation(groups[name]).astype(int).tolist()
        for name in GROUPS
    }
    cursors = {name: 0 for name in GROUPS}

    def draw(name: str, count: int) -> list[int]:
        result: list[int] = []
        while len(result) < count:
            remaining = len(pools[name]) - cursors[name]
            take = min(count - len(result), remaining)
            result.extend(pools[name][cursors[name]:cursors[name] + take])
            cursors[name] += take
            if cursors[name] == len(pools[name]):
                pools[name] = (
                    rngs[name].permutation(groups[name]).astype(int).tolist()
                )
                cursors[name] = 0
        return result

    if batch_size == 64:
        cv_count, first_manual, second_manual = 45, 10, 9
    else:
        first_manual = max(1, round(batch_size * 0.15))
        second_manual = max(1, round(batch_size * 0.15))
        cv_count = batch_size - first_manual - second_manual
        if cv_count < 1:
            raise ValueError("batch_size produces an empty cv draw")
    batch_count = math.ceil(len(groups["cv"]) / cv_count)
    draw_counts = {name: 0 for name in GROUPS}
    batches: list[list[int]] = []
    for batch_index in range(batch_count):
        active_count, context_count = (
            (first_manual, second_manual)
            if batch_index % 2 == 0
            else (second_manual, first_manual)
        )
        counts = {
            "cv": cv_count,
            "manual_active": active_count,
            "manual_context": context_count,
        }
        batch: list[int] = []
        for name in GROUPS:
            batch.extend(draw(name, counts[name]))
            draw_counts[name] += counts[name]
        order = np.random.default_rng(
            seed + epoch * 1009 + 100_003 + batch_index
        ).permutation(len(batch))
        batches.append([batch[int(index)] for index in order])

    total_draws = sum(draw_counts.values())
    report = {
        "source_row_counts": {name: len(groups[name]) for name in GROUPS},
        "draw_counts": draw_counts,
        "draw_ratios": {
            name: draw_counts[name] / total_draws for name in GROUPS
        },
        "seed": int(seed),
        "epoch": int(epoch),
        "batch_count": batch_count,
    }
    return batches, report
