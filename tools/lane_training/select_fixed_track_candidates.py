from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PRIORITY_ORDER = [
    "crossroad correction",
    "ordinary and sharp-turn preservation",
    "vehicle speed/recovery behavior",
    "cv average error",
    "independent manual references",
]


def _metrics(report: dict) -> dict[str, float]:
    try:
        cross = report["sets"]["cross_train"]
        sequence = cross["sequence"]
        cv = report["sets"]["cv_validation"]
        values = {
            "direction": float(sequence["direction_accuracy"]),
            "recall": float(sequence["active_recall"]),
            "false_steer": float(sequence["false_steer_rate"]),
            "cross_mae": float(cross["kappa_mae"]),
            "speed_mae": float(cv["speed_mae"]),
            "cv_kappa": float(cv["kappa_mae"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation report is missing required metrics") from error
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("evaluation report contains non-finite required metrics")
    return values


def _normalized_ranks(
    reports: list[dict], metric_rows: list[dict[str, float]], key: str, *, high: bool,
) -> dict[int, float]:
    ordered = sorted(
        range(len(reports)),
        key=lambda index: (
            -metric_rows[index][key] if high else metric_rows[index][key],
            int(reports[index]["epoch"]),
        ),
    )
    denominator = max(len(ordered) - 1, 1)
    return {index: rank / denominator for rank, index in enumerate(ordered)}


def shortlist_candidates(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("no evaluation reports")
    metrics = [_metrics(report) for report in reports]
    cross_index = min(
        range(len(reports)),
        key=lambda index: (
            -metrics[index]["direction"],
            -metrics[index]["recall"],
            metrics[index]["false_steer"],
            metrics[index]["cross_mae"],
            int(reports[index]["epoch"]),
        ),
    )
    cv_index = min(
        range(len(reports)),
        key=lambda index: (
            metrics[index]["speed_mae"],
            metrics[index]["cv_kappa"],
            int(reports[index]["epoch"]),
        ),
    )
    rank_specs = [
        ("direction", True),
        ("recall", True),
        ("false_steer", False),
        ("cross_mae", False),
        ("speed_mae", False),
        ("cv_kappa", False),
    ]
    rank_maps = [
        _normalized_ranks(reports, metrics, key, high=high)
        for key, high in rank_specs
    ]
    balanced_index = min(
        range(len(reports)),
        key=lambda index: (
            max(rank[index] for rank in rank_maps),
            sum(rank[index] for rank in rank_maps) / len(rank_maps),
            int(reports[index]["epoch"]),
        ),
    )
    role_indices = {
        "crossroad_best": cross_index,
        "cv_preservation_best": cv_index,
        "balanced": balanced_index,
    }
    roles = {}
    candidates_by_epoch = {}
    for role, index in role_indices.items():
        report = reports[index]
        item = {
            "epoch": int(report["epoch"]),
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report.get("checkpoint_sha256"),
        }
        roles[role] = item
        candidate = candidates_by_epoch.setdefault(item["epoch"], {
            **item,
            "roles": [],
        })
        candidate["roles"].append(role)
    return {
        "schema_version": 1,
        "roles": roles,
        "candidates": [candidates_by_epoch[key] for key in sorted(candidates_by_epoch)],
        "priority_order": PRIORITY_ORDER,
        "selected": False,
        "vehicle_test_performed": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    index = json.loads(args.evaluation_index.read_text(encoding="utf-8"))
    result = shortlist_candidates(index["reports"])
    if args.output.exists():
        raise FileExistsError(f"selection output already exists: {args.output}")
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
