"""Stateless cross-road feature extraction for offline inspection.

The first migration stage only reports a candidate score.  It does not make a
route decision or issue any vehicle command.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossFeatures:
    score: float
    status: str
    width_ratio: float
    expanded_rows: int


def detect_cross_candidate(widths: np.ndarray) -> CrossFeatures:
    """Estimate whether lane widths contain a cross-road expansion.

    ``widths`` contains one positive lane width per valid image row, ordered
    from far to near.  Cross roads tend to create several adjacent rows that
    are much wider than the robust median.  This is only a feature extractor;
    temporal confirmation will be added when collection-time control is built.
    """

    widths = np.asarray(widths, dtype=np.float32)
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size < 12:
        return CrossFeatures(0.0, "unknown", 1.0, 0)

    median_width = float(np.median(widths))
    if median_width <= 0:
        return CrossFeatures(0.0, "unknown", 1.0, 0)

    width_ratio = float(np.percentile(widths, 90) / median_width)
    expanded_rows = int(np.count_nonzero(widths > median_width * 1.35))
    row_fraction = expanded_rows / float(widths.size)

    ratio_score = np.clip((width_ratio - 1.20) / 0.80, 0.0, 1.0)
    rows_score = np.clip(row_fraction / 0.25, 0.0, 1.0)
    score = float(0.65 * ratio_score + 0.35 * rows_score)
    status = "candidate" if score >= 0.55 and expanded_rows >= 3 else "normal"
    return CrossFeatures(score, status, width_ratio, expanded_rows)
