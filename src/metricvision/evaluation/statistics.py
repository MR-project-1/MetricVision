"""Paired statistics for comparing two methods on the same sequence.

KITTI: the devkit segments (start frame, length) are identical for every method, so per-segment errors form paired
samples. A Wilcoxon signed-rank test asks whether one method's errors are systematically lower; a bootstrap over
segments gives a confidence interval of the mean difference. Segments overlap (start frames 10 apart), so samples are
not independent; the block bootstrap resamples whole start frames to respect that, and the p-value should be read as
indicative.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import asdict, dataclass  # Result container.

import numpy as np  # Maths.
from scipy import stats  # Wilcoxon test.


@dataclass(frozen=True)
class PairedComparison:
    """Comparison of method B against method A on paired samples (negative difference = B better).

    Attributes:
        mean_a: Mean error of A.
        mean_b: Mean error of B.
        mean_diff: Mean of (B - A).
        ci_low: Lower bound of the 95 % bootstrap CI of mean_diff.
        ci_high: Upper bound.
        p_value: Two-sided Wilcoxon signed-rank p-value.
        win_rate: Fraction of samples where B < A.
        num_samples: Paired samples.
    """

    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    win_rate: float
    num_samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a plain dictionary.

        Returns:
            Fields.
        """
        return asdict(self)


def paired_comparison(
    errors_a: np.ndarray, errors_b: np.ndarray, blocks: np.ndarray | None = None, resamples: int = 2000, seed: int = 0
) -> PairedComparison:
    """Wilcoxon signed-rank test and (block) bootstrap CI of the mean paired difference.

    Args:
        errors_a: (N,) errors of the reference method.
        errors_b: (N,) errors of the compared method (same samples).
        blocks: (N,) block label per sample (e.g. segment start frame) for the block bootstrap; None = i.i.d.
        resamples: Bootstrap resamples.
        seed: Random seed.

    Returns:
        The comparison.
    """
    a, b = np.asarray(errors_a, dtype=np.float64), np.asarray(errors_b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = b - a
    if diff.size < 5:
        return PairedComparison(
            float(np.mean(a)) if a.size else float("nan"),
            float(np.mean(b)) if b.size else float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            int(diff.size),
        )
    p_value = float(stats.wilcoxon(diff, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
    rng = np.random.default_rng(seed)
    labels = np.asarray(blocks)[ok] if blocks is not None else np.arange(diff.size)
    unique = np.unique(labels)
    groups = [np.flatnonzero(labels == u) for u in unique]
    means = np.empty(resamples)
    for i in range(resamples):  # Resample whole blocks.
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        means[i] = diff[idx].mean()
    return PairedComparison(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(diff.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        p_value=p_value,
        win_rate=float(np.mean(diff < 0)),
        num_samples=int(diff.size),
    )
