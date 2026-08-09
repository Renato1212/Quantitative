"""Purged, embargoed cross-validation (C4).

Forward-looking labels overlap. An observation anchored at ``t0`` is not resolved until
its barrier is touched or its clock runs out, so a training sample whose label window
reaches into the test window has seen the test period. Ordinary k-fold — even
``TimeSeriesSplit`` — leaves that overlap in place and inflates every out-of-sample
number computed on top of it.

Two corrections, in order:

*Purge.* Drop training samples whose ``[t0, window_end]`` intersects the test block's own
span. This is the correction that matters most and the one people skip.

*Embargo.* Drop a further stretch of training samples that begin just after the test
block ends. Purging handles overlap in the labels; the embargo handles serial correlation
in the features, which does not stop at the label boundary. The default is the label
horizon, as ``CLAUDE.md`` C4 requires.

Removing either because it "loses too much data" is refused — the lost data is precisely
the data that was contaminated, and keeping it does not recover information, it only
hides the contamination.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    test_span: tuple[datetime, datetime]
    purged: int
    embargoed: int


class PurgedKFold:
    """Contiguous, chronological folds with label-overlap purging and an embargo.

    ``t0s`` and ``window_ends`` are the anchor and label-resolution timestamps of each
    sample, in the sample's own order. Samples are sorted by anchor internally, and the
    indices yielded refer to the original order.
    """

    def __init__(
        self,
        t0s: np.ndarray,
        window_ends: np.ndarray,
        *,
        n_splits: int = 5,
        embargo: timedelta,
    ):
        t0s = np.asarray(t0s, dtype="datetime64[ns]")
        window_ends = np.asarray(window_ends, dtype="datetime64[ns]")
        if t0s.shape != window_ends.shape:
            raise ValueError("t0s and window_ends must be the same length")
        if len(t0s) < n_splits:
            raise ValueError(f"{len(t0s)} samples cannot make {n_splits} folds")
        if (window_ends < t0s).any():
            raise ValueError("a label cannot resolve before its anchor")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self._order = np.argsort(t0s, kind="stable")
        self.t0s = t0s
        self.window_ends = window_ends
        self.n_splits = n_splits
        self.embargo = np.timedelta64(int(embargo.total_seconds()), "s")

    def __len__(self) -> int:
        return self.n_splits

    def split(self) -> Iterator[Fold]:
        """Yield folds. Test blocks are contiguous in time and never shuffled (C3)."""
        blocks = np.array_split(self._order, self.n_splits)
        for test_idx in blocks:
            if test_idx.size == 0:
                continue
            start = self.t0s[test_idx].min()
            end = max(self.window_ends[test_idx].max(), self.t0s[test_idx].max())

            # Purge: any sample whose own window touches the test block's span.
            overlaps = (self.t0s <= end) & (self.window_ends >= start)
            # Embargo: samples anchored in the stretch just after the test block.
            embargoed = (self.t0s > end) & (self.t0s <= end + self.embargo)

            keep = ~overlaps & ~embargoed
            keep[test_idx] = False
            train_idx = np.flatnonzero(keep)

            in_test = np.zeros(len(self.t0s), bool)
            in_test[test_idx] = True
            yield Fold(
                train=train_idx,
                test=np.sort(test_idx),
                test_span=(start.astype("datetime64[us]").item(), end.astype("datetime64[us]").item()),
                purged=int((overlaps & ~in_test).sum()),
                embargoed=int(embargoed.sum()),
            )

    def report(self) -> str:
        lines = [f"PurgedKFold: {self.n_splits} folds, embargo {self.embargo}"]
        for i, fold in enumerate(self.split(), 1):
            lines.append(
                f"  fold {i}: train {fold.train.size:>5}  test {fold.test.size:>5}  "
                f"purged {fold.purged:>5}  embargoed {fold.embargoed:>4}"
            )
        return "\n".join(lines)


def train_holdout_masks(t0s: np.ndarray, boundary: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split at a boundary. The holdout is opened once, at the end (C3)."""
    stamps = np.asarray(t0s, dtype="datetime64[ns]")
    cut = np.datetime64(boundary.replace(tzinfo=None), "ns")
    return stamps < cut, stamps >= cut
