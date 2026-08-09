"""Purged, embargoed cross-validation (C4).

The property that matters: no training sample in any fold may have a label window that
reaches into that fold's test span. If that ever holds only approximately, every
out-of-sample number in Phase 5 is inflated and nothing downstream is trustworthy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.validation.purged import PurgedKFold, train_holdout_masks

HORIZON = timedelta(minutes=120)


def _samples(n: int = 200, spacing: timedelta = timedelta(minutes=20)):
    start = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    t0s = np.array([np.datetime64((start + i * spacing).replace(tzinfo=None), "ns") for i in range(n)])
    ends = t0s + np.timedelta64(int(HORIZON.total_seconds()), "s")
    return t0s, ends


@pytest.fixture
def splitter():
    t0s, ends = _samples()
    return PurgedKFold(t0s, ends, n_splits=5, embargo=HORIZON)


def test_no_training_label_window_touches_the_test_span(splitter):
    """The purge. Overlapping labels are the reason ordinary k-fold cannot be used here."""
    for fold in splitter.split():
        start, end = (np.datetime64(x, "ns") for x in fold.test_span)
        train_t0 = splitter.t0s[fold.train]
        train_end = splitter.window_ends[fold.train]
        overlap = (train_t0 <= end) & (train_end >= start)
        assert not overlap.any(), f"{overlap.sum()} training labels overlap the test span"


def test_embargo_removes_samples_starting_just_after_the_test_block(splitter):
    """The embargo. Purging handles the labels; serial correlation outlasts them."""
    for fold in splitter.split():
        _, end = (np.datetime64(x, "ns") for x in fold.test_span)
        train_t0 = splitter.t0s[fold.train]
        inside = (train_t0 > end) & (train_t0 <= end + splitter.embargo)
        assert not inside.any()


def test_train_and_test_never_intersect(splitter):
    for fold in splitter.split():
        assert not set(fold.train.tolist()) & set(fold.test.tolist())


def test_test_blocks_partition_the_sample(splitter):
    """Contiguous and chronological (C3): every sample is tested exactly once."""
    seen = np.concatenate([fold.test for fold in splitter.split()])
    assert sorted(seen.tolist()) == list(range(splitter.t0s.size))


def test_test_blocks_are_contiguous_in_time(splitter):
    spans = [fold.test_span for fold in splitter.split()]
    starts = [s for s, _ in spans]
    assert starts == sorted(starts)


def test_purging_actually_removes_something(splitter):
    """A purge that never fires is a purge that is silently misconfigured."""
    assert sum(fold.purged for fold in splitter.split()) > 0
    assert sum(fold.embargoed for fold in splitter.split()) > 0


def test_wider_embargo_never_grows_the_training_set():
    t0s, ends = _samples()
    small = PurgedKFold(t0s, ends, n_splits=5, embargo=timedelta(minutes=10))
    large = PurgedKFold(t0s, ends, n_splits=5, embargo=timedelta(hours=6))
    for a, b in zip(small.split(), large.split()):
        assert b.train.size <= a.train.size


def test_longer_labels_purge_more():
    """Overlap scales with the label horizon, so the purge must too."""
    t0s, short = _samples()
    long = t0s + np.timedelta64(8, "h")
    a = PurgedKFold(t0s, short, n_splits=5, embargo=HORIZON)
    b = PurgedKFold(t0s, long, n_splits=5, embargo=HORIZON)
    assert sum(f.purged for f in b.split()) > sum(f.purged for f in a.split())


def test_unsorted_input_is_handled(splitter):
    """Indices refer to the caller's order, whatever order that is."""
    t0s, ends = _samples(60)
    shuffle = np.random.default_rng(0).permutation(60)
    scrambled = PurgedKFold(t0s[shuffle], ends[shuffle], n_splits=3, embargo=HORIZON)
    for fold in scrambled.split():
        start, end = (np.datetime64(x, "ns") for x in fold.test_span)
        overlap = (scrambled.t0s[fold.train] <= end) & (scrambled.window_ends[fold.train] >= start)
        assert not overlap.any()


def test_label_resolving_before_its_anchor_is_rejected():
    t0s, _ = _samples(10)
    with pytest.raises(ValueError, match="cannot resolve before"):
        PurgedKFold(t0s, t0s - np.timedelta64(1, "h"), n_splits=2, embargo=HORIZON)


def test_too_few_samples_is_rejected():
    t0s, ends = _samples(3)
    with pytest.raises(ValueError, match="cannot make"):
        PurgedKFold(t0s, ends, n_splits=5, embargo=HORIZON)


def test_holdout_mask_is_chronological():
    t0s, _ = _samples(100)
    boundary = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc) + timedelta(minutes=20 * 60)
    train, holdout = train_holdout_masks(t0s, boundary)
    assert train.sum() + holdout.sum() == 100
    assert t0s[train].max() < t0s[holdout].min()
