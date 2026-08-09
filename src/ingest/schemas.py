"""Pandera schema contracts, enforced on read and on write.

A schema failing loudly at the boundary is worth more than a downstream statistic that
is quietly wrong. Everything that crosses into or out of ``data/`` is validated.

The bar contract encodes the invariant that leak L1 turns on: a bar carries both the
timestamp it opened at and the timestamp it closed at, and ``close_ts`` is strictly
greater. Point-in-time truncation filters on ``close_ts``, so a bar is only visible once
it has finished forming. A single-timestamp bar makes that distinction unrepresentable,
which is why one is refused here.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData

UTC_TS = pl.Datetime("us", "UTC")


def _strictly_increasing(data: PolarsData) -> bool:
    col = data.lazyframe.select(data.key).collect().to_series()
    return bool(col.is_sorted(descending=False)) and col.n_unique() == len(col)


def _non_decreasing(data: PolarsData) -> bool:
    col = data.lazyframe.select(data.key).collect().to_series()
    return bool(col.is_sorted(descending=False))


TickSchema = pa.DataFrameSchema(
    name="ticks",
    strict=False,
    columns={
        "ts": pa.Column(UTC_TS, checks=pa.Check(_non_decreasing, name="non_decreasing")),
        "price": pa.Column(pl.Float64, pa.Check.gt(0)),
        "size": pa.Column(pl.Int64, pa.Check.gt(0)),
        # Exchange-provided aggressor: +1 buyer initiated, -1 seller initiated, 0 unknown.
        "aggressor": pa.Column(pl.Int8, pa.Check.isin([-1, 0, 1])),
        "contract": pa.Column(pl.Utf8),
        "session_date": pa.Column(pl.Date),
    },
)


BarSchema = pa.DataFrameSchema(
    name="bars",
    strict=False,
    columns={
        "open_ts": pa.Column(UTC_TS),
        "close_ts": pa.Column(UTC_TS, checks=pa.Check(_strictly_increasing, name="unique_increasing")),
        "open": pa.Column(pl.Float64, pa.Check.gt(0)),
        "high": pa.Column(pl.Float64, pa.Check.gt(0)),
        "low": pa.Column(pl.Float64, pa.Check.gt(0)),
        "close": pa.Column(pl.Float64, pa.Check.gt(0)),
        "volume": pa.Column(pl.Int64, pa.Check.gt(0)),
        "dollar_volume": pa.Column(pl.Float64, pa.Check.gt(0)),
        "vwap": pa.Column(pl.Float64, pa.Check.gt(0)),
        "n_ticks": pa.Column(pl.Int64, pa.Check.gt(0)),
        "session_date": pa.Column(pl.Date),
        "contract": pa.Column(pl.Utf8),
        "in_rth": pa.Column(pl.Boolean),
    },
    checks=[
        pa.Check(
            lambda d: d.lazyframe.select(
                (pl.col("high") >= pl.max_horizontal("open", "close")).all()
            ).collect().item(),
            name="high_covers_body",
            error="high must be at least max(open, close)",
        ),
        pa.Check(
            lambda d: d.lazyframe.select(
                (pl.col("low") <= pl.min_horizontal("open", "close")).all()
            ).collect().item(),
            name="low_covers_body",
            error="low must be at most min(open, close)",
        ),
        pa.Check(
            lambda d: d.lazyframe.select((pl.col("high") >= pl.col("low")).all()).collect().item(),
            name="high_ge_low",
        ),
        pa.Check(
            lambda d: d.lazyframe.select(
                (pl.col("close_ts") > pl.col("open_ts")).all()
            ).collect().item(),
            name="bar_has_duration",
            error="close_ts must be strictly after open_ts — see leak L1",
        ),
        pa.Check(
            lambda d: d.lazyframe.select(
                ((pl.col("vwap") <= pl.col("high")) & (pl.col("vwap") >= pl.col("low"))).all()
            ).collect().item(),
            name="vwap_within_range",
        ),
    ],
)


SessionSchema = pa.DataFrameSchema(
    name="sessions",
    strict=False,
    columns={
        "session_date": pa.Column(pl.Date, checks=pa.Check(_strictly_increasing, name="unique_increasing")),
        "rth_open": pa.Column(UTC_TS),
        "rth_close": pa.Column(UTC_TS),
        "eth_open": pa.Column(UTC_TS),
        "eth_close": pa.Column(UTC_TS),
        "is_early_close": pa.Column(pl.Boolean),
    },
    checks=[
        pa.Check(
            lambda d: d.lazyframe.select(
                (pl.col("rth_close") > pl.col("rth_open")).all()
            ).collect().item(),
            name="rth_window_positive",
        ),
    ],
)


def validate(schema: pa.DataFrameSchema, frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and return the frame, so this can wrap a write."""
    return schema.validate(frame, lazy=True)
