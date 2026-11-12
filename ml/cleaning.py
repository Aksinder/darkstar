"""Shared training-data cleaning on the recorder's quality flags.

The recorder tags slots whose energy values are NOT trustworthy measurements
(snapshot fallbacks, clamps, zeroed spikes) in ``slot_observations.quality_flags``.
Those tags were recorded but ignored by training. This module is the single place
that decides which tagged rows may teach which model; PV and load training
(``ml/train.py``), the 7-day slot means (``backend/learning/baseline.py``) and the
comparison scripts all go through it.

Flag tokens are a fixed contract with ``backend/recorder.py``. A token that never
occurs in the data (older rows, or a recorder that does not write it yet) is a
no-op: nothing matches, nothing is dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# quality_flags keys marking a row whose load_kwh was DERIVED from the energy-balance
# identity instead of read from a load meter:
#   "load_rescued"  backend/recorder.py — load register glitched/froze, so
#                   load = pv + import - export + discharge - charge - EV - water.
#   "repaired"      backend/learning/repair.py — statistics backfill, the same
#                   identity on hourly sums spread flat over the four quarters.
# Such a value folds every metering error and conversion loss of the other channels
# into "load", and it cannot be cross-checked (the balance tripwire abstains on it).
DERIVED_LOAD_FLAG_KEYS: tuple[str, ...] = ("load_rescued", "repaired")

# PV target (pv_kwh, trained as pv_kwh - physics): the value is a single power
# snapshot x 15 min or a counter that may omit the AC-coupled array — not weather.
PV_BAD_FLAGS: tuple[str, ...] = ("pv_unmeasured",)

# Load target (load_kwh = BASE load = total - EV - water - cyclic):
#   water_unmeasured / ev_unmeasured  the subtracted term is a snapshot guess (or 0),
#                                     so the base load is off by the unmeasured part.
#   base_load_clamped                 managed loads exceeded the total; base load was
#                                     clamped to 0 (or a floor) — an artefact.
#   load_spike_zeroed                 validation replaced an implausible load by 0.
LOAD_BAD_FLAGS: tuple[str, ...] = (
    "water_unmeasured",
    "ev_unmeasured",
    "base_load_clamped",
    "load_spike_zeroed",
)

# Name of the one cross-channel rule: a balance-derived load computed FROM an
# unmeasured PV value (load = pv + import - ...) inherits the PV error.
DERIVED_FROM_UNMEASURED_PV = "pv_unmeasured+derived_load"

# Starvation guard: cleaning is abandoned for a target (unfiltered rows are used,
# with a warning) when it would remove more than this share of the rows ...
MAX_DROP_FRACTION: float = 0.5
# ... or leave fewer rows than the trainer needs although the unfiltered set had
# enough (min_rows is passed by the caller, normally TrainingConfig.min_samples).


def flag_keys(raw: Any) -> set[str]:
    """Return the set of flag tokens present in one quality_flags value.

    Accepts the recorder's JSON object (``{"pv_unmeasured": "snapshot"}``), a JSON
    list of tokens, or an already parsed dict/list. A key whose value is ``false`` or
    ``null`` counts as absent. NULL/NaN/empty -> no flags. A truncated or malformed
    JSON string is matched on the quoted token as a plain substring (see ``has_flag``).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return set()
    flags: Any = raw
    if isinstance(raw, str):
        if not raw.strip():
            return set()
        try:
            flags = json.loads(raw)
        except ValueError:
            return set()
    if isinstance(flags, dict):
        return {str(k) for k, v in flags.items() if v is not None and v is not False}
    if isinstance(flags, list):
        return {str(item) for item in flags if isinstance(item, str)}
    return set()


def has_flag(raw: Any, tokens: tuple[str, ...]) -> bool:
    """True if any of ``tokens`` is present in one quality_flags value."""
    if isinstance(raw, str):
        try:
            json.loads(raw)
        except ValueError:
            # Malformed/truncated JSON: fall back to a quoted-substring match.
            return any(f'"{token}"' in raw for token in tokens)
    keys = flag_keys(raw)
    return any(token in keys for token in tokens)


def flag_mask(df: pd.DataFrame, tokens: tuple[str, ...]) -> pd.Series:
    """Boolean mask of rows whose quality_flags contain any of ``tokens``.

    A frame without a quality_flags column has no flags (all False).
    """
    if "quality_flags" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["quality_flags"].map(lambda raw: has_flag(raw, tokens)).astype(bool)


@dataclass
class CleaningReport:
    """What the cleaning did for one target."""

    target: str
    rows_in: int = 0
    rows_out: int = 0
    # rule name -> rows matched by that rule (a row can match several rules)
    dropped_by_rule: dict[str, int] = field(default_factory=dict)
    # True when the starvation guard discarded the cleaning (rows_out == rows_in)
    starved: bool = False

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_by_rule": dict(self.dropped_by_rule),
            "starved": self.starved,
        }


def _rule_masks(df: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    if target == "pv":
        return {token: flag_mask(df, (token,)) for token in PV_BAD_FLAGS}
    if target == "load":
        masks = {token: flag_mask(df, (token,)) for token in LOAD_BAD_FLAGS}
        masks[DERIVED_FROM_UNMEASURED_PV] = flag_mask(df, PV_BAD_FLAGS) & flag_mask(
            df, DERIVED_LOAD_FLAG_KEYS
        )
        return masks
    raise ValueError(f"Unknown cleaning target: {target!r} (expected 'pv' or 'load')")


def bad_row_mask(df: pd.DataFrame, target: str) -> pd.Series:
    """True for rows that must not be used as a ``target`` ('pv' or 'load') sample."""
    bad = pd.Series(False, index=df.index)
    for mask in _rule_masks(df, target).values():
        bad |= mask
    return bad


def clean_training_rows(
    df: pd.DataFrame,
    target: str,
    *,
    min_rows: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop rows whose quality flags make them unfit as a ``target`` training sample.

    Returns (cleaned frame, report). The input frame is never modified; when nothing
    matches, the SAME frame object is returned (absent flags = exact no-op).

    Starvation guard: if the rules together would remove more than
    ``MAX_DROP_FRACTION`` of the rows, or leave fewer than ``min_rows`` when the
    unfiltered frame had at least that many, the unfiltered frame is returned with
    ``report.starved`` set and a warning logged — a model trained on tagged rows
    beats no model, and a rule matching most of the data points at the recorder,
    not at the weather.
    """
    report = CleaningReport(target=target, rows_in=len(df), rows_out=len(df))
    if df.empty:
        return df, report

    bad = pd.Series(False, index=df.index)
    for name, mask in _rule_masks(df, target).items():
        hits = int(mask.sum())
        if hits:
            report.dropped_by_rule[name] = hits
            bad |= mask

    n_bad = int(bad.sum())
    if n_bad == 0:
        return df, report

    rules = ", ".join(f"{name}={hits}" for name, hits in report.dropped_by_rule.items())
    kept = len(df) - n_bad
    starved = n_bad / len(df) > MAX_DROP_FRACTION or (kept < min_rows <= len(df))
    if starved:
        report.starved = True
        log(
            f"Warning: {target} training-data cleaning would remove {n_bad}/{len(df)} "
            f"rows ({rules}), leaving {kept} — starvation guard: training on the "
            "UNFILTERED rows instead. Check the recorder's measurement sources."
        )
        return df, report

    report.rows_out = kept
    log(
        f"{target} training-data cleaning: removed {n_bad}/{len(df)} rows flagged "
        f"as not measured ({rules})"
    )
    return df[~bad].copy(), report
