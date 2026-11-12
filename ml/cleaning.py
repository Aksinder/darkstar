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
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("darkstar.ml.cleaning")

# quality_flags keys marking a row whose load_kwh was DERIVED from the energy-balance
# identity instead of read from a load meter:
#   "load_rescued"  backend/recorder.py — load register glitched/froze, so
#                   load = pv + import - export + discharge - charge - EV - water.
#   "repaired"      backend/learning/repair.py — statistics backfill, the same
#                   identity on hourly sums spread flat over the four quarters.
# Such a value folds every metering error and conversion loss of the other channels
# into "load", and it cannot be cross-checked (the balance tripwire abstains on it).
DERIVED_LOAD_FLAG_KEYS: tuple[str, ...] = ("load_rescued", "repaired")

# Value shapes written by the recorder (a flag is a key, present only when it fired):
#   pv_unmeasured      "snapshot" | "counter"
#   water_unmeasured   {heater_id_or_sensor: "unavailable" | "snapshot"}
#   ev_unmeasured      {charger_id_or_sensor: "unavailable" | "snapshot"}
#   base_load_clamped  true — only when the stored load_kwh IS the clamped 0.0 (a
#                      balance-rescued row carries load_rescued and a real value)
#   load_spike_zeroed  sorted LIST of the zeroed columns, e.g. ["load_kwh", "pv_kwh"];
#                      fires for ANY zeroed energy column, so it is matched per column.
SPIKE_ZEROED_FLAG = "load_spike_zeroed"

# PV target (pv_kwh, trained as pv_kwh - physics): the value is a single power
# snapshot x 15 min or a counter that may omit the AC-coupled array — not weather.
# Plus load_spike_zeroed naming pv_kwh: the stored 0.0 would train residual = -physics.
PV_BAD_FLAGS: tuple[str, ...] = ("pv_unmeasured",)
PV_SPIKE_ZEROED = "load_spike_zeroed[pv_kwh]"

# Load target (load_kwh = BASE load = total - EV - water - cyclic):
#   water_unmeasured / ev_unmeasured  the subtracted term is a snapshot guess (or 0,
#                                     leaving the device inside base load), so the
#                                     base load is off by the unmeasured part.
#   base_load_clamped                 managed loads exceeded the total; base load was
#                                     clamped to 0 — an artefact.
#   load_spike_zeroed                 only when the list names load_kwh: validation
#                                     replaced an implausible load by 0.
LOAD_BAD_FLAGS: tuple[str, ...] = (
    "water_unmeasured",
    "ev_unmeasured",
    "base_load_clamped",
    SPIKE_ZEROED_FLAG,
)

# Per-device flags (value = {device: "unavailable" | "snapshot"}).
DEVICE_FLAGS: tuple[str, ...] = ("water_unmeasured", "ev_unmeasured")

# OFFLINE-DEVICE RULE. The recorder writes "unavailable" for every slot in which a
# device's power sensor has no numeric state. For a short gap that means the device may
# have been running unseen, so the base load is suspect. But a device that is switched
# off for weeks (winterised spa, a caravan heater unplugged for the season) is
# "unavailable" the whole time while drawing nothing: its base load is CORRECT, and
# dropping those rows would remove the newest, highest-weighted data.
#   A device counts as OFFLINE for an uninterrupted run of "unavailable" slots — no
#   slot in between where that device was measured or snapshot-read — that spans at
#   least OFFLINE_MIN_HOURS from its first to its last slot and holds at least
#   OFFLINE_MIN_ROWS rows. Rows of such a run are not dropped on that device's
#   account. Shorter "unavailable" runs, every "snapshot", and any other device in
#   the same row still drop it.
# A run is judged as a whole, so an outage that started less than 24 h ago is still a
# gap (dropped) and becomes "offline" (kept, retroactively) once it has lasted a day.
OFFLINE_STATE = "unavailable"
OFFLINE_MIN_HOURS: float = 24.0
OFFLINE_MIN_ROWS: int = 48

# Cross-channel rules for a balance-DERIVED load (load = pv + import - export + ...):
# it inherits the error of an unmeasured PV value, and of any zeroed energy column.
DERIVED_FROM_UNMEASURED_PV = "pv_unmeasured+derived_load"
DERIVED_FROM_ZEROED_COLUMN = "load_spike_zeroed+derived_load"

# Starvation guard: cleaning is abandoned for a target (unfiltered rows are used,
# with a warning) when it would remove more than this share of the rows ...
MAX_DROP_FRACTION: float = 0.5
# ... or leave fewer rows than the trainer needs although the unfiltered set had
# enough (min_rows is passed by the caller, normally TrainingConfig.min_samples).


# Kill switches under forecasting: (both default true = the behaviour described here).
CLEANING_ENABLED_KEY = "training_data_cleaning_enabled"
SLOT_MEAN_FALLBACK_ENABLED_KEY = "load_slot_mean_fallback_enabled"


def forecasting_switch(config: dict[str, Any] | None, key: str, default: bool = True) -> bool:
    """Read a boolean switch under ``forecasting``; unset/unreadable -> ``default``.

    Accepts real booleans and the usual spellings (false/off/no/0, true/on/yes/1).
    """
    forecasting_cfg: Any = (config or {}).get("forecasting")
    raw: Any = forecasting_cfg.get(key) if isinstance(forecasting_cfg, dict) else None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int | float):
        return raw != 0
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("false", "off", "no", "0"):
            return False
        if text in ("true", "on", "yes", "1"):
            return True
    return default


def parse_flags(raw: Any) -> dict[str, Any] | None:
    """Parse one quality_flags value into {token: value}.

    Accepts the recorder's JSON object, a JSON list of tokens (each -> True), or an
    already parsed dict/list. NULL/NaN/empty/other types -> {} (no flags). Returns
    None for a string that is not valid JSON (see ``has_flag`` for how that matches).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    flags: Any = raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            flags = json.loads(raw)
        except ValueError:
            return None
    if isinstance(flags, dict):
        return {str(k): v for k, v in flags.items()}
    if isinstance(flags, list):
        return {str(item): True for item in flags if isinstance(item, str)}
    return {}


def _fired(value: Any) -> bool:
    """A key whose value is false/null counts as absent; any other value fired."""
    return value is not None and value is not False


def has_flag(raw: Any, tokens: tuple[str, ...]) -> bool:
    """True if any of ``tokens`` is present (and not false/null) in a flags value.

    A truncated or otherwise malformed JSON string is matched on the quoted token as
    a plain substring.
    """
    flags = parse_flags(raw)
    if flags is None:
        return any(f'"{token}"' in str(raw) for token in tokens)
    return any(_fired(flags.get(token)) for token in tokens)


def spike_zeroed_column(raw: Any, column: str) -> bool:
    """True if load_spike_zeroed says ``column`` was zeroed by validation.

    The flag value is a list of column names. A bare string is read as one column.
    Any other fired value (e.g. a legacy ``true``) names no column, so it is taken to
    mean what the flag's name says: load_kwh only. Malformed JSON: both the quoted
    flag and the quoted column must occur in the string.
    """
    flags = parse_flags(raw)
    if flags is None:
        text = str(raw)
        return f'"{SPIKE_ZEROED_FLAG}"' in text and f'"{column}"' in text
    value = flags.get(SPIKE_ZEROED_FLAG)
    if not _fired(value):
        return False
    if isinstance(value, list | tuple | set):
        return column in value
    if isinstance(value, str):
        return value == column
    return column == "load_kwh"


def flag_mask(df: pd.DataFrame, tokens: tuple[str, ...]) -> pd.Series:
    """Boolean mask of rows whose quality_flags contain any of ``tokens``.

    A frame without a quality_flags column has no flags (all False).
    """
    if "quality_flags" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["quality_flags"].map(lambda raw: has_flag(raw, tokens)).astype(bool)


def spike_zeroed_mask(df: pd.DataFrame, column: str) -> pd.Series:
    """Boolean mask of rows whose load_spike_zeroed list names ``column``."""
    if "quality_flags" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["quality_flags"].map(lambda raw: spike_zeroed_column(raw, column)).astype(bool)


def device_states(raw: Any, token: str) -> dict[str, str]:
    """{device: state} for one per-device flag (water_unmeasured / ev_unmeasured).

    A fired value that is not a dict (legacy ``true``, a bare string, malformed JSON
    containing the quoted token) names no device: it is returned as {"?": "unknown"},
    which can never qualify as an offline device and therefore always drops the row.
    """
    flags = parse_flags(raw)
    if flags is None:
        return {"?": "unknown"} if f'"{token}"' in str(raw) else {}
    value = flags.get(token)
    if not _fired(value):
        return {}
    if isinstance(value, dict):
        return {str(device): str(state) for device, state in value.items()}
    return {"?": "unknown"}


def device_flag_masks(df: pd.DataFrame, token: str) -> tuple[pd.Series, dict[str, int]]:
    """(rows to drop for ``token``, {device: rows kept as offline device}).

    Applies the OFFLINE-DEVICE RULE above per device. Needs a slot_start column to
    order the rows and measure a run; without one nothing is exempted.
    """
    drop = pd.Series(False, index=df.index)
    kept: dict[str, int] = {}
    if "quality_flags" not in df.columns or df.empty:
        return drop, kept
    states = df["quality_flags"].map(lambda raw: device_states(raw, token))
    flagged = states.map(bool)
    if not flagged.any():
        return drop, kept
    if "slot_start" not in df.columns:
        return flagged.astype(bool), kept

    when = df["slot_start"]
    if not pd.api.types.is_datetime64_any_dtype(when):
        when = pd.to_datetime(when, format="ISO8601", utc=True, errors="coerce")
    order = when.sort_values(kind="stable").index
    devices = sorted({device for row in states[flagged] for device in row})
    for device in devices:
        state = states.loc[order].map(lambda row, d=device: row.get(d))
        unavailable = state == OFFLINE_STATE
        offline = pd.Series(False, index=order)
        if unavailable.any():
            run_id = (~unavailable).cumsum()[unavailable]
            times = when.loc[order][unavailable]
            span_h = (
                times.groupby(run_id).transform("max") - times.groupby(run_id).transform("min")
            ).dt.total_seconds() / 3600.0 + 0.25
            rows = times.groupby(run_id).transform("size")
            offline.loc[span_h.index] = (
                (span_h >= OFFLINE_MIN_HOURS) & (rows >= OFFLINE_MIN_ROWS)
            ).to_numpy()
        n_offline = int(offline.sum())
        if n_offline:
            kept[device] = n_offline
        drop |= (state.notna() & ~offline).reindex(df.index)
    return drop.astype(bool), kept


@dataclass
class CleaningReport:
    """What the cleaning did for one target."""

    target: str
    rows_in: int = 0
    rows_out: int = 0
    # rule name -> rows matched by that rule (a row can match several rules)
    dropped_by_rule: dict[str, int] = field(default_factory=dict)
    # "flag:device" -> rows NOT dropped on that device's account because it was
    # offline (see OFFLINE-DEVICE RULE); they may still be dropped by another rule
    kept_offline_device: dict[str, int] = field(default_factory=dict)
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
            "kept_offline_device": dict(self.kept_offline_device),
            "starved": self.starved,
        }


def _rule_masks(
    df: pd.DataFrame, target: str, kept_offline: dict[str, int] | None = None
) -> dict[str, pd.Series]:
    if target == "pv":
        masks = {token: flag_mask(df, (token,)) for token in PV_BAD_FLAGS}
        masks[PV_SPIKE_ZEROED] = spike_zeroed_mask(df, "pv_kwh")
        return masks
    if target == "load":
        masks: dict[str, pd.Series] = {}
        for token in LOAD_BAD_FLAGS:
            if token == SPIKE_ZEROED_FLAG:
                masks[token] = spike_zeroed_mask(df, "load_kwh")
            elif token in DEVICE_FLAGS:
                masks[token], offline = device_flag_masks(df, token)
                if kept_offline is not None:
                    kept_offline.update({f"{token}:{dev}": n for dev, n in offline.items()})
            else:
                masks[token] = flag_mask(df, (token,))
        derived = flag_mask(df, DERIVED_LOAD_FLAG_KEYS)
        masks[DERIVED_FROM_UNMEASURED_PV] = derived & flag_mask(df, PV_BAD_FLAGS)
        masks[DERIVED_FROM_ZEROED_COLUMN] = derived & flag_mask(df, (SPIKE_ZEROED_FLAG,))
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
    ``report.starved`` set and a WARNING logged (logger, not ``log``) — a model trained on tagged rows
    beats no model, and a rule matching most of the data points at the recorder,
    not at the weather.
    """
    report = CleaningReport(target=target, rows_in=len(df), rows_out=len(df))
    if df.empty:
        return df, report

    bad = pd.Series(False, index=df.index)
    for name, mask in _rule_masks(df, target, report.kept_offline_device).items():
        hits = int(mask.sum())
        if hits:
            report.dropped_by_rule[name] = hits
            bad |= mask

    if report.kept_offline_device:
        offline = ", ".join(f"{name}={n}" for name, n in report.kept_offline_device.items())
        log(
            f"{target} training-data cleaning: kept rows of offline devices "
            f"('unavailable' for >= {OFFLINE_MIN_HOURS:g} h, base load unaffected): {offline}"
        )

    n_bad = int(bad.sum())
    if n_bad == 0:
        return df, report

    rules = ", ".join(f"{name}={hits}" for name, hits in report.dropped_by_rule.items())
    kept = len(df) - n_bad
    starved = n_bad / len(df) > MAX_DROP_FRACTION or (kept < min_rows <= len(df))
    if starved:
        report.starved = True
        logger.warning(
            "%s training-data cleaning would remove %d/%d rows (%s), leaving %d — "
            "starvation guard: training on the UNFILTERED rows instead. Check the "
            "recorder's measurement sources.",
            target,
            n_bad,
            len(df),
            rules,
            kept,
        )
        return df, report

    report.rows_out = kept
    log(
        f"{target} training-data cleaning: removed {n_bad}/{len(df)} rows flagged "
        f"as not measured ({rules})"
    )
    return df[~bad].copy(), report
