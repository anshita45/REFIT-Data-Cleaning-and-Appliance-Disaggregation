"""Washing-machine cycle detection (one rule, used for real data and for predictions)."""
import numpy as np
import pandas as pd

from . import config as C


def runs_from_bool(on: np.ndarray, merge_gap: int = C.MERGE_GAP_MIN, min_len: int = C.MIN_CYCLE_MIN, is_gap: np.ndarray = None):
    """Positions [s, e) of ON runs; runs separated by <= merge_gap OFF samples are merged
    UNLESS the gap contains missing/NaN data (to prevent false merging across outages);
    runs shorter than min_len are dropped."""
    on = np.asarray(on, dtype=bool)
    if not on.any():
        return []
    d = np.diff(np.r_[0, on.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ms, me = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        gap_has_nan = is_gap is not None and np.any(is_gap[me[-1]:s])
        if (s - me[-1] <= merge_gap) and not gap_has_nan:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    return [(s, e) for s, e in zip(ms, me) if e - s >= min_len]


def detect_cycles(power: pd.Series, house_id: int = None) -> pd.DataFrame:
    """Cycles in a 1-minute power series (W). Missing data (NaN) does not bridge cycles.
    Cycles that touch or contain missing data are flagged touches_gap=True."""
    v = power.to_numpy(dtype=float)
    nan = np.isnan(v)
    on = np.where(nan, False, v > C.ON_THRESHOLD_W)
    rows = []
    for s, e in runs_from_bool(on, is_gap=nan):
        # Calculate energy in Wh as sum of valid minute readings / 60
        valid_span = v[s:e]
        valid_power = valid_span[~np.isnan(valid_span)]
        if len(valid_power) == 0:
            continue
        energy_wh = float(np.sum(valid_power) / 60.0)
        if energy_wh < C.MIN_CYCLE_WH:
            continue
        lo, hi = max(s - 1, 0), min(e + 1, len(v))
        has_internal_gap = bool(nan[s:e].any())
        touches_boundary_gap = bool(nan[lo:s].any() or nan[e:hi].any())
        
        row = dict(
            i0=s, i1=e, start=power.index[s], end=power.index[e - 1],
            duration_min=e - s, peak_w=float(np.nanmax(valid_span)),
            mean_w=float(np.nanmean(valid_span)),
            energy_wh=energy_wh,
            touches_gap=has_internal_gap or touches_boundary_gap)
        if house_id is not None:
            row["house"] = house_id
        rows.append(row)
    
    cols = (["house"] if house_id is not None else []) + [
        "i0", "i1", "start", "end", "duration_min", "peak_w", "mean_w", "energy_wh", "touches_gap"
    ]
    return pd.DataFrame(rows, columns=cols)
