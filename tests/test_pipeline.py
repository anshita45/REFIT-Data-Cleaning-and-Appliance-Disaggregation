"""Unit test suite for the REFIT NILM Washer project.

Covers:
1. Duplicate timestamp handling (identical vs conflicting).
2. Gap detection and interpolation (conservative <= 2 min fill vs outage flagging).
3. Cycle detection rules (thresholds, duration, energy, outage non-merging).
4. Metric calculations (MAE, RMSE, SAE, NDE, Precision, Recall, F1, energy bias).
5. Zero leakage validation (aggregate-only input, no target channel in features).
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.events import detect_cycles, runs_from_bool
from src.model import build_features, minute_metrics
from src.utils import fill_short_gaps, nan_runs


# -----------------------------------------------------------------------------
# 1. Duplicate Timestamp Handling
# -----------------------------------------------------------------------------
def test_duplicate_timestamp_handling():
    """Verify that identical duplicates are deduplicated and conflicting readings are averaged."""
    t0 = pd.Timestamp("2024-01-01 12:00:00")
    t1 = pd.Timestamp("2024-01-01 12:01:00")
    t2 = pd.Timestamp("2024-01-01 12:02:00")

    # Construct raw records with duplicates
    df = pd.DataFrame({
        "time": [t0, t0, t1, t1, t2],
        "Aggregate": [100.0, 100.0, 200.0, 400.0, 50.0],  # t0 identical, t1 conflicting (200 & 400)
        "Appliance1": [10.0, 10.0, 20.0, 40.0, 5.0],
    }).set_index("time")

    # Identical duplicate count
    reset_df = df.reset_index()
    identical_count = int(reset_df.duplicated(keep=False).sum())
    assert identical_count == 2, "Should identify the two identical rows at t0"

    # Conflicting duplicate count on Aggregate
    dup_ts = df.index.duplicated(keep=False)
    g = df.loc[dup_ts, "Aggregate"].groupby(level=0).nunique(dropna=True)
    conflicting_count = int((g > 1).sum())
    assert conflicting_count == 1, "Should identify t1 as having conflicting aggregate values"

    # Mean resolution
    resolved = df.groupby(level=0).mean().sort_index()
    assert len(resolved) == 3
    assert resolved.loc[t0, "Aggregate"] == 100.0
    assert resolved.loc[t1, "Aggregate"] == 300.0  # mean of 200 and 400
    assert resolved.loc[t2, "Aggregate"] == 50.0


# -----------------------------------------------------------------------------
# 2. Gap Detection and Conservative Interpolation
# -----------------------------------------------------------------------------
def test_gap_detection_and_interpolation():
    """Verify that gaps <= MAX_FILL_GAP_MIN (2 min) are interpolated, while longer outages remain NaN."""
    # 10 minute series with a 2-minute gap and a 4-minute gap
    idx = pd.date_range("2024-01-01 00:00", periods=12, freq="1min")
    s = pd.Series([100.0, np.nan, np.nan, 130.0, np.nan, np.nan, np.nan, np.nan, 200.0, 210.0, 220.0, 230.0], index=idx)

    # Detect NaN runs
    runs = nan_runs(s)
    assert len(runs) == 2
    assert runs[0] == (1, 2)  # pos 1, length 2 (<= 2 min)
    assert runs[1] == (4, 4)  # pos 4, length 4 (> 2 min)

    # Conservative filling with MAX_FILL_GAP_MIN = 2
    filled = fill_short_gaps(s, max_gap=2)

    # The 2-min gap should be linearly interpolated
    assert not np.isnan(filled.iloc[1])
    assert not np.isnan(filled.iloc[2])
    assert np.isclose(filled.iloc[1], 110.0)
    assert np.isclose(filled.iloc[2], 120.0)

    # The 4-min outage must stay NaN
    assert np.isnan(filled.iloc[4])
    assert np.isnan(filled.iloc[5])
    assert np.isnan(filled.iloc[6])
    assert np.isnan(filled.iloc[7])


# -----------------------------------------------------------------------------
# 3. Cycle Detection Logic
# -----------------------------------------------------------------------------
def test_cycle_detection_heuristics():
    """Verify active threshold, duration, energy filter, and gap-bridging rules."""
    idx = pd.date_range("2024-01-01 00:00", periods=60, freq="1min")
    p = pd.Series(0.0, index=idx)

    # Valid cycle: 15 minutes, power 2000 W (Energy = 2000 * 15 / 60 = 500 Wh)
    p.iloc[5:20] = 2000.0

    # Short blip: 5 minutes at 1000 W (< 10 min threshold)
    p.iloc[30:35] = 1000.0

    # Low energy event: 12 minutes at 10 W (< 20 W ON_THRESHOLD)
    p.iloc[40:52] = 10.0

    cycles = detect_cycles(p, house_id=1)
    assert len(cycles) == 1, "Only the 15-min 2000W event should be detected as a cycle"
    c0 = cycles.iloc[0]
    assert c0.duration_min == 15
    assert c0.peak_w == 2000.0
    assert np.isclose(c0.energy_wh, 500.0)
    assert c0.house == 1
    assert not c0.touches_gap


def test_cycles_do_not_merge_across_missing_outages():
    """Verify that separate active periods separated by a NaN outage are NOT merged together."""
    idx = pd.date_range("2024-01-01 00:00", periods=40, freq="1min")
    p = pd.Series(0.0, index=idx)

    # First cycle: 10 mins active
    p.iloc[2:12] = 1500.0

    # 3-min NaN outage
    p.iloc[12:15] = np.nan

    # Second cycle: 12 mins active
    p.iloc[15:27] = 1800.0

    cycles = detect_cycles(p)
    # Because of the NaN outage, runs_from_bool must not merge them into a single 25-minute cycle
    assert len(cycles) == 2, "Separate runs separated by a NaN gap must not be merged"
    assert cycles.iloc[0].duration_min == 10
    assert cycles.iloc[1].duration_min == 12
    # Both cycles touch the NaN outage, so touches_gap should be True
    assert cycles.iloc[0].touches_gap
    assert cycles.iloc[1].touches_gap


# -----------------------------------------------------------------------------
# 4. Disaggregation Metric Calculations
# -----------------------------------------------------------------------------
def test_metric_calculations():
    """Verify that disaggregation metrics match analytical ground truths and handle zeros."""
    y = np.array([0.0, 0.0, 100.0, 200.0, 0.0])
    p = np.array([0.0, 0.0, 120.0, 180.0, 0.0])

    # True energy = (100 + 200) = 300
    # Pred energy = (120 + 180) = 300 -> SAE = 0, bias = 0%
    # Errors = [0, 0, +20, -20, 0] -> MAE = 8.0, RMSE = sqrt(800/5) = 12.649
    m = minute_metrics(y, p, on_thr=20.0)
    assert np.isclose(m["MAE_W"], 8.0)
    assert np.isclose(m["RMSE_W"], np.sqrt(160.0))
    assert np.isclose(m["SAE"], 0.0)
    assert np.isclose(m["energy_bias_pct"], 0.0)
    assert np.isclose(m["precision"], 1.0)
    assert np.isclose(m["recall"], 1.0)
    assert np.isclose(m["F1"], 1.0)

    # Test baseline always-off
    p_off = np.zeros_like(y)
    m_off = minute_metrics(y, p_off, on_thr=20.0)
    assert np.isclose(m_off["NDE"], 1.0), "NDE of always-off should analytically be 1.0"
    assert np.isclose(m_off["energy_bias_pct"], -100.0)


# -----------------------------------------------------------------------------
# 5. Zero Feature Leakage
# -----------------------------------------------------------------------------
def test_feature_engineering_zero_leakage():
    """Verify that build_features relies strictly on the aggregate series and has correct shape."""
    idx = pd.date_range("2024-01-01 00:00", periods=100, freq="1min")
    agg = pd.Series(np.random.uniform(50, 3000, size=100), index=idx, name="Aggregate")

    feat = build_features(agg)
    assert isinstance(feat, pd.DataFrame)
    assert len(feat) == 100
    assert "agg" in feat.columns
    assert "floor" in feat.columns
    assert "excess" in feat.columns
    assert "hour_sin" in feat.columns
    assert "hour_cos" in feat.columns
    # Ensure no appliance column leaked
    for col in feat.columns:
        assert "Appliance" not in col
        assert "Washer" not in col
