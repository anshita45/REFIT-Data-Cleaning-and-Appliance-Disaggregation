"""Section 1: inspect and lightly clean RAW REFIT data (House 1) at 1-minute resolution."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import find_house_files, read_house
from .utils import fill_short_gaps, log, nan_runs, save_table


def inspect(raw: pd.DataFrame) -> dict:
    idx = raw.index
    good = raw[idx.notna()]
    rep = {}
    rep["rows_in_file"] = len(raw)
    rep["unparsable_timestamps"] = int(idx.isna().sum())
    gi = good.index
    rep["first_timestamp"] = str(gi.min())
    rep["last_timestamp"] = str(gi.max())
    rep["out_of_order_rows"] = int((gi[1:] < gi[:-1]).sum())          # file order
    
    # Detailed duplicate analysis
    dup_ts = gi.duplicated(keep=False)
    rep["rows_with_duplicate_timestamp"] = int(dup_ts.sum())
    
    # Identical rows (timestamp + all readings identical)
    reset_good = good.reset_index()
    identical_dups = int(reset_good.duplicated(keep=False).sum())
    rep["identical_duplicate_rows"] = identical_dups
    
    if dup_ts.any() and "Aggregate" in good:
        g = good.loc[dup_ts, "Aggregate"].groupby(level=0).nunique(dropna=True)
        rep["duplicate_timestamps_with_conflicting_aggregate"] = int((g > 1).sum())
    else:
        rep["duplicate_timestamps_with_conflicting_aggregate"] = 0

    s = gi.drop_duplicates().sort_values()
    dt = pd.Series(s[1:] - s[:-1]).dt.total_seconds()
    rep["median_interval_s"] = float(dt.median())
    rep["share_interval_eq_median"] = float((dt == dt.median()).mean())
    rep["p95_interval_s"] = float(dt.quantile(0.95))
    rep["intervals_over_60s"] = int((dt > 60).sum())
    rep["intervals_over_1h"] = int((dt > 3600).sum())
    rep["longest_interval_h"] = float(dt.max() / 3600)
    return rep


def channel_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    agg = raw["Aggregate"] if "Aggregate" in raw else None
    for c in raw.columns:
        v = raw[c]
        p_cap = C.MAX_POWER_W if c == "Aggregate" else C.MAX_APPLIANCE_W
        rows.append(dict(
            channel=c, readings=int(v.notna().sum()), missing=int(v.isna().sum()),
            negative=int((v < 0).sum()), over_limit=int((v > p_cap).sum()),
            zero_share=float((v == 0).mean()), p99_99=float(v.quantile(0.9999)),
            max=float(v.max()),
            above_aggregate=int((v > agg * 1.1 + 50).sum()) if (agg is not None and c != "Aggregate") else np.nan))
    return pd.DataFrame(rows)


def clean(raw: pd.DataFrame):
    log_ = {}
    df = raw[raw.index.notna()].copy()
    n_bad = 0
    # Channel-aware validity filtering
    for c in df.columns:
        p_cap = C.MAX_POWER_W if c == "Aggregate" else C.MAX_APPLIANCE_W
        bad = (df[c] < 0) | (df[c] > p_cap)
        n_bad += int(bad.sum())
        df.loc[bad, c] = np.nan
    log_["impossible_values_set_to_nan"] = n_bad
    
    # Duplicate resolution rule:
    # 1. Exact duplicate rows are dropped
    # 2. Conflicting timestamps are merged by taking the mean of valid readings
    n_dups = int(df.index.duplicated().sum())
    log_["duplicate_rows_merged_by_mean"] = n_dups
    df = df.groupby(level=0).mean().sort_index()
    
    # Resample to 1-minute means
    m = df.resample(C.RESAMPLE).mean()
    log_["minutes_total"] = len(m)
    log_["minutes_without_any_reading_before_fill"] = int(m["Aggregate"].isna().sum())
    
    # Conservative short-gap filling (<= MAX_FILL_GAP_MIN minutes)
    filled = m.copy()
    for c in filled.columns:
        filled[c] = fill_short_gaps(filled[c], C.MAX_FILL_GAP_MIN)
    
    log_["aggregate_minutes_interpolated"] = int(m["Aggregate"].isna().sum() - filled["Aggregate"].isna().sum())
    runs = nan_runs(filled["Aggregate"])
    log_["outage_runs_flagged"] = len(runs)
    log_["outage_minutes_flagged"] = int(sum(l for _, l in runs))
    log_["longest_outage_h"] = float(max([l for _, l in runs], default=0) / 60)
    
    # Retain explicit masks for missingness / outages / interpolation
    filled["outage"] = filled["Aggregate"].isna()
    filled["interpolated"] = (m["Aggregate"].isna() & filled["Aggregate"].notna())
    return filled, log_, runs


def plot_before_after(raw, cleaned, runs, house):
    chans = [c for c in C.RAW_PLOT_CHANNELS if c in raw.columns]
    # Window: 2 days around the longest outage if there is one, else the middle of the record
    if runs:
        s, l = max(runs, key=lambda r: r[1])
        centre = cleaned.index[min(s + l // 2, len(cleaned) - 1)]
    else:
        centre = cleaned.index[len(cleaned) // 2]
    lo, hi = centre - pd.Timedelta(days=1), centre + pd.Timedelta(days=1)
    rb = raw[raw.index.notna()].sort_index().loc[lo:hi]
    ca = cleaned.loc[lo:hi]
    fig, ax = plt.subplots(len(chans), 2, figsize=(14, 2.6 * len(chans)), sharex="col", squeeze=False)
    for i, c in enumerate(chans):
        # Raw readings plotted as scatter points to avoid connecting lines across gaps
        ax[i, 0].plot(rb.index, rb[c], ".", ms=1.5, lw=0, color="tab:red")
        # Cleaned 1-minute series (line breaks naturally where NaN outages exist)
        ax[i, 1].plot(ca.index, ca[c], lw=0.9, color="tab:blue")
        ax[i, 0].set_ylabel(f"{c} (W)")
        out = ca["outage"].to_numpy()
        ax[i, 1].fill_between(ca.index, 0, 1, where=out, transform=ax[i, 1].get_xaxis_transform(),
                              color="grey", alpha=0.35, lw=0, label="Outage (>2 min)" if i == 0 else "")
    ax[0, 0].set_title("Before: Raw Readings (irregular, duplicates, sensor spikes)")
    ax[0, 1].set_title(f"After: 1-min Resampled (duplicates merged, short gaps <= {C.MAX_FILL_GAP_MIN} min filled)\n(grey shading = outage, left unfilled)")
    fig.suptitle(f"House {house}: Raw vs Cleaned Power Profiles (2-day representative window)", y=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    C.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG / "01_raw_before_after.png", dpi=130)
    plt.close(fig)


def run(house: int = C.RAW_HOUSE):
    paths = find_house_files(C.RAW_DIR, house)
    log.info("Reading raw file(s): %s", [p.name for p in paths])
    raw = read_house(paths)
    log.info("Raw rows loaded: %d, Columns: %s", len(raw), list(raw.columns))
    rep = inspect(raw)
    ch = channel_table(raw)
    cleaned, clog, runs = clean(raw)
    rep.update({f"clean_{k}": v for k, v in clog.items()})
    rep["files"] = ", ".join(p.name for p in paths)
    rep["columns"] = list(raw.columns)

    save_table(pd.DataFrame(dict(check=list(rep.keys()), result=[str(v) for v in rep.values()])), "raw_quality")
    save_table(ch, "raw_channels", nd=3)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "raw_quality.json").write_text(json.dumps(rep, indent=2))
    plot_before_after(raw, cleaned, runs, house)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_pickle(C.PROC_DIR / f"raw_house{house}_1min.pkl")
    log.info("Section 1 done: %s", {k: clog[k] for k in clog})
    return rep


if __name__ == "__main__":
    run()
