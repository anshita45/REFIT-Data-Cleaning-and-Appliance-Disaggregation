"""Section 2: washing-machine EDA on the official CLEAN REFIT data (1-minute)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .events import detect_cycles
from .io_utils import to_local
from .prep import load_minute
from .utils import log, nan_runs, save_table

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_study(houses=None, force=False):
    """Load houses, run the washer sanity check, return (data, excluded_table)."""
    houses = houses or C.STUDY_HOUSES
    data, excluded = {}, []
    for h in houses:
        try:
            df = load_minute(h, force)
        except FileNotFoundError as e:
            excluded.append((h, f"file missing: {e}"))
            continue
        col = C.WASHER_CHANNEL.get(h)
        if col is None or col not in df.columns:
            excluded.append((h, "no washer channel configured/found"))
            continue
        cyc = detect_cycles(df[col])
        peak = float(df[col].max()) if df[col].notna().any() else 0.0
        if peak < C.MIN_PEAK_W or len(cyc) < C.MIN_CYCLES:
            excluded.append((h, f"{col} does not look like a washer (peak {peak:.0f} W, {len(cyc)} cycles)"))
            continue
        data[h] = df
    exc = pd.DataFrame(excluded, columns=["house", "reason"])
    save_table(exc if len(exc) else pd.DataFrame(dict(house=["-"], reason=["none excluded"])), "excluded_houses")
    if len(data) < C.MIN_HOUSES_REQUIRED:
        log.warning("Only %d usable houses (<%d required). Check WASHER_CHANNEL against the readme.",
                    len(data), C.MIN_HOUSES_REQUIRED)
    return data, exc


def availability(data):
    rows, cycles, all_cycles_list = [], {}, []
    for h, df in data.items():
        col = C.WASHER_CHANNEL[h]
        w, a = df[col], df["Aggregate"]
        both = w.notna() & a.notna()
        cyc = detect_cycles(w, house_id=h)
        cycles[h] = cyc
        all_cycles_list.append(cyc)
        good = cyc[~cyc.touches_gap]
        runs = nan_runs(w)
        days = max((df.index[-1] - df.index[0]).days, 1)
        valid_count = int(both.sum())
        missing_count = int((~both).sum())
        missing_pct = 100.0 * float((~both).mean())
        longest_missing_h = float(max([l for _, l in runs], default=0) / 60)
        off_share = 100.0 * float((w[both] <= C.ON_THRESHOLD_W).mean()) if both.any() else np.nan
        active_share = 100.0 * float((w[both] > C.ON_THRESHOLD_W).mean()) if both.any() else np.nan

        rows.append(dict(
            house=h, channel=col, first=str(df.index[0].date()), last=str(df.index[-1].date()),
            span_days=days, valid_observations=valid_count, missing_minutes=missing_count,
            missing_pct=missing_pct, longest_missing_h=longest_missing_h,
            off_time_pct=off_share, active_time_pct=active_share,
            cycles_detected=len(cyc), valid_cycles=len(good),
            outage_intervals=len(runs),
            median_duration_min=float(good.duration_min.median()) if len(good) else np.nan,
            median_energy_wh=float(good.energy_wh.median()) if len(good) else np.nan,
            median_peak_w=float(good.peak_w.median()) if len(good) else np.nan))
    tab = pd.DataFrame(rows)
    save_table(tab, "household_availability")
    save_table(tab, "washer_availability")
    
    if all_cycles_list:
        cycle_summary_df = pd.concat(all_cycles_list, ignore_index=True)
        save_table(cycle_summary_df, "cycle_summary")
    return tab, cycles


def patterns(data, cycles):
    """Hour-of-day and day-of-week usage (cycle starts per observed day, local time)."""
    hour_share, dow_rate = {}, {}
    peak_rows = []
    for h, df in data.items():
        col = C.WASHER_CHANNEL[h]
        both = df[col].notna() & df["Aggregate"].notna()
        loc = to_local(df.index)
        day = pd.Series(loc.normalize(), index=df.index)
        day_ok = both.groupby(day.values).mean() >= 0.8          # days with >=80% coverage
        c = cycles[h]
        c = c[~c.touches_gap]
        ls = to_local(pd.DatetimeIndex(c.start))
        c = c.assign(hour=ls.hour, dow=ls.dayofweek, day=ls.normalize())
        c = c[c.day.map(day_ok).fillna(False).astype(bool)]
        hs = c.hour.value_counts().reindex(range(24), fill_value=0)
        hour_share[h] = hs / max(hs.sum(), 1)
        n_days = pd.Series(day_ok[day_ok].index.dayofweek).value_counts().reindex(range(7), fill_value=0)
        dw = c.dow.value_counts().reindex(range(7), fill_value=0)
        dow_rate[h] = dw / n_days.replace(0, np.nan)
        # share of washer energy in the evening peak
        p = df[col].where(both, 0).fillna(0)
        hh = pd.Series(loc.hour, index=df.index)
        in_peak = (hh >= C.PEAK_HOURS[0]) & (hh < C.PEAK_HOURS[1])
        peak_rows.append(dict(house=h, peak_window=f"{C.PEAK_HOURS[0]}-{C.PEAK_HOURS[1]}h",
                              washer_kwh=p.sum() / 60000, peak_share_pct=100 * p[in_peak].sum() / max(p.sum(), 1e-9)))
    save_table(pd.DataFrame(peak_rows), "peak_share")

    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for h in hour_share:
        ax[0].plot(range(24), 100 * hour_share[h], marker=".", label=f"H{h}")
        ax[1].plot(range(7), dow_rate[h].values, marker="o", label=f"H{h}")
    ax[0].set(title="Cycle starts by hour of day (local)", xlabel="hour", ylabel="% of a house's cycles")
    ax[1].set(title="Cycles per observed day, by weekday", ylabel="cycles / day", xticks=range(7), xticklabels=DOW)
    ax[0].legend(ncol=2, fontsize=7)
    heat = pd.DataFrame({h: hour_share[h] for h in hour_share}).mean(axis=1)
    ax[2].bar(range(24), 100 * heat, color="tab:blue")
    ax[2].set(title="Average across houses: share of cycles by hour", xlabel="hour", ylabel="%")
    fig.tight_layout()
    fig.savefig(C.FIG / "03_usage_patterns.png", dpi=130)
    plt.close(fig)


def representative_day(data, cycles):
    best = None
    for h, df in data.items():
        col = C.WASHER_CHANNEL[h]
        ok = df[col].notna() & df["Aggregate"].notna()
        cov = ok.groupby(df.index.normalize()).mean()
        e = df[col].fillna(0).groupby(df.index.normalize()).sum() / 60
        cand = e[(cov >= 0.95) & (e > 0)]
        if len(cand):
            d = cand.idxmax()
            if best is None or cand.max() > best[2]:
                best = (h, d, cand.max())
    if best is None:
        return
    h, d, _ = best
    df, col = data[h], C.WASHER_CHANNEL[h]
    day = df.loc[d:d + pd.Timedelta(hours=23, minutes=59)]
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    ax[0].plot(day.index, day["Aggregate"], lw=0.8, color="tab:blue")
    ax[0].set_ylabel("Aggregate (W)")
    ax[1].plot(day.index, day[col], lw=0.9, color="tab:orange")
    ax[1].axhline(C.ON_THRESHOLD_W, ls="--", lw=0.8, color="grey", label=f"ON threshold ({C.ON_THRESHOLD_W} W)")
    ax[1].set_ylabel("Washing machine (W)")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"House {h}, {d.date()}: 24-Hour Representative Trace (Aggregate vs Washing Machine)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(C.FIG / "02_representative_day.png", dpi=130)
    fig.savefig(C.FIG / "02_daily_aggregate_washer.png", dpi=130)
    plt.close(fig)

    # detected cycles (small multiples), same house
    c = cycles[h]
    c = c[~c.touches_gap]
    pick = c.iloc[np.linspace(0, len(c) - 1, min(6, len(c))).astype(int)]
    fig, ax = plt.subplots(2, 3, figsize=(14, 6), squeeze=False)
    for a, (_, r) in zip(ax.ravel(), pick.iterrows()):
        lo, hi = max(r.i0 - 20, 0), min(r.i1 + 20, len(df))
        seg = df[col].iloc[lo:hi]
        a.plot(range(lo - r.i0, hi - r.i0), seg.values, lw=0.9)
        a.axvspan(0, r.duration_min, color="tab:green", alpha=0.18, label="Active cycle")
        a.axhline(C.ON_THRESHOLD_W, ls="--", lw=0.7, color="grey")
        a.set_title(f"{r.start:%Y-%m-%d %H:%M}\n{r.duration_min:.0f} min, {r.energy_wh:.0f} Wh, peak {r.peak_w:.0f} W", fontsize=8)
        a.set_xlabel("minutes from detected start")
        a.set_ylabel("Power (W)")
    fig.suptitle(f"House {h}: Individual Detected Washing-Machine Cycles (green = active span)", y=1.0)
    fig.tight_layout()
    fig.savefig(C.FIG / "02b_detected_cycles.png", dpi=130)
    fig.savefig(C.FIG / "03_detected_cycles.png", dpi=130)
    plt.close(fig)


def cross_house(cycles):
    rows = []
    for h, c in cycles.items():
        for _, r in c[~c.touches_gap].iterrows():
            rows.append((h, r.duration_min, r.energy_wh, r.peak_w))
    d = pd.DataFrame(rows, columns=["house", "duration_min", "energy_wh", "peak_w"])
    hs = sorted(d.house.unique())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, lab in zip(ax, ["energy_wh", "duration_min", "peak_w"], ["Energy per cycle (Wh)", "Duration (min)", "Peak power (W)"]):
        a.boxplot([d.loc[d.house == h, col] for h in hs], showfliers=False)
        a.set_xticks(range(1, len(hs) + 1), [f"H{h}" for h in hs])
        a.set_title(lab)
        a.set_ylabel(lab)
    fig.suptitle("Cycle characteristics across houses (outliers hidden)")
    fig.tight_layout()
    fig.savefig(C.FIG / "04_cross_house.png", dpi=130)
    fig.savefig(C.FIG / "04_cross_house_comparison.png", dpi=130)
    plt.close(fig)
    save_table(d.groupby("house").agg(cycles=("energy_wh", "size"), energy_p25=("energy_wh", lambda x: x.quantile(.25)),
                                      energy_med=("energy_wh", "median"), energy_p75=("energy_wh", lambda x: x.quantile(.75)),
                                      dur_med=("duration_min", "median"), peak_med=("peak_w", "median")).reset_index(),
               "cycle_stats_by_house")


def concurrency(data, cycles):
    """How often is another heavy appliance running while the washer is on?"""
    rows = []
    for h, df in data.items():
        wc = C.WASHER_CHANNEL[h]
        others = [c for c in df.columns if c.startswith("Appliance") and c != wc]
        heavy = (df[others] > C.HEAVY_APPLIANCE_W).any(axis=1)
        on = df[wc] > C.ON_THRESHOLD_W
        rows.append(dict(house=h, washer_on_min=int(on.sum()),
                         overlap_with_other_heavy_pct=100 * float(heavy[on].mean()) if on.any() else np.nan))
    save_table(pd.DataFrame(rows), "concurrency")


def run(houses=None, force=False):
    C.FIG.mkdir(parents=True, exist_ok=True)
    data, _ = load_study(houses, force)
    if not data:
        raise SystemExit("No usable houses found. Check data/clean and config.WASHER_CHANNEL.")
    tab, cycles = availability(data)
    patterns(data, cycles)
    representative_day(data, cycles)
    cross_house(cycles)
    concurrency(data, cycles)
    log.info("Section 2 done for houses %s", list(data))
    return data, cycles


if __name__ == "__main__":
    run()
