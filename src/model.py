"""Section 3: washing-machine disaggregation from the aggregate signal (1-minute).

Design in one paragraph
-----------------------
Leave-one-house-out (LOHO): each house is the TEST set once, a model is trained on the other
houses only. Inside the training houses the LAST 20% of each timeline is held out as a validation
set (time-ordered, not random) to pick hyper-parameters and the on/off "gate". The test house is
never seen in training, validation or tuning. Features come from the AGGREGATE signal only; no
other appliance channel is ever used as an input.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .eda_washer import load_study
from .events import detect_cycles, runs_from_bool
from .io_utils import to_local
from .utils import log, save_table


# ------------------------------------------------------------------ features
def build_features(agg: pd.Series) -> pd.DataFrame:
    """Centred-window features (offline disaggregation: past AND future minutes are used,
    like Seq2Point). Always computed from the aggregate only."""
    a = agg.astype("float32")
    f = {"agg": a}
    floor = a.rolling(61, center=True, min_periods=15).min()
    f["floor"] = floor
    f["excess"] = a - floor
    for w in (5, 15, 30, 61):
        r = a.rolling(w, center=True, min_periods=max(2, w // 3))
        f[f"mean{w}"], f[f"std{w}"], f[f"max{w}"] = r.mean(), r.std(), r.max()
    for k in (1, 2, 3, 5, 10, 20, 30):
        f[f"lead{k}"] = a.shift(-k) - a
        f[f"lag{k}"] = a.shift(k) - a
    f["d1"] = a.diff()
    loc = to_local(a.index)
    hr = loc.hour + loc.minute / 60
    f["hour_sin"] = pd.Series(np.sin(2 * np.pi * hr / 24), index=a.index, dtype="float32")
    f["hour_cos"] = pd.Series(np.cos(2 * np.pi * hr / 24), index=a.index, dtype="float32")
    f["dow"] = pd.Series(loc.dayofweek, index=a.index, dtype="float32")
    return pd.DataFrame(f).astype("float32")


# ------------------------------------------------------------------ metrics
def minute_metrics(y, p, on_thr):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    on_t, on_p = y > C.ON_THRESHOLD_W, p > on_thr
    tp = int((on_t & on_p).sum())
    prec = tp / max(int(on_p.sum()), 1)
    rec = tp / max(int(on_t.sum()), 1)
    y_sum = max(y.sum(), 1e-9)
    p_sum = p.sum()
    return dict(
        MAE_W=float(np.abs(err).mean()),
        RMSE_W=float(np.sqrt((err ** 2).mean())),
        SAE=float(abs(p_sum - y_sum) / y_sum),                         # relative total-energy error
        energy_bias_pct=float(100.0 * (p_sum - y_sum) / y_sum),         # signed energy percentage bias
        abs_energy_err_pct=float(100.0 * abs(p_sum - y_sum) / y_sum),  # absolute energy percentage error
        NDE=float((err ** 2).sum() / max((y ** 2).sum(), 1e-9)),        # 1.0 == always-off
        precision=prec, recall=rec,
        F1=float(2 * prec * rec / max(prec + rec, 1e-9)))


def cycle_scores(y_full, p_full, pred_on, idx):
    """Cycle-level view on a full-length (NaN where invalid) series."""
    true = detect_cycles(pd.Series(y_full, index=idx))
    true = true[~true.touches_gap]
    true_on = np.nan_to_num(y_full, nan=0.0) > C.ON_THRESHOLD_W
    pred_runs = runs_from_bool(pred_on)
    recalled, e_true, e_pred = 0, [], []
    for _, r in true.iterrows():
        if pred_on[r.i0:r.i1].mean() >= 0.5:
            recalled += 1
        e_true.append(r.energy_wh)
        e_pred.append(float(np.nansum(p_full[r.i0:r.i1]) / 60))
    false_c = sum(1 for s, e in pred_runs if true_on[s:e].sum() == 0)
    per = true[["start", "duration_min", "energy_wh"]].copy()
    per["pred_energy_wh"] = e_pred
    return dict(true_cycles=len(true), cycle_recall=recalled / max(len(true), 1),
                false_cycles=false_c, pred_cycles=len(pred_runs)), per


# ------------------------------------------------------------------ data
def prepare(data, agg_transform=None):
    out = {}
    for h, df in data.items():
        agg = df["Aggregate"]
        if agg_transform is not None:
            agg = agg_transform(agg)
        y = df[C.WASHER_CHANNEL[h]]
        out[h] = dict(X=build_features(agg), y=y.to_numpy(float), idx=df.index,
                      valid=(agg.notna() & y.notna()).to_numpy())
    return out


def sample_training(X, y, rng):
    """All ON minutes + a fraction of OFF minutes; OFF minutes up-weighted to keep the prior."""
    on = y > C.ON_THRESHOLD_W
    keep = on | (rng.random(len(y)) < C.OFF_KEEP_FRAC)
    w = np.where(on, 1.0, 1.0 / C.OFF_KEEP_FRAC)
    sel = np.flatnonzero(keep)
    if len(sel) > C.MAX_TRAIN_ROWS:                      # random cap; weights keep the ON/OFF balance
        sel = np.sort(rng.choice(sel, C.MAX_TRAIN_ROWS, replace=False))
    return X[sel], y[sel], w[sel]


def stack(prep, houses, part=None):
    Xs, ys = [], []
    for h in houses:
        pos = np.flatnonzero(prep[h]["valid"])
        cut = int((1 - C.VAL_FRAC) * len(pos))
        sel = pos if part is None else (pos[:cut] if part == "train" else pos[cut:])
        Xs.append(prep[h]["X"].to_numpy()[sel])
        ys.append(prep[h]["y"][sel])
    return np.vstack(Xs), np.concatenate(ys)


def fit(X, y, cfg, seed):
    rng = np.random.default_rng(seed)
    Xs, ys, w = sample_training(X, y, rng)
    m = HistGradientBoostingRegressor(random_state=seed, **cfg)
    m.fit(Xs, ys, sample_weight=w)
    return m


def choose_gate(y, p):
    best, best_f1 = C.GATE_GRID_W[0], -1
    for g in C.GATE_GRID_W:
        f1 = minute_metrics(y, np.where(p > g, p, 0), 0)["F1"]
        if f1 > best_f1:
            best, best_f1 = g, f1
    return best


# ------------------------------------------------------------------ CV
def run_cv(data, agg_transform=None, tune=True, tag="main"):
    prep = prepare(data, agg_transform)
    houses = list(prep)
    rows, preds, cyc_tables, choices = [], {}, [], []
    for test_h in houses:
        train_hs = [h for h in houses if h != test_h]
        # ---- tune on a time-ordered validation block of the training houses
        Xtr, ytr = stack(prep, train_hs, "train")
        Xva, yva = stack(prep, train_hs, "val")
        cfgs = C.HGB_CONFIGS if tune else {"A": C.HGB_CONFIGS["A"]}
        best = None
        for name, cfg in cfgs.items():
            m = fit(Xtr, ytr, cfg, C.SEED)
            pv = np.clip(m.predict(Xva), 0, None)
            rmse = minute_metrics(yva, pv, C.ON_THRESHOLD_W)["RMSE_W"]
            if best is None or rmse < best[0]:
                best = (rmse, name, pv)
        _, cname, pv = best
        gate = choose_gate(yva, pv)
        choices.append(dict(test_house=test_h, config=cname, gate_W=gate))
        # ---- refit on ALL training-house data with the chosen config, score on the unseen house
        Xall, yall = stack(prep, train_hs, None)
        model = fit(Xall, yall, C.HGB_CONFIGS[cname], C.SEED)
        d = prep[test_h]
        v = d["valid"]
        p_raw = np.full(len(v), np.nan)
        p_raw[v] = np.clip(model.predict(d["X"].to_numpy()[v]), 0, None)
        p_gate = np.where(p_raw > gate, p_raw, 0.0)
        p_gate[~v] = np.nan
        # baselines (fit on training houses only)
        trx = pd.concat([pd.Series(prep[h]["y"][prep[h]["valid"]],
                                   index=to_local(prep[h]["idx"][prep[h]["valid"]]).hour) for h in train_hs])
        prof = trx.groupby(level=0).mean().reindex(range(24), fill_value=0)
        hours = to_local(d["idx"]).hour
        p_prof = np.where(v, prof.reindex(hours).to_numpy(), np.nan)
        p_off = np.where(v, 0.0, np.nan)
        y_full = np.where(v, d["y"], np.nan)
        # ---- test-size stats (context for readers)
        valid_test_min = int(v.sum())
        test_cyc = detect_cycles(pd.Series(y_full, index=d["idx"]))
        test_cyc = test_cyc[~test_cyc.touches_gap]
        n_test_cycles = len(test_cyc)
        washer_energy = float(np.nansum(y_full[v]))
        agg_energy = float(np.nansum(data[test_h]["Aggregate"].to_numpy(float)[v]))
        washer_share_pct = 100.0 * washer_energy / max(agg_energy, 1e-9)
        test_size = dict(valid_test_min=valid_test_min,
                         test_cycles=n_test_cycles,
                         washer_energy_share_pct=round(washer_share_pct, 2))
        models = {"always_off": (p_off, C.ON_THRESHOLD_W), "hourly_profile": (p_prof, C.ON_THRESHOLD_W),
                  "gbm_raw": (p_raw, C.ON_THRESHOLD_W), "gbm_gated": (p_gate, 0.0)}
        preds[test_h] = dict(y=y_full, idx=d["idx"], **{k: pv_[0] for k, pv_ in models.items()})
        for name, (pp, thr) in models.items():
            m_ = minute_metrics(y_full[v], pp[v], thr)
            c_, per = cycle_scores(y_full, pp, np.nan_to_num(pp, nan=0.0) > thr, d["idx"])
            rows.append(dict(test_house=test_h, model=name, **m_, **c_, **test_size))
            if name == "gbm_gated":
                cyc_tables.append(per.assign(house=test_h))
        log.info("[%s] test house %s done (cfg %s, gate %s W)", tag, test_h, cname, gate)
    res = pd.DataFrame(rows)
    return res, preds, pd.concat(cyc_tables, ignore_index=True), pd.DataFrame(choices)


# ------------------------------------------------------------------ outputs
def summarise(res):
    cols = ["MAE_W", "RMSE_W", "SAE", "NDE", "precision", "recall", "F1", "cycle_recall", "false_cycles"]
    g = res.groupby("model")[cols]
    mean, std = g.mean(), g.std()
    out = pd.DataFrame({"model": mean.index})
    for c in cols:
        out[c] = [f"{mean.loc[m, c]:.3f} ± {std.loc[m, c]:.3f}" for m in mean.index]
    order = ["always_off", "hourly_profile", "gbm_raw", "gbm_gated"]
    return out.set_index("model").loc[[m for m in order if m in set(out.model)]].reset_index()


def plot_main(preds, res):
    sub = res[res.model == "gbm_gated"].sort_values("NDE")
    h = int(sub.iloc[len(sub) // 2].test_house)               # median-NDE house: not cherry-picked
    d = preds[h]
    y = np.nan_to_num(d["y"])
    cyc = detect_cycles(pd.Series(d["y"], index=d["idx"]))
    cyc = cyc[~cyc.touches_gap]
    r = cyc.loc[cyc.energy_wh.idxmax()]
    lo, hi = max(int(r.i0) - 90, 0), min(int(r.i1) + 90, len(y))
    t = d["idx"][lo:hi]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    agg = None
    ax[1].plot(t, d["y"][lo:hi], color="black", lw=1.4, label="actual washer")
    ax[1].plot(t, d["hourly_profile"][lo:hi], color="tab:green", lw=1, label="hourly-profile baseline")
    ax[1].plot(t, d["gbm_gated"][lo:hi], color="tab:red", lw=1.2, label="model (gated)")
    ax[1].set_ylabel("Washer power (W)")
    ax[1].legend(loc="upper right", fontsize=8)
    d_agg = d.get("agg")
    if d_agg is not None:
        ax[0].plot(t, d_agg[lo:hi], color="tab:blue", lw=0.9)
    ax[0].set_ylabel("Aggregate (W)")
    ax[0].set_title(f"Test house {h} (never seen in training): the highest-energy cycle, ±90 min")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(C.FIG / "05_actual_vs_predicted.png", dpi=130)
    plt.close(fig)

    # one window per test house
    hs = list(preds)
    fig, ax = plt.subplots(int(np.ceil(len(hs) / 2)), 2, figsize=(14, 2.6 * int(np.ceil(len(hs) / 2))), squeeze=False)
    for a, hh in zip(ax.ravel(), hs):
        dd = preds[hh]
        cy = detect_cycles(pd.Series(dd["y"], index=dd["idx"]))
        cy = cy[~cy.touches_gap]
        rr = cy.loc[cy.energy_wh.idxmax()]
        l, u = max(int(rr.i0) - 45, 0), min(int(rr.i1) + 45, len(dd["y"]))
        a.plot(range(l, u), dd["y"][l:u], "k", lw=1.2)
        a.plot(range(l, u), dd["gbm_gated"][l:u], "r", lw=1)
        a.set_title(f"House {hh}", fontsize=9)
    fig.suptitle("Actual (black) vs model (red), highest-energy cycle in each test house")
    fig.tight_layout()
    fig.savefig(C.FIG / "05b_actual_vs_predicted_all_houses.png", dpi=110)
    plt.close(fig)


def plot_metrics(res):
    p = res.pivot(index="test_house", columns="model", values="NDE")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    p[[c for c in ["always_off", "hourly_profile", "gbm_raw", "gbm_gated"] if c in p]].plot.bar(ax=ax[0])
    ax[0].set(title="NDE per test house (lower is better; 1.0 = same as always-off)", xlabel="test house")
    ax[0].axhline(1, color="grey", ls="--", lw=0.8)
    q = res.pivot(index="test_house", columns="model", values="F1")
    q[[c for c in ["hourly_profile", "gbm_raw", "gbm_gated"] if c in q]].plot.bar(ax=ax[1])
    ax[1].set(title="Minute-level ON/OFF F1 per test house (higher is better)", xlabel="test house")
    fig.tight_layout()
    fig.savefig(C.FIG / "06_metrics_by_house.png", dpi=130)
    plt.close(fig)


def failure_analysis(preds, per_cycle, data):
    pc = per_cycle.copy()
    pc["err_wh"] = pc.pred_energy_wh - pc.energy_wh
    pc["abs_err_wh"] = pc.err_wh.abs()
    # scatter of cycle energy
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pc.energy_wh, pc.pred_energy_wh, s=8, alpha=0.5)
    lim = [0, max(pc.energy_wh.max(), pc.pred_energy_wh.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xlabel="actual energy per cycle (Wh)", ylabel="predicted (Wh)", title="Cycle energy: predicted vs actual (all test houses)")
    fig.tight_layout()
    fig.savefig(C.FIG / "07_cycle_energy_scatter.png", dpi=130)
    plt.close(fig)
    save_table(pd.DataFrame(dict(
        cycles=[len(pc)], median_abs_err_wh=[pc.abs_err_wh.median()],
        median_pct_err=[float((pc.abs_err_wh / pc.energy_wh).median() * 100)],
        p90_pct_err=[float((pc.abs_err_wh / pc.energy_wh).quantile(0.9) * 100)])), "cycle_energy_error")

    worst = pc.sort_values("abs_err_wh", ascending=False).head(2)
    rows = []
    fig, ax = plt.subplots(len(worst), 1, figsize=(12, 3.8 * len(worst)), squeeze=False)
    for a, (_, r) in zip(ax[:, 0], worst.iterrows()):
        h = int(r.house)
        d, df = preds[h], data[h]
        pos = df.index.get_indexer([r.start])[0]
        lo, hi = max(pos - 60, 0), min(pos + int(r.duration_min) + 60, len(df))
        wc = C.WASHER_CHANNEL[h]
        others = [c for c in df.columns if c.startswith("Appliance") and c != wc]
        heavy = (df[others].iloc[lo:hi] > C.HEAVY_APPLIANCE_W).any(axis=1)
        t = df.index[lo:hi]
        a.plot(t, df["Aggregate"].iloc[lo:hi], color="tab:blue", lw=0.8, label="aggregate")
        a.plot(t, d["y"][lo:hi], "k", lw=1.2, label="actual washer")
        a.plot(t, d["gbm_gated"][lo:hi], "r", lw=1, label="model")
        a.fill_between(t, 0, 1, where=heavy.to_numpy(), transform=a.get_xaxis_transform(), color="orange", alpha=0.25, lw=0,
                       label="another heavy appliance on")
        a.set_title(f"House {h}, {r.start:%Y-%m-%d %H:%M}: actual {r.energy_wh:.0f} Wh vs predicted {r.pred_energy_wh:.0f} Wh", fontsize=9)
        a.legend(fontsize=7, loc="upper right")
        rows.append(dict(house=h, start=str(r.start), actual_wh=r.energy_wh, predicted_wh=r.pred_energy_wh,
                         other_heavy_minutes=int(heavy.sum())))
    fig.tight_layout()
    fig.savefig(C.FIG / "08_failure_cases.png", dpi=130)
    plt.close(fig)
    save_table(pd.DataFrame(rows), "failure_cases")


def split_design(data, prep_valid=None):
    rows = []
    for h, df in data.items():
        ok = df[C.WASHER_CHANNEL[h]].notna() & df["Aggregate"].notna()
        t = df.index[ok.to_numpy()]
        cut = t[int((1 - C.VAL_FRAC) * len(t))]
        rows.append(dict(house=h, data_from=str(t[0].date()), data_to=str(t[-1].date()),
                         validation_block_starts=str(cut.date()), role="test (once, as held-out house) / train (other folds)"))
    save_table(pd.DataFrame(rows), "split_design")


def resolution_study(data):
    out = []
    for k in C.RESOLUTIONS_MIN:
        if k == 1:
            tf = None
        else:
            def tf(a, k=k):
                return a.groupby(a.index.floor(f"{k}min")).transform("mean")
        res, _, _, _ = run_cv(data, agg_transform=tf, tune=False, tag=f"res{k}")
        g = res[res.model == "gbm_gated"][["NDE", "SAE", "F1", "cycle_recall"]].mean()
        base = res[res.model == "hourly_profile"][["NDE"]].mean()
        out.append(dict(aggregate_resolution_min=k, NDE_model=g.NDE, NDE_hourly_profile=base.NDE,
                        SAE=g.SAE, F1=g.F1, cycle_recall=g.cycle_recall))
    t = pd.DataFrame(out)
    save_table(t, "resolution_study", nd=3)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(t.aggregate_resolution_min, t.NDE_model, "o-", label="model")
    ax.plot(t.aggregate_resolution_min, t.NDE_hourly_profile, "s--", label="hourly-profile baseline")
    ax.set(xlabel="aggregate resolution (min)", ylabel="mean NDE across test houses", title="What coarser meter data does to the model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "09_resolution_study.png", dpi=130)
    plt.close(fig)


def run(houses=None, with_resolution=True):
    C.FIG.mkdir(parents=True, exist_ok=True)
    data, _ = load_study(houses)
    if len(data) < 2:
        raise SystemExit("Need at least 2 usable houses for leave-one-house-out.")
    split_design(data)
    res, preds, per_cycle, choices = run_cv(data, tune=True)
    for h in preds:                       # keep aggregate for the plots
        preds[h]["agg"] = data[h]["Aggregate"].to_numpy(float)
    save_table(res, "metrics_per_house", nd=3)
    save_table(res, "model_metrics", nd=3)
    save_table(summarise(res), "metrics_summary")
    save_table(choices, "tuning_choices")
    # ---- test-size context table (one row per test house)
    size_cols = ["test_house", "valid_test_min", "test_cycles", "washer_energy_share_pct"]
    ts = res[res.model == "gbm_gated"][size_cols].copy()
    ts = ts.rename(columns={"test_house": "house"})
    save_table(ts, "test_sizes", nd=2)
    plot_main(preds, res)
    plot_metrics(res)
    failure_analysis(preds, per_cycle, data)
    if with_resolution:
        resolution_study(data)
    (C.RESULTS / "metrics.json").write_text(json.dumps(res.to_dict(orient="records"), default=str, indent=1))
    log.info("Section 3 done.")
    return res


if __name__ == "__main__":
    run()
