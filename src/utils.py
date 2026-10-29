import logging
import numpy as np
import pandas as pd

from . import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("refit")


def _fmt(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, nd: int = 2) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for tup in df.astype(object).itertuples(index=False, name=None):
        out.append("| " + " | ".join(_fmt(v, nd) for v in tup) + " |")
    return "\n".join(out) + "\n"


def save_table(df: pd.DataFrame, name: str, nd: int = 2):
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (C.TABLES / f"{name}.md").write_text(md_table(df, nd))
    df.to_csv(C.TABLES / f"{name}.csv", index=False)


def nan_runs(s: pd.Series):
    """Return list of (start_pos, length) for each run of NaN."""
    na = s.isna().to_numpy()
    if not na.any():
        return []
    d = np.diff(np.r_[0, na.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts, ends - starts))


def fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Linear-interpolate NaN runs of <= max_gap samples. Longer runs stay NaN."""
    out = s.copy()
    interp = s.interpolate(method="linear", limit_area="inside")
    for start, length in nan_runs(s):
        if length <= max_gap and start > 0 and start + length < len(s):
            out.iloc[start:start + length] = interp.iloc[start:start + length]
    return out
