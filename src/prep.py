"""Official CLEAN REFIT files -> 1-minute DataFrame (cached)."""
import json

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import find_house_file, read_house
from .utils import fill_short_gaps, log


def process_clean_house(h: int) -> pd.DataFrame:
    path = find_house_file(C.CLEAN_DIR, h)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    chans = [c for c in header if c.startswith("Aggregate") or c.startswith("Appliance")]
    use = chans + (["Issues"] if "Issues" in header else [])
    df = read_house(path, use)
    n_rows = len(df)
    df = df[df.index.notna()]

    n_issue = 0
    if "Issues" in df.columns:
        bad = df.pop("Issues").fillna(0) == 1
        n_issue = int(bad.sum())
        if C.MASK_ISSUES:
            df.loc[bad, :] = np.nan

    n_dup = int(df.index.duplicated().sum())
    df = df.groupby(level=0).mean().sort_index()           # duplicate rule: mean
    m = df.resample(C.RESAMPLE).mean()                      # NaN where a minute has no reading
    for c in m.columns:
        m[c] = fill_short_gaps(m[c], C.MAX_FILL_GAP_MIN)
    m = m.astype("float32")

    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    m.to_pickle(C.PROC_DIR / f"house{h}_1min.pkl")
    (C.PROC_DIR / f"house{h}_meta.json").write_text(json.dumps(dict(
        file=path.name, raw_rows=n_rows, duplicate_timestamps=n_dup,
        issue_rows=n_issue, masked_issue_rows=n_issue if C.MASK_ISSUES else 0)))
    log.info("house %s: %s rows -> %s one-minute rows (%s issue rows)", h, n_rows, len(m), n_issue)
    return m


def load_minute(h: int, force: bool = False) -> pd.DataFrame:
    p = C.PROC_DIR / f"house{h}_1min.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    return process_clean_house(h)
