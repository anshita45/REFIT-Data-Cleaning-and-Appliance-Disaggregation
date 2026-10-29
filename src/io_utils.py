import re
from pathlib import Path

import pandas as pd

from . import config as C


def find_house_files(folder, house: int) -> list[Path]:
    """Find all CSV parts for a given house (e.g. Part1 and Part2)."""
    pat = re.compile(rf"house[_\s-]*0*{house}(?!\d)", re.I)
    cands = sorted(p for p in Path(folder).rglob("*.csv") if pat.search(p.stem))
    if not cands:
        raise FileNotFoundError(
            f"No CSV for house {house} under {folder}. Did you unzip the archive there? "
            f"(see README, 'Get the data')")
    return cands


def find_house_file(folder, house: int) -> Path:
    """Find primary/first CSV for a given house."""
    return find_house_files(folder, house)[0]


def read_house(path, channels=None) -> pd.DataFrame:
    """Read one or more REFIT CSVs (e.g. multi-part raw files). Returns DataFrame indexed by parsed timestamp,
    FILE ORDER KEPT (so ordering problems stay visible). Unparsable timestamps become NaT."""
    if isinstance(path, (list, tuple)):
        dfs = [read_house(p, channels) for p in path]
        return pd.concat(dfs, ignore_index=False)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    tcol = "Unix" if "Unix" in header else "Time"
    if tcol not in header:
        raise KeyError(f"{Path(path).name}: no 'Time' or 'Unix' column. Header: {header}")
    want = [c for c in header if c not in ("Time", "Unix")] if channels is None else list(channels)
    missing = [c for c in want if c not in header]
    if missing:
        raise KeyError(f"{Path(path).name}: columns {missing} not found. Header: {header}")
    df = pd.read_csv(path, usecols=[tcol] + want)
    if tcol == "Unix":
        idx = pd.to_datetime(pd.to_numeric(df[tcol], errors="coerce"), unit="s")
    else:
        idx = pd.to_datetime(df[tcol], errors="coerce")
    df = df.drop(columns=[tcol]).apply(pd.to_numeric, errors="coerce")
    df.index = pd.DatetimeIndex(idx, name="time")
    return df


def to_local(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive timestamps -> naive household wall-clock time (see config.ASSUME_UTC)."""
    if not C.ASSUME_UTC:
        return idx
    return idx.tz_localize("UTC").tz_convert(C.LOCAL_TZ).tz_localize(None)
