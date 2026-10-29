"""Download and unpack the REFIT archives into data/raw and data/clean.
    python -m src.prepare_data            # both
    python -m src.prepare_data --clean    # only the cleaned data (smaller)
Needs `pip install py7zr`. The archives are large (several GB unpacked); you can also download and
extract them by hand from the links in the README."""
import argparse
import urllib.request

from . import config as C

URLS = {
    "raw": "https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z",
    "clean": "https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z",
}


def get(kind):
    import py7zr
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    arc = C.DATA_DIR / f"{kind}.7z"
    if not arc.exists():
        print("downloading", URLS[kind])
        urllib.request.urlretrieve(URLS[kind], arc)
    out = C.RAW_DIR if kind == "raw" else C.CLEAN_DIR
    out.mkdir(parents=True, exist_ok=True)
    print("extracting to", out)
    with py7zr.SevenZipFile(arc) as z:
        z.extractall(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    for k in (["raw"] if a.raw and not a.clean else ["clean"] if a.clean and not a.raw else ["raw", "clean"]):
        get(k)
