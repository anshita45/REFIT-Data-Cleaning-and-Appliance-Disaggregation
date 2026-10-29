"""Run the whole pipeline:  python -m src.run_all  [--houses 1 2 3 5 6 7 8 9] [--skip-resolution] [--force]"""
import argparse

from . import clean_raw, eda_washer, model, render_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, nargs="*", help="override config.STUDY_HOUSES")
    ap.add_argument("--skip-resolution", action="store_true", help="skip the 5/15/30-minute study (slowest part)")
    ap.add_argument("--force", action="store_true", help="rebuild the 1-minute cache from the CSVs")
    a = ap.parse_args()
    clean_raw.run()
    eda_washer.run(a.houses, force=a.force)
    model.run(a.houses, with_resolution=not a.skip_resolution)
    render_docs.run()


if __name__ == "__main__":
    main()
