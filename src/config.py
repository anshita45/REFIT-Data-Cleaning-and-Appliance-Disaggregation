"""Central settings. Change values here, not inside the scripts.

Paths can be redirected with environment variables (used by the synthetic tests):
    REFIT_DATA_DIR, REFIT_RESULTS_DIR
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("REFIT_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # unzipped REFIT_RAW_081116.7z goes here
CLEAN_DIR = DATA_DIR / "clean"      # unzipped CLEAN_REFIT_081116.7z goes here
PROC_DIR = DATA_DIR / "processed"   # 1-minute cache written by the pipeline
RESULTS = Path(os.environ.get("REFIT_RESULTS_DIR", ROOT / "results"))
FIG = RESULTS / "figures"
TABLES = RESULTS / "tables"

# ---------------------------------------------------------------- cleaning
RESAMPLE = "1min"
MAX_FILL_GAP_MIN = 2        # conservative: fill at most 2 consecutive missing 1-min bins; longer = outage
MAX_POWER_W = 25_000        # impossible for a UK home (100 A fuse x 230 V = 23 kW)
MAX_APPLIANCE_W = 4_000     # maximum plausible single-appliance plug load (Murray et al. 2017)
MASK_ISSUES = True          # cleaned files: drop rows where the official 'Issues' flag == 1
RAW_HOUSE = 1
RAW_PLOT_CHANNELS = ["Aggregate", "Appliance1", "Appliance5"]  # house 1: fridge, washer

# Timestamps: REFIT documentation says GMT. Hour-of-day analysis should follow the
# household's wall clock, so we convert UTC -> Europe/London. Set False if you find
# the timestamps are already local time.
ASSUME_UTC = True
LOCAL_TZ = "Europe/London"

# ---------------------------------------------------------------- washing machine
# Washer channel per house. Verified against CLEAN_READ_ME_081116.txt and Table 3.
# House 4 has two washers, so it is left out on purpose.
# In House 9: Appliance3 is the washing machine, Appliance2 is the washer-dryer.
WASHER_CHANNEL = {
    1: "Appliance5", 2: "Appliance2", 3: "Appliance6", 5: "Appliance3",
    6: "Appliance2", 7: "Appliance5", 8: "Appliance4", 9: "Appliance3",
    10: "Appliance5",
}
STUDY_HOUSES = [1, 2, 3, 5, 6, 7, 8, 9]

# Cycle rule (plain language): a cycle starts when washer power goes above
# ON_THRESHOLD_W, short pauses (<= MERGE_GAP_MIN minutes at/below threshold, e.g. soak
# or drain phases) are bridged, and anything shorter than MIN_CYCLE_MIN or using
# less than MIN_CYCLE_WH is ignored (door-sensor blips, standby noise).
ON_THRESHOLD_W = 20
MERGE_GAP_MIN = 5
MIN_CYCLE_MIN = 10
MIN_CYCLE_WH = 20
HEAVY_APPLIANCE_W = 500     # other appliance counts as "heavy" above this (concurrency analysis)
PEAK_HOURS = (17, 21)       # local hours used for the peak-time share (Section 4)

# Sanity check for a washer channel
MIN_PEAK_W = 300
MIN_CYCLES = 20
MIN_HOUSES_REQUIRED = 4

# ---------------------------------------------------------------- model
SEED = 42
OFF_KEEP_FRAC = 0.25        # training: keep this share of OFF minutes (re-weighted)
MAX_TRAIN_ROWS = 400_000    # cap on rows per model fit (keeps real-data runs to minutes)
VAL_FRAC = 0.20             # last 20% of each training house's timeline = validation
GATE_GRID_W = [10, 20, 30, 50, 80]
HGB_CONFIGS = {
    "A": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50),
    "B": dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
              l2_regularization=1.0),
    "C": dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100),
}
RESOLUTIONS_MIN = [1, 5, 15, 30]   # resolution study
