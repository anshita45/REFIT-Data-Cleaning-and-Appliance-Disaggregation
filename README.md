# REFIT: data cleaning and washing-machine disaggregation

A reproducible pipeline that (1) inspects and cleans raw REFIT House 1 data, (2) explores washing-machine use across households, and (3) estimates washing-machine power from the whole-home signal at 1-minute resolution, with a leave-one-house-out evaluation against simple baselines.

Other deliverables: [DATA.md](DATA.md) · [RESULTS.md](RESULTS.md) · [Recommendation.md](Recommendation.md)

## What I built

| Step | Code | Output |
|---|---|---|
| 1. Raw inspection and cleaning (House 1) | `src/clean_raw.py` | `results/tables/raw_*.md`, `results/figures/01_*` |
| 2. Washing-machine EDA (cleaned data) | `src/eda_washer.py` | `results/figures/02-04_*`, tables |
| 3. Disaggregation model + evaluation | `src/model.py` | `results/figures/05-09_*`, tables |
| Shared | `config.py` (all settings), `events.py` (cycle rule), `prep.py` (1-min cache) | |

## Get the data and run

```bash
pip install -r requirements.txt
python -m src.prepare_data          # downloads + unpacks both archives into data/raw and data/clean
                                    # (or download from the links below and unzip there yourself)
python -m src.run_all               # everything, then fills the tables in the .md files
python -m src.run_all --skip-resolution   # faster: skips the 5/15/30-minute study
```

Individual steps: `python -m src.clean_raw`, `python -m src.eda_washer`, `python -m src.model`.
Everything is controlled from `src/config.py`. Seeds are fixed (`SEED = 42`).

Data sources (University of Strathclyde): [raw](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z), [cleaned](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z), [readme](https://pureportal.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt), [dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-loadmeasurements/).

## Assumptions and preprocessing decisions

- **Timestamps** are treated as UTC/GMT and converted to Europe/London for hour-of-day analysis (`ASSUME_UTC` in config). If the files turn out to be local time already, set it to `False`.
- **Duplicates:** rows with identical timestamps are merged by taking the mean.
- **Impossible values (raw):** negative values, and values above 25 kW (a UK home on a 100 A supply cannot draw that), are set to missing, not clipped.
- **Resampling:** 1-minute mean of the readings in each minute. Minutes with no reading stay missing.
- **Gaps:** up to 5 minutes are linearly interpolated. Longer gaps are **not** filled; they are flagged as outages and excluded from training and scoring.
- **Cleaned data:** rows the official files mark with `Issues = 1` are masked (`MASK_ISSUES`).
- **Washer channels:** mapped per house in `config.WASHER_CHANNEL`. The EDA step rejects any house whose channel does not behave like a washer (peak power and cycle count check) and lists it in `results/tables/excluded_houses.md`. House 4 has two washers and is left out.
- **Active cycle:** see "Cycle rule" in [RESULTS.md](RESULTS.md).

## Validation design (no leakage)

- **Leave-one-house-out:** each house is the test set once; a model is trained on the *other* houses only. This tests the situation that matters for a utility: a home the model has never seen.
- **Tuning** (3 small hyper-parameter sets and an on/off "gate") uses a **time-ordered** validation block: the last 20% of each *training* house. Random row splits are never used, because neighbouring minutes overlap in their input windows and would leak.
- The test house is not used for training, tuning, or choosing the gate.
- **Inputs are the aggregate signal only.** Other appliance channels are used for analysis (e.g. overlap) but never as model inputs.
- Training uses all ON minutes plus 25% of OFF minutes (re-weighted to keep the true ON/OFF ratio) and a cap of 400k rows per fit, for speed.

## Model and baselines

- **Model:** gradient boosting (`HistGradientBoostingRegressor`) on ~45 features from a window of aggregate power centred on each minute (±30 min): rolling mean/std/max, differences to neighbouring minutes, an "always-on floor" estimate, and local hour/weekday. Predictions are clipped at 0. A "gated" version sets predictions below a tuned threshold to 0.
- **Baselines:** `always_off` (predict 0) and `hourly_profile` (average washer power at each hour of the day in the training houses).
- **Metrics:** see [RESULTS.md](RESULTS.md) for why each is used.

## Trade-offs

- Gradient boosting over a neural network: faster to run on a laptop, easier to explain, and strong on a single mostly-OFF appliance. The cost is hand-made features and no learned sequence memory.
- Centred windows use future minutes, so this is *offline* disaggregation (analysis of recorded data), not real-time.
- One washer type per UK home is assumed; the model is not trained to separate two washers.
- Hyper-parameter search is deliberately small. More tuning would add little relative to house-to-house variation.

## Research used

- Murray, Stankovic and Stankovic (2017), *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*, Scientific Data 4:160122. Used for dataset description and limitations.
- Zhang, Zhong, Wang, Goddard and Sutton (2018), *Sequence-to-point learning with neural networks for non-intrusive load monitoring*, AAAI. Idea adopted: predict the appliance power at the *middle* of a window of aggregate readings. I implemented this with gradient boosting on window features, not a neural network.
- Kelly and Knottenbelt (2015), *Neural NILM: Deep neural networks applied to energy disaggregation*, BuildSys. Idea adopted: evaluating on **unseen houses**, and using a threshold to turn power into ON/OFF.
- Batra et al. (2014), *NILMTK: An open source toolkit for non-intrusive load monitoring*, e-Energy. Metrics adopted: MAE, normalised disaggregation error and the ON/OFF F1 family.
- Hart (1992), *Nonintrusive appliance load monitoring*, Proc. IEEE. Background on the problem.

✏️ TODO: open each paper and confirm the reference details and that the "idea adopted" lines are accurate for what you actually read.

## AI tools used

✏️ TODO (be accurate): I used Claude (Anthropic) to help draft the code structure and the documentation templates. I ran the pipeline on the real REFIT data myself, checked the washer-channel mapping against the dataset readme, and wrote the interpretation of the results.

## Possible improvements

- Seq2Point / small 1D-CNN and compare to the gradient-boosting model on the same split.
- Train one model per washer type (front-loader with heater vs others) instead of one pooled model.
- Use a state-based (HMM-style) cycle model so predicted cycles keep a realistic shape.
- Calibrate the gate per house from a few weeks of its own data, if the utility has any ground truth.
- Add temperature as an input (hot-wash use varies by season).
- Probability-of-use output instead of watts, for coarse (15/30-minute) data.
