# RESULTS

All numbers below are written automatically by `python -m src.run_all`. The interpretation lines marked ✏️ TODO are yours to write after looking at your own plots.

## 1. Washing-machine EDA

**Homes and periods used:** see the availability table (date range per house).

<!-- TABLE:washer_availability -->
<!-- /TABLE -->

### Cycle rule (plain language)

A cycle **starts** when the washer's power goes above **20 W** (standby draws a few watts; real activity draws far more). Short quiet spells of **5 minutes or less** inside a wash (soaking, draining) do not end the cycle. A candidate is only counted if it lasts **at least 10 minutes** and uses **at least 20 Wh**, which removes blips. Cycles that touch a missing-data gap are excluded from statistics because their length and energy may be cut short. All thresholds are in `src/config.py`.

### (i) One representative day

![day](results/figures/02_representative_day.png)

✏️ TODO: interpret (what the aggregate does when the washer runs; how large the washer is relative to the whole-home load; what else is happening).

### (ii) Detected cycles

![cycles](results/figures/02b_detected_cycles.png)

✏️ TODO: interpret (typical shape: heating peak, drum phases, spin; does the rule capture the whole cycle?).

### Usage patterns by hour and day of week

![patterns](results/figures/03_usage_patterns.png)

✏️ TODO: interpret (morning/evening peaks? weekend vs weekday?).

### (iii) Cross-house comparison

![cross](results/figures/04_cross_house.png)

<!-- TABLE:cycle_stats_by_house -->
<!-- /TABLE -->

✏️ TODO: interpret (which houses differ and plausible reasons, e.g. hot vs cold wash, machine size).

### How often is another heavy appliance on at the same time?

<!-- TABLE:concurrency -->
<!-- /TABLE -->

## 2. Disaggregation model

### Setup

- **Target:** washer power (W) each minute. **Input:** aggregate power only.
- **Houses and periods:** each house is held out once as the test set; all its valid minutes in its date range are used for testing. Validation (for tuning) is the last 20% in time of each training house.

<!-- TABLE:split_design -->
<!-- /TABLE -->

- **Why this avoids leakage:** the test house never appears in training or tuning, and validation is a later time block, not random rows, so overlapping input windows cannot straddle train and validation.
- **Models:** `always_off`, `hourly_profile` (baselines), `gbm_raw`, `gbm_gated` (gradient boosting, optionally with a tuned on/off gate).
- Tuning choices made on validation data only:

<!-- TABLE:tuning_choices -->
<!-- /TABLE -->

### Metrics and why

| Metric | Meaning | Why it is here |
|---|---|---|
| MAE / RMSE (W) | average / large-error-sensitive power error | Easy to read, but MAE alone **flatters always-off** because the washer is off ~95% of the time |
| SAE | error in total energy, relative | What matters for billing-type insights and savings |
| NDE | squared error divided by squared signal; **1.0 = same as always-off** | Shows whether the model beats "do nothing"; hard to fake on a mostly-off appliance |
| Precision / recall / F1 | minute-level ON/OFF detection | Did it find the *when* |
| Cycle recall, false cycles | did we find each real cycle, and how many invented cycles | What a user would notice |

### Results (mean ± standard deviation across the test houses)

> **Comparability note:** Our cleaning (5-minute gap interpolation, outage exclusion, Issues-flag masking) and the varying test-set sizes per house (see table below) mean that raw metric numbers are not directly comparable to published NILM benchmarks that use different preprocessing, different houses, or a fixed test window.

<!-- TABLE:metrics_summary -->
<!-- /TABLE -->

Per test house:

<!-- TABLE:metrics_per_house -->
<!-- /TABLE -->

#### Test-set sizes

The table below shows, for each test house, the number of valid test minutes, the number of complete washer cycles in the test data, and the washer's share of total household energy. Houses with fewer cycles or a smaller energy share present a harder disaggregation problem (the signal is rarer and more buried in the aggregate).

<!-- TABLE:test_sizes -->
<!-- /TABLE -->

![metrics](results/figures/06_metrics_by_house.png)

✏️ TODO: interpret honestly (does the model beat both baselines on every house? where is it closest to the baseline? what does the spread across houses say?).

### Actual vs predicted (evaluation data)

The house shown is the one with the **median** score, not the best.

![pred](results/figures/05_actual_vs_predicted.png)

All test houses:

![pred all](results/figures/05b_actual_vs_predicted_all_houses.png)

✏️ TODO: interpret (does the heating peak line up? how are the quieter drum phases handled?).

### Cycle energy

<!-- TABLE:cycle_energy_error -->
<!-- /TABLE -->

![scatter](results/figures/07_cycle_energy_scatter.png)

### Failure cases

<!-- TABLE:failure_cases -->
<!-- /TABLE -->

![failures](results/figures/08_failure_cases.png)

✏️ TODO: describe the two failures you see (typical candidates: another heavy appliance running at the same time; a low-power or unusual programme the model has not seen in other houses; a house with a different machine type). Say what you would try next.

### What coarser meter data does (feeds the Section 4 answer)

The same model is re-trained and re-tested after replacing the aggregate by its 5-, 15-, or 30-minute block average (the washer is still scored per minute). This isolates how much information is lost.

<!-- TABLE:resolution_study -->
<!-- /TABLE -->

![res](results/figures/09_resolution_study.png)

✏️ TODO: interpret.

## What I would do next (if given more time)

✏️ TODO: 3 bullets, ordered by expected value, each with the reason.
