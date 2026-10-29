# Recommendation: washing-machine disaggregation for utility use

*One-page summary for a utility audience. Includes the answers to Section 4.*

## What the model does

It estimates a home's washing-machine power minute by minute using only the whole-home smart-meter signal at 1-minute resolution. It was tested on **homes it had never seen** (leave-one-house-out), against a "predict nothing" baseline and a time-of-day-average baseline.

<!-- TABLE:metrics_summary -->
<!-- /TABLE -->

## Suitable uses

✏️ TODO: fill from your results. Candidates, to keep only if your numbers support them:
- Estimating how many washes a home does and roughly when (cycle recall, false cycles).
- Estimating washing-machine energy over weeks (SAE), for portfolio-level insight.
- Identifying *time-of-day* habits for tariff or demand-flexibility programmes.

## Limits (do not use for)

- Billing or compliance decisions about an individual home.
- Individual cycle energy where the error table shows large spread (see `cycle_energy_error`).
- Any home with data at 15- or 30-minute resolution (see below).
- Homes with different machine types or supply conditions from the training homes (e.g. India).

---

## Section 4 answers

### Opportunity: shift washing out of the evening peak

How much of each home's washing energy falls in the local evening peak window:

<!-- TABLE:peak_share -->
<!-- /TABLE -->

✏️ TODO: choose the opportunity your numbers support. Template if the evening share is large:
*"Homes that wash mostly in the evening peak get a time-of-use nudge (a message or small bill credit for starting delayed-start washes after the peak). Realistic, because washing is time-flexible and many machines have a delay timer."*

**How to measure it:** run a randomised trial. Split similar homes into a nudge group and a no-nudge group. Compare, before vs after, the **share of washing energy in the peak window** (the model supplies this), and use the difference between the two groups as the effect (difference-in-differences). Success = a statistically clear drop in peak share in the nudge group with no rise in total washing energy. The model's own error should be checked on a small set of homes with real washer sub-meters, so that a drop in "predicted" share is not a model artefact.

### Limitations

- **Overlapping appliances:** when a dryer, dishwasher or kettle runs together with the washer, their signals add up and the washer's share is hard to separate (see the concurrency and failure-case tables).
- **Missing data:** outages are left empty, not filled, so those periods give no estimate. Cycles cut by an outage are excluded from statistics.
- **Household differences:** machines, programmes and routines differ between homes. The leave-one-house-out spread shows how much accuracy changes from one home to the next.
- **UK to India:** the model learned from UK washers, which usually heat water electrically and show a high, clear heating peak. Many Indian machines (often top-loading, often using cold or externally heated water) have a much lower and different power signature, so a UK-trained model would likely miss or mis-size cycles. Other differences: supply voltage swings and power cuts, inverter/UPS back-up, larger appliances such as air-conditioners overlapping the washer's signal, and different habits and tariffs. **The model must be re-trained and re-checked on Indian homes (with washer sub-meters for a sample) before use.**

### What gets harder with 15- or 30-minute data

A wash cycle lasts about 1-2 hours, so at 15- or 30-minute readings it covers only a few data points, and each point is an *average* that blends the washer's brief high-power heating phase with its quiet phases. The sharp rise and fall that identifies the washer disappear, start and end times become uncertain by up to the reading interval, and a washer's average is easily confused with a kettle or a heater. What stays feasible is a *statistical* estimate (how much washing energy over weeks, or which hours are likely), not detection of individual cycles. The resolution study in RESULTS.md quantifies this:

<!-- TABLE:resolution_study -->
<!-- /TABLE -->

✏️ TODO: add one sentence reading the table (at what resolution does the model stop beating the hourly-profile baseline, if it does?).
