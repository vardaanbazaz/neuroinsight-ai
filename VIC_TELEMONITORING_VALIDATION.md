# VIC-2 Severity Validation — UCI Parkinson's Telemonitoring Dataset

Exploratory analysis only. No formula implemented in the pipeline, no other repo file touched. Follow-up to `INDEX_PROPOSAL.md` (where VIC was derived) and `VIC_EXTERNAL_VALIDATION.md` (where VIC's binary PD/healthy signal was tested on a second dataset). This document tests a different, narrower question on a third dataset from the same Oxford research group: does the acoustic-instability construct correlate with **disease severity** rather than PD/healthy status.

## Summary — does this validate VIC?

**VIC itself cannot be tested here — this dataset has no `spread2`, `D2`, or `spread1` column, so VIC's formula (`log10(Jitter% × Shimmer:APQ3 × spread2 × 1000)`) is not computable, full stop. No substitute was used.** What was tested instead: a two-ingredient construct built only from the columns that do exist here — Jitter(%) and Shimmer:APQ3 — named **VIC-2** to keep it clearly distinct from VIC.

- At the correct, independence-respecting unit of analysis — **subject-level aggregation, n=42 patients** — VIC-2 (and every individual ingredient tested) shows a **weak, statistically non-significant** correlation with both motor_UPDRS and total_UPDRS. Best subject-level r's: RPDE (r=0.238–0.277), PPE (r=0.248–0.261), VIC-2's log form (r=0.223–0.228) — all p>0.05, none clearing the n=42 critical value (|r|≈0.304 for p<0.05, two-tailed, df=40).
- Naive row-level correlation (n=5,875, ignoring that ~140 rows come from each of only 42 patients) produces small but "significant" p-values (p as low as 1e-36) on trivial effect sizes (r=0.04–0.16) — this is the inflation the task warned about, and it is not used as the headline result.
- A within-subject (repeated-measures) check — does a given patient's instability track their own UPDRS as it moves over the 4–6 month trial — comes back **essentially null to slightly negative** (r=−0.01 to −0.05) for every feature tested, including DFA/RPDE/PPE. Instability measures do not visibly track a patient's own worsening over this window.
- **Conclusion: the acoustic-instability concept does not show a clear relationship with disease severity in this dataset.** The signal that separated PD from healthy voices in the Oxford dataset does not obviously carry over to grading severity within an all-PD cohort. This is consistent with (not contradicting) `VIC_EXTERNAL_VALIDATION.md`'s finding on the Naranjo dataset — a second independent test where VIC-adjacent features also underperformed relative to their Oxford, PD-vs-healthy numbers.

---

## 1. Dataset and construct

`data/parkinsons_updrs.data` — 5,875 rows, 22 columns, 42 subjects, 101–168 recordings per subject (mean 139.9, matching the task's "~140 recordings per patient" description). All 42 subjects have PD; there is no binary status column and no healthy control group. Targets: `motor_UPDRS`, `total_UPDRS` (continuous, higher = more severe).

**VIC-2 (not VIC):**
- `product = Jitter(%) × Shimmer:APQ3`
- `log_product = log10(Jitter(%) × Shimmer:APQ3 × 1000)` — mirrors VIC's log-product structure and `×1000` scale constant, minus the missing `spread2` term.

Both ingredients were confirmed present with the expected names and value ranges in this file (`Jitter(%)`, `Shimmer:APQ3`), documented in `data/parkinsons_updrs.names` as coming from the same Little/Tsanas methodology as the Oxford dataset. No claim is made that the extraction pipeline is byte-identical to the original Oxford `pd_data.txt` — same caveat `VIC_EXTERNAL_VALIDATION.md` raised for the Naranjo dataset applies here too, though same-lab provenance makes it more likely than for a third-party dataset.

Also tested individually, per the task: `DFA`, `RPDE`, `PPE` — the closest available cousins to the nonlinear-dynamics half of the original fPI/VIC family (not spread2/D2 themselves; RPDE/DFA/PPE are documented in `parkinsons_updrs.names` as this dataset's own nonlinear/dynamical measures).

## 2. Handling the repeated-measures structure

Correlating all 5,875 rows independently is wrong here: rows are not independent draws — each patient contributes ~140 correlated observations (same vocal tract, same disease trajectory, same recording device), so a naive row-level correlation is pseudoreplication. It inflates apparent significance (huge effective N) without adding 5,875 independent pieces of evidence about whether the *construct* relates to *severity across patients*.

Three views were computed, in order of how much they're trusted:

1. **Subject-level aggregation (headline).** `groupby('subject#').mean()` collapses each patient's ~140 recordings (and their motor/total UPDRS, which drift slowly over the trial) to one row per patient. n=42, each row now an independent patient. This is the correct denominator for "does this construct distinguish more-severe from less-severe patients," and is reported as the deciding evidence, matching how `INDEX_PROPOSAL.md` and `VIC_EXTERNAL_VALIDATION.md` used subject-grouped/subject-level numbers as their headline metric over row-level ones.
2. **Row-level (naive), shown only for contrast.** All 5,875 rows treated as independent. Included specifically to demonstrate the inflation the task asked to guard against — not used to support any conclusion.
3. **Within-subject (repeated-measures correlation).** Each feature and each UPDRS score demeaned by subject (`x - subject_mean(x)`), then Pearson r computed on the residuals, with degrees of freedom corrected for the number of groups (`n_rows - n_subjects - 1`, the standard rmcorr adjustment). This asks a genuinely different question from (1): not "do more-severe patients tend to have higher instability" (between-subject/trait) but "as a given patient's own recordings vary over the trial, does instability move with their own UPDRS" (within-subject/state, i.e. does VIC-2 track a patient's own progression). Reported as a robustness/context check, not the headline.

A mixed-effects model (random intercept per subject) was considered but not used — for a simple bivariate correlation question, subject-level aggregation and the within-subject decomposition together answer both halves of what a mixed model would show (between- and within-subject association) without adding model-fitting complexity for a two-column construct.

## 3. Results

### 3a. Subject-level aggregated (n=42) — HEADLINE

| Feature | r vs motor_UPDRS | p | r vs total_UPDRS | p |
|---|---|---|---|---|
| Jitter(%) | 0.181 | 0.251 | 0.159 | 0.313 |
| Shimmer:APQ3 | 0.158 | 0.318 | 0.146 | 0.356 |
| **VIC-2 (product)** | 0.116 | 0.465 | 0.092 | 0.562 |
| **VIC-2 (log_product)** | 0.223 | 0.157 | 0.228 | 0.146 |
| DFA | −0.123 | 0.439 | −0.123 | 0.436 |
| RPDE | 0.238 | 0.128 | 0.277 | 0.076 |
| PPE | 0.261 | 0.095 | 0.248 | 0.114 |

n=42, df=40. Critical |r| for p<0.05 (two-tailed) ≈ 0.304. **Nothing in this table clears it.** RPDE vs total_UPDRS (r=0.277, p=0.076) comes closest.

### 3b. Row-level (naive, n=5,875) — shown only to demonstrate inflation, not trusted

| Feature | r vs motor_UPDRS | p | r vs total_UPDRS | p |
|---|---|---|---|---|
| Jitter(%) | 0.085 | 7.4e-11 | 0.074 | 1.2e-08 |
| Shimmer:APQ3 | 0.084 | 9.9e-11 | 0.079 | 1.1e-09 |
| VIC-2 (product) | 0.053 | 5.5e-05 | 0.041 | 1.6e-03 |
| VIC-2 (log_product) | 0.127 | 1.4e-22 | 0.134 | 7.0e-25 |
| DFA | −0.116 | 4.0e-19 | −0.114 | 2.7e-18 |
| RPDE | 0.129 | 4.3e-23 | 0.157 | 1.1e-33 |
| PPE | 0.162 | 5.1e-36 | 0.156 | 2.1e-33 |

Every one of these is "significant" at p<10⁻⁴ and every one of these r's is smaller in magnitude than its subject-level counterpart above. This is exactly the effect the task flagged: with ~140 pseudo-replicated rows per patient, an effect that is not distinguishable from zero at n=42 becomes p<10⁻²² through sample-size inflation alone. **Do not cite these p-values as evidence of anything** — they measure the size of the dataset, not the strength of the relationship.

### 3c. Within-subject (repeated-measures correlation)

| Feature | r vs motor_UPDRS | p | r vs total_UPDRS | p |
|---|---|---|---|---|
| Jitter(%) | −0.013 | 0.322 | −0.009 | 0.480 |
| Shimmer:APQ3 | −0.038 | 0.0034 | −0.037 | 0.0045 |
| VIC-2 (product) | −0.016 | 0.219 | −0.011 | 0.385 |
| VIC-2 (log_product) | −0.029 | 0.025 | −0.029 | 0.026 |
| DFA | −0.045 | 5.8e-04 | −0.027 | 0.039 |
| RPDE | −0.046 | 4.3e-04 | −0.038 | 0.0040 |
| PPE | −0.025 | 0.054 | −0.023 | 0.078 |

n=5,875 rows, dof=5,832 (n − 42 subjects − 1). Effect sizes are near zero across the board (|r|≤0.046). Several reach nominal significance only because of the same large-n effect as 3b, not because the within-patient relationship is meaningfully non-zero. Note the **sign flips negative** for every feature here versus the (mostly positive) subject-level table in 3a — between-patient trait differences trend in the expected direction (weakly), but within a given patient's own trial window, rising instability does not track rising UPDRS; if anything the raw numbers trend the opposite way, though the effect is too small to hang a conclusion on either way.

## 4. Comparison to the Oxford VIC result (context, not equivalence)

This is not an apples-to-apples comparison — Oxford tested VIC (3 ingredients, all present) against a **binary PD-vs-healthy** label; this dataset tests VIC-2 (2 ingredients, `spread2` unavailable) against a **continuous severity score within an all-PD cohort**. Both the formula and the outcome variable differ. With that caveat:

| | Oxford VIC, 3 ingredients, PD-vs-healthy, subject-grouped (n=32 groups) | Telemonitoring VIC-2, 2 ingredients, severity, subject-level (n=42) |
|---|---|---|
| Standalone r (best single ingredient/construct) | 0.489 (VIC itself) | 0.277 (RPDE, closest cousin; VIC-2 log itself: 0.223–0.228) |
| Statistically distinguishable from 0? | Yes | No |

The Oxford number answers "can this construct tell a PD voice from a healthy one" — a question with two well-separated populations. The telemonitoring number answers "within a group of people who already have PD, does the same construct track how severe it is" — a narrower-range, harder regression problem, on top of only having 2 of VIC's 3 ingredients available. The drop is consistent with — not necessarily caused by — either explanation (harder question, or missing ingredient, or both); this analysis cannot cleanly separate them, same limitation `VIC_EXTERNAL_VALIDATION.md` flagged for its own comparison.

## 5. Limitations — read before citing this document

- **VIC was not tested. VIC-2 was.** `spread2`/`spread1`/`D2` do not exist in this dataset (confirmed against `data/parkinsons_updrs.names`'s full 16-feature list). No substitute was used for spread2, and no number in this document should be read as a VIC result.
- **n=42 subjects is small** for detecting r~0.2–0.3 effects reliably — the subject-level table's "not significant" results are also consistent with a true small-to-moderate effect that this sample is simply underpowered to detect, not proof of zero effect. The honest reading is "no clear evidence of a relationship here," not "proven absent."
- **Severity range may be compressed.** All 42 subjects were "early-stage" PD patients recruited to the same trial — the UPDRS range covered is narrower than a full disease-vs-health spectrum, which mechanically caps how large any correlation with an acoustic measure can be, independent of whether the underlying biology is real.
- **Subject-level aggregation trades granularity for validity.** Averaging ~140 recordings per patient discards within-trial variation (including the slow UPDRS drift the trial was designed to capture) to get a valid n=42 unit of analysis. The within-subject check (3c) is an attempt to recover some of that signal, and it does not show a positive relationship either.
- **Same-lab provenance helps but doesn't guarantee identical Jitter(%)/Shimmer:APQ3 extraction** between this telemonitoring dataset and the original Oxford `pd_data.txt` — both were produced by overlapping authorship (Tsanas/Little) using documented, standard formulas, which is stronger provenance than the third-party Naranjo comparison in `VIC_EXTERNAL_VALIDATION.md`, but not proof of an identical extraction pipeline.
- **This document does not conclude VIC is wrong.** It concludes: on a third dataset, using only the two of VIC's three ingredients that are available, the acoustic-instability concept does not show a statistically or practically convincing relationship with PD severity. Whether the full 3-ingredient VIC formula (with `spread2`) would fare differently on a severity task remains untested — that validation, like on the Naranjo dataset, was never run because the ingredient doesn't exist here.

---

*No pipeline file, README, notebook, `generate_clean_assets.py`, `INDEX_PROPOSAL.md`, or `VIC_EXTERNAL_VALIDATION.md` was modified in producing this document. Analysis script run from a scratch directory, not committed to the repo.*
