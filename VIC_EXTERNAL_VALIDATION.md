# VIC External Validation — UCI Replicated Acoustic Features Dataset

Exploratory analysis only. No formula implemented in the pipeline, no other repo file touched. Follow-up to `INDEX_PROPOSAL.md` — read that first for how VIC was derived on the Oxford dataset.

## Summary — does this validate VIC?

**No. VIC as defined (`log10(Jitter% × Shimmer:APQ3 × spread2 × 1000)`) cannot be computed on this dataset, and this is not a technicality — it's the headline finding.**

- Of VIC's three ingredients, only **two** have even a reasonably-comparable column here: `Jitter_rel` (~ MDVP:Jitter(%)) and `Shim_APQ3` (~ Shimmer:APQ3). **`spread2` has no equivalent anywhere in this dataset's 44 features** — its nonlinear-measure family here is RPDE/DFA/PPE/GNE, a different set of constructs with different formulas, not a renamed or rescaled spread1/spread2/D2. Per the task instructions, no substitute was used and no "VIC" number was fabricated from 2 of 3 ingredients.
- As a bonus, non-VIC diagnostic: even the two available parents, alone or combined, perform **much worse** here than their Oxford counterparts did — point-biserial r drops (0.252 / 0.301 vs Oxford's 0.278 / 0.348), and subject-level 5-fold CV AUC for the 2-parent combination is **0.628**, far below the 0.834 the same two features (plus spread2) scored as VIC's parents on Oxford, and far below this very dataset's own full 44-feature raw set (AUC 0.903).
- **Confidence in this validation: low, but the direction of the signal is not reassuring.** The missing ingredient means the real test (does VIC itself transfer) was never run. What *was* run — VIC's two transferable ingredients — did not hold up. Combined with only 80 subjects here (same order of magnitude as Oxford's 32-subject grouped analysis), this is a genuine limitation for VIC's generality, not a confirmation, and should be reported as such rather than spun.

---

## 1. Dataset inventory

`data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv` — 240 rows × 48 columns. 4 metadata columns (`ID` object/str, `Recording` int64 1–3, `Status` int64 0/1, `Gender` int64 0/1) + 44 float64 acoustic features. 80 unique subject IDs (`CONT-##` / `PARK-##` pattern), exactly 3 rows per subject, `Status` and `Gender` constant within subject (verified — 0 mismatches). 40 subjects `Status=1` (PD), 40 `Status=0` (healthy) — balanced, matching the task's stated 40/40 split.

Per-feature ranges (row-level, n=240), grouped as the source documentation groups them:

| Feature | Group | Min | Max | Mean | Std |
|---|---|---|---|---|---|
| Jitter_rel | Jitter (pitch perturbation) | 0.148 | 6.838 | 0.584 | 0.536 |
| Jitter_abs | Jitter | 0.000007 | 0.000550 | 0.000044 | 0.000045 |
| Jitter_RAP | Jitter | 0.000678 | 0.043843 | 0.003172 | 0.003373 |
| Jitter_PPQ | Jitter | 0.001036 | 0.065199 | 0.003532 | 0.004449 |
| Shim_loc | Shimmer (amplitude perturbation) | 0.007444 | 0.192600 | 0.038428 | 0.023213 |
| Shim_dB | Shimmer | 0.064989 | 1.747600 | 0.336832 | 0.205905 |
| Shim_APQ3 | Shimmer | 0.003344 | 0.113240 | 0.021499 | 0.013787 |
| Shim_APQ5 | Shimmer | 0.004103 | 0.120760 | 0.023468 | 0.014402 |
| Shi_APQ11 | Shimmer | 0.006459 | 0.142440 | 0.028671 | 0.016610 |
| HNR05 | Harmonics-to-noise (0–500 Hz band) | 22.22 | 101.21 | 59.33 | 14.54 |
| HNR15 | HNR (0–1500 Hz) | 26.27 | 109.65 | 63.67 | 15.62 |
| HNR25 | HNR (0–2500 Hz) | 33.16 | 120.71 | 74.20 | 16.44 |
| HNR35 | HNR (0–3500 Hz) | 36.49 | 128.29 | 80.19 | 17.19 |
| HNR38 | HNR (0–3800 Hz) | 36.91 | 129.99 | 81.51 | 17.40 |
| RPDE | Nonlinear dynamics | 0.163 | 0.536 | 0.311 | 0.066 |
| DFA | Nonlinear dynamics | 0.411 | 0.784 | 0.613 | 0.078 |
| PPE | Nonlinear dynamics (Pitch Period Entropy) | 0.004 | 0.908 | 0.272 | 0.230 |
| GNE | Nonlinear dynamics (Glottal-to-Noise Excitation) | 0.847 | 0.987 | 0.918 | 0.034 |
| MFCC0–MFCC12 | Spectral (13 coefficients) | ~0.57–2.08 | | ~1.30–1.36 | ~0.20–0.26 |
| Delta0–Delta12 | Spectral derivatives (13 coefficients) | ~0.62–2.04 | | ~1.33–1.35 | ~0.20–0.23 |

(MFCC/Delta rows collapsed for space — all 26 are float64, similar 0.6–2.1 range, no outliers of note.)

Source description (UCI dataset 489, Naranjo et al.): 4 jitter measures (relative, absolute, RAP, PPQ), 5 shimmer measures (local, dB, APQ3, APQ5, APQ11), 5 HNR measures at increasing frequency-band cutoffs, 4 "additional nonlinear measures" (RPDE, DFA, PPE, GNE), and 26 MFCC/delta-MFCC spectral coefficients. **This is the complete nonlinear-measure list for this dataset — there is no fifth or sixth nonlinear feature hiding under a different name. spread1, spread2, and D2 (the Little et al. nonlinear dysphonia measures the Oxford dataset uses) are simply not part of this feature set.**

## 2. Ingredient equivalence — rigorous check, not name-matching

| VIC ingredient (Oxford) | Candidate column (Naranjo) | Verdict | Basis |
|---|---|---|---|
| `MDVP:Jitter(%)` — relative pitch-period perturbation | `Jitter_rel` | **Reasonably comparable, not identical** | Same underlying construct (standard "local/relative jitter"), but different unit scale and almost certainly different extraction software. Oxford's column, despite being named "%", stores a raw fraction (range 0.00168–0.03316, mean 0.00622 ≈ 0.622%). Naranjo's `Jitter_rel` stores the number as a literal percent already (range 0.148–6.838, mean 0.584 ≈ 0.584%) — same order of magnitude once you account for the ×100 naming difference, which is consistent with the same physical quantity, but the two pipelines (Oxford: Kay Elemetrics MDVP; Naranjo: documented as a custom/Praat-style extraction for sustained-vowel recordings) are not proven to compute local jitter identically frame-to-frame. Treated as a usable proxy, not a byte-for-byte match. |
| `Shimmer:APQ3` — 3-point amplitude perturbation quotient | `Shim_APQ3` | **Reasonably comparable, higher confidence** | Both are explicitly documented as "3-point Amplitude Perturbation Quotient" — the same standard, named algorithm (not just a similar name). Ranges overlap well (Oxford 0.00455–0.05647, Naranjo 0.00334–0.11324, both unitless ratios, same order of magnitude). Still a different extraction pipeline than Oxford's MDVP, so treated as a proxy, but the strongest of the three. |
| `spread2` — nonlinear measure of fundamental-frequency spread/dispersion | **none** | **No equivalent — missing ingredient** | This dataset's nonlinear-measure family is RPDE, DFA, PPE, GNE. PPE (Pitch Period Entropy) is the closest in spirit (also from Little et al.'s methodology, also F0-related) but it is a **different formula measuring a different quantity** — entropy of the pitch-period sequence after removing expected physiological F0 variation, not the nonlinear F0-spread statistic spread2 computes. GNE measures glottal-to-noise excitation (a noise-domain quantity), not F0 spread at all. Neither is a valid substitute, and per the task's instructions none was used. |

**Conclusion: 2 of 3 ingredients have a usable (not exact) proxy; 1 of 3 has none. VIC itself is therefore not computable on this dataset.**

## 3. Replication handling

Averaged the 3 replications per subject into one subject-level row (`groupby('ID').mean()` for the 44 features; `Status`/`Gender` taken as `first()` since constant within subject — verified). Result: **80 rows, 40 PD / 40 healthy**, matching the task's stated cohort. All downstream correlation and CV work below uses this subject-level table, not the 240 raw rows — no i.i.d. assumption over replications.

## 4. What was computed, given the missing ingredient

VIC itself: **not computed** — stopping short per the task instructions rather than substituting a loosely related feature for spread2.

What *is* legitimate to test: the two ingredients that do have proxies, alone and combined, and — as an explicitly labeled non-VIC diagnostic — their log-product with the same `×1000` scaling VIC uses, so the shape of the transform (not the missing ingredient) is not what's being blamed for the numbers below.

**Point-biserial correlation with `Status` (subject-level, n=80):**

| Feature | r | p |
|---|---|---|
| `Jitter_rel` | 0.252 | 0.024 |
| `Shim_APQ3` | 0.301 | 0.0067 |
| `log10(Jitter_rel × Shim_APQ3 × 1000)` — **NOT VIC, 2 ingredients only** | 0.286 | 0.010 |

For context, Oxford's row-level r for the same two ingredients were 0.278 (`Jitter(%)`) and 0.348 (`Shimmer:APQ3`) — in the same ballpark, modestly lower here. Oxford's full VIC (all 3 ingredients) scored r=0.489, well above anything achievable from 2 ingredients on either dataset.

## 5. Cross-validated logistic regression, subject-level (n=80)

`Pipeline(StandardScaler → LogisticRegression(max_iter=5000))`, `StratifiedKFold(n_splits=5, shuffle=True)`, averaged over 10 seeds (0–9), mean±std over those 50 folds. Subject-level rows are already independent post-aggregation (each subject appears once), so plain `StratifiedKFold` is sufficient — no residual grouping to protect against.

| Feature set | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| `Jitter_rel` alone | 0.611±0.101 | 0.566±0.112 | 0.624±0.111 |
| `Shim_APQ3` alone | 0.614±0.101 | 0.574±0.117 | 0.649±0.119 |
| `Jitter_rel` + `Shim_APQ3` (2 raw parents) | 0.603±0.084 | 0.558±0.095 | 0.628±0.101 |
| `log10(Jitter_rel × Shim_APQ3 × 1000)` — **NOT VIC** | 0.549±0.106 | 0.548±0.104 | 0.646±0.107 |
| Full 44-feature raw set | 0.806±0.099 | 0.803±0.105 | 0.903±0.063 |

Majority-class baseline accuracy on this balanced (40/40) cohort: **0.500**.

## 6. Comparison against the Oxford numbers (`INDEX_PROPOSAL.md`)

| | Oxford (VIC, 3 ingredients, subject-grouped CV, n=32 subjects) | Naranjo (2-ingredient proxy, 5-fold CV, n=80 subjects) |
|---|---|---|
| Standalone r (index/proxy vs status) | 0.489 | 0.286 |
| Index/proxy alone — Accuracy | 0.795 | 0.549 |
| Index/proxy alone — F1 | 0.871 | 0.548 |
| Index/proxy alone — AUC | 0.833 | 0.646 |
| Parents alone (raw, uncombined) — Accuracy | 0.789 | 0.603–0.614 |
| Parents alone (raw, uncombined) — AUC | 0.834 | 0.624–0.649 |
| Full raw feature set — Accuracy | 0.780 (22 features) | 0.806 (44 features) |
| Full raw feature set — AUC | 0.734 (22 features) | 0.903 (44 features) |

Two things stand out, and neither favors VIC:

1. **Every VIC-related number drops substantially** on the Naranjo cohort — correlation nearly halves, AUC drops ~0.19–0.21 points for both the proxy index and its raw parents. Even granting that `spread2` (the single strongest-correlating ingredient on Oxford, r=0.455 alone) is missing here and would likely have added real signal, a ~0.2 AUC drop on the 2 ingredients that *are* comparable is a large gap, not noise — this dataset's own 50-fold std on these features is ≤0.12.
2. **The relationship between "one composite feature" and "the full raw set" inverts.** On Oxford, VIC alone (AUC 0.833) *beat* the entire 22-feature raw set (AUC 0.734) under grouped CV — the headline finding of `INDEX_PROPOSAL.md`, attributed there to the 22-feature model overfitting on only ~26 training subjects per fold. Here, the full 44-feature raw set (AUC 0.903) crushes every jitter/shimmer-based feature combination by a wide margin. That's consistent with the same overfitting story (80 subjects vs. 32 gives a 44-feature model much more room to fit cleanly), but it also means **this dataset does not reproduce the specific condition (small-n overfitting of a large raw set) that made VIC look advantageous on Oxford.** Whatever is carrying this dataset's very high full-feature AUC — most likely the HNR bands or MFCC/delta-MFCC spectral features, untested individually here — is not jitter or shimmer.

## 7. Limitations — read before citing this document

- **The real test didn't run.** VIC needs all three ingredients; only two had proxies. Nothing here proves or disproves VIC's actual 3-feature formula on an independent cohort — that validation remains undone.
- **"Reasonably comparable" is not "identical."** Both `Jitter_rel` and `Shim_APQ3` are proxies from a different extraction pipeline (documented separately from Oxford's Kay Elemetrics MDVP software) on a different, smaller cohort with its own recording protocol. Some (or all) of the performance drop in §6 could be pipeline/population mismatch rather than a genuine failure of the jitter+shimmer signal to generalize — this analysis cannot cleanly separate those two explanations.
- **n=80 subjects is still small**, same order of magnitude as Oxford's 32-subject grouped analysis; single-digit-percent accuracy swings from fold-to-fold noise are expected and the 10-seed averaging only partially controls for that (see the ±0.08–0.12 stds above).
- **Balanced (40/40) vs. Oxford's imbalanced (147/48) cohort** changes what "majority baseline" means (0.500 here vs. 0.754 on Oxford) — accuracy and F1 numbers across the two tables are not on a perfectly level footing for that reason; AUC is the more comparable metric of the three.
- **This document does not conclude VIC is wrong.** It concludes: one of VIC's three ingredients has no analog in this dataset, and the two that do have analogs underperformed their Oxford counterparts by a wide margin. That is a real, reportable limitation on how far the Oxford-derived VIC result generalizes — not a confirmation, and not a refutation of the untested full formula.

---

*No pipeline file, README, notebook, `generate_clean_assets.py`, or `INDEX_PROPOSAL.md` was modified in producing this document. Analysis run from a scratch script, not committed to the repo.*
