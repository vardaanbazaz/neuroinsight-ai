# Feature Importance Validation

Validates two claims in the README's Research Insights section (§10) against this project's own leak-free methodology (StratifiedGroupKFold, subject IDs from `name`, 5 folds × 10 seeds [0–9], raw 22 acoustic features + VIC). Both claims pre-date this repo's subject-grouped CV standard and had never been tested under it.

> **Claim A (Predictive Signal Strength):** "Perturbation measures like `PPE` and amplitude variations (`Shimmer`) exhibit high predictive signal strength."
> **Claim B (Complexity Metrics):** "Signal complexity (`D2`) and Recurrence Period Density Entropy (`RPDE`) prove important in differentiating healthy voice profiles."

## Summary verdict: partially holds — and not for the reasons the original bullets give

Both claims bundle two features together as if they behave alike. They don't.

| Feature | Verdict | Basis |
|---|---|---|
| `PPE` | **Holds, with a caveat** | Top-3 permutation, #1 gain, positive in all 10 seeds. But `spread1` is r=0.96 correlated with it — the two are near-interchangeable, and dropping PPE alone costs almost nothing because `spread1` covers for it. The real finding is "the spread/PPE cluster carries signal," not "PPE specifically." |
| `Shimmer` (as a family) | **Does not hold** | Permuting all 6 shimmer columns together costs **0 AUC** (−0.006 ± 0.049). Individual shimmer columns look "important" by gain only because the tree has to split on *something*, and shimmer is one of several redundant options — not because shimmer carries unique signal. |
| `D2` | **Holds — genuinely, and better-supported than PPE** | Weak on gain (rank 17/23) but real on both permutation-AUC (rank 2/23) and a drop-column refit (removing it costs 0.027 AUC, the second-largest drop of any feature tested). Positive in all 10 seeds. This is the one place a corroborating third method changed the initial read — see Methodology. |
| `RPDE` | **Does not hold** | Near-zero on both methods (perm rank 9 of 0.002, gain rank 15), straddles zero across seeds (range −0.019 to +0.020), and removing it from the model *improves* AUC slightly in the drop-column check. No evidence of signal. |

Net: the original two-bullet framing (PPE+Shimmer paired, D2+RPDE paired) doesn't match the data — each pair splits into one real and one null feature. Recommendation at the bottom replaces both bullets accordingly.

## Methodology

- Same CV as `generate_clean_assets.py`: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` for `seed` in 0–9, groups = subject ID extracted from `name` (`(S\d+)`), 50 folds total.
- Model: `xgb.XGBClassifier(eval_metric='logloss', random_state=42)`, fit fresh per fold (no scaling, matching XGBoost's row in the main pipeline).
- Feature set: all 22 raw Oxford acoustic features + `VIC` (`log10(Jitter% × Shimmer:APQ3 × spread2 × 1000)`), 23 columns total.
- **Permutation importance, AUC-based (not accuracy).** `permutation_importance`'s default accuracy scoring was tried first and abandoned: validation folds hold ~39 samples, so the smallest possible accuracy change is 1/39 ≈ 0.026 — a quantization floor that swallowed most of the ranking below the top 4–5 features (this is the same "lead with AUC, not recall" issue §10 already raises for SVM). All permutation numbers below are mean AUC-drop across 50 folds × 10 repeats per fold (500 permutations per feature), each measured against that fold's own fitted model.
- **Gain-based importance**: XGBoost's built-in `get_score(importance_type='gain')` per fold model, averaged first within each seed's 5 folds, then across the 10 seeds. "Folds used" reports how many of the 50 fold-models actually split on that feature at least once (gain is only meaningful when it does).
- **Two collinearity problems required extra handling, not just single-column permutation:**
  1. **Exact duplicates.** `Shimmer:DDA ≡ 3 × Shimmer:APQ3` (r = 0.99999996) and `Jitter:DDP ≡ 3 × MDVP:RAP` (r = 0.99999960). XGBoost picks one twin and discards the other (`Shimmer:DDA` used in only 4/50 folds, `Jitter:DDP` in 7/50) — those two columns' near-zero individual importance is guaranteed by construction, not a measurement of "shimmer" or "jitter" in general.
  2. **Near-duplicates.** `spread1`–`PPE` r = 0.96, `NHR`–`MDVP:RAP` r = 0.92. The tree's choice between each pair is close to arbitrary (`spread1`'s gain has a std of 4.61 on a mean of 5.24 across the 10 seeds — it's sometimes barely used, sometimes heavily), which is why individual-column rankings for these pairs disagree between permutation and gain.
  3. **Fix applied:** in addition to per-column permutation, ran **block (grouped) permutation** — shuffling every column in a named family together with one shared row-permutation per repeat — for Shimmer, Jitter, Pitch, HNR/NHR, Nonlinear-complexity (RPDE/DFA/D2), and Spread/PPE. This tests "does this concept carry information the rest of the model can't recover elsewhere," which is what the README claims are actually asserting (a family/concept, not one specific column).
  4. **Corroborating check:** a drop-column refit (retrain 50 folds with the named feature or family removed entirely, compare mean OOF-fold AUC to the full 23-feature baseline). This is the only method immune to the "the model reads the correlated twin instead" failure mode, and is what overturned the initial accuracy-scored read on `D2`.
- **`VIC` contains `spread2` as an ingredient** (see formula above), so `spread2`'s individual importance and `VIC`'s are mutually masked to some degree; the Spread/PPE block below excludes `VIC` for this reason. Flagged, not resolved — untangling it would need refitting without `spread2` present at all, which changes what VIC *is*.
- All ± figures below are population std (`ddof=0`) across the 50 folds, matching `generate_clean_assets.py`'s convention.

## Ranked tables

### Top 8 by permutation importance (AUC drop, mean ± std over 50 folds)

| Rank | Feature | AUC drop |
|---|---|---|
| 1 | `spread2` | 0.0559 ± 0.0520 |
| 2 | `D2` | 0.0445 ± 0.0536 |
| 3 | `PPE` | 0.0326 ± 0.0422 |
| 4 | `MDVP:Fhi(Hz)` | 0.0282 ± 0.0665 |
| 5 | `MDVP:RAP` | 0.0256 ± 0.0434 |
| 6 | `NHR` | 0.0146 ± 0.0391 |
| 7 | `spread1` | 0.0135 ± 0.0270 |
| 8 | `MDVP:Shimmer(dB)` | 0.0062 ± 0.0252 |

### Top 8 by gain-based importance (mean ± std over 10 seed-level means)

| Rank | Feature | Gain | Folds used (of 50) |
|---|---|---|---|
| 1 | `PPE` | 7.51 ± 1.76 | 50 |
| 2 | `spread1` | 5.24 ± 4.61 | 49 |
| 3 | `MDVP:Fo(Hz)` | 1.97 ± 0.43 | 50 |
| 4 | `Shimmer:APQ5` | 1.87 ± 0.57 | 45 |
| 5 | `spread2` | 1.61 ± 0.24 | 50 |
| 6 | `MDVP:APQ` | 1.43 ± 0.51 | 42 |
| 7 | `HNR` | 1.38 ± 0.73 | 45 |
| 8 | `Shimmer:APQ3` | 1.34 ± 1.32 | 44 |

### Features that rank highly on one method but not the other

Real divergence, not noise — this is the "importance is method-specific" pattern the task asked to watch for:

| Feature | Perm rank | Gain rank | Read |
|---|---|---|---|
| `D2` | 2 | 17 | Rarely the tree's chosen split (low gain), but removing it hurts held-out AUC — confirmed by drop-column refit below. Gain undercredits it. |
| `MDVP:Fo(Hz)` | 19 (negative) | 3 | Frequently split on, but permuting it doesn't hurt predictions — the tree is using it in a way that's easily compensated elsewhere. |
| `MDVP:APQ` | 22 (negative) | 6 | Same pattern as `MDVP:Fo(Hz)`: used often, not needed. |
| `HNR` | 21 (negative) | 7 | Same pattern. |
| `Shimmer:APQ5` | 12 | 4 | Used often (45/50 folds), individually near-zero permutation effect. |
| `Jitter:DDP` | 11 | 22 | Rarely used (7/50 — it's the RAP duplicate), so its gain is genuinely near-zero by construction, but its rare individual permutation reads mildly positive; treat as noise on 7 folds, not signal. |

The general shape: **gain-based importance tends to reward whichever feature the tree happens to split on when several are redundant; permutation/drop-column reward features whose information genuinely isn't recoverable elsewhere.** For a dataset this collinear (Jitter and Shimmer sub-measures are themselves highly inter-correlated), gain rankings alone are not a safe basis for "this feature matters" claims — which is exactly the situation the two existing README bullets are in.

## Direct check against the two existing claims

### Claim A — PPE and Shimmer

**PPE**: perm rank 3 (AUC drop 0.0326 ± 0.0422), gain rank 1 (7.51 ± 1.76, used in all 50 folds), positive in all 10 seeds (seed range 0.0156–0.0456). **Real signal — claim holds for PPE specifically.** Caveat: `spread1` (r=0.96 with PPE) is comparably important by both methods (perm rank 7, gain rank 2) and a drop-column check shows removing PPE alone costs only 0.005 AUC (see table below) because `spread1` substitutes for it. The honest statement is "the spread1/PPE cluster carries signal," not "PPE uniquely does."

**Shimmer** — checked every variant individually, since the claim doesn't say which one:

| Shimmer variant | Perm rank (of 23) | Perm AUC drop | Gain rank | Gain (folds used) |
|---|---|---|---|---|
| `MDVP:Shimmer(dB)` | 8 | 0.0062 ± 0.0252 | 11 | 1.06 (37/50) |
| `MDVP:Shimmer` | 10 | 0.0009 ± 0.0054 | 13 | 1.02 (23/50) |
| `Shimmer:APQ5` | 12 | 0.0003 ± 0.0200 | 4 | 1.87 (45/50) |
| `Shimmer:DDA` | 13 | 0.00005 ± 0.0011 | 23 | 0.04 (4/50, exact 3× duplicate of APQ3, tree discards it) |
| `Shimmer:APQ3` | 16 | −0.0030 ± 0.0190 | 8 | 1.34 (44/50) |
| `MDVP:APQ` | 22 | −0.0131 ± 0.0493 | 6 | 1.43 (42/50) |

None of the six shimmer variants clears rank 7 on permutation importance, and the **shimmer family as a block** (all 6 columns permuted together) costs **0 AUC** (−0.006 ± 0.049) — confirmed by the drop-column refit, which shows dropping all 6 shimmer columns from training doesn't change mean AUC (0.7514 vs. 0.7496 full-set baseline — actually a hair *higher* without them, well inside noise). **Shimmer does not hold as a predictive-signal claim.** The gain-based ranks (4, 6, 8, 11, 13) that make individual shimmer columns look important are an artifact of the tree needing to split on *some* redundant amplitude-perturbation measure — not evidence the family carries unique information.

### Claim B — D2 and RPDE

**D2**: perm rank 2 (AUC drop 0.0445 ± 0.0536, positive in all 10 seeds, range 0.0157–0.0664) but gain rank 17 (0.80 ± 0.22, used in 49/50 folds but rarely the primary split). Given the gain/permutation split, this was checked with a third method — drop-column refit — before reaching a verdict:

| Config | Mean AUC over 50 folds |
|---|---|
| Full 23-feature set | 0.7496 ± 0.1867 |
| Drop `spread2` | 0.7118 ± 0.1795 (−0.038) |
| Drop `D2` | 0.7223 ± 0.2076 (−0.027) |
| Drop `PPE` | 0.7446 ± 0.1910 (−0.005) |
| Drop `RPDE` | 0.7598 ± 0.1818 (+0.010) |
| Drop Shimmer family (6 cols) | 0.7514 ± 0.1873 (+0.002) |

D2's removal costs the second-largest AUC drop of anything tested, right behind `spread2` and ahead of `PPE`. **D2 holds — genuinely, and by more direct evidence than the low gain rank alone would suggest.** The gain metric undercredits it because the tree usually finds a cheaper split elsewhere first; that doesn't mean D2 is redundant, since removing it entirely (drop-column) does hurt.

**RPDE**: perm rank 9 (AUC drop 0.0020 ± 0.0287 — an order of magnitude below D2/spread2/PPE), gain rank 15 (0.82 ± 0.21), and straddles zero across seeds (range −0.019 to +0.020 — roughly half the seeds show it as mildly *harmful* to include). The drop-column check confirms this: removing RPDE from training *improves* mean AUC slightly (0.7598 vs. 0.7496 baseline). **RPDE does not hold.** No method finds real signal in it on this dataset under this CV scheme.

## Honesty check: is this ranking stable, or just this run's luck?

Per-fold and per-seed variance is real and worth stating plainly, same spirit as INDEX_PROPOSAL.md's model-ranking-instability note — with only 32 subjects and ~6 per fold, individual-fold permutation std is often larger than the mean effect (e.g. `MDVP:Fhi(Hz)`'s 0.028 ± 0.067 spans zero at the per-fold level). But the **seed-level means** (averaging each seed's 5 folds first) are a materially more stable signal, and on that basis the split between "real" and "null" features is not close:

- **Stable positive across all 10 seeds** (never crosses zero seed-to-seed): `spread2` (0.035–0.093), `D2` (0.016–0.066), `PPE` (0.016–0.046), `MDVP:Fhi(Hz)` (0.006–0.066).
- **Straddles zero across seeds** (sign flips depending on seed): `RPDE` (−0.019 to +0.020), `Shimmer:APQ3` (−0.017 to +0.006), `Shimmer:APQ5` (−0.006 to +0.007), `MDVP:Shimmer` (−0.001 to +0.006, effectively flat).

So this isn't a case of "the ranking is too noisy to say anything" — the top tier (`spread2`, `D2`, `PPE`) and the null tier (`RPDE`, all Shimmer variants) are cleanly separated by seed-stability, not just by the point estimate. The genuinely unstable part of this dataset's feature importance is the *middle of the pack* (ranks ~10–20), which is exactly where the small-sample accuracy-scoring problem lives — not a place either README claim was making a bet on.

## Recommendation for the two Research Insights bullets

Replace both existing bullets with one bullet per verified finding, so the wording matches what actually held up rather than what the pre-existing text assumed:

- *(Replaces "Predictive Signal Strength")* **The spread/entropy cluster — `spread2`, `spread1`, and `PPE` — carries this dataset's clearest signal**, ranking in the top 3–7 by permutation importance, top 5 by gain, and stable (never sign-flipping) across all 10 CV seeds. `spread2` alone accounts for the largest single-feature AUC drop when removed (−0.038). `PPE` and `spread1` are near-duplicates of each other (r=0.96) — the model uses whichever one a given tree happens to split on first, so credit the pair, not PPE specifically.
- *(New, replaces "amplitude variations (Shimmer)")* **Shimmer measures — individually or as a family — do not show unique predictive signal once subject-level leakage is removed.** Individual variants look important by gain (because the model must split on *some* amplitude-perturbation measure), but permuting all 6 shimmer columns together costs 0 AUC, and dropping them from training entirely doesn't hurt performance either. Treat prior "shimmer is important" language as an artifact of single-split-CV / non-grouped importance methods, not a robust finding here.
- *(Replaces "Complexity Metrics")* **`D2` (correlation dimension) is a genuine, if under-appreciated, contributor** — weak by the standard gain-based importance ranking (17th of 23) but the second-largest AUC drop of any feature when removed entirely, and positive across all 10 CV seeds. `RPDE`, its usual complexity-metric pairing, shows no such effect on either metric and should not be grouped with D2 in future writing — removing it doesn't hurt, and in some seeds mildly helps.
