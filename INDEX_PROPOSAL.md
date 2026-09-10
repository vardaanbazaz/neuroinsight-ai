# Acoustic Index Proposal — Replacement for fPI

Exploratory analysis only. No formula implemented in the pipeline. No other repo file touched.

## Methodology summary (read before trusting numbers)

- Dataset: `data/pd_data.txt`, 195 recordings, **32 unique subjects** (extracted from the `name` column, e.g. `phon_R01_S01_1` → subject `S01`), 147 PD / 48 healthy. **Majority-class baseline: accuracy = 0.754, F1 (predict-all-PD) = 0.860.** Any candidate must clear both meaningfully to mean anything — a number that lands *on* these values is the model predicting PD for everything, not finding signal.
- Two CV schemes, because plain row-level k-fold leaks: sibling recordings of the same subject can land in both train and test.
  - Row-level: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` — the split the task asked for, mean±std over 50 folds.
  - Subject-grouped (the number that actually generalizes): `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`, averaged over 10 seeds (0–9), mean±std over those 10 runs. This is reported as the deciding evidence below, not a footnote — see the correction note under Candidate 2.
- Models: `Pipeline(StandardScaler → LogisticRegression(max_iter=5000))`. A decision stump (`max_depth=1`) was also tried but only ever splits on the single strongest raw feature regardless of what else is in the set, so it can't show a multi-feature index's value — mentioned once below, not used to decide anything.
- **Fold-safety on the composite itself, not just the classifier:** CCC (candidate 2) is a z-score combination, and z-scoring requires a mean/std estimate. Computing that over the whole dataset before cross-validation leaks test-fold information into the feature, even with the classifier itself in a clean pipeline. CCC's z-scores are therefore fit on train-fold rows only and applied to test rows, refit per fold, both here and in any real use. PFLI and NRR are pure row-wise functions (no fitted parameters), so they carry no such risk.
- Metrics: accuracy, F1 (PD class = `status`=1, sklearn's default `pos_label`), ROC-AUC.
- Univariate ranking: point-biserial correlation (= Pearson r against the 0/1 label), ANOVA F-score, mutual information (`random_state=42`) across all 22 raw features.
- Known domain constraint checked before proposing any log form: **`spread1` is negative for every row** (range −7.96 to −2.43), so any log-product containing it is mathematically undefined. It's excluded from log-based candidates for that reason, not by oversight.
- The original fPI trio (DFA, D2, spread2) is not reused as a set. Where a candidate reuses one member of that trio, it's called out explicitly (spec allows "not the same trio").

## Candidate comparison

| Candidate | Features | Math form | Standalone r with status | Index-alone (row CV) acc / F1 / AUC | Index-alone (subject-grouped CV, 10 seeds) acc / F1 / AUC | Parents-only (subject-grouped CV) acc / F1 / AUC |
|---|---|---|---|---|---|---|
| **VIC (recommended)** | Jitter(%), Shimmer:APQ3, spread2 | log-product: `log10(Jitter% × APQ3 × spread2 × 1000)` | **0.489** (highest) | 0.817 / 0.886 / 0.835 | **0.795 / 0.871 / 0.833** | 0.789 / 0.866 / 0.834 |
| CCC | RPDE, HNR, Fo(Hz) | weighted z-score sum: `z(RPDE) − z(HNR) − z(Fo)`, z fit per-fold | 0.467 | 0.805 / 0.878 / 0.802 | 0.779 / 0.861 / 0.789 | 0.785 / 0.865 / 0.695 |
| NRR | NHR, Shimmer:APQ3, HNR | ratio: `(NHR + APQ3) / HNR` | 0.185 (weak) | 0.754 / 0.860 / 0.769 | 0.756 / 0.860 / 0.757 — **acc/F1 identical to the majority-class baseline; only AUC shows real ranking signal** | 0.706 / 0.820 / 0.711 |
| Raw 22 features (no index) | — | — | — | 0.854 / 0.906 / 0.895 | 0.780 / 0.857 / 0.734 | — |

**Why the "raw22 + index" comparison isn't in this table:** CCC and NRR are linear/near-linear functions of columns already inside the raw 22, so a logistic regression can already reconstruct them internally — adding the column can't change the fitted model (measured delta was ≤0.002 across all three candidates, pure fold noise, both CV schemes). VIC's log-product is genuinely nonlinear so it could in principle move a linear model, and empirically it doesn't either, by a similarly small margin — with 195 rows and 22 already-informative features, one more derived column rarely changes what a linear classifier learns. **The comparison that actually tests these indices is "index alone vs. its own parent features alone"** (columns 5–6): does the composite compress several raw signals into one without losing much?

## Recommendation

**VIC (Vocal Instability Compound).** Reasoning, led by the subject-grouped numbers (the ones that generalize to a new patient, not just a new recording of a patient already in the training fold):

- Highest standalone correlation (r = 0.489) and, decisively, the best subject-grouped performance of the three: 0.795 accuracy / 0.833 AUC as a *single feature* — beating the CCC (0.779 / 0.789), NRR (0.756 / 0.757), and even the **entire raw 22-feature set** (0.780 / 0.734).
- Near-zero compression loss: its three raw parents together score 0.789 acc / 0.834 AUC under the same grouped CV — VIC alone matches that almost exactly (0.795 / 0.833), meaning the log-product keeps essentially all the discriminative power of three columns in one.
- Tight variance across seeds (±0.014–0.015) compared to CCC (±0.018–0.030) and the raw set (±0.021–0.042), so this isn't a lucky split.
- Note on what the AUC number does and doesn't test: `log10` is monotonic, so VIC's AUC is mathematically identical to the AUC of the bare product `Jitter% × APQ3 × spread2` (and the ×1000 constant changes nothing either) — the 0.833 AUC is evidence for *this feature grouping and its product structure*, not for the log wrapper specifically. The log earns its place in the accuracy/F1 numbers (it linearizes the relationship for a linear classifier and gives a roughly-normal, human-readable scale), and in the physiological rationale below — not in AUC.
- Why a single feature (0.833 AUC) beats the entire 22-feature raw set (0.734 AUC) under grouped CV: a 22-feature logistic regression fit on ~156 training rows, with the jitter and shimmer families internally correlated at r ≥ 0.90 (see §1), overfits — and subject-grouped CV is exactly what exposes that (row-level CV hides it: 0.895 vs. 0.835, raw-22 ahead there, because leaking sibling recordings into both train and test masks the overfitting). This is a property of the raw set at this sample size, not a claim that one feature is intrinsically better than twenty-two.
- Physically coherent: three cycle-to-cycle *perturbation* measures from distinct sub-systems of phonation (frequency micro-instability, amplitude micro-instability, nonlinear frequency spread) — see rationale below, not just three correlated numbers.
- Log-product form is mathematically valid (all three parents strictly positive) and fits the physiology: vocal-fold perturbation compounds multiplicatively cycle over cycle, which a log-product is built to model — the same reasoning the original paper used, applied to different, better-motivated features.

**Correction note on an earlier draft of this analysis:** a first pass reported a single (non-repeated) `GroupKFold` split where CCC appeared to beat VIC on subject-grouped AUC (0.828 vs. 0.759). That table also computed CCC's z-scores over the whole dataset rather than per-fold. Both problems favored CCC by chance/leak; fixing the z-score leak and averaging over 10 stratified-grouped seeds (the numbers used throughout this document) reverses that — VIC wins on every subject-grouped metric. Flagging this here rather than quietly replacing it, since it's the sort of small-N flip that's easy to get backwards.

CCC is a reasonable runner-up: it clears the raw-22 baseline on AUC (0.789 vs. 0.734) using one feature, and its rationale (clarity + dynamical complexity + baseline pitch) is the most clinically varied of the three. NRR is weakest: its accuracy and F1 alone (0.756 / 0.860) are the majority-class baseline almost exactly — it is not clearing the bar, it is sitting on it. Only its AUC (0.757) shows the ratio construction is finding some real ranking signal, and even that trails both other candidates.

---

## 1. Fresh univariate analysis (all 22 features)

Point-biserial r (Pearson r against the 0/1 status label), ANOVA F-score, and mutual information, computed independently of the existing notebooks:

| Feature | Point-biserial r | F-score | MI |
|---|---|---|---|
| spread1 | 0.565 | 90.4 | 0.221 |
| PPE | 0.531 | 75.8 | 0.254 |
| spread2 | 0.455 | 50.3 | 0.192 |
| MDVP:Fo(Hz) | −0.384 | 33.3 | 0.207 |
| MDVP:Flo(Hz) | −0.380 | 32.6 | 0.174 |
| MDVP:Shimmer | 0.367 | 30.1 | 0.100 |
| MDVP:APQ | 0.364 | 29.5 | 0.181 |
| HNR | −0.362 | 29.0 | 0.132 |
| Shimmer:APQ5 | 0.351 | 27.1 | 0.114 |
| MDVP:Shimmer(dB) | 0.351 | 27.1 | 0.111 |
| Shimmer:APQ3 | 0.348 | 26.5 | 0.109 |
| Shimmer:DDA | 0.348 | 26.5 | 0.109 |
| D2 | 0.340 | 25.3 | 0.018 |
| MDVP:Jitter(Abs) | 0.339 | 25.0 | 0.182 |
| RPDE | 0.309 | 20.3 | 0.037 |
| MDVP:PPQ | 0.289 | 17.5 | 0.098 |
| MDVP:Jitter(%) | 0.278 | 16.2 | 0.115 |
| MDVP:RAP | 0.267 | 14.8 | 0.092 |
| Jitter:DDP | 0.267 | 14.8 | 0.092 |
| DFA | 0.232 | 11.0 | 0.095 |
| NHR | 0.189 | 7.2 | 0.115 |
| MDVP:Fhi(Hz) | −0.166 | 5.5 | 0.140 |

All 22 are significant at p<0.05; the ranking by r, F-score, and MI is broadly consistent (spread1/PPE/spread2 dominate, Fhi is weakest).

**Redundancy check (within-family correlations)** — this shaped which features got grouped together below:
- Jitter family (Jitter%, Jitter Abs, RAP, PPQ, DDP): pairwise r ≥ 0.90 — effectively one signal.
- Shimmer family (Shimmer, Shimmer dB, APQ3, APQ5, APQ, DDA): pairwise r ≥ 0.90 — effectively one signal.
- spread1 / PPE: r = 0.96 — near-duplicates; never used together in a candidate below.
- NHR / HNR: r = −0.71 — related but not redundant (different measurement convention, see NRR rationale).
- RPDE / DFA / D2 (the nonlinear-dynamics family the original fPI drew two of its three terms from): mutually weak, r between −0.17 and 0.24 — genuinely three separate signals despite the shared "family" label.

## 2–3. Candidate groups: acoustic rationale + math form

### Candidate 1 (recommended): VIC — Jitter(%), Shimmer:APQ3, spread2

**What each feature physically is:**
- `MDVP:Jitter(%)` — cycle-to-cycle variation in the *pitch period* (how much the vocal folds fail to vibrate at a stable frequency). Frequency-domain instability.
- `Shimmer:APQ3` — cycle-to-cycle variation in *amplitude* (how much loudness wavers cycle to cycle). Amplitude-domain instability — a different physical axis from jitter even though both stem from imprecise vocal-fold closure.
- `spread2` — a nonlinear measure of fundamental-frequency spread/variation, capturing broader dispersion in pitch beyond simple jitter.

**Why these three, not the original trio:** none of DFA/D2/spread2 as a set; only spread2 is reused, replacing DFA and D2 with the two dominant, physically distinct perturbation families (frequency and amplitude) that PD dysarthria is clinically known to disrupt (reduced laryngeal control → both frequency and amplitude micro-instability).

**Math form — log-product:** `VIC = log10(Jitter% × APQ3 × spread2 × 1000)` (the ×1000 is a fixed scale constant only, chosen to keep the log's argument in a convenient range — it's a constant multiplicative shift before the log, so it doesn't affect rank-correlation or a downstream model's fit).

**Why log-product here:** the same justification as the original fPI, applied to different, better-motivated features — all three are strictly-positive, small-magnitude ratios representing compounding, multiplicative degradation (cycle-to-cycle error compounds over many cycles, which is a multiplicative not additive process). A log-product turns that multiplicative compounding into an additive, roughly-normal-shaped score, which is exactly what log-transforms are for. All three parents are confirmed strictly positive in this dataset (minimum observed values 0.0017, 0.0046, 0.0063 respectively), so the log is always defined — this is exactly why spread1 (which goes negative) was excluded from any log-based candidate.

### Candidate 2: CCC (Clarity–Complexity Composite) — RPDE, HNR, MDVP:Fo(Hz)

**What each feature physically is:**
- `RPDE` (Recurrence Period Density Entropy) — how periodic vs. chaotic the vocal-fold vibration cycle is; PD voices lose clean periodicity.
- `HNR` (Harmonics-to-Noise Ratio, dB) — how "clear" vs. breathy/noisy the voice sounds; higher = clearer.
- `MDVP:Fo(Hz)` — average fundamental pitch; PD speech often shows altered baseline pitch (monotone, reduced range) as a rigidity symptom.

**Why these three:** spans three genuinely different measurement domains — nonlinear dynamical complexity, spectral noise ratio, and baseline pitch level — none of them from the original trio at all.

**Math form — weighted z-score sum, fit per fold:** `CCC = z(RPDE) − z(HNR) − z(Fo)`, signs chosen so all three terms point the same direction (higher CCC = more PD-like: RPDE up, HNR down, Fo down are all associated with status=1 per the correlation table above). The mean/std used for each z-score are estimated from training rows only in every fold — see the fold-safety note above; a deployed version would freeze the μ/σ learned during training and apply them to new recordings, not recompute them on each new batch.

**Why z-score sum here:** the three parents live on wildly different natural scales and units (RPDE ∈ [0,1] unitless, HNR in dB ranging 8–33, Fo in Hz ranging 88–260) — a raw sum or product would be dominated by whichever feature has the largest numeric range. Z-scoring first puts them on comparable footing, then a signed sum lets each contribute a genuinely additive vote, which fits three independent-ish severity signals better than a multiplicative form would.

### Candidate 3: NRR (Noise-Roughness Ratio) — NHR, Shimmer:APQ3, HNR

**What each feature physically is:**
- `NHR` — noise-to-harmonics energy ratio (unitless); higher = noisier voice.
- `HNR` — the (roughly) inverse convention, in dB; higher = clearer voice.
- `Shimmer:APQ3` — amplitude perturbation (see above), a physically distinct axis from either noise measure.

**Why these three:** NHR and HNR both come from the "noise" family, but are deliberately paired *because* they're expressed on incompatible conventions (a small unitless ratio vs. a decibel scale) that partially — not perfectly — invert each other (r = −0.71, not −1.0, so there's real complementary information, not pure redundancy). Shimmer:APQ3 adds a genuinely independent amplitude-perturbation axis. This candidate is the one most vulnerable to the "one family wearing two hats" concern — flagged here rather than hidden, and it's also the weakest performer below.

**Math form — ratio:** `NRR = (NHR + APQ3) / HNR`. A ratio is the natural form when the numerator terms (both "badness" quantities, roughly comparable small-ratio scales) sit awkwardly against a denominator on a totally different scale (dB) and pointing the opposite clinical direction (higher HNR = healthier) — dividing by HNR both rescales it and flips its direction for free, without needing a manually-chosen sign like the z-score approach.

## 4. Standalone + predictive testing detail

**Row-level `RepeatedStratifiedKFold(5×10)`, LogisticRegression:**

| Feature set | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Raw 22 (no index) | 0.854±0.053 | 0.906±0.034 | 0.895±0.052 |
| Raw 22 + VIC | 0.852±0.053 | 0.904±0.035 | 0.898±0.050 |
| Raw 22 + CCC | 0.854±0.053 | 0.906±0.034 | 0.895±0.052 |
| Raw 22 + NRR | 0.854±0.053 | 0.906±0.034 | 0.895±0.051 |
| VIC alone | 0.817±0.053 | 0.886±0.032 | 0.835±0.061 |
| CCC alone | 0.805±0.055 | 0.878±0.034 | 0.802±0.074 |
| NRR alone | 0.754±0.013 | 0.860±0.008 | 0.769±0.084 |
| VIC's 3 parents (raw, uncombined) | 0.803±0.057 | 0.876±0.036 | 0.849±0.058 |
| CCC's 3 parents (raw, uncombined) | 0.818±0.061 | 0.886±0.039 | 0.799±0.072 |
| NRR's 3 parents (raw, uncombined) | 0.743±0.044 | 0.847±0.028 | 0.748±0.076 |

**Subject-grouped `StratifiedGroupKFold(5, shuffle=True)`, averaged over 10 seeds, LogisticRegression (CCC fit per-fold, see fold-safety note):**

| Feature set | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Raw 22 (no index) | 0.780±0.021 | 0.857±0.018 | 0.734±0.042 |
| VIC alone | **0.795±0.014** | 0.871±0.009 | **0.833±0.015** |
| CCC alone (per-fold z-score) | 0.779±0.018 | 0.861±0.012 | 0.789±0.030 |
| NRR alone | 0.756±0.000 | 0.860±0.000 | 0.757±0.027 |
| VIC's 3 parents (raw, scaled) | 0.789±0.017 | 0.866±0.011 | 0.834±0.014 |
| CCC's 3 parents (raw, scaled) | 0.785±0.018 | 0.865±0.014 | 0.695±0.047 |
| NRR's 3 parents (raw, scaled) | 0.706±0.022 | 0.820±0.017 | 0.711±0.029 |

Note the `acc=0.756±0.000` for NRR alone: the model predicts the majority class (PD) in effectively every fold across every seed — that's the degenerate solution, not real classification, and the zero std confirms it's not noise.

**Reading the two CV tables together:** row-level and subject-grouped CV agree that VIC is the strongest single-feature compression of its three parents (index-alone AUC essentially matches parents-only AUC in both: 0.835 vs. 0.849 row-level, 0.833 vs. 0.834 grouped). They disagree somewhat on CCC's grouped-AUC ranking relative to its own parents (0.789 vs. 0.695 — the composite actually helps there, an interesting result on its own, though it doesn't change VIC being the top pick). Neither table is a claim that an index beats the full 22-feature raw set outright — it never will at this sample size; the comparison that matters is index-vs.-own-parents.

With only 195 rows / 32 subjects, expect real fold-to-fold noise: row-level CV shows ±0.03–0.08 swings on AUC; subject-grouped CV (only ~6 subjects per fold) needs the 10-seed average used here rather than a single split, which is why an earlier single-split version of this table (see the correction note above) briefly gave a different answer.

## 5. Side-by-side recap

See comparison table at the top of the document. Top pick: **VIC (Vocal Instability Compound)**, on both row-level and subject-grouped evidence. Runner-up: **CCC**, notably strong on AUC using its own single value versus its own parents grouped. **NRR** included for completeness and honesty about a weaker result — sitting on the majority baseline for accuracy/F1, with only AUC showing it's doing anything at all.

## 6. Proposed name for the winning index

> **Vocal Instability Compound (VIC)**

`VIC = log10(Jitter% × APQ3 × spread2 × 1000)`

Rationale: describes what the index measures (compounded cycle-to-cycle vocal instability, across frequency, amplitude, and spread) without echoing "fPI," "Feature Performance Index," or "Frequency Parkinson's Indicator" in wording or structure. "Compound" rather than "Index"/"Score"/"Indicator" was chosen deliberately, both because it reflects the multiplicative (compounding) math and because it avoids the original naming pattern outright. Checked against, and does not collide with, existing published or standard voice/speech-processing terms: Dysphonia Severity Index (DSI), Acoustic Voice Quality Index (AVQI), GRBAS, CAPE-V, and — since the obvious short form "PIC"/"CIS"-style acronyms were considered and rejected — Continuous Interleaved Sampling (CIS, a standard cochlear-implant coding strategy) and Linear Predictive Coding (LPC), both of which are established speech-signal-processing acronyms that a "P"/"L"-first alternative would have collided with. "VIC" was checked against the same list and against this dataset's own feature names (RAP, PPQ, DDP, APQ, DDA, RPDE, DFA, NHR, HNR, PPE) and does not overlap.

---

*No pipeline file, README, notebook, or asset-generator script was modified in producing this document. Analysis scripts used to produce these numbers were run from a scratch directory, not committed to the repo.*
