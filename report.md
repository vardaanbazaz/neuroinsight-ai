# NeuroInsight-AI — Repository Audit Report

**Scope:** Full read-only audit of `neuroinsight-ai` (commit `9aeabc6`, branch `main`). No files modified as part of this audit.

---

## 1. Project Understanding

### 1.1 What it actually does, end to end

- **Data**: `data/pd_data.xlsx` / `data/pd_data.txt` — the classic **UCI "Oxford Parkinson's Disease Detection Dataset"** (Little et al.), 195 voice-recording rows from 31 people (23 PD, 8 healthy), 22 acoustic features + `name` + `status`.
- **Features → Model**: Standard tabular pipeline — scale/normalize numeric features, optionally reduce Jitter/Shimmer families via PCA or drop collinear columns, split train/test, fit DT / RF / SVM / KNN / XGBoost, report accuracy/precision/recall/F1 and ROC/PR curves.
- **Output**: A trained classifier that outputs `status` (0 = healthy, 1 = PD) probability from a single row of vocal acoustic measurements — a research demo, not a shipped inference pipeline (no saved model artifact, no `predict.py`, no CLI/API).

### 1.2 The fPI concept — does the math match the README?

**No — and this is the most important finding in this audit.** Full detail in §6 (Attribution) and §5, but summarizing the technical mismatch here since it bears directly on "does the math match":

- README (`README.md:122`) states: `fPI = log10(DFA × D2 × spread2)`, and calls it the **"Feature Performance Index."**
- This formula is only ever computed in **one file in the entire repo**: `generate_clean_assets.py:38`. It is **never computed in any of the five notebooks** (`eda.ipynb`, `pd_models.ipynb`, `xgboost_classifier.ipynb`, `pd_alpha.ipynb`, `pd_beta.ipynb`) — verified by parsing every code cell's source in all five files; the string `fPI`/`log10` appears nowhere in notebook source, only coincidentally inside base64 image blobs in cell outputs.
- The README's own "Notebook Showcase" (`README.md:166`) claims `eda.ipynb` "explores the math behind the Feature Performance Index (fPI)." It does not — that notebook (33 cells) never references DFA, D2, or spread2 together, never computes a log, never has a cell mentioning fPI.
- Even where `generate_clean_assets.py` computes fPI, it appends it as a 23rd feature while **leaving DFA, D2, and spread2 in `X` unchanged** (`generate_clean_assets.py:38-41`) — so the model trains on the engineered index *and* its three raw ingredients simultaneously. That's not "feature selection to reduce multicollinearity" (which the README claims as pipeline step 4); it's the opposite — it manufactures a deterministic linear-in-log-space combination of three already-present columns and hands all four to a tree model, which will absorb it as redundant capacity, not as reduced dimensionality.
- The **source formula for fPI is not this project's own invention** — see §6. The original paper defines it as `fPI = log10(D2 × DFA) × spread2` (log applied to D2·DFA only, then multiplied by spread2 outside the log) and calls it the **"Frequency Parkinson's Indicator,"** not "Feature Performance Index." `README.md:122` and `generate_clean_assets.py:38` both use `log10(DFA × D2 × spread2)` — log applied to the full triple product. These are **not the same function** (`log(ab)·c ≠ log(abc)`). So this repo has silently changed both the name and the formula of someone else's named index while presenting it as original.

**Conclusion**: the headline scientific claim of the README is disconnected from the codebase that supposedly demonstrates it, and is a mutated restatement of a named external paper's result, uncredited.

---

## 2. Architecture & Structure

### 2.1 Folder map

```
neuroinsight-ai/
├── README.md                      duplicate of README_neuroinsight_ai.md (byte-identical)
├── README_neuroinsight_ai.md      duplicate — orphaned leftover, likely from a rename
├── LICENSE                        MIT, sole copyright "Vardaan" (see §6)
├── .gitignore                     standard Python/Jupyter template, appropriate
├── requirements.txt               8 packages, unpinned upper bounds, 2 imports MISSING (see §3.2)
├── generate_clean_assets.py       undocumented script that produces the README's headline
│                                  metrics table + 2 of 4 images — NOT referenced anywhere in README
├── data/
│   ├── pd_data.xlsx               used by all 5 notebooks
│   └── pd_data.txt                used only by generate_clean_assets.py (preferred path)
├── notebooks/
│   ├── eda.ipynb                  33 cells; EDA + F-tests + correlation heatmaps + single 70/30 split
│   │                              (SVM/RF/DT baselines); no fPI
│   ├── pd_models.ipynb            125 cells; repeats eda.ipynb's EDA, adds Ridge/RidgeCV regression
│   │                              on a binary target, GridSearchCV for DT/RF/SVM, a PCA-based
│   │                              feature scheme, KNN; no fPI
│   └── xgboost_classifier.ipynb   47 cells; PCA feature scheme, DT/RF/XGBoost single-split fit;
│                                  no fPI, no hyperparameter tuning, no 5-fold CV
├── docs/
│   ├── report.pdf                 Turnitin submission cover page for "Mrinal Bhan" + the actual
│                                  source paper "fPI: A Novel Index for Predictive Analysis of
│                                  Parkinson's Disease..." by Gautam Gupta, Mrinal Bhan, Sahil
│                                  Nimsarkar (IIIT Naya Raipur) — see §6
│   ├── parkinsons_prediction.pdf  slide deck "Created By: Mrinal Bhan & Gautam Gupta"
│   └── problem_statement.pdf      UCI dataset card + assignment brief ("Try at least three ML
│                                  approaches...") — this is a course assignment prompt, not this
│                                  project's own research question
├── assets/
│   ├── architecture.jpg           AI-generated workflow diagram, fine as decorative asset
│   ├── decision_tree.png          plot_tree() output — see §3.3, feature names never passed in,
│                                  so it shows x[5], x[8], x[0]... not PPE/fPI as README claims
│   ├── model_metrics.png          reproducibly generated by generate_clean_assets.py
│   └── roc_pr_curves.jpg          NOT reproducibly generated — script writes .png, repo has .jpg
└── testing/experimental/
    ├── pd_alpha.ipynb             89 cells — earlier draft of pd_models.ipynb, same content,
    │                              broken/incomplete in places (see §3.1)
    └── pd_beta.ipynb              47 cells — later draft; genuinely broken (NameError on run,
                                   see §3.1)
```

### 2.2 Structural inconsistencies vs. README

| README claim | Reality |
|---|---|
| `README.md:89` lists `assets/architecture.png` | File is `assets/architecture.jpg` |
| `README.md:92` lists `assets/roc_pr_curves.png` | File is `assets/roc_pr_curves.jpg` |
| `README.md:41` "partitioning into stratified train/test folds to prevent data leakage" | **Zero** occurrences of `stratify=` in any notebook (`grep stratify` across `notebooks/`, `testing/` returns nothing); every `train_test_split` call omits it. Only `generate_clean_assets.py` uses `StratifiedKFold`. |
| `README.md:43` "Feature Selection: Filtering and correlation-based selection to reduce multicollinearity" | Three *different, mutually inconsistent* feature schemes exist across the notebooks (raw-drop in `eda.ipynb`, PCA-reduced in `pd_models.ipynb`/`xgboost_classifier.ipynb`, all-raw+fPI in `generate_clean_assets.py`) — none is "the" pipeline; the README describes a single linear pipeline that doesn't exist in the code. |
| `README.md:166` eda.ipynb "explores the math behind fPI" | Not present in the notebook (§1.2). |
| Repo structure section (`README.md:62-98`) | Does not mention `generate_clean_assets.py` at all, despite it being the actual source of the flagship metrics table and 2 of the 4 images. |

### 2.3 Orphaned / unreferenced files

- **`README_neuroinsight_ai.md`** — byte-identical to `README.md` (`diff` confirms zero differences). GitHub renders only `README.md` by default; this second file is dead weight and, worse, will silently drift out of sync the next time someone edits one and not the other.
- **`generate_clean_assets.py`** — functionally the most important file in the repo (it's what actually produced the numbers everyone will read first), yet it's invisible in the README's file tree, notebook showcase, and has no docstring/header comment explaining what it is or when to run it.

---

## 3. Code Quality

### 3.1 Notebooks — line-by-line issues

**`notebooks/eda.ipynb`**
- Cell 0: 20+ imports, at least 9 unused in this notebook — `math`, `variance_inflation_factor`, `ttest_ind`, `chi2_contingency`, `resample`, `smt` (statsmodels.stats.multitest), `LogisticRegression`, `Naive_Bayes`, `GridSearchCV`, `confusion_matrix`, `cross_val_score`, `make_classification`, `XGBClassifier` (imported, never instantiated).
- Cell 1: hardcoded relative path `'../data/pd_data.xlsx'` — notebook only runs if the working directory is exactly `notebooks/`; no `os.path` guard, unlike `generate_clean_assets.py:31-33` which does have one.
- Cells 9, 23: `ftest_results.append({...}, ignore_index=True)` — `DataFrame.append` was **removed in pandas 2.0** (deprecated since 1.4). `requirements.txt` pins `pandas>=1.4.0` with no upper bound, so a fresh `pip install` today pulls pandas 2.x/3.x and this cell **raises `AttributeError` on execution**. This breaks 4 of the 5 notebooks (`eda`, `pd_models`, `pd_alpha`, `pd_beta` all share this line).
- Cell 19-20: `StandardScaler().fit_transform(df)` fit on the **full 195-row dataset**, then split into train/test at cell 25 — classic **scale-before-split leakage**: test-set statistics contaminate the scaler used to transform the training set.
- Cell 25: `train_test_split(X, y, test_size=0.3, random_state=20)` — no `stratify=y`, contradicting the README's explicit claim of stratified splitting.
- No docstrings on any function except `f_test` (cell 9) — the rest of the notebook is unstructured chained cells with no explanatory markdown between major sections beyond two one-liners.

**`notebooks/pd_models.ipynb`** (125 cells — largest notebook)
- Same unused-import list, same hardcoded path, same `.append()` breakage, same fit-before-split leakage (cell 27), same missing `stratify` (cells 38, 98).
- **Real correctness bug**, cell 63:
  ```python
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(X_train, y_train)
  predKNN = knn.predict(X_test)
  print(classification_report(y_test, yhat))   # <-- yhat, not predKNN
  ```
  `yhat` is a leftover variable from the `for n in range(2,Ks)` sweep two cells earlier (cell 61) — it holds predictions from `n_neighbors=9` (the last loop iteration), not the `n_neighbors=3` model just fitted. The printed classification report is **for the wrong model**. This exact bug is repeated at cell 120 with `n_neighbors=2`.
- PCA fit on the full un-split dataset (cells 73, 85) — same leakage pattern as scaling, now for dimensionality reduction: the "Jitter_pc"/"Shimmer_pc" components used downstream in `xgboost_classifier.ipynb` were derived using information from rows that later become the test set.
- Ridge/RidgeCV regression (cells 38-42) fits linear regression + MSE/R²/adjusted-R² against a **binary 0/1 classification target** — a legitimate exploratory move but never mentioned in the README's methodology or notebook showcase, and is not treated as a classifier (no threshold, no accuracy).
- `random_state` values are inconsistent across cells with no rationale: 20 (main split), 890 (Ridge split), 30/100/150/250 range in `GridSearchCV` param grids, 0 (permutation importance) — makes results non-reproducible fold-to-fold and undermines the notion of a single canonical pipeline.
- 5 different `GridSearchCV` grids searched but **none of the resulting best estimators feed into the README's headline metrics table** — that table comes exclusively from `generate_clean_assets.py`'s untuned, default-hyperparameter models.

**`notebooks/xgboost_classifier.ipynb`**
- Same shared header cell with the same unused imports.
- Cell 41:
  ```python
  fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
  tree.plot_tree(clf, filled=True)
  fig.savefig('imagename.png')          # hardcoded placeholder filename, not 'decision_tree.png'
  tree.plot_tree(clf);                  # second, redundant, unfilled call — dead code, no output captured
  ```
  No `feature_names=X.columns` passed to `plot_tree` — this is confirmed visually in `assets/decision_tree.png`, whose nodes read `x[5] <= -6.649`, `x[8] <= 0.113`, `x[0] <= 118.141` etc., **not** feature names. The README (`README.md:158`) claims this image lets you "examine splits on features such as `PPE` and the fPI components" — it is structurally impossible to verify that claim from the image as generated, since no feature is named in it.
  - Separately, `'imagename.png'` ≠ `assets/decision_tree.png`; the actual repo asset must have been renamed and moved by hand outside any script, so nothing currently in the repo reproduces it.
- Only model reported is a single default-hyperparameter fit (cell 45-46, no `GridSearchCV`) despite the README (`README.md:44`) claiming "hyperparameter tuning using Grid Search" for the XGBoost step specifically — the actual grid searches happen only for DT/RF/SVM in `pd_models.ipynb`, never for XGBoost anywhere.

**`testing/experimental/pd_alpha.ipynb`** — essentially a duplicate/earlier draft of `pd_models.ipynb` (same cell content through cell 39 where comparison stopped); path is `../../data/pd_data.xlsx` (two levels up, correct for its deeper location).

**`testing/experimental/pd_beta.ipynb`** — genuinely broken if run top-to-bottom: cell 6 references `amplitude_p_values` and `frequency_p_values` before they are ever defined anywhere in the notebook (those computations exist in `pd_alpha.ipynb`/`pd_models.ipynb` but were deleted or reordered here) — **`NameError` on execution**. The README's own label for this folder — "Legacy files and sandbox tests" (`README.md:96`) — is accurate; no complaint there, just confirming it.

### 3.2 Reproducibility — can someone clone and run this end-to-end?

**No, not as-is.** Concrete blockers:

1. **`statsmodels` is imported in all 5 notebooks** (`import statsmodels.api as sm`, `from statsmodels.stats.outliers_influence import variance_inflation_factor`, `import statsmodels.stats.multitest as smt` — each notebook's cell 0) but **is absent from `requirements.txt`**. It is not a transitive dependency of any listed package. `pip install -r requirements.txt` followed by opening any notebook fails immediately at the first code cell with `ModuleNotFoundError: No module named 'statsmodels'`.
2. **`pandas>=1.4.0`** with no upper bound will resolve to pandas 2.x/3.x today, and `DataFrame.append` (used in `eda.ipynb`, `pd_models.ipynb`, `pd_alpha.ipynb`, `pd_beta.ipynb`) was removed in pandas 2.0 — those cells crash.
3. No `stratify=y` anywhere means results aren't even reproducible run-to-run relative to what the README describes, independent of the crash.
4. No trained model is persisted (no `.model`/`.pkl`/`.json` XGBoost file) despite the README's "Sample Prediction Workflow" (`README.md:172-202`) showing commented-out `xgb_model.load_model(...)` — there is nothing to load; the sample workflow is aspirational pseudocode, not something a cloner can actually run.
5. `generate_clean_assets.py` is the one artifact that *does* run cleanly top-to-bottom (verified by reading it in full — no leakage, correct stratified CV, no undefined names) — but it isn't mentioned in the README at all, so a newcomer has no way to know it's the thing to run.

### 3.3 Data leakage / methodology summary

| Location | Issue |
|---|---|
| `eda.ipynb:19-20`, `pd_models.ipynb:27`, `xgboost_classifier.ipynb:11` | `StandardScaler` fit on full dataset before split |
| `pd_models.ipynb:73,85`, `xgboost_classifier.ipynb:22,29` | `PCA` fit on full dataset before split |
| all `train_test_split` calls in notebooks | no `stratify=y`, contradicting README |
| `generate_clean_assets.py` | **This is the one place the pipeline is done correctly** — per-fold `StandardScaler` fit inside the CV loop (`generate_clean_assets.py:87-90`), `StratifiedKFold` (`:74`), out-of-fold probabilities for ROC/PR curves. Ironically the only methodologically sound code in the repo is the one file the README never shows anyone. |

---

## 4. Configuration & Metadata

- **`.gitignore`** (`.gitignore:1-146`) — standard, comprehensive Python/Jupyter/OS template. Appropriate for this project; no changes needed. `.venv` correctly excluded and confirmed not tracked (`git ls-files` has no `.venv/*`).
- **`requirements.txt`** — incomplete (missing `statsmodels`; `scipy` also imported directly (`from scipy import stats`) but not listed — it happens to install transitively via scikit-learn, so it doesn't break, but relying on an undeclared transitive dependency is fragile). No upper-bound pins, no lockfile, no Python version file — see §3.2.
- **`LICENSE`** — MIT, `Copyright (c) 2026 Vardaan`. Legally fine for the code as written, but see §6: the sole-copyright framing sits awkwardly next to a "novel index" and methodology lifted from a named 3-author academic paper with no reference back to them anywhere in the license, README, or code.
- **README.md vs README_neuroinsight_ai.md** — confirmed via `diff`: **byte-for-byte identical**, 221 lines each. `README.md` is the one GitHub renders; `README_neuroinsight_ai.md` is dead weight, almost certainly left over from an early filename choice before renaming to the repo-standard `README.md`. **Delete it.**
- **Internal/image links in README.md**:
  - `assets/architecture.jpg` (`README.md:36`) — resolves ✓ (file exists, matches extension).
  - `assets/model_metrics.png` (`README.md:151`) — resolves ✓.
  - `assets/roc_pr_curves.jpg` (`README.md:154`) — resolves ✓ (file exists as `.jpg`, matches what's linked — but disagrees with the repo-structure table earlier in the same document, and with what `generate_clean_assets.py` actually writes; see §2.2).
  - `notebooks/eda.ipynb`, `notebooks/pd_models.ipynb`, `notebooks/xgboost_classifier.ipynb` (`README.md:166-168`) — resolve ✓.
  - `LICENSE` (`README.md:6`) — resolves ✓.
  - Badge links (`README.md:5-10`) all point to `(#)` (no-op anchors) except the LICENSE badge — cosmetic, low priority.

---

## 5. Redundancies & Cleanup

| File | Status | Recommendation |
|---|---|---|
| `README_neuroinsight_ai.md` | Exact duplicate of `README.md` | Delete |
| `testing/experimental/pd_alpha.ipynb` | Earlier draft of `pd_models.ipynb`, same boilerplate, same bugs, incomplete | Keep only if you want to show iteration history; otherwise delete — it adds 2.3MB of embedded output images for zero unique signal |
| `testing/experimental/pd_beta.ipynb` | Later draft, contains a hard `NameError` if executed | Delete, or fix and relabel — a portfolio visitor who opens it and hits an undefined-name crash reads as sloppy |
| `generate_clean_assets.py` | Not a duplicate, but effectively invisible/orphaned relative to docs | Keep, but document it prominently (see punch-list) |
| `assets/roc_pr_curves.jpg` vs. script output `assets/roc_pr_curves.png` | Filename drift between generator and generated artifact | Reconcile — either change the script to write `.jpg`, or convert/commit a `.png` and fix the README's `<img>` tag |
| Five copies of the same 40-line import cell (`eda.ipynb`, `pd_models.ipynb`, `xgboost_classifier.ipynb`, `pd_alpha.ipynb`, `pd_beta.ipynb`) | Not a "file" redundancy but a strong duplication smell — identical unused-import block copy-pasted five times | Not urgent to fix given notebooks are being consolidated anyway, but worth noting as the clearest signal these notebooks all trace back to one shared original file (see §6) rather than being independently authored |
| `data/pd_data.txt` and `data/pd_data.xlsx` | Both genuinely used (txt by the script, xlsx by notebooks) | Keep both, not redundant |

**Copied-but-not-adapted markers**: the shared 40-line import cell at the top of all five notebooks (`# from sklearn.metrics import plot_confusion_matrix` commented out, `Naive_Bayes` aliasing `GaussianNB` that's never used, `make_classification` import never used) reads as boilerplate carried over from a shared template/original notebook rather than something written fresh for each file — consistent with the provenance finding in §6.

---

## 6. Attribution & Provenance — the critical finding

This is not a minor housekeeping issue; it changes how the whole README should be framed.

**`docs/report.pdf`** page 1 is a **Turnitin plagiarism-checker cover sheet**: `Submission date: 08-May-2023`, `Submission ID: 2086656757`, `File name: MLA_Final_1.docx`, **`by Mrinal Bhan`**. Page 2 is the actual paper it wraps:

> **"fPI: A Novel Index for Predictive Analysis of Parkinson's Disease Using Acoustic Sound Feature"**
> Authors: **¹Gautam Gupta, ²Mrinal Bhan, ³Sahil Nimsarkar** — Data Science & AI, **International Institute of Information Technology, Naya Raipur, India**, with `@iiitnr.edu.in` institutional emails visible for all three.

Page 3 of that same PDF gives the paper's own definition:

> "We formulate a novel index called **Frequency Parkinson's Indicator (fPI)**... `fPI = log₁₀(D2 × DFA) × spread2`"

`docs/parkinsons_prediction.pdf` is the companion slide deck, title page: **"Created By: Mrinal Bhan & Gautam Gupta."**

**What this means for this repo, concretely:**

1. The **entire premise of the README's title** ("...the Feature Performance Index (fPI)...") and its framing as *this* repository's "custom-engineered" contribution (`README.md:16`: "By utilizing a custom-engineered **Feature Performance Index (fPI)**...") is a **rebrand of someone else's named, already-published index** ("Frequency Parkinson's Indicator"), with **the formula itself altered** (`log(ab)·c` → `log(abc)`) and **zero citation** anywhere in the README, notebooks, or LICENSE to Gupta, Bhan, Nimsarkar, or IIIT Naya Raipur.
2. The dataset description, problem framing ("investigate predictive capability... compare DT/SVM/RF/XGBoost"), and even the specific models compared trace directly to `docs/problem_statement.pdf`'s assignment brief ("Try at least three different machine learning approaches to PD identification... Which approach is better and why?") — this reads as a course/capstone assignment for a specific class at IIIT Naya Raipur, not an independently conceived "applied ML research project," as the README's Hero Section (`README.md:16`) presents it.
3. `LICENSE:3` assigns sole copyright to "Vardaan" for a codebase whose central intellectual contribution (the index, the paper title, the abstract's claims) is co-authored work by three other named people, uncredited.
4. `git log --all` shows only 3 commits, all authored by `vardaanbazaz <vardaanbazaz@gmal.com>` — meaning the original collaborative history (if there ever was a shared repo with Gupta/Bhan/Nimsarkar as committers) was **not preserved**; this repo was seeded fresh, so there is no git-level record of the original team's contribution either.

**This needs to be fixed before this goes on a portfolio.** As currently written, a technical reviewer who finds `docs/report.pdf` in the repo (which is public, committed, and linked from nothing but sitting right there in `docs/`) will read this as an unattributed reuse of classmates'/collaborators' published research presented as solo original work. That is a materially worse look than any of the code-quality issues above, and it's an easy, fast fix: add a clear "Acknowledgments / Origin" section naming the original paper and authors, correct the fPI formula and name (or explicitly state you're using a modified variant and why), and clarify what *you* (Vardaan) actually added on top (e.g., the 5-fold stratified CV pipeline in `generate_clean_assets.py`, which — worth noting — genuinely is clean, original, and the best-engineered file in the repo, and is a legitimate thing to take credit for).

---

## 7. Portfolio Readiness

### Rating: 4/10

**Reasoning**: The README itself is polished, well-formatted, and reads confidently — on a skim, it looks like a strong portfolio piece. But every layer beneath the surface has a problem: the headline metrics table isn't reproducible from anything the README points a reader to; the notebooks it showcases don't contain the technique (fPI) the whole project is named after; a fresh clone crashes on a missing dependency; the one demonstrative image (`decision_tree.png`) can't show what it's captioned to show; and the central "novel index" is an uncredited, subtly altered reuse of a named external paper sitting in the repo's own `docs/` folder. A technical reviewer who does exactly what this audit did — open the notebooks and the PDFs — will find all of this within minutes. None of it is unfixable, and most of it is fast.

### Punch-list

**Critical (fix before linking this anywhere):**
1. Add an "Origin / Acknowledgments" section to the README crediting Gautam Gupta, Mrinal Bhan, Sahil Nimsarkar, and IIIT Naya Raipur for the original fPI concept/paper; clarify what part of the project is your own extension (the 5-fold CV pipeline in `generate_clean_assets.py` is genuinely yours to claim).
2. Fix the dataset description: `README.md:104` claims "198 subjects (140 with Parkinson's, 58 healthy controls)"; the actual data (`data/pd_data.txt`, verified by loading) is **195 rows, 147 PD, 48 healthy**, from 31 people. All three numbers are wrong.
3. Correct the fPI formula (`log10(DFA × D2 × spread2)` vs. the source paper's `log10(D2 × DFA) × spread2`) — decide which one you're actually using and make the README, `generate_clean_assets.py`, and any future notebook code agree.
4. Add `statsmodels` to `requirements.txt` (hard blocker — every notebook's first cell fails without it).
5. Pin or upper-bound `pandas` (or replace `.append()` calls with `pd.concat()` in `eda.ipynb`, `pd_models.ipynb`, `pd_alpha.ipynb`, `pd_beta.ipynb`) so a fresh install doesn't crash on cell 1 of most notebooks.
6. Either compute fPI inside the showcased notebooks (so the claim in `README.md:166` is true) or stop claiming `eda.ipynb` explores it.
7. Fix `assets/decision_tree.png` reproducibility: pass `feature_names=X.columns` to `tree.plot_tree`, save directly to `assets/decision_tree.png` (not `imagename.png`), so the image actually shows what the README says it shows.
8. Delete `README_neuroinsight_ai.md`.

**Nice-to-haves:**
9. Reference `generate_clean_assets.py` in the README (repo structure + a "how to reproduce these numbers" note) — right now it's the single most important file and it's undocumented.
10. Fix the `knn`/`yhat` mismatch in `pd_models.ipynb` cells 63 and 120.
11. Add `stratify=y` to every `train_test_split` call, and move all `StandardScaler`/`PCA` fitting inside the train-split boundary (or inside a CV loop, matching `generate_clean_assets.py`'s already-correct pattern) to remove the leakage flagged in §3.3.
12. Reconcile `assets/roc_pr_curves.jpg` vs. the script's `.png` output — pick one extension and make the script and repo agree.
13. Fix the two filename typos in the README's repo-structure section (`architecture.png`→`.jpg`, `roc_pr_curves.png`→`.jpg`, or vice versa if you standardize on `.png`).
14. Trim unused imports from the shared notebook header cell (9+ unused names per notebook × 5 notebooks).
15. Decide the fate of `testing/experimental/` — either fix `pd_beta.ipynb`'s `NameError` or drop the folder; a portfolio visitor who opens a notebook and hits a crash on cell 6 reflects worse than not having the folder at all.

**Optional polish:**
16. Add a single consistent `random_state` (you already use 42 in `generate_clean_assets.py`) across every notebook instead of the current 20/30/100/150/250/890/0 scatter.
17. Save an actual trained XGBoost model file so the "Sample Prediction Workflow" code block in the README (`README.md:172-202`) is runnable rather than pseudocode with commented-out lines.
18. Consider consolidating `eda.ipynb` + `pd_models.ipynb` + `xgboost_classifier.ipynb` into one coherent, single-feature-scheme pipeline notebook that actually matches the README's 7-step workflow diagram — right now the diagram (`assets/architecture.jpg`) describes a pipeline none of the three notebooks individually implements.
