# CONSISTENCY_CHECK.md — README_DRAFT.md vs. Current Repo State (re-verification pass)

**Scope:** Read-only re-audit of `README_DRAFT.md` against the current repo, focused on re-verifying the fixes applied in the last session, plus a full re-run of the original checklist categories. No files edited, no git commands run.

## VERDICT: ✅ PASS — safe to promote to README.md

All previously-flagged critical/high issues are confirmed fixed and consistent with the repo. No new discrepancies found.

---

## Summary

| Check | Result |
|---|---|
| 1. File/path references | ✅ Pass — `testing/` reference removed, all listed files exist |
| 2. Section numbering / anchors | ✅ Pass — all headers unchanged, all anchors resolve correctly |
| 3. Number consistency (prose vs. tables) | ✅ Pass — subject count now 32 everywhere |
| 4. Image-text consistency | ✅ Pass |
| 5. Claim traceability (§8/§9/§10/§13) | ✅ Pass — all numbers trace to source docs |
| 6. Implementation Status accuracy | ✅ Pass — VIC wiring language now matches code |
| 7. Acknowledgments presence | ✅ Pass |

---

## Re-verification of items fixed last session

### VIC wiring language (§3, §8, §11, §12, §14)

`generate_clean_assets.py:38-39,46` confirmed:
```
df['VIC'] = np.log10(df['MDVP:Jitter(%)'] * df['Shimmer:APQ3'] * df['spread2'] * 1000)
X = df.drop(columns=['name', 'status'])
```
VIC is computed and stays in `X` (23-column matrix) — matches README's claim.

- **§3** (line 50): "validated offline on this dataset (§8) and **live in the trained pipeline**" — correct, no contradiction.
- **§8 Implementation Status** (line 150-152): "**VIC is live in the trained pipeline.**" — correct. Also confirms the old "legacy leftover fPI/DFA×D2 column" paragraph is gone; `grep -n "DFA\|fPI" generate_clean_assets.py` returns no such computation, and README no longer describes one. Consistent.
- **§11** (line 210): "VIC is live in `generate_clean_assets.py` (§8) but not yet wired into any notebook" — accurate: this is a claim about notebooks, not the pipeline, and notebooks genuinely don't compute VIC (checked — no `VIC` string in any of the three `.ipynb` files' visible pipeline code referenced by README). No contradiction with §8.
- **§12** (line 218): "the VIC feature engineering step that [§8](#implementation-status) confirms is live in `generate_clean_assets.py`" — correct.
- **§14** (line 266): reframed as "**VIC Notebook Integration**... even though `generate_clean_assets.py` (§8) already computes it" — correct, no longer implies VIC isn't in the pipeline.

No stale "not yet wired into the trained pipeline" / "not yet part of any trained/saved model" phrasing remains anywhere (grep confirmed zero matches).

### Subject count (§6)

Line 110: "195 voice recordings from **32 subjects**" — matches `data/pd_data.txt` (32 unique `S\d+` IDs, 195 data rows), `INDEX_PROPOSAL.md`, `FEATURE_IMPORTANCE_VALIDATION.md`, and both VIC validation docs. Fixed correctly.

### §5 file tree and root-doc list

- `testing/experimental/` entry removed from the tree — directory confirmed absent from repo (`[[ -d testing ]]` → false).
- `data/` listing now includes `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` and `parkinsons_updrs.data`/`.names` alongside `pd_data.xlsx`/`pd_data.txt` — all four confirmed present in `data/` at exactly those names.
- Root-doc paragraph (line 104) now lists `INDEX_PROPOSAL.md`, `VIC_EXTERNAL_VALIDATION.md`, `VIC_TELEMONITORING_VALIDATION.md`, **and `FEATURE_IMPORTANCE_VALIDATION.md`** — all four confirmed present at repo root, all four cited elsewhere in the doc.

---

## Full re-check of other categories

- **Section numbering/anchors**: all `## N.` headers unchanged (§1–§14 present, in order). Anchor links (`#8-methodology--vocal-instability-compound-vic`, `#9-external-validation--limitations`, `#13-research-insights`, `#implementation-status`) all resolve to the correct header via GitHub's slug rules. No broken links.
- **Image-text consistency**: `assets/architecture.jpg`, `model_metrics.png`, `roc_pr_curves.jpg`, `decision_tree.png` all exist and are referenced at the same paths as before; no text changes around these sections this session.
- **Claim traceability**: §8/§9/§13 numbers (r=0.489, AUC 0.833/0.734, Naranjo r=0.286/AUC 0.646, telemonitoring r≈0.23/p>0.14, PPE/D2/RPDE/Shimmer importance figures) unchanged from prior pass and still trace to `INDEX_PROPOSAL.md`, `VIC_EXTERNAL_VALIDATION.md`, `VIC_TELEMONITORING_VALIDATION.md`, `FEATURE_IMPORTANCE_VALIDATION.md`.
- **Acknowledgments**: present, unchanged, second block in file (lines 14-18), names Gupta/Bhan/Nimsarkar/IIIT Naya Raipur.

---

## Remaining open items (pre-existing, low severity, not part of this fix list — not blocking)

- **[LOW]** Fig. 4 pooling-inflation caveat (line ~199) still frames the OOF-pooling AUC inflation as a Decision-Tree-specific quirk, though KNN and SVM show smaller but real pooling effects too. Cosmetic; not a factual error.
- **[NIT]** Line 218 uses "§8" as link text for `#implementation-status` (a subsection of §8, not §8's own anchor) — link resolves fine, just slightly imprecise labeling.

Neither was in scope for this session's fixes and neither blocks promotion.
