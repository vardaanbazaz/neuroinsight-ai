import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Configure clean, modern light style suitable for README and LinkedIn media
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2

# Set random seed
np.random.seed(42)

# 1. Load dataset
data_path = 'data/pd_data.txt'
if not os.path.exists(data_path):
    data_path = 'data/pd_data.xlsx'
    df = pd.read_excel(data_path)
else:
    df = pd.read_csv(data_path)

# Compute Vocal Instability Compound (VIC = log10(Jitter% * Shimmer:APQ3 * spread2 * 1000))
df['VIC'] = np.log10(df['MDVP:Jitter(%)'] * df['Shimmer:APQ3'] * df['spread2'] * 1000)

# Extract subject IDs from name (e.g. phon_R01_S01_1 -> S01) for group-aware CV,
# before dropping the name column. Not used as a model feature.
groups = df['name'].str.extract(r'(S\d+)')[0].values

# Separate features X and target y
X = df.drop(columns=['name', 'status'])
y = df['status'].values

# Models configuration (strictly 5 requested models)
models_config = {
    'XGBoost': {
        'model_fn': lambda: xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'scale': False,
        'color': '#2563EB'  # Vibrant Royal Blue
    },
    'Random Forest': {
        'model_fn': lambda: RandomForestClassifier(random_state=42),
        'scale': False,
        'color': '#7C3AED'  # Purple
    },
    'SVM': {
        'model_fn': lambda: SVC(probability=True, random_state=42),
        'scale': True,
        'color': '#059669'  # Emerald Green
    },
    'Decision Tree': {
        'model_fn': lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
        'scale': False,
        'color': '#D97706'  # Amber / Warm Gold
    },
    'KNN': {
        'model_fn': lambda: KNeighborsClassifier(),
        'scale': True,
        'color': '#DC2626'  # Crimson Red
    }
}

# 5-Fold Subject-Grouped Stratified Cross-Validation, averaged over 10 seeds
# (0-9). With only ~6 subjects per fold a single split can flip model
# rankings (see INDEX_PROPOSAL.md's own earlier-draft correction note), so
# every reported number here is mean +/- std over 50 folds (5 folds x 10
# seeds), same scheme INDEX_PROPOSAL.md used for VIC's own validation.
SEEDS = range(10)
STD_DDOF = 0  # population std throughout: np.std default, and pandas .std(ddof=0) below

eval_start = time.time()

metrics_summary = []
oof_probabilities = {}
seed_diagnostics = {}  # name -> list of per-seed mean dicts (10 entries), all models
fold_degeneracy = {}  # name -> (n folds predicting all-PD, n folds with recall==1.000, n folds total)

for name, config in models_config.items():
    oof_probs_sum = np.zeros(len(y))
    oof_probs_count = np.zeros(len(y))
    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []
    per_seed_summaries = []
    n_all_pd_folds, n_recall1_folds = 0, 0

    for seed in SEEDS:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        seed_accs, seed_precs, seed_recs, seed_f1s = [], [], [], []
        oof_probs_this_seed = np.zeros(len(y))

        for train_idx, val_idx in skf.split(X, y, groups=groups):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y[train_idx], y[val_idx]

            if config['scale']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = config['model_fn']()
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]

            # Average OOF probability per sample across the 10 seeds (each
            # seed covers every sample exactly once) for stable ROC/PR curves.
            oof_probs_sum[val_idx] += probs
            oof_probs_count[val_idx] += 1
            oof_probs_this_seed[val_idx] = probs

            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)

            if np.all(preds == 1):
                n_all_pd_folds += 1
            if rec == 1.0:
                n_recall1_folds += 1

            fold_accs.append(acc)
            fold_precs.append(prec)
            fold_recs.append(rec)
            fold_f1s.append(f1)
            seed_accs.append(acc)
            seed_precs.append(prec)
            seed_recs.append(rec)
            seed_f1s.append(f1)

        seed_fpr, seed_tpr, _ = roc_curve(y, oof_probs_this_seed)
        per_seed_summaries.append({
            'Seed': seed,
            'Accuracy': np.mean(seed_accs),
            'Precision': np.mean(seed_precs),
            'Recall': np.mean(seed_recs),
            'F1-score': np.mean(seed_f1s),
            'AUC': auc(seed_fpr, seed_tpr)
        })

    seed_diagnostics[name] = per_seed_summaries
    oof_probabilities[name] = oof_probs_sum / oof_probs_count

    seed_df = pd.DataFrame(per_seed_summaries)
    metrics_summary.append({
        'Model': name,
        'Accuracy': np.mean(fold_accs),
        'Accuracy_std': np.std(fold_accs),
        'Accuracy_seed_std': seed_df['Accuracy'].std(ddof=STD_DDOF),
        'Precision': np.mean(fold_precs),
        'Precision_std': np.std(fold_precs),
        'Precision_seed_std': seed_df['Precision'].std(ddof=STD_DDOF),
        'Recall': np.mean(fold_recs),
        'Recall_std': np.std(fold_recs),
        'Recall_seed_std': seed_df['Recall'].std(ddof=STD_DDOF),
        'F1-score': np.mean(fold_f1s),
        'F1-score_std': np.std(fold_f1s),
        'F1-score_seed_std': seed_df['F1-score'].std(ddof=STD_DDOF),
        'AUC_seed_mean': seed_df['AUC'].mean(),
        'AUC_seed_std': seed_df['AUC'].std(ddof=STD_DDOF)
    })
    fold_degeneracy[name] = (n_all_pd_folds, n_recall1_folds, len(fold_recs))

# Majority-class baseline (predict PD for everyone), computed over the same
# 50 folds so it is exactly comparable to the model rows above — this is the
# bar INDEX_PROPOSAL.md sets (analytic full-dataset figures: acc 0.754 /
# F1 0.860; the fold-averaged numbers below will differ slightly by
# fold composition, that's expected).
base_accs, base_precs, base_recs, base_f1s = [], [], [], []
for seed in SEEDS:
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for _, val_idx in skf.split(X, y, groups=groups):
        y_val = y[val_idx]
        preds = np.ones_like(y_val)
        base_accs.append(accuracy_score(y_val, preds))
        base_precs.append(precision_score(y_val, preds, zero_division=0))
        base_recs.append(recall_score(y_val, preds, zero_division=0))
        base_f1s.append(f1_score(y_val, preds, zero_division=0))

baseline_row = {
    'Model': 'Majority baseline (predict all-PD)',
    'Accuracy': np.mean(base_accs), 'Accuracy_std': np.std(base_accs), 'Accuracy_seed_std': np.nan,
    'Precision': np.mean(base_precs), 'Precision_std': np.std(base_precs), 'Precision_seed_std': np.nan,
    'Recall': np.mean(base_recs), 'Recall_std': np.std(base_recs), 'Recall_seed_std': np.nan,
    'F1-score': np.mean(base_f1s), 'F1-score_std': np.std(base_f1s), 'F1-score_seed_std': np.nan,
    'AUC_seed_mean': np.nan, 'AUC_seed_std': np.nan
}

eval_elapsed = time.time() - eval_start

res_df = pd.DataFrame(metrics_summary)

print(f"=== 5-Fold Subject-Grouped Stratified CV, averaged over {len(SEEDS)} seeds "
      f"(0-{len(SEEDS)-1}), 50 folds total per model ===")
print("(+/- is per-fold std over all 50 folds, per the task spec; seed-level std — how much "
      "the 10 seed-means themselves vary, INDEX_PROPOSAL.md's convention — printed alongside "
      "since it's much tighter and the two are easy to conflate.)")
full_table = pd.concat([pd.DataFrame([baseline_row]), res_df], ignore_index=True)
display_df = pd.DataFrame({
    'Model': full_table['Model'],
    'Accuracy': full_table.apply(lambda r: f"{r['Accuracy']:.3f} +/- {r['Accuracy_std']:.3f} (seed +/- {r['Accuracy_seed_std']:.3f})" if pd.notna(r['Accuracy_seed_std']) else f"{r['Accuracy']:.3f} +/- {r['Accuracy_std']:.3f}", axis=1),
    'Precision': full_table.apply(lambda r: f"{r['Precision']:.3f} +/- {r['Precision_std']:.3f} (seed +/- {r['Precision_seed_std']:.3f})" if pd.notna(r['Precision_seed_std']) else f"{r['Precision']:.3f} +/- {r['Precision_std']:.3f}", axis=1),
    'Recall': full_table.apply(lambda r: f"{r['Recall']:.3f} +/- {r['Recall_std']:.3f} (seed +/- {r['Recall_seed_std']:.3f})" if pd.notna(r['Recall_seed_std']) else f"{r['Recall']:.3f} +/- {r['Recall_std']:.3f}", axis=1),
    'F1-score': full_table.apply(lambda r: f"{r['F1-score']:.3f} +/- {r['F1-score_std']:.3f} (seed +/- {r['F1-score_seed_std']:.3f})" if pd.notna(r['F1-score_seed_std']) else f"{r['F1-score']:.3f} +/- {r['F1-score_std']:.3f}", axis=1),
})
print(display_df.to_string(index=False))
print(f"\n10-seed evaluation loop took {eval_elapsed:.1f}s "
      f"({len(models_config)} models x 10 seeds x 5 folds = {len(models_config) * 50} total fits).")

# Per-fold degeneracy check (flagged last session on the single seed-42
# split, where SVM recall was exactly 1.000): a seed-level recall mean can
# hide the fold-by-fold picture, so count folds directly — all-PD
# prediction (zero predicted negatives) and recall==1.000 (zero false
# negatives; a weaker condition than all-PD) separately, all 5 models.
print("\n=== Per-fold degeneracy check (all-PD prediction / recall==1.000 folds, 50 folds each) ===")
for name in models_config.keys():
    n_all_pd, n_rec1, n_total = fold_degeneracy[name]
    print(f"{name:15s} predicted all-PD in {n_all_pd}/{n_total} folds, recall==1.000 in {n_rec1}/{n_total} folds")

# ROC/PR curves below use OOF probabilities averaged per-sample across the
# 10 seeds (variance-reduction / light bagging), not a single-seed OOF
# curve, so the legend AUC runs a bit above the mean of the 10 per-seed
# AUCs and isn't directly comparable to INDEX_PROPOSAL.md's single-model
# AUCs. Printing both for the record — this is also the primary evidence
# for the SVM finding below, more informative than precision/recall alone.
print("\n=== ROC-AUC: 10-seed-averaged-probs (curve/legend value) vs mean of per-seed AUCs ===")
for name in models_config.keys():
    ensemble_fpr, ensemble_tpr, _ = roc_curve(y, oof_probabilities[name])
    ensemble_auc = auc(ensemble_fpr, ensemble_tpr)
    row = next(r for r in metrics_summary if r['Model'] == name)
    print(f"{name:15s} averaged-probs AUC = {ensemble_auc:.3f}   |   "
          f"per-seed AUC mean +/- std = {row['AUC_seed_mean']:.3f} +/- {row['AUC_seed_std']:.3f}")

# SVM recall=1.000 degeneracy check: per-seed recall/precision/AUC spread,
# 10 seeds, compared against the majority-baseline row above.
svm_diag_df = pd.DataFrame(seed_diagnostics['SVM'])
print("\n=== SVM per-seed diagnostic ===")
print(svm_diag_df.to_string(index=False))
svm_recall_mean, svm_recall_std = svm_diag_df['Recall'].mean(), svm_diag_df['Recall'].std(ddof=STD_DDOF)
svm_prec_mean, svm_prec_std = svm_diag_df['Precision'].mean(), svm_diag_df['Precision'].std(ddof=STD_DDOF)
svm_auc_mean, svm_auc_std = svm_diag_df['AUC'].mean(), svm_diag_df['AUC'].std(ddof=STD_DDOF)
svm_f1_mean = svm_diag_df['F1-score'].mean()
n_all_pd_svm, n_rec1_svm, n_total_svm = fold_degeneracy['SVM']
print(f"SVM recall across seeds: {svm_recall_mean:.3f} +/- {svm_recall_std:.3f} "
      f"(min {svm_diag_df['Recall'].min():.3f}, max {svm_diag_df['Recall'].max():.3f})")
print(f"SVM precision across seeds: {svm_prec_mean:.3f} +/- {svm_prec_std:.3f} "
      f"(min {svm_diag_df['Precision'].min():.3f}, max {svm_diag_df['Precision'].max():.3f})")
print(f"SVM AUC across seeds: {svm_auc_mean:.3f} +/- {svm_auc_std:.3f} "
      f"(min {svm_diag_df['AUC'].min():.3f}, max {svm_diag_df['AUC'].max():.3f})")
baseline_f1 = np.mean(base_f1s)
baseline_acc = np.mean(base_accs)
print(f"Majority-baseline (predict all-PD), same 50 folds: accuracy {baseline_acc:.3f}, F1 {baseline_f1:.3f}.")
svm_ensemble_fpr, svm_ensemble_tpr, _ = roc_curve(y, oof_probabilities['SVM'])
svm_ensemble_auc = auc(svm_ensemble_fpr, svm_ensemble_tpr)
if svm_auc_mean < 0.65:
    print(f"SVM AUC stays near-chance (per-seed {svm_auc_mean:.3f} +/- {svm_auc_std:.3f}, "
          f"averaged-probs {svm_ensemble_auc:.3f}) across all 10 "
          f"seeds — clearly lowest of the 5 models (XGBoost/RF/KNN sit at ~0.74-0.77) — so SVM's "
          f"high recall ({svm_recall_mean:.3f}) and F1 ({svm_f1_mean:.3f}, barely above the "
          f"{baseline_f1:.3f} majority-baseline F1) are NOT evidence of learned signal, they're "
          f"near-baseline ranking dressed up by a threshold that predicts PD often (all-PD in "
          f"{n_all_pd_svm}/{n_total_svm} folds, recall==1.000 in {n_rec1_svm}/{n_total_svm} folds). "
          f"State plainly: SVM may not be learning meaningful signal here. Lead with AUC, not recall.")
else:
    print("SVM recall settled below the seed-42 1.000 once grouping/averaging was fixed, and both "
          f"AUC ({svm_auc_mean:.3f}) and F1 ({svm_f1_mean:.3f} vs baseline {baseline_f1:.3f}) clear "
          "the majority baseline by a real margin — the seed-42 degeneracy does not reproduce.")

os.makedirs('assets', exist_ok=True)

# -------------------------------------------------------------
# 2. SAVE ASSET 1: PERFORMANCE MULTI-BAR CHART (assets/model_metrics.png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#F8FAFC')

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metrics_names))
width = 0.15

for i, row in res_df.iterrows():
    model_name = row['Model']
    vals = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-score']]
    errs = [row['Accuracy_std'], row['Precision_std'], row['Recall_std'], row['F1-score_std']]
    color = models_config[model_name]['color']
    offset = (i - len(res_df)/2 + 0.5) * width
    rects = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                    error_kw={'ecolor': '#334155', 'elinewidth': 1, 'capthick': 1},
                    label=model_name, color=color, alpha=0.9, edgecolor='#FFFFFF', linewidth=1)

    # Label bar values
    for rect, err in zip(rects, errs):
        h = rect.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h + err),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E293B')

## Majority-baseline reference line, per metric (computed above, same 50
## folds as the model bars — not hardcoded). Drawn as short dashed segments
## spanning each metric's own group of bars, rather than one continuous
## line across the whole axis, so it stays readable next to the error bars.
baseline_vals = [baseline_row['Accuracy'], baseline_row['Precision'],
                  baseline_row['Recall'], baseline_row['F1-score']]
group_half_width = (len(res_df) * width) / 2
for i, (xc, bv) in enumerate(zip(x, baseline_vals)):
    ax.plot([xc - group_half_width, xc + group_half_width], [bv, bv],
            linestyle='--', color='#64748B', linewidth=1.6, zorder=5,
            label='Majority-class baseline (predict all-PD)' if i == 0 else None)

ax.set_ylabel('Score (10-Seed Averaged Subject-Grouped CV, Mean +/- Std)', fontsize=11, fontweight='bold', color='#0F172A', labelpad=10)
ax.set_title('Classifier Performance Metrics Comparison (5-Fold Subject-Grouped CV, 10-Seed Avg)',
             fontsize=14, fontweight='bold', color='#0F172A', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold', color='#1E293B')
ax.set_ylim(0.5, 1.18)  # headroom for error bars + value labels (SVM recall ~0.94+0.09)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')

# Legend
legend = ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=10, loc='upper left')
legend.get_frame().set_boxstyle('round,pad=0.5')
plt.setp(legend.get_texts(), color='#0F172A')

plt.tight_layout()
plt.savefig('assets/model_metrics.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved assets/model_metrics.png successfully!")

# -------------------------------------------------------------
# 3. SAVE ASSET 2: ROC & PRECISION-RECALL 2-PANEL CURVES (assets/roc_pr_curves.jpg)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax1.set_facecolor('#F8FAFC')
ax2.set_facecolor('#F8FAFC')

for name in models_config.keys():
    probs = oof_probabilities[name]
    color = models_config[name]['color']
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc_val = auc(fpr, tpr)
    lw = 2.5 if name == 'XGBoost' else 1.8
    ax1.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC = {roc_auc_val:.3f})')
    
    # Precision-Recall Curve
    prec_c, rec_c, _ = precision_recall_curve(y, probs)
    pr_auc_val = average_precision_score(y, probs)
    ax2.plot(rec_c, prec_c, color=color, lw=lw, label=f'{name} (AP = {pr_auc_val:.3f})')

# ROC Panel styling
ax1.plot([0, 1], [0, 1], linestyle='--', lw=1.2, color='#94A3B8', label='Random Chance (AUC = 0.500)')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', color='#0F172A')
ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold', color='#0F172A')
ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', color='#0F172A', pad=12)
ax1.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
leg1 = ax1.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=9.5, loc='lower right')
leg1.get_frame().set_boxstyle('round,pad=0.4')
plt.setp(leg1.get_texts(), color='#0F172A')

# PR Panel styling
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold', color='#0F172A')
ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold', color='#0F172A')
ax2.set_title('Precision-Recall (PR) Curve', fontsize=13, fontweight='bold', color='#0F172A', pad=12)
ax2.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
leg2 = ax2.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=9.5, loc='lower left')
leg2.get_frame().set_boxstyle('round,pad=0.4')
plt.setp(leg2.get_texts(), color='#0F172A')

fig.suptitle('Model Evaluation & Discriminative Power (5-Fold Subject-Grouped CV, 10-Seed Avg OOF Probs)',
             fontsize=15, fontweight='bold', color='#0F172A', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('assets/roc_pr_curves.jpg', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved assets/roc_pr_curves.jpg successfully!")
