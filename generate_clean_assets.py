import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
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

# Compute Feature Performance Index (fPI = log10(DFA * D2 * spread2))
df['fPI'] = np.log10(df['DFA'] * df['D2'] * df['spread2'])

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

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics_summary = []
oof_probabilities = {}

for name, config in models_config.items():
    oof_probs = np.zeros(len(y))
    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []
    
    for train_idx, val_idx in skf.split(X, y):
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
        
        oof_probs[val_idx] = probs
        
        fold_accs.append(accuracy_score(y_val, preds))
        fold_precs.append(precision_score(y_val, preds, zero_division=0))
        fold_recs.append(recall_score(y_val, preds, zero_division=0))
        fold_f1s.append(f1_score(y_val, preds, zero_division=0))
        
    oof_probabilities[name] = oof_probs
    
    metrics_summary.append({
        'Model': name,
        'Accuracy': np.mean(fold_accs),
        'Precision': np.mean(fold_precs),
        'Recall': np.mean(fold_recs),
        'F1-score': np.mean(fold_f1s)
    })

res_df = pd.DataFrame(metrics_summary)
print("=== 5-Fold Stratified Cross-Validation Performance Summary ===")
print(res_df.to_string(index=False))

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
    color = models_config[model_name]['color']
    offset = (i - len(res_df)/2 + 0.5) * width
    rects = ax.bar(x + offset, vals, width, label=model_name, color=color, alpha=0.9, edgecolor='#FFFFFF', linewidth=1)
    
    # Label bar values
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E293B')

ax.set_ylabel('Score (5-Fold Stratified CV Mean)', fontsize=11, fontweight='bold', color='#0F172A', labelpad=10)
ax.set_title('Classifier Performance Metrics Comparison (5-Fold CV)', 
             fontsize=14, fontweight='bold', color='#0F172A', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold', color='#1E293B')
ax.set_ylim(0.5, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')

# Legend
legend = ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=10, loc='lower right')
legend.get_frame().set_boxstyle('round,pad=0.5')
plt.setp(legend.get_texts(), color='#0F172A')

plt.tight_layout()
plt.savefig('assets/model_metrics.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved assets/model_metrics.png successfully!")

# -------------------------------------------------------------
# 3. SAVE ASSET 2: ROC & PRECISION-RECALL 2-PANEL CURVES (assets/roc_pr_curves.png)
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

fig.suptitle('Model Evaluation & Discriminative Power (5-Fold Stratified CV)', 
             fontsize=15, fontweight='bold', color='#0F172A', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('assets/roc_pr_curves.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved assets/roc_pr_curves.png successfully!")
