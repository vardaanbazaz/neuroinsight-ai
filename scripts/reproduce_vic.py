"""Reproduce the VIC vs raw-22 subject-grouped CV result from docs/INDEX_PROPOSAL.md.

Protocol: StratifiedGroupKFold(5, shuffle=True, random_state=seed), seeds 0-9,
Pipeline(StandardScaler -> LogisticRegression(max_iter=5000)), fit per fold.
Groups = subject ID parsed from the `name` column. VIC is a row-wise function
(no fitted parameters), so it needs no fold-safe fitting.

Run from repo root: python scripts/reproduce_vic.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = range(10)


def load():
    df = pd.read_csv("data/pd_data.txt")
    df["VIC"] = np.log10(df["MDVP:Jitter(%)"] * df["Shimmer:APQ3"] * df["spread2"] * 1000)
    groups = df["name"].str.extract(r"(S\d+)")[0].values
    y = df["status"].values
    raw22 = df.drop(columns=["name", "status", "VIC"])
    return df, raw22, y, groups


def evaluate(X, y, groups):
    per_seed = []  # one dict per seed
    fold_auc_all = []
    for seed in SEEDS:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        accs, f1s, fold_aucs = [], [], []
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, y, groups=groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            oof[va] = p
            accs.append(accuracy_score(y[va], p > 0.5))
            f1s.append(f1_score(y[va], p > 0.5, zero_division=0))
            if len(np.unique(y[va])) == 2:
                fold_aucs.append(roc_auc_score(y[va], p))
        per_seed.append({
            "acc": np.mean(accs), "f1": np.mean(f1s),
            "auc_foldmean": np.mean(fold_aucs),     # docs/INDEX_PROPOSAL.md AUC: mean of per-fold AUCs, per seed
            "auc_pooled": roc_auc_score(y, oof),    # for reference: AUC of the pooled OOF vector (generate_clean_assets.py's per-seed AUC)
        })
    return pd.DataFrame(per_seed)


def main():
    df, raw22, y, groups = load()
    print(f"rows={len(df)} subject IDs={len(set(groups))} PD={y.sum()} healthy={(y == 0).sum()}")
    sets = {
        "VIC alone": df[["VIC"]],
        "Raw 22": raw22,
        "VIC parents (Jitter%, APQ3, spread2)": df[["MDVP:Jitter(%)", "Shimmer:APQ3", "spread2"]],
    }
    for name, X in sets.items():
        r = evaluate(X, y, groups)
        print(f"\n{name}  (mean +/- std over 10 seeds)")
        for col in ["acc", "f1", "auc_foldmean", "auc_pooled"]:
            print(f"  {col:13s} {r[col].mean():.3f} +/- {r[col].std(ddof=0):.3f}")


if __name__ == "__main__":
    main()
