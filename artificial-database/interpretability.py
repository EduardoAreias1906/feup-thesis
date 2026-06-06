"""
=============================================================================
interpretability.py — RQ5: Actionability / What drives success
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

RQ5 asks which behavioral components most contribute to outcomes, and how to
surface them for teachers. We train a Random Forest to predict success (pass)
from each student's behavioral profile, then extract feature importance
(thesis 4.3.4: Gini importance / SHAP), plus the DIRECTION of each effect so
the message is actionable ("more X -> more likely to pass").

Per-student features:
  - action MIX: proportion of each behavioral action (act__* / total)
  - time-of-day MIX: proportion of activity in each bin (time__*)
  - volume: total actions, sessions, active days, active weeks

Three views of importance (robust, not relying on one method):
  1. Gini importance      (built-in, fast, but biased to high-variance feats)
  2. Permutation importance (more trustworthy; how much accuracy drops when a
                             feature is shuffled)
  3. Direction of effect   (pass rate in top vs bottom third of each feature)
  + SHAP if installed (bonus: ranks features by mean impact on predictions)

Inputs:
  data/student_week_features.csv     (act__/time__ per student-week)
  data/academic_outcomes.csv         (Pass_Fail)

Outputs:
  results/rq5_importance.png
  results/rq5_importance.csv

Usage:
  python interpretability.py
  (optional, for SHAP bonus:  pip install shap)
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd

COL_STUDENT = "Student_ID"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat_path = os.path.join(args.data_dir, "student_week_features.csv")
    out_path = os.path.join(args.data_dir, "academic_outcomes.csv")
    for p in (feat_path, out_path):
        if not os.path.exists(p):
            print(f"  Not found: {p}")
            return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("RQ5 — What behavioral components drive success?")
    print("=" * 60)

    feats = pd.read_csv(feat_path)
    outcomes = pd.read_csv(out_path)

    act_cols = [c for c in feats.columns if c.startswith("act__")]
    time_cols = [c for c in feats.columns if c.startswith("time__")]

    # ---- Per-student aggregation ----
    print("  Aggregating per-student behavioral profile...")
    g = feats.groupby(COL_STUDENT)
    act_sum = g[act_cols].sum()
    time_sum = g[time_cols].sum()
    # proportions (shape of behavior)
    act_prop = act_sum.div(act_sum.sum(axis=1).replace(0, 1), axis=0)
    time_prop = time_sum.div(time_sum.sum(axis=1).replace(0, 1), axis=0)
    vol = g.agg(total_actions=("total_actions", "sum"),
                n_sessions=("n_sessions", "sum"),
                active_days=("active_days", "sum"))
    vol["active_weeks"] = g.size()

    X = act_prop.join(time_prop).join(vol)
    # clean feature names for display
    X.columns = [c.replace("act__", "").replace("time__", "TOD:") for c in X.columns]

    # ---- Label ----
    outcomes["passed"] = (outcomes["Pass_Fail"].astype(str).str.strip().str.lower()
                          == "pass").astype(int)
    label = (outcomes.groupby(COL_STUDENT)["passed"].mean() >= 0.5).astype(int)

    data = X.join(label.rename("passed"), how="inner").dropna()
    y = data["passed"].values
    Xm = data.drop(columns=["passed"])
    feat_names = list(Xm.columns)
    print(f"  {len(data):,} students, {len(feat_names)} features, "
          f"pass rate {y.mean()*100:.1f}%")

    # ---- Train RF ----
    Xtr, Xte, ytr, yte = train_test_split(Xm.values, y, test_size=0.25,
                                          random_state=args.seed, stratify=y)
    rf = RandomForestClassifier(n_estimators=400, random_state=args.seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    print(f"  Model AUC on test set: {auc:.3f}")

    # ---- 1. Gini importance ----
    gini = pd.Series(rf.feature_importances_, index=feat_names)

    # ---- 2. Permutation importance ----
    print("  Computing permutation importance (this can take a moment)...")
    perm = permutation_importance(rf, Xte, yte, n_repeats=10,
                                  random_state=args.seed, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=feat_names)

    # ---- 3. Direction of effect (pass rate: top third vs bottom third) ----
    print("\n  Direction of effect for the most important behaviors:")
    top_feats = perm_imp.sort_values(ascending=False).head(8).index
    directions = {}
    for f in top_feats:
        vals = Xm[f]
        lo_thr, hi_thr = vals.quantile(0.33), vals.quantile(0.67)
        lo_pass = data[vals <= lo_thr]["passed"].mean() * 100
        hi_pass = data[vals >= hi_thr]["passed"].mean() * 100
        arrow = "UP  ↑" if hi_pass > lo_pass else "DOWN↓"
        directions[f] = (lo_pass, hi_pass)
        print(f"    {f:22s}: low={lo_pass:5.1f}%  high={hi_pass:5.1f}%  "
              f"pass when high -> {arrow}")

    # ---- Combine + save ----
    imp = pd.DataFrame({"gini": gini, "permutation": perm_imp}).sort_values(
        "permutation", ascending=False)
    imp.to_csv(os.path.join(args.results_dir, "rq5_importance.csv"))

    # ---- Plot top features by permutation importance ----
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#3fb950" if directions.get(f, (0, 1))[1] >=
              directions.get(f, (1, 0))[0] else "#f85149"
              for f in top.index]
    ax.barh(top.index, top["permutation"].values, color=colors)
    ax.set_xlabel("Permutation importance (drop in accuracy when shuffled)")
    ax.set_title(f"What signals success? (RF AUC={auc:.2f})\n"
                 "green = more of this -> more likely to pass; red = less likely")
    fig.tight_layout()
    p = os.path.join(args.results_dir, "rq5_importance.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n  Saved {p}")

    # ---- 4. SHAP (optional bonus) ----
    try:
        import shap
        print("\n  Computing SHAP values (TreeExplainer)...")
        expl = shap.TreeExplainer(rf)
        sample = Xte[:2000]
        sv = expl.shap_values(sample)
        # binary classifier: take class-1 contributions
        sv1 = sv[1] if isinstance(sv, list) else sv
        mean_abs = np.abs(sv1).mean(axis=0)
        shap_imp = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
        print("  Top 8 features by mean |SHAP|:")
        for f, v in shap_imp.head(8).items():
            print(f"    {f:22s} {v:.4f}")
    except ImportError:
        print("\n  (shap not installed — skipping SHAP bonus. "
              "Optional: pip install shap)")
    except Exception as e:
        print(f"\n  (SHAP step skipped: {e})")

    print(f"\n  Saved {os.path.join(args.results_dir, 'rq5_importance.csv')}")
    print()


if __name__ == "__main__":
    main()