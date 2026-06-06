"""
=============================================================================
early_prediction.py — RQ4: Predictive Utility of Choreographies
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

RQ4 asks: does knowing a student's choreography improve EARLY prediction of
success, beyond the volume metrics teachers already have? And how early can
the signal be detected?

Success = course PASS (aligned with Beck's retention/pass framing for at-risk
learners). A student is labelled "passed" if their majority of courses are
Pass.

NO DATA LEAKAGE (thesis 4.3.3): features are computed ONLY from the FIRST N
weeks of each student's enrolment. The label (pass/fail) comes from the final
outcome. We run N = 4 and N = 8 weeks to answer "how early?".

Two models compared (Random Forest, chosen for interpretability -> RQ5):
  BASELINE      : volume only (logins, total actions, sessions, active days,
                  avg session length) from the first N weeks.
  EXPERIMENTAL  : baseline + the student's dominant choreography in weeks 1..N
                  (one-hot encoded).

If EXPERIMENTAL beats BASELINE, the "shape" of behaviour adds predictive power
beyond raw volume.

Validation: stratified train/test split (75/25). Metrics: accuracy, F1, ROC-AUC.

Inputs:
  data/activity_logs.csv           (events; has Archetype col — NOT used here)
  data/academic_outcomes.csv       (Pass_Fail per student-subject)
  results/student_week_clusters.csv (cluster per student-week)

Outputs:
  results/rq4_results.csv
  results/rq4_comparison.png

Usage:
  python early_prediction.py
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd

COL_STUDENT = "Student_ID"
COL_TIME = "Timestamp"
COL_SESSION = "Session_ID"
COL_ACTION = "Action_Type"


def build_early_features(logs, clusters, first_n_weeks):
    """
    Build per-student feature vectors using only their first N weeks of activity.

    Data leakage prevention: "data leakage" happens when information from the
    future (post-outcome period) is used to train or evaluate a prediction model.
    In education this is a common pitfall — if features include Week 30 engagement
    patterns, then the "early" model has effectively already seen behaviour that
    only happens AFTER the student has passed or failed. The model would perform
    well in retrospect but be useless in real-time.

    Here we compute week_idx = weeks since each student's OWN first week (not a
    global calendar week). We then discard any events with week_idx >= first_n_weeks,
    so ALL features reflect only what was observable in the first N weeks of
    enrolment. The label (pass/fail) still comes from the FINAL outcome and is
    never used as a feature. This simulates a teacher checking a dashboard after
    exactly N weeks with no hindsight.
    """
    logs = logs.copy()
    logs[COL_TIME] = pd.to_datetime(logs[COL_TIME])
    logs["week_start"] = (
        logs[COL_TIME] - pd.to_timedelta(logs[COL_TIME].dt.weekday, unit="D")
    ).dt.normalize()

    # each student's first enrolment week
    first_week = logs.groupby(COL_STUDENT)["week_start"].transform("min")
    week_idx = ((logs["week_start"] - first_week).dt.days // 7)
    logs["week_idx"] = week_idx
    early = logs[logs["week_idx"] < first_n_weeks].copy()

    # ---- Volume features (baseline) ----
    vol = early.groupby(COL_STUDENT).agg(
        n_logins=(COL_ACTION, lambda s: (s == "Login").sum()),
        total_actions=(COL_ACTION, "size"),
        n_sessions=(COL_SESSION, "nunique"),
        active_days=(COL_TIME, lambda s: s.dt.normalize().nunique()),
        total_seconds=("Duration_Seconds", "sum"),
    )
    vol["avg_session_len"] = vol["total_seconds"] / vol["n_sessions"].replace(0, 1)

    # ---- Dominant cluster in first N weeks (experimental add-on) ----
    cl = clusters.copy()
    cl["week_start"] = pd.to_datetime(cl["week_start"])
    fw = cl.groupby(COL_STUDENT)["week_start"].transform("min")
    cl["week_idx"] = ((cl["week_start"] - fw).dt.days // 7)
    cl_early = cl[cl["week_idx"] < first_n_weeks]
    dom = (cl_early.groupby(COL_STUDENT)["cluster"]
           .agg(lambda s: s.value_counts().idxmax())
           .rename("dominant_cluster"))

    feats = vol.join(dom, how="inner")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--weeks", default="4,8",
                    help="Comma-separated early-window sizes (default 4,8)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logs_path = os.path.join(args.data_dir, "activity_logs.csv")
    out_path = os.path.join(args.data_dir, "academic_outcomes.csv")
    cl_path = os.path.join(args.results_dir, "student_week_clusters.csv")
    for p in (logs_path, out_path, cl_path):
        if not os.path.exists(p):
            print(f"  Not found: {p}")
            return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("RQ4 — Early Prediction of Success (Pass)")
    print("=" * 60)

    print("  Loading data...")
    logs = pd.read_csv(logs_path, usecols=[COL_STUDENT, COL_TIME, COL_SESSION,
                                           COL_ACTION, "Duration_Seconds"])
    outcomes = pd.read_csv(out_path)
    clusters = pd.read_csv(cl_path)

    # ---- Label: student passed (majority of their courses) ----
    outcomes["passed"] = (outcomes["Pass_Fail"].astype(str).str.strip().str.lower()
                          == "pass").astype(int)
    label = (outcomes.groupby(COL_STUDENT)["passed"].mean() >= 0.5).astype(int)
    label.name = "passed"
    print(f"  Pass rate overall: {label.mean()*100:.1f}%")

    windows = [int(w) for w in args.weeks.split(",")]
    rows = []

    for n in windows:
        print(f"\n  --- First {n} weeks ---")
        feats = build_early_features(logs, clusters, n)
        data = feats.join(label, how="inner").dropna(subset=["passed"])
        data["dominant_cluster"] = data["dominant_cluster"].astype(int)

        vol_cols = ["n_logins", "total_actions", "n_sessions",
                    "active_days", "total_seconds", "avg_session_len"]
        y = data["passed"].values

        # one-hot the cluster for the experimental model
        cl_dummies = pd.get_dummies(data["dominant_cluster"], prefix="cluster")
        X_base = data[vol_cols].values
        X_exp = np.hstack([data[vol_cols].values, cl_dummies.values])

        # ---- How a Random Forest predicts ----
        # A single decision tree asks a sequence of binary questions about features
        # (e.g. "is total_actions > 120?") and assigns the majority class of
        # training points that reach each leaf. A Random Forest grows many trees,
        # each trained on a random bootstrap sample of the training data and using
        # a random subset of features at each split candidate. This "bagging +
        # feature randomness" makes the trees decorrelated — their individual
        # errors tend to cancel when averaged, producing a more stable and
        # generalizable probability estimate than any single tree.
        # class_weight="balanced" compensates for pass/fail imbalance by giving
        # the minority class (fail) proportionally higher weight during training,
        # preventing the model from simply predicting the majority class always.
        #
        # ---- Why ROC-AUC, not accuracy? ----
        # If 80% of students pass, a trivial classifier that predicts "pass" for
        # everyone achieves 80% accuracy — often beating naive models. ROC-AUC
        # (area under the Receiver Operating Characteristic curve) is immune to
        # class imbalance: it measures the probability that the model ranks a
        # randomly chosen passer ABOVE a randomly chosen failer. 0.5 = no better
        # than random; 1.0 = perfect rank ordering. This is the right metric when
        # the goal is to identify at-risk students from a mixed population, because
        # it captures how well the model ORDERS students by risk rather than whether
        # a fixed threshold yields the right binary label.
        results_n = {}
        for name, X in [("baseline", X_base), ("experimental", X_exp)]:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=args.seed, stratify=y)
            rf = RandomForestClassifier(n_estimators=300, random_state=args.seed,
                                        class_weight="balanced", n_jobs=-1)
            rf.fit(Xtr, ytr)
            pred = rf.predict(Xte)
            proba = rf.predict_proba(Xte)[:, 1]
            acc = accuracy_score(yte, pred)
            f1 = f1_score(yte, pred)
            auc = roc_auc_score(yte, proba)
            results_n[name] = (acc, f1, auc)
            print(f"    {name:12s}: acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
            rows.append({"weeks": n, "model": name, "accuracy": acc,
                         "f1": f1, "auc": auc})

        d_auc = results_n["experimental"][2] - results_n["baseline"][2]
        print(f"    -> AUC gain from adding choreography: {d_auc:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.results_dir, "rq4_results.csv"), index=False)

    # ---- Plot AUC comparison ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, color in [("baseline", "#8b949e"), ("experimental", "#1f6feb")]:
        sub = res[res["model"] == model]
        ax.plot(sub["weeks"], sub["auc"], "o-", color=color,
                label=model, linewidth=2, markersize=8)
    ax.set_xlabel("Early window (first N weeks)")
    ax.set_ylabel("ROC-AUC (pass prediction)")
    ax.set_title("Does choreography improve early prediction?")
    ax.set_xticks(windows)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.results_dir, "rq4_comparison.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n  Saved {p}")
    print(f"  Saved {os.path.join(args.results_dir, 'rq4_results.csv')}")
    print()


if __name__ == "__main__":
    main()