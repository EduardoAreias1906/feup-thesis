"""
=============================================================================
outcome_association.py — RQ3: Association between Choreographies & Success
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

RQ3 asks how the discovered choreographies relate to academic success, and
tests H1 (regularity -> success), H2 (balanced participation), H5
(disengagement -> weaker outcomes). Section 4.3.2 also requires controlling
for student background (at-risk, IEP).

Granularity bridge:
  - clusters are per STUDENT-WEEK
  - outcomes are per STUDENT-SUBJECT
  We summarise each student by their DOMINANT choreography (the cluster where
  they spent the most weeks), then attach outcomes.

Analyses:
  1. Outcomes by dominant choreography: mean Final_Grade, Quiz, Completion,
     and Pass rate per profile.
  2. Statistical significance: one-way ANOVA across profiles on Final_Grade
     (are the differences real, not noise?), plus effect size (eta-squared).
  3. Background control: same grade comparison split by at-risk status, so we
     can see whether the choreography effect holds within at-risk and within
     non-at-risk students (not just because profiles differ in at-risk mix).

Inputs:
  results/student_week_clusters.csv   (Student_ID, week_start, cluster)
  data/academic_outcomes.csv          (Student_ID, Subject, Final_Grade, ...)
  data/demographics.csv               (Student_ID, At_Risk_Status, IEP_Status, ...)

Outputs:
  results/outcomes_by_cluster.csv
  results/grade_by_cluster.png
  results/grade_by_cluster_atrisk.png

Usage:
  python outcome_association.py
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd
from cluster_names import name as cluster_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    clusters_path = os.path.join(args.results_dir, "student_week_clusters.csv")
    outcomes_path = os.path.join(args.data_dir, "academic_outcomes.csv")
    demo_path = os.path.join(args.data_dir, "demographics.csv")
    for p in (clusters_path, outcomes_path, demo_path):
        if not os.path.exists(p):
            print(f"  Not found: {p}")
            return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("RQ3 — Choreography vs Academic Success")
    print("=" * 60)

    clusters = pd.read_csv(clusters_path)
    outcomes = pd.read_csv(outcomes_path)
    demo = pd.read_csv(demo_path)

    n_clusters = int(clusters["cluster"].nunique())

    # ---- Dominant choreography per student (mode of weekly clusters) ----
    print("  Computing each student's dominant choreography...")
    dom = (clusters.groupby("Student_ID")["cluster"]
           .agg(lambda s: s.value_counts().idxmax())
           .rename("dominant_cluster").reset_index())

    # ---- Average outcomes per student (across their subjects) ----
    out_num = outcomes.copy()
    out_num["passed"] = (out_num["Pass_Fail"].astype(str).str.strip().str.lower()
                         == "pass").astype(int)
    per_student = out_num.groupby("Student_ID").agg(
        final_grade=("Final_Grade", "mean"),
        quiz=("Quiz_Average", "mean"),
        completion=("Course_Completion_Percentage", "mean"),
        pass_rate=("passed", "mean"),
    ).reset_index()

    # ---- Merge ----
    df = dom.merge(per_student, on="Student_ID", how="inner")
    df = df.merge(demo[["Student_ID", "At_Risk_Status", "IEP_Status"]],
                  on="Student_ID", how="left")

    # ---- 1. Outcomes by dominant choreography ----
    summary = df.groupby("dominant_cluster").agg(
        n_students=("Student_ID", "count"),
        avg_final_grade=("final_grade", "mean"),
        avg_quiz=("quiz", "mean"),
        avg_completion=("completion", "mean"),
        pass_rate=("pass_rate", "mean"),
    ).round(2)
    summary["pass_rate"] = (summary["pass_rate"] * 100).round(1)

    print("\n  Outcomes by dominant choreography:")
    print("  (pass_rate in %)")
    print(summary.to_string())

    # ---- 2. ANOVA on Final_Grade across clusters ----
    # One-way ANOVA tests whether the mean Final_Grade differs significantly
    # across choreography groups, or whether the observed spread could plausibly
    # arise from random sampling variation alone.
    #
    # F-statistic: the ratio of variance BETWEEN group means to variance WITHIN
    # groups. A large F means the groups differ more than expected by chance.
    # F = 1.0 would mean "between-group variance equals within-group variance"
    # (no signal above noise).
    #
    # p-value: the probability of observing an F this large if all choreographies
    # truly had the same population mean (the null hypothesis). p < 0.05 is the
    # conventional threshold for rejecting the null — i.e. the differences are
    # unlikely to be noise.
    #
    # eta-squared (η²): the effect size — what FRACTION of total grade variance
    # is explained by choreography membership. Interpretation:
    #   < 0.06 = small, ~0.14 = medium, > 0.14 = large (Cohen's conventions).
    # A statistically significant result with a tiny η² means "real but trivially
    # small" (common with large n). A large η² means choreography explains a
    # substantial portion of outcome variance — the effect is practically
    # meaningful, not just statistically detectable.
    try:
        from scipy import stats
        groups = [df[df["dominant_cluster"] == c]["final_grade"].values
                  for c in range(n_clusters)]
        groups = [g for g in groups if len(g) > 1]
        f_stat, p_val = stats.f_oneway(*groups)

        # eta-squared (effect size), computed manually to avoid scipy warnings
        all_vals = np.concatenate(groups)
        grand = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = float(((all_vals - grand) ** 2).sum())
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

        print("\n  ANOVA — Final_Grade differs across choreographies?")
        print(f"    F = {f_stat:.1f}, p = {p_val:.2e}")
        print(f"    eta-squared (effect size) = {eta2:.3f}")
        sig = "YES — differences are statistically significant" if p_val < 0.05 \
              else "no significant difference"
        print(f"    -> {sig}")
    except ImportError:
        print("\n  (scipy not installed — skipping ANOVA. pip install scipy)")

    # ---- 3. Background control: grade by cluster, split by at-risk ----
    # "Confounding" is when a third variable influences both the predictor
    # (choreography) and the outcome (grade), making a naive comparison misleading.
    # Here: at-risk students are more likely to be in the Low_Engagement
    # choreography AND more likely to have lower grades — for entirely separate
    # reasons (prior academic history, socioeconomic factors). If we only compare
    # choreographies without accounting for at-risk status, we may conclude that
    # Low_Engagement causes lower grades when in fact it is partially or fully
    # a proxy for being at-risk.
    #
    # The fix is to look at choreography effects WITHIN each at-risk group:
    #   "Among non-at-risk students, does choreography still predict grade?"
    #   "Among at-risk students, does choreography still matter?"
    # If yes — the choreography has a real DIRECT effect on grade beyond the
    # at-risk confound. If the effect vanishes within each group, choreography
    # was just a proxy for at-risk status and has no independent predictive value.
    df["at_risk"] = (df["At_Risk_Status"].astype(str).str.strip().str.lower()
                     .isin(["true", "1", "yes", "at-risk", "at_risk"]))
    print("\n  Mean Final_Grade by choreography, split by background:")
    ctrl = df.groupby(["dominant_cluster", "at_risk"])["final_grade"].mean().unstack()
    cnt = df.groupby(["dominant_cluster", "at_risk"])["final_grade"].size().unstack()
    ctrl.columns = ["not_at_risk" if c is False else "at_risk" for c in ctrl.columns]
    cnt.columns = ["not_at_risk" if c is False else "at_risk" for c in cnt.columns]
    # Ensure both columns exist
    for col in ["not_at_risk", "at_risk"]:
        if col not in ctrl.columns:
            ctrl[col] = np.nan
            cnt[col] = 0
    ctrl = ctrl[["not_at_risk", "at_risk"]]
    cnt = cnt[["not_at_risk", "at_risk"]].fillna(0).astype(int)
    print("  (mean grade, with n students in parentheses)")
    for c in sorted(ctrl.index):
        gn = ctrl.loc[c, "not_at_risk"]; gr = ctrl.loc[c, "at_risk"]
        nn = cnt.loc[c, "not_at_risk"]; nr = cnt.loc[c, "at_risk"]
        gn_s = f"{gn:.1f}" if not np.isnan(gn) else "  -- "
        gr_s = f"{gr:.1f}" if not np.isnan(gr) else "  -- "
        print(f"    Cluster {c}: not_at_risk={gn_s} (n={nn:5d})   "
              f"at_risk={gr_s} (n={nr:5d})")
    print("\n  NOTE: in this dataset some cluster x group cells are empty/tiny")
    print("  because archetypes were generated correlated with at-risk status.")
    print("  Where both groups exist (e.g. cluster 0), at-risk students score")
    print("  lower within the SAME choreography -> at-risk has its own effect,")
    print("  partially confounded with choreography by construction.")

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(8, 5))
    order = summary.sort_values("avg_final_grade", ascending=False).index
    ax.bar([cluster_name(c) for c in order],
           summary.loc[order, "avg_final_grade"].values, color="#1f6feb")
    ax.set_ylabel("Mean Final Grade")
    ax.set_xlabel("Dominant choreography (cluster)")
    ax.set_title("Final grade by choreography")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 100)
    for i, c in enumerate(order):
        ax.text(i, summary.loc[c, "avg_final_grade"] + 1,
                f"{summary.loc[c, 'avg_final_grade']:.0f}", ha="center")
    fig.tight_layout()
    p1 = os.path.join(args.results_dir, "grade_by_cluster.png")
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"\n  Saved {p1}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_clusters)
    w = 0.38
    def safe(col, c):
        if c in ctrl.index and not np.isnan(ctrl.loc[c, col]):
            return ctrl.loc[c, col]
        return 0.0
    g_not = [safe("not_at_risk", c) for c in range(n_clusters)]
    g_risk = [safe("at_risk", c) for c in range(n_clusters)]
    ax.bar(x - w/2, g_not, w, label="Not at-risk", color="#3fb950")
    ax.bar(x + w/2, g_risk, w, label="At-risk", color="#f85149")
    # mark empty cells
    for c in range(n_clusters):
        if g_not[c] == 0:
            ax.text(c - w/2, 2, "n/a", ha="center", fontsize=8, rotation=90)
        if g_risk[c] == 0:
            ax.text(c + w/2, 2, "n/a", ha="center", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([cluster_name(c) for c in range(n_clusters)],
                                         rotation=20, ha="right")
    ax.set_ylabel("Mean Final Grade"); ax.set_ylim(0, 100)
    ax.set_title("Final grade by choreography, controlled for at-risk status")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(args.results_dir, "grade_by_cluster_atrisk.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"  Saved {p2}")

    summary.to_csv(os.path.join(args.results_dir, "outcomes_by_cluster.csv"))
    print(f"  Saved {os.path.join(args.results_dir, 'outcomes_by_cluster.csv')}")
    print()


if __name__ == "__main__":
    main()