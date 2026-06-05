"""
=============================================================================
context_stability.py — RQ2 part (b): Cross-Context Stability
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

RQ2 also asks whether the choreographies are universal behavioral habits or
artefacts of a specific grade / school / region. Two analyses:

  1. PROFILE DISTRIBUTION ACROSS CONTEXTS
     How are the 4 choreographies distributed across Grade_Level, School_ID,
     and State? If the same profiles appear everywhere (in similar-ish
     proportions) they are robust; if some context is dominated by one
     profile, the choreography is context-dependent.

  2. PROFILE SIGNATURE CONSISTENCY ACROSS GRADES (spirit of H_Robustness)
     Do the clusters keep the same behavioral "signature" (same dominant
     actions) when we look at them within different grade bands? We compare
     each cluster's mean action profile computed separately for lower grades
     (K-5) vs upper grades (6-12), using cosine similarity. High similarity =
     the choreography means the same thing regardless of grade.

NOTE (honest framing): the synthetic generator does NOT vary behavior by
subject, so an H_Robustness test *by subject* is not meaningful here and is
omitted. Grade / school / state DO carry real signal (at-risk mix differs by
grade), so those are tested.

Inputs:
  results/student_week_clusters.csv     (Student_ID, week_start, cluster)
  data/demographics.csv                 (Student_ID, Grade_Level, School_ID, State, ...)
  data/student_week_features.csv        (the act__/time__ feature matrix)

Outputs:
  results/profile_by_grade.png
  results/profile_by_state.png
  results/context_distribution.csv
  prints signature-consistency cosine similarities

Usage:
  python context_stability.py
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    clusters_path = os.path.join(args.results_dir, "student_week_clusters.csv")
    demo_path = os.path.join(args.data_dir, "demographics.csv")
    feat_path = os.path.join(args.data_dir, "student_week_features.csv")
    for p in (clusters_path, demo_path, feat_path):
        if not os.path.exists(p):
            print(f"  Not found: {p}")
            return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("RQ2 (b) — Cross-Context Stability")
    print("=" * 60)

    clusters = pd.read_csv(clusters_path)
    demo = pd.read_csv(demo_path, usecols=["Student_ID", "Grade_Level",
                                           "School_ID", "State"])
    df = clusters.merge(demo, on="Student_ID", how="left")
    n_clusters = int(df["cluster"].nunique())
    print(f"  {len(df):,} student-weeks across "
          f"{df['Grade_Level'].nunique()} grades, "
          f"{df['School_ID'].nunique()} schools, "
          f"{df['State'].nunique()} states.")

    # ---- 1a. Distribution by grade ----
    by_grade = pd.crosstab(df["Grade_Level"], df["cluster"], normalize="index") * 100
    by_grade = by_grade.reindex(columns=range(n_clusters), fill_value=0)
    print("\n  Choreography share (%) by Grade_Level:")
    with pd.option_context("display.float_format", lambda x: f"{x:5.1f}"):
        print(by_grade.to_string())

    # ---- 1b. Distribution by state ----
    by_state = pd.crosstab(df["State"], df["cluster"], normalize="index") * 100
    by_state = by_state.reindex(columns=range(n_clusters), fill_value=0)

    # ---- Variability across contexts ----
    print("\n  Spread of each choreography across contexts (std of % share):")
    print("    (low std = the profile appears in similar proportion everywhere "
          "= robust)")
    for c in range(n_clusters):
        print(f"    Cluster {c}: grade-std={by_grade[c].std():4.1f}  "
              f"state-std={by_state[c].std():4.1f}")

    # ---- Plots: stacked bars by grade and by state ----
    def stacked(tab, title, fname, xlabel):
        fig, ax = plt.subplots(figsize=(11, 5))
        bottom = np.zeros(len(tab))
        for c in range(n_clusters):
            ax.bar(tab.index.astype(str), tab[c].values, bottom=bottom,
                   label=f"Cluster {c}")
            bottom += tab[c].values
        ax.set_ylabel("Share of student-weeks (%)")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(ncol=n_clusters, fontsize=8)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        path = os.path.join(args.results_dir, fname)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  Saved {path}")

    print()
    stacked(by_grade, "Choreography distribution by grade level",
            "profile_by_grade.png", "Grade level")
    stacked(by_state, "Choreography distribution by state",
            "profile_by_state.png", "State")

    # ---- 2. Signature consistency across grade bands (H_Robustness spirit) ----
    print("\n  Signature consistency across grade bands (K-5 vs 6-12):")
    feats = pd.read_csv(feat_path)
    act_cols = [c for c in feats.columns if c.startswith("act__")]

    # normalize each week to proportions (same as clustering 'shape')
    counts = feats[act_cols].values.astype(float)
    sums = counts.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    prop = counts / sums
    feats_prop = pd.DataFrame(prop, columns=act_cols)
    feats_prop["Student_ID"] = feats["Student_ID"].values
    feats_prop["week_start"] = feats["week_start"].values

    merged = feats_prop.merge(clusters, on=["Student_ID", "week_start"]) \
                       .merge(demo[["Student_ID", "Grade_Level"]], on="Student_ID")

    # Grade_Level is text ("K", "Grade 1", ..., "Grade 12"). Convert to a number.
    def grade_to_num(g):
        s = str(g).strip()
        if s.upper() in ("K", "KINDERGARTEN"):
            return 0
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else np.nan

    merged["grade_num"] = merged["Grade_Level"].map(grade_to_num)
    merged["band"] = np.where(merged["grade_num"] <= 5, "K-5", "6-12")

    print(f"    {'cluster':>8} | cosine(K-5 profile, 6-12 profile)")
    sims = []
    for c in range(n_clusters):
        sub = merged[merged["cluster"] == c]
        lo = sub[sub["band"] == "K-5"][act_cols].mean().values
        hi = sub[sub["band"] == "6-12"][act_cols].mean().values
        sim = cosine(lo, hi)
        sims.append(sim)
        print(f"    {c:>8} | {sim:.4f}")
    print(f"\n    Mean signature similarity: {np.nanmean(sims):.4f}")
    print("    (1.0 = identical behavior across grade bands = robust profile)")

    # Save distribution table
    by_grade.to_csv(os.path.join(args.results_dir, "context_distribution.csv"))
    print(f"\n  Saved {os.path.join(args.results_dir, 'context_distribution.csv')}")
    print()


if __name__ == "__main__":
    main()