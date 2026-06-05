"""
=============================================================================
temporal_stability.py — RQ2 part (a): Temporal Stability of Choreographies
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

RQ2 asks whether the discovered choreographies are consistent habits rather
than week-specific noise. This script measures, over time, how stable each
student's choreography (cluster) membership is from one week to the next.

Two complementary views:

  1. WEEK-OVER-WEEK ARI (the thesis metric, section 4.3.1)
     For each pair of consecutive calendar weeks, take the students present in
     BOTH weeks and compute the Adjusted Rand Index between their cluster
     labels in week W and week W+1. High ARI = the grouping of students is
     preserved across weeks (stable behavioral types). Also reports the simple
     "retention" (% of students who kept the exact same cluster).

  2. TRANSITION MATRIX
     Across all students' consecutive (7-day-apart) weeks, how often does a
     student in cluster i move to cluster j? Row-normalized to %. The diagonal
     is the "stickiness" of each choreography.

Input:  results/student_week_clusters.csv  (Student_ID, week_start, cluster)
Output: results/temporal_ari.png           (ARI + retention over time)
        results/transition_matrix.png       (heatmap)
        results/transition_matrix.csv

NOTE (honest framing for the thesis): in this dataset each student has a fixed
underlying profile, so stability is expected to be HIGH by design. Interpret
the result as confirming the measurement works and quantifying the residual
week-to-week variation, not as an independent empirical discovery.

Usage:
  python temporal_stability.py
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    path = os.path.join(args.results_dir, "student_week_clusters.csv")
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        print("    Run run_clustering.py first.")
        return

    from sklearn.metrics import adjusted_rand_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("RQ2 (a) — Temporal Stability")
    print("=" * 60)

    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    n_clusters = df["cluster"].nunique()
    print(f"  {len(df):,} student-weeks, {df['Student_ID'].nunique():,} students, "
          f"{n_clusters} clusters.")

    # ---- 1. Week-over-week ARI ----
    weeks = sorted(df["week_start"].unique())
    rows = []
    for i in range(len(weeks) - 1):
        w1, w2 = weeks[i], weeks[i + 1]
        gap = (pd.Timestamp(w2) - pd.Timestamp(w1)).days
        a = df[df["week_start"] == w1].set_index("Student_ID")["cluster"]
        b = df[df["week_start"] == w2].set_index("Student_ID")["cluster"]
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        la, lb = a.loc[common], b.loc[common]
        ari = adjusted_rand_score(la, lb)
        retention = float((la.values == lb.values).mean()) * 100
        rows.append({"week": pd.Timestamp(w1), "gap_days": gap,
                     "n_common": len(common), "ari": ari,
                     "retention_pct": retention})

    ts = pd.DataFrame(rows)
    print("\n  Week-over-week ARI:")
    print(f"    mean ARI       = {ts['ari'].mean():.3f}")
    print(f"    std  ARI       = {ts['ari'].std():.3f}")
    print(f"    min / max ARI  = {ts['ari'].min():.3f} / {ts['ari'].max():.3f}")
    print(f"    mean retention = {ts['retention_pct'].mean():.1f}% "
          f"(students keeping the same cluster week to week)")

    # Plot ARI + retention over time
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(ts["week"], ts["ari"], "o-", color="#1f6feb", label="ARI")
    ax1.set_ylabel("ARI (week vs next)", color="#1f6feb")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor="#1f6feb")
    ax1.set_xlabel("Week")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ts["week"], ts["retention_pct"], "s--", color="#bc8cff",
             alpha=0.7, label="Retention %")
    ax2.set_ylabel("Same-cluster retention (%)", color="#bc8cff")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#bc8cff")
    fig.suptitle("Temporal stability of weekly choreographies")
    fig.tight_layout()
    p1 = os.path.join(args.results_dir, "temporal_ari.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"  Saved {p1}")

    # ---- 2. Transition matrix (consecutive 7-day weeks) ----
    df_sorted = df.sort_values(["Student_ID", "week_start"]).copy()
    g = df_sorted.groupby("Student_ID")
    df_sorted["next_cluster"] = g["cluster"].shift(-1)
    df_sorted["next_week"] = g["week_start"].shift(-1)
    df_sorted["gap"] = (df_sorted["next_week"] - df_sorted["week_start"]).dt.days
    adj = df_sorted[df_sorted["gap"] == 7].dropna(subset=["next_cluster"])
    adj["next_cluster"] = adj["next_cluster"].astype(int)

    tm = pd.crosstab(adj["cluster"], adj["next_cluster"], normalize="index") * 100
    tm = tm.reindex(index=range(n_clusters), columns=range(n_clusters),
                    fill_value=0)

    print("\n  Transition matrix (row = from, col = to, % of next week):")
    with pd.option_context("display.float_format", lambda x: f"{x:5.1f}"):
        print(tm.to_string())
    print("\n  Stickiness (diagonal — % staying in same choreography):")
    for c in range(n_clusters):
        print(f"    Cluster {c}: {tm.loc[c, c]:5.1f}%")

    # Heatmap
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.heatmap(tm, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                    cbar_kws={"label": "% of transitions"})
        ax.set_title("Week-to-week cluster transitions (%)")
        ax.set_xlabel("To cluster (next week)")
        ax.set_ylabel("From cluster (this week)")
        fig.tight_layout()
        p2 = os.path.join(args.results_dir, "transition_matrix.png")
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        print(f"\n  Saved {p2}")
    except ImportError:
        print("\n  (seaborn not installed — skipping heatmap)")

    tm.to_csv(os.path.join(args.results_dir, "transition_matrix.csv"))
    print(f"  Saved {os.path.join(args.results_dir, 'transition_matrix.csv')}")
    print()


if __name__ == "__main__":
    main()