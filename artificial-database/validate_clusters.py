"""
=============================================================================
validate_clusters.py — Validate discovered clusters against true archetypes
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

The clustering was UNSUPERVISED: k-means never saw the archetype labels.
This script checks whether the clusters it found actually correspond to the
5 archetypes that generated the data. This is the key validation for RQ1:
if the pipeline rediscovers the known structure on synthetic data, we can
trust it on the real Pearson data later.

It joins:
  results/student_week_clusters.csv   (Student_ID, week_start, cluster)
  data/activity_logs.csv              (provides the per-student Archetype)

and produces:
  - a cluster x archetype contingency table (counts + row %)
  - the dominant archetype and purity of each cluster
  - Adjusted Rand Index (ARI) and Normalized Mutual Info (NMI) between the
    discovered clustering and the true archetypes (overall agreement scores)
  - results/cluster_vs_archetype.csv

NOTE: validation is done at the STUDENT-WEEK level. Each student-week's true
archetype is the archetype of that student (archetype is fixed per student).

Usage:
  python validate_clusters.py
=============================================================================
"""

import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    clusters_path = os.path.join(args.results_dir, "student_week_clusters.csv")
    logs_path = os.path.join(args.data_dir, "activity_logs.csv")

    for p in (clusters_path, logs_path):
        if not os.path.exists(p):
            print(f"  Not found: {p}")
            return

    print("=" * 60)
    print("Cluster Validation against True Archetypes (RQ1)")
    print("=" * 60)

    clusters = pd.read_csv(clusters_path)

    # Get the (Student_ID -> Archetype) mapping from the logs.
    # usecols keeps memory low on the big file.
    print("  Reading archetype labels from activity_logs.csv ...")
    arche = pd.read_csv(logs_path, usecols=["Student_ID", "Archetype"])
    student_archetype = (
        arche.drop_duplicates("Student_ID").set_index("Student_ID")["Archetype"]
    )
    print(f"  Found {student_archetype.nunique()} archetypes across "
          f"{len(student_archetype):,} students.")

    # Attach true archetype to each student-week
    clusters["archetype"] = clusters["Student_ID"].map(student_archetype)
    clusters = clusters.dropna(subset=["archetype"])

    # ---- Contingency table: cluster (rows) x archetype (cols) ----
    ct = pd.crosstab(clusters["cluster"], clusters["archetype"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    print("\n  Cluster x Archetype  (row % — what each cluster is made of):\n")
    with pd.option_context("display.width", None,
                           "display.max_columns", None,
                           "display.float_format", lambda x: f"{x:5.1f}"):
        print(ct_pct.to_string())

    # ---- Dominant archetype + purity per cluster ----
    print("\n  Dominant archetype per cluster:")
    for c in ct.index:
        row = ct.loc[c]
        dom = row.idxmax()
        purity = row.max() / row.sum() * 100
        print(f"    Cluster {c}: {dom:18s} (purity {purity:4.1f}%, "
              f"n={row.sum():,})")

    # ---- Overall agreement scores ----
    try:
        from sklearn.metrics import (adjusted_rand_score,
                                     normalized_mutual_info_score)
        ari = adjusted_rand_score(clusters["archetype"], clusters["cluster"])
        nmi = normalized_mutual_info_score(clusters["archetype"],
                                           clusters["cluster"])
        print("\n  Overall agreement (discovered clusters vs true archetypes):")
        print(f"    Adjusted Rand Index (ARI):       {ari:.3f}")
        print(f"    Normalized Mutual Info (NMI):    {nmi:.3f}")
        print("    (1.0 = perfect recovery; 0.0 = random. For behavioral data,")
        print("     0.4-0.7 already indicates the structure was largely recovered.)")
    except ImportError:
        print("\n  (scikit-learn not available — skipping ARI/NMI.)")

    # Save the table
    out = os.path.join(args.results_dir, "cluster_vs_archetype.csv")
    ct.to_csv(out)
    print(f"\n  Saved counts table -> {out}")
    print()


if __name__ == "__main__":
    main()