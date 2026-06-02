"""
=============================================================================
run_clustering.py — Stage 3 Pattern Discovery (k-means + UMAP)
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

Takes student_week_features.csv and discovers the virtual choreographies:

  1. Normalize each student-week to PROPORTIONS (cluster on behavior SHAPE,
     not volume) + standardize so rare-but-discriminative actions count.
  2. Run k-means for a range of k; pick k with Elbow + Silhouette (RQ1).
  3. Project with UMAP to 2D to visualize cluster separation (RQ2).
  4. Characterize each cluster by its most OVER-represented features, so you
     can label them (e.g. the "Night Crammer" choreography).
  5. Save a per-(student, week) cluster label table for the downstream
     temporal-stability (ARI) and outcome-association (RQ3) analyses.

Outputs (in ./results/):
  k_selection.png            — Elbow + Silhouette plots
  umap_clusters.png          — 2D UMAP colored by cluster
  cluster_profiles.csv       — mean profile + volume per cluster
  student_week_clusters.csv  — Student_ID, week_start, cluster

Usage:
  python run_clustering.py                 # auto-pick k by silhouette
  python run_clustering.py --k 5           # force k = 5 after seeing the plot
  python run_clustering.py --k-max 8 --umap-sample 20000

Requirements:
  pip install scikit-learn umap-learn matplotlib seaborn pandas numpy
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd

COL_STUDENT = "Student_ID"
COL_WEEK    = "week_start"
META_COLS   = ["n_sessions", "active_days", "total_actions"]


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if "__" in c]
    if not feature_cols:
        raise ValueError("No feature columns (with '__') found. "
                         "Did you run build_weekly_features.py?")
    return df, feature_cols


def normalize_shape(X):
    """L1-normalize each row to proportions (focus on behavior shape)."""
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # guard against empty rows
    return X / row_sums


def select_k(X_scaled, k_min, k_max, seed, out_dir):
    """Compute inertia (elbow) and silhouette across k; save plot."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(k_min, k_max + 1))
    inertias, silhouettes = [], []

    print(f"  Searching k = {k_min}..{k_max} ...")
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        # silhouette on a subsample (full is O(n^2) and too slow)
        sil = silhouette_score(X_scaled, labels,
                               sample_size=min(10000, len(X_scaled)),
                               random_state=seed)
        silhouettes.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:14,.0f}  silhouette={sil:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ks, inertias, "o-", color="#1f6feb")
    ax1.set_title("Elbow Method (inertia)")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.grid(alpha=0.3)
    ax2.plot(ks, silhouettes, "o-", color="#bc8cff")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "k_selection.png")
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {plot_path}")

    best_k = ks[int(np.argmax(silhouettes))]
    return best_k, dict(zip(ks, silhouettes))


def characterize(df, feature_cols, X_prop, labels, out_dir):
    """Describe each cluster by its most over-represented features."""
    global_mean = X_prop.mean(axis=0)
    profiles = []
    print("\n" + "=" * 60)
    print("  CLUSTER CHARACTERIZATION")
    print("=" * 60)

    for c in sorted(np.unique(labels)):
        mask = labels == c
        size = mask.sum()
        share = size / len(labels) * 100
        cluster_mean = X_prop[mask].mean(axis=0)
        # over-representation = how much higher than the global average
        lift = cluster_mean - global_mean
        top_idx = np.argsort(lift)[::-1][:6]

        print(f"\n  Cluster {c}  ({size:,} student-weeks, {share:.1f}%)")
        print(f"    Avg total_actions/week: "
              f"{df.loc[mask, 'total_actions'].mean():.1f}  |  "
              f"sessions: {df.loc[mask, 'n_sessions'].mean():.1f}  |  "
              f"active days: {df.loc[mask, 'active_days'].mean():.1f}")
        print("    Most over-represented actions:")
        for i in top_idx:
            print(f"      {feature_cols[i]:34s} "
                  f"{cluster_mean[i]*100:5.1f}%  (global {global_mean[i]*100:4.1f}%)")

        row = {"cluster": c, "n_student_weeks": size, "share_pct": round(share, 1),
               "avg_total_actions": round(df.loc[mask, 'total_actions'].mean(), 1),
               "avg_sessions": round(df.loc[mask, 'n_sessions'].mean(), 1),
               "avg_active_days": round(df.loc[mask, 'active_days'].mean(), 1)}
        for i, col in enumerate(feature_cols):
            row[col] = round(float(cluster_mean[i]), 4)
        profiles.append(row)

    prof_df = pd.DataFrame(profiles)
    prof_path = os.path.join(out_dir, "cluster_profiles.csv")
    prof_df.to_csv(prof_path, index=False)
    print(f"\n  Saved {prof_path}")


def plot_umap(X_scaled, labels, sample, seed, out_dir):
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(X_scaled)
    if sample < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample, replace=False)
    else:
        idx = np.arange(n)

    print(f"\n  Running UMAP on {len(idx):,} points (this can take a minute)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        random_state=seed, n_components=2)
    emb = reducer.fit_transform(X_scaled[idx])

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels[idx],
                         cmap="tab10", s=4, alpha=0.5)
    ax.set_title("UMAP projection of weekly choreographies")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    legend = ax.legend(*scatter.legend_elements(),
                       title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    path = os.path.join(out_dir, "umap_clusters.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    ap = argparse.ArgumentParser(description="k-means + UMAP on weekly features")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--features", default=None,
                    help="Path to student_week_features.csv")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--k", type=int, default=None,
                    help="Force a specific k (skip auto-selection)")
    ap.add_argument("--umap-sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feat_path = args.features or os.path.join(args.data_dir, "student_week_features.csv")
    if not os.path.exists(feat_path):
        print(f"  Not found: {feat_path}")
        print("    Run build_weekly_features.py first.")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 3 Pattern Discovery — k-means + UMAP")
    print("=" * 60)
    print(f"  Reading {feat_path} ...")
    df, feature_cols = load_features(feat_path)
    print(f"  {len(df):,} student-weeks, {len(feature_cols)} features.")

    # 1. Shape normalization + standardization
    X_counts = df[feature_cols].values.astype(float)
    X_prop = normalize_shape(X_counts)          # proportions (for characterization)
    X_scaled = StandardScaler().fit_transform(X_prop)  # for distance-based k-means

    # 2. Choose k
    if args.k is None:
        best_k, sils = select_k(X_scaled, args.k_min, args.k_max,
                                args.seed, args.out_dir)
        print(f"\n  Suggested k (highest silhouette): {best_k}")
        print("  (Check k_selection.png — re-run with --k N to override.)")
        k = best_k
    else:
        k = args.k
        print(f"  Using k = {k} (user-specified).")

    # 3. Final clustering
    print(f"\n  Fitting final k-means with k={k} ...")
    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
    labels = km.fit_predict(X_scaled)

    # 4. Characterize clusters
    characterize(df, feature_cols, X_prop, labels, args.out_dir)

    # 5. Save labels
    out = df[[COL_STUDENT, COL_WEEK]].copy()
    out["cluster"] = labels
    labels_path = os.path.join(args.out_dir, "student_week_clusters.csv")
    out.to_csv(labels_path, index=False)
    print(f"\n  Saved {labels_path}")

    # 6. UMAP visualization
    try:
        plot_umap(X_scaled, labels, args.umap_sample, args.seed, args.out_dir)
    except ImportError:
        print("\n  (umap-learn not installed — skipping UMAP plot. "
              "Install with: pip install umap-learn)")

    print("\n  Done. Next: temporal stability (ARI) and outcome association (RQ3).")
    print()


if __name__ == "__main__":
    main()