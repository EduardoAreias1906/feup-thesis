"""
=============================================================================
build_weekly_features.py — Stage 3 Vectorization (Feature Engineering)
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

Transforms the event-level activity_logs.csv into a WEEKLY feature matrix:
one row per (Student_ID, week), where the columns are the action-frequency
vector combined with time-of-day bins (Section 4.2.3 of the dissertation).

This is the input for k-means clustering (RQ1) and UMAP visualization (RQ2).

Unit of analysis: STUDENT-WEEK. All student-weeks are pooled so k-means learns
GLOBAL cluster definitions; each student then has a sequence of weekly cluster
labels, enabling the week-to-week ARI temporal-stability test (RQ1).

Expected columns in activity_logs.csv:
  Student_ID, Timestamp, Session_ID, Action_Type,
  Subject, Resource_Category, Time_Of_Day, Duration_Seconds

Feature columns: 17 action types x 4 time-of-day bins = 68 frequency features.
  Morning 06:00-11:59 | Afternoon 12:00-17:59 | Evening 18:00-21:59 |
  Night 22:00-05:59 (wraps midnight; captures late-night crammers)

Plus 3 metadata columns (not clustered on; kept for the RQ4 volume baseline
and for characterizing clusters): n_sessions, active_days, total_actions.

Outputs RAW COUNTS. Scaling/normalization happens in the clustering step.

Usage:
  python build_weekly_features.py
  python build_weekly_features.py --data-dir ./data --min-actions 3
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd

# Column names as they appear in YOUR activity_logs.csv
COL_STUDENT = "Student_ID"
COL_TIME    = "Timestamp"
COL_SESSION = "Session_ID"
COL_ACTION  = "Action_Type"

ACTION_TYPES = [
    "Login", "Logout",
    "Content_Study", "Video_Watch",
    "Assessment_Start", "Assessment_Submit",
    "Assignment_Start", "Assignment_Submit",
    "Synchronous_Join", "Synchronous_Leave",
    "Forum_View", "Forum_Post",
    "Feedback_Review",
    "Dashboard_View", "Grade_Check",
    "Resource_Download",
    "Page_Navigation",
]

TIME_BINS = ["Morning", "Afternoon", "Evening", "Night"]


def assign_time_bin(hours: pd.Series) -> np.ndarray:
    """Map hour-of-day (0-23) to one of four bins. Night wraps midnight."""
    conditions = [
        (hours >= 6) & (hours < 12),
        (hours >= 12) & (hours < 18),
        (hours >= 18) & (hours < 22),
    ]
    choices = ["Morning", "Afternoon", "Evening"]
    return np.select(conditions, choices, default="Night")


def build_features(logs: pd.DataFrame, min_actions: int) -> pd.DataFrame:
    print("  Parsing timestamps...")
    logs[COL_TIME] = pd.to_datetime(logs[COL_TIME])
    logs["hour"] = logs[COL_TIME].dt.hour
    logs["date"] = logs[COL_TIME].dt.normalize()

    # Week identifier = Monday 00:00 of that week
    logs["week_start"] = (
        logs[COL_TIME] - pd.to_timedelta(logs[COL_TIME].dt.weekday, unit="D")
    ).dt.normalize()

    print("  Assigning time-of-day bins...")
    logs["time_bin"] = assign_time_bin(logs["hour"])
    logs["feature"] = logs[COL_ACTION] + "__" + logs["time_bin"]

    print("  Aggregating counts per student-week x feature...")
    counts = (
        logs.groupby([COL_STUDENT, "week_start", "feature"])
        .size()
        .unstack("feature", fill_value=0)
    )

    full_cols = [f"{a}__{b}" for a in ACTION_TYPES for b in TIME_BINS]
    counts = counts.reindex(columns=full_cols, fill_value=0)

    print("  Computing per-week metadata (sessions, active days, totals)...")
    meta = logs.groupby([COL_STUDENT, "week_start"]).agg(
        n_sessions=(COL_SESSION, "nunique"),
        active_days=("date", "nunique"),
        total_actions=(COL_ACTION, "size"),
    )

    features = counts.join(meta).reset_index()

    before = len(features)
    features = features[features["total_actions"] >= min_actions].copy()
    dropped = before - len(features)
    if dropped:
        print(f"  Dropped {dropped:,} weeks with < {min_actions} actions.")

    features.sort_values([COL_STUDENT, "week_start"], inplace=True)
    features.reset_index(drop=True, inplace=True)
    return features


def print_summary(features: pd.DataFrame):
    feature_cols = [c for c in features.columns if "__" in c]
    n_students = features[COL_STUDENT].nunique()
    weeks_per_student = features.groupby(COL_STUDENT).size()

    print("\n" + "=" * 60)
    print("  WEEKLY FEATURE MATRIX - SUMMARY")
    print("=" * 60)
    print(f"  Rows (student-weeks):     {len(features):,}")
    print(f"  Unique students:          {n_students:,}")
    print(f"  Feature columns:          {len(feature_cols)} (action x time-bin)")
    print(f"  Date range:               {features['week_start'].min().date()} "
          f"-> {features['week_start'].max().date()}")
    print()
    print("  Weeks per student:")
    print(f"    min={weeks_per_student.min()}  "
          f"median={int(weeks_per_student.median())}  "
          f"max={weeks_per_student.max()}  "
          f"mean={weeks_per_student.mean():.1f}")
    print()
    block = features[feature_cols].values
    sparsity = (block == 0).mean() * 100
    print(f"  Feature-matrix sparsity:  {sparsity:.1f}% zeros")
    print()
    print("  Most frequent actions overall (sum across all weeks):")
    totals = features[feature_cols].sum().sort_values(ascending=False)
    for name, val in totals.head(8).items():
        print(f"    {name:32s} {int(val):>10,}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Build weekly feature matrix for clustering"
    )
    parser.add_argument("--data-dir", default="./data",
                        help="Directory containing activity_logs.csv")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <data-dir>/student_week_features.csv)")
    parser.add_argument("--min-actions", type=int, default=1,
                        help="Drop weeks with fewer than this many actions (default: 1)")
    args = parser.parse_args()

    logs_path = os.path.join(args.data_dir, "activity_logs.csv")
    if not os.path.exists(logs_path):
        print(f"  Not found: {logs_path}")
        print("    Run the generator first.")
        return

    out_path = args.output or os.path.join(args.data_dir, "student_week_features.csv")

    print("=" * 60)
    print("Stage 3 Vectorization - Weekly Feature Engineering")
    print("=" * 60)
    print(f"  Reading {logs_path} ...")
    logs = pd.read_csv(logs_path)
    print(f"  Loaded {len(logs):,} events.")

    expected = [COL_STUDENT, COL_TIME, COL_SESSION, COL_ACTION]
    missing = [c for c in expected if c not in logs.columns]
    if missing:
        print(f"  Missing expected columns: {missing}")
        print(f"    Columns found: {list(logs.columns)}")
        return

    features = build_features(logs, args.min_actions)
    features.to_csv(out_path, index=False)

    print_summary(features)
    print(f"  Saved -> {out_path}")
    print()
    print("  Next step: clustering (k-means + UMAP) on the 68 feature columns.")
    print()


if __name__ == "__main__":
    main()