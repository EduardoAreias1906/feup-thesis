"""
=============================================================================
build_weekly_features.py — Stage 3 Vectorization (Feature Engineering)
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

Transforms event-level activity_logs.csv into a WEEKLY feature matrix:
one row per (Student_ID, week). The vector deliberately separates two
questions so neither drowns the other:

  WHAT the student does  -> proportion of each meaningful action type
  WHEN they do it        -> proportion of activity in each time-of-day bin

WHY THIS DESIGN (important):
A previous version used 17 actions x 4 time bins = 68 "Action__Bin" columns.
That made k-means split students mostly by HOUR OF DAY (a Morning cluster, an
Afternoon cluster...), because each action got fragmented across 4 columns and
the time dimension dominated. It also let universal actions (Login, Logout,
Page_Navigation) — which everyone does and which carry no behavioral signal —
sit at the top of every cluster.

Fix:
  1. DROP universal/structural actions that don't discriminate behavior
     (Login, Logout). They just bound sessions.
  2. Keep WHAT and WHEN as two SEPARATE, balanced blocks instead of a single
     68-col cross product. "Content_Study" is now one feature regardless of
     hour; the hour is summarized separately in 4 time-share columns.

This focuses clustering on behavioral SHAPE (what mix of activities), which is
what the hypotheses (H2 sync/async, H3 feedback, H5 disengagement) are about,
while still preserving chronotype as a smaller, non-dominant signal.

Unit of analysis: STUDENT-WEEK. All student-weeks pooled -> global k-means
clusters -> each student becomes a sequence of weekly labels (enables the
week-to-week ARI temporal-stability test, RQ1).

Output columns:
  act__<ActionType>   : count of that action that week (raw count)
  time__<Bin>         : count of events in that time bin that week (raw count)
  n_sessions, active_days, total_actions : metadata (NOT clustered on)

Scaling/normalization happens in run_clustering.py.

Usage:
  python build_weekly_features.py
  python build_weekly_features.py --data-dir ./data --min-actions 3
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd

COL_STUDENT = "Student_ID"
COL_TIME    = "Timestamp"
COL_SESSION = "Session_ID"
COL_ACTION  = "Action_Type"

# Actions that carry behavioral meaning (used as WHAT features).
# Login/Logout are EXCLUDED: everyone does them, they only bound sessions.
BEHAVIORAL_ACTIONS = [
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

# Actions excluded from the WHAT block (still counted in metadata totals).
EXCLUDED_ACTIONS = ["Login", "Logout"]

TIME_BINS = ["Morning", "Afternoon", "Evening", "Night"]


def assign_time_bin(hours: pd.Series) -> np.ndarray:
    """
    Map hour-of-day (0-23) to one of four coarse bins.

    Four bins keeps the WHEN block balanced in size with the WHAT block (~15
    action features vs 4 time features). Finer bins (hourly) would let the
    time dimension dominate k-means distances. Night wraps midnight (22-05h)
    and is the default/catch-all so no hour is left unlabelled.
    """
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

    logs["time_bin"] = assign_time_bin(logs["hour"])

    # ---- WHAT block: counts per behavioral action (Login/Logout dropped) ----
    print("  Building WHAT block (action-type counts)...")
    behav = logs[logs[COL_ACTION].isin(BEHAVIORAL_ACTIONS)]

    # groupby([student, week, action]).size() produces a Series with a
    # 3-level MultiIndex (Student_ID, week_start, Action_Type) and a count value.
    # .unstack(COL_ACTION) pivots the Action_Type level into columns, turning a
    # "long" format (one row per event type) into a "wide" format (one row per
    # student-week, one column per action). fill_value=0 ensures that actions a
    # student never performed that week get 0 rather than NaN, so the resulting
    # matrix is dense and suitable for k-means distance calculations.
    what = (
        behav.groupby([COL_STUDENT, "week_start", COL_ACTION])
        .size()
        .unstack(COL_ACTION, fill_value=0)
    )
    # reindex guarantees every action column is present in a fixed order,
    # even if no student performed that action in the loaded dataset.
    what = what.reindex(columns=BEHAVIORAL_ACTIONS, fill_value=0)
    what.columns = [f"act__{c}" for c in what.columns]

    # ---- WHEN block: counts per time-of-day bin (all events) ----
    # All events (including Login/Logout) count toward WHEN: the clock time of
    # a login is just as valid a chronotype signal as the clock time of studying.
    # The same groupby+unstack pattern is used — pivot time_bin into columns.
    print("  Building WHEN block (time-of-day counts)...")
    when = (
        logs.groupby([COL_STUDENT, "week_start", "time_bin"])
        .size()
        .unstack("time_bin", fill_value=0)
    )
    when = when.reindex(columns=TIME_BINS, fill_value=0)
    when.columns = [f"time__{c}" for c in when.columns]

    # ---- Metadata (not clustered on) ----
    print("  Computing per-week metadata (sessions, active days, totals)...")
    meta = logs.groupby([COL_STUDENT, "week_start"]).agg(
        n_sessions=(COL_SESSION, "nunique"),
        active_days=("date", "nunique"),
        total_actions=(COL_ACTION, "size"),
    )

    features = what.join(when, how="outer").join(meta, how="outer").reset_index()
    features = features.fillna(0)

    before = len(features)
    features = features[features["total_actions"] >= min_actions].copy()
    dropped = before - len(features)
    if dropped:
        print(f"  Dropped {dropped:,} weeks with < {min_actions} actions.")

    features.sort_values([COL_STUDENT, "week_start"], inplace=True)
    features.reset_index(drop=True, inplace=True)
    return features


def print_summary(features: pd.DataFrame):
    act_cols  = [c for c in features.columns if c.startswith("act__")]
    time_cols = [c for c in features.columns if c.startswith("time__")]
    n_students = features[COL_STUDENT].nunique()
    weeks_per_student = features.groupby(COL_STUDENT).size()

    print("\n" + "=" * 60)
    print("  WEEKLY FEATURE MATRIX - SUMMARY")
    print("=" * 60)
    print(f"  Rows (student-weeks):     {len(features):,}")
    print(f"  Unique students:          {n_students:,}")
    print(f"  WHAT features (actions):  {len(act_cols)}  (Login/Logout excluded)")
    print(f"  WHEN features (time):     {len(time_cols)}")
    print(f"  Date range:               {features['week_start'].min().date()} "
          f"-> {features['week_start'].max().date()}")
    print()
    print("  Weeks per student:")
    print(f"    min={weeks_per_student.min()}  "
          f"median={int(weeks_per_student.median())}  "
          f"max={weeks_per_student.max()}  "
          f"mean={weeks_per_student.mean():.1f}")
    print()
    print("  Total action counts (WHAT block, summed across all weeks):")
    totals = features[act_cols].sum().sort_values(ascending=False)
    for name, val in totals.items():
        print(f"    {name:28s} {int(val):>12,}")
    print()
    print("  Time-of-day distribution (WHEN block):")
    tt = features[time_cols].sum()
    tt_pct = tt / tt.sum() * 100
    for name in time_cols:
        print(f"    {name:18s} {tt_pct[name]:5.1f}%")
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
    print("  Next step: re-run clustering (python run_clustering.py).")
    print()


if __name__ == "__main__":
    main()