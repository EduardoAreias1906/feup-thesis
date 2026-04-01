"""
=============================================================================
Synthetic Data Generator for "Virtual Choreographies of Learning"
Master's Thesis — Eduardo Salé Areias (FEUP / INESC TEC)
=============================================================================

Generates four interlinked CSV files matching the Data Specification:
  A. students.csv          — Demographics & context (Section 2.A)
  B. activity_logs.csv     — Timestamped event-level logs (Section 2.B)
  C. interactions.csv      — Teacher/peer interaction metadata (Section 2.C)
  D. academic_outcomes.csv — Grades, scores, completion (Section 2.D)

Design principles
-----------------
  - 2 000 students, K–12, spread across 32 Connections Academy schools.
  - 5 behavioral archetypes baked in so that clustering (k-means, UMAP)
    will surface meaningful choreographies aligned with hypotheses H1–H5.
  - At-risk students (~30%) have shifted distributions for engagement,
    matching the literature (Beck 2023/2024, Toppin & Toppin 2016).
  - Temporal realism: academic year Sep 2025 – Jun 2026, with
    weekday/weekend patterns, school breaks, and session-level coherence.

Usage
-----
  python generate_synthetic_data.py                     # defaults

Requirements
------------
  pip install pandas numpy
=============================================================================
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

SCHOOLS = [f"CA-{state}" for state in [
    "AL", "AZ", "CA", "CO", "FL", "GA", "ID", "IN", "IA", "KS",
    "LA", "ME", "MD", "MI", "MN", "MO", "MT", "NV", "NJ", "NM",
    "NC", "OH", "OK", "OR", "PA", "SC", "TX", "UT", "VA", "WA",
    "WI", "WY",
]]  # 32 schools

GRADE_LEVELS = list(range(0, 13))  # 0 = Kindergarten, 1–12

ACADEMIC_YEAR_START = datetime(2025, 9, 1)
ACADEMIC_YEAR_END   = datetime(2026, 6, 15)

# School breaks — no activity generated on these dates
BREAKS = [
    (datetime(2025, 11, 24), datetime(2025, 11, 28)),  # Thanksgiving
    (datetime(2025, 12, 20), datetime(2026, 1, 5)),     # Winter break
    (datetime(2026, 3, 16), datetime(2026, 3, 20)),     # Spring break
]

# Semantic action types (Table 4.1 in thesis)
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

# Maps each action to the resource category from the data spec
RESOURCE_CATEGORIES = {
    "Login":               "System",
    "Logout":              "System",
    "Content_Study":       "Asynchronous Content",
    "Video_Watch":         "Asynchronous Content",
    "Assessment_Start":    "Evaluation",
    "Assessment_Submit":   "Evaluation",
    "Assignment_Start":    "Evaluation",
    "Assignment_Submit":   "Evaluation",
    "Synchronous_Join":    "Live Classroom",
    "Synchronous_Leave":   "Live Classroom",
    "Forum_View":          "Social",
    "Forum_Post":          "Social",
    "Feedback_Review":     "Metacognition",
    "Dashboard_View":      "Navigation",
    "Grade_Check":         "Navigation",
    "Resource_Download":   "Asynchronous Content",
    "Page_Navigation":     "Navigation",
}

INTERACTION_TYPES = [
    "Message_Sent_To_Teacher",
    "Message_Received_From_Teacher",
    "Feedback_Received",
    "Forum_Post",
    "Forum_Reply_Received",
    "Live_Question_Asked",
]

SUBJECTS = [
    "Mathematics", "English Language Arts", "Science",
    "Social Studies", "Art", "Physical Education",
]

ENROLLMENT_REASONS = [
    "Medical/Health", "Bullying/Social", "Academic Struggles",
    "Family Preference", "Geographic/Rural", "Advanced Coursework",
    "Schedule Flexibility", "Other",
]

# ═══════════════════════════════════════════════════════════════════════════
# BEHAVIORAL ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════
# Each archetype shapes how activity logs are generated. Parameters are
# tuned so that unsupervised clustering will discover distinct groups
# aligned with the thesis hypotheses (H1–H5).

ARCHETYPES = {
    "Steady_Worker": {
        # H1: regular temporal rhythm → strong success
        "weight": 0.25,
        "sessions_per_week": (4.5, 0.8),        # (mean, std)
        "actions_per_session": (18, 4),
        "study_hour_center": 10,                 # peak study time (10 AM)
        "study_hour_spread": 2.0,                # tight consistency
        "weekend_activity_prob": 0.30,
        "sync_ratio": 0.35,                      # good sync/async balance
        "feedback_freq": 0.20,                   # frequent feedback review
        "interaction_per_week": (3.0, 1.0),
        "inactivity_gap_days": (1.0, 0.5),       # short gaps between sessions
        "grade_mean": 85, "grade_std": 7,
        "completion_prob": 0.95,
        "dropout_prob": 0.03,
    },
    "Balanced_Engager": {
        # H2: strong sync + async balance → good outcomes
        "weight": 0.20,
        "sessions_per_week": (4.0, 1.0),
        "actions_per_session": (15, 5),
        "study_hour_center": 14,                 # afternoon worker
        "study_hour_spread": 3.0,
        "weekend_activity_prob": 0.25,
        "sync_ratio": 0.45,                      # highest live participation
        "feedback_freq": 0.25,
        "interaction_per_week": (4.0, 1.5),
        "inactivity_gap_days": (1.5, 0.8),
        "grade_mean": 80, "grade_std": 8,
        "completion_prob": 0.90,
        "dropout_prob": 0.05,
    },
    "Night_Crammer": {
        # H5 (partial): irregular bursts, late night → moderate outcomes
        "weight": 0.20,
        "sessions_per_week": (3.0, 1.5),
        "actions_per_session": (22, 7),           # long intense bursts
        "study_hour_center": 22,                  # late night
        "study_hour_spread": 2.5,
        "weekend_activity_prob": 0.50,            # weekend cramming
        "sync_ratio": 0.10,                       # rarely joins live sessions
        "feedback_freq": 0.08,
        "interaction_per_week": (1.0, 0.8),
        "inactivity_gap_days": (3.0, 1.5),        # bigger gaps then bursts
        "grade_mean": 68, "grade_std": 12,
        "completion_prob": 0.70,
        "dropout_prob": 0.12,
    },
    "Minimal_Browser": {
        # H4 inverse: fragmented, sporadic navigation → weak outcomes
        "weight": 0.20,
        "sessions_per_week": (2.0, 1.0),
        "actions_per_session": (8, 3),
        "study_hour_center": 12,
        "study_hour_spread": 5.0,                 # very inconsistent timing
        "weekend_activity_prob": 0.15,
        "sync_ratio": 0.05,
        "feedback_freq": 0.04,
        "interaction_per_week": (0.5, 0.5),
        "inactivity_gap_days": (4.0, 2.0),
        "grade_mean": 55, "grade_std": 14,
        "completion_prob": 0.50,
        "dropout_prob": 0.25,
    },
    "Disengaged_Ghost": {
        # H5: long inactivity, minimal interaction → worst outcomes
        "weight": 0.15,
        "sessions_per_week": (1.0, 0.8),
        "actions_per_session": (5, 2),
        "study_hour_center": 15,
        "study_hour_spread": 6.0,                 # no consistent time
        "weekend_activity_prob": 0.10,
        "sync_ratio": 0.02,
        "feedback_freq": 0.02,
        "interaction_per_week": (0.2, 0.3),
        "inactivity_gap_days": (7.0, 3.0),        # week-long gaps
        "grade_mean": 40, "grade_std": 15,
        "completion_prob": 0.25,
        "dropout_prob": 0.45,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def is_break(date: datetime) -> bool:
    """Check if a date falls within a school break."""
    for start, end in BREAKS:
        if start.date() <= date.date() <= end.date():
            return True
    return False


def make_student_id(index: int) -> str:
    """Create a realistic-looking anonymized student ID."""
    return f"STU-{index:06d}"


def assign_archetype(rng: np.random.Generator, is_at_risk: bool) -> str:
    """
    Assign a behavioral archetype. At-risk students have shifted
    probabilities toward less-engaged profiles, matching the literature.
    """
    names = list(ARCHETYPES.keys())
    base_weights = np.array([ARCHETYPES[n]["weight"] for n in names])

    if is_at_risk:
        # Shift: less likely Steady/Balanced, more likely Minimal/Disengaged
        shift = np.array([-0.08, -0.05, +0.03, +0.05, +0.05])
        weights = np.clip(base_weights + shift, 0.02, None)
    else:
        weights = base_weights.copy()

    weights /= weights.sum()
    return rng.choice(names, p=weights)


def generate_session_actions(rng: np.random.Generator,
                             params: dict,
                             session_start: datetime,
                             subject: str) -> List[dict]:
    """
    Generate a coherent sequence of actions for one study session.
    The sequence follows a realistic flow: Login → activities → Logout.
    """
    n_actions = max(3, int(rng.normal(*params["actions_per_session"])))
    actions = []
    current_time = session_start

    # --- Login ---
    actions.append({
        "timestamp": current_time,
        "action_type": "Login",
        "duration_seconds": 0,
    })
    current_time += timedelta(seconds=int(rng.integers(5, 30)))

    # --- Core activities ---
    is_sync_session = rng.random() < params["sync_ratio"]

    for i in range(n_actions - 2):  # -2 for Login/Logout
        gap = int(rng.exponential(45) + 10)  # seconds between actions
        current_time += timedelta(seconds=gap)

        if is_sync_session and i == 0:
            # Synchronous session block
            action = "Synchronous_Join"
            duration = int(rng.normal(2700, 600))  # ~45 min live lesson
            actions.append({
                "timestamp": current_time,
                "action_type": action,
                "duration_seconds": max(300, duration),
            })
            current_time += timedelta(seconds=max(300, duration))
            actions.append({
                "timestamp": current_time,
                "action_type": "Synchronous_Leave",
                "duration_seconds": 0,
            })
            current_time += timedelta(seconds=int(rng.integers(10, 60)))
            continue

        # Choose action based on probabilities shaped by archetype
        roll = rng.random()
        if roll < 0.25:
            action = "Content_Study"
            duration = int(rng.normal(300, 120))    # ~5 min reading
        elif roll < 0.40:
            action = "Video_Watch"
            duration = int(rng.normal(480, 180))    # ~8 min video
        elif roll < 0.52:
            action = "Page_Navigation"
            duration = int(rng.exponential(20) + 5)
        elif roll < 0.62:
            action = "Assignment_Start" if rng.random() < 0.5 else "Assessment_Start"
            duration = int(rng.normal(600, 240))    # ~10 min
        elif roll < 0.70:
            action = "Assignment_Submit" if rng.random() < 0.5 else "Assessment_Submit"
            duration = int(rng.integers(5, 30))
        elif roll < 0.78:
            action = "Dashboard_View"
            duration = int(rng.integers(10, 60))
        elif roll < 0.84:
            action = "Resource_Download"
            duration = int(rng.integers(5, 20))
        elif roll < 0.88:
            action = "Grade_Check"
            duration = int(rng.integers(15, 90))
        elif roll < 0.92 and rng.random() < params["feedback_freq"] * 3:
            action = "Feedback_Review"
            duration = int(rng.normal(120, 60))
        elif roll < 0.96:
            action = "Forum_View"
            duration = int(rng.integers(30, 180))
        else:
            action = "Forum_Post"
            duration = int(rng.normal(180, 90))

        actions.append({
            "timestamp": current_time,
            "action_type": action,
            "duration_seconds": max(5, duration),
        })
        current_time += timedelta(seconds=max(5, duration))

    # --- Logout ---
    current_time += timedelta(seconds=int(rng.integers(5, 60)))
    actions.append({
        "timestamp": current_time,
        "action_type": "Logout",
        "duration_seconds": 0,
    })

    return actions


# ═══════════════════════════════════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_students(rng: np.random.Generator, num_students: int) -> pd.DataFrame:
    """Generate Table A: Student Demographics & Context."""
    print(f"  Generating {num_students} students...")

    rows = []
    for i in range(num_students):
        student_id = make_student_id(i)
        grade = rng.choice(GRADE_LEVELS, p=_grade_distribution())
        school = rng.choice(SCHOOLS)

        # ~30% at-risk, higher in middle/high school
        at_risk_base = 0.30 if grade >= 6 else 0.20
        is_at_risk = rng.random() < at_risk_base
        has_iep_504 = rng.random() < (0.25 if is_at_risk else 0.08)

        enrollment_start = ACADEMIC_YEAR_START + timedelta(
            days=int(rng.integers(0, 14))
        )

        archetype = assign_archetype(rng, is_at_risk)
        params = ARCHETYPES[archetype]

        # Some students withdraw early (dropout)
        dropped = rng.random() < params["dropout_prob"]
        if dropped:
            weeks_enrolled = max(4, int(rng.normal(16, 6)))
            enrollment_end = enrollment_start + timedelta(weeks=weeks_enrolled)
            if enrollment_end > ACADEMIC_YEAR_END:
                enrollment_end = ACADEMIC_YEAR_END
                dropped = False
        else:
            enrollment_end = ACADEMIC_YEAR_END

        # Enrollment reason — at-risk students skew toward certain reasons
        if is_at_risk:
            reason_weights = [0.20, 0.25, 0.30, 0.05, 0.05, 0.02, 0.08, 0.05]
        else:
            reason_weights = [0.05, 0.05, 0.05, 0.35, 0.15, 0.15, 0.15, 0.05]
        enrollment_reason = rng.choice(ENROLLMENT_REASONS, p=reason_weights)

        # Assign 3–6 subjects depending on grade level
        n_subjects = 3 if grade <= 2 else (4 if grade <= 5 else min(6, rng.integers(4, 7)))
        student_subjects = list(rng.choice(SUBJECTS, size=n_subjects, replace=False))

        rows.append({
            "student_id": student_id,
            "grade_level": grade,
            "school": school,
            "enrollment_start": enrollment_start.strftime("%Y-%m-%d"),
            "enrollment_end": enrollment_end.strftime("%Y-%m-%d"),
            "at_risk": is_at_risk,
            "iep_504_plan": has_iep_504,
            "enrollment_reason": enrollment_reason,
            "subjects": ";".join(student_subjects),
            "_archetype": archetype,        # internal only, dropped before saving
            "_withdrew_early": dropped,     # internal only, dropped before saving
        })

    return pd.DataFrame(rows)


def _grade_distribution() -> np.ndarray:
    """Slightly more students in grades 6–10 (typical for virtual schools)."""
    weights = np.array([
        3,   # K
        4, 4, 4, 5, 5,   # 1–5
        8, 9, 10, 11, 11, # 6–10
        9, 7,             # 11–12
    ], dtype=float)
    return weights / weights.sum()


def generate_activity_logs(rng: np.random.Generator,
                           students_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table B: Activity Logs (time-series event data)."""
    total = len(students_df)
    all_logs = []

    for idx, student in students_df.iterrows():
        if idx % 500 == 0:
            print(f"  Activity logs: student {idx + 1}/{total}...")

        params = ARCHETYPES[student["_archetype"]]
        subjects = student["subjects"].split(";")
        enroll_start = datetime.strptime(student["enrollment_start"], "%Y-%m-%d")
        enroll_end = datetime.strptime(student["enrollment_end"], "%Y-%m-%d")

        # Walk through each day of enrollment
        current_date = enroll_start
        session_counter = 0

        while current_date <= enroll_end:
            # Skip breaks
            if is_break(current_date):
                current_date += timedelta(days=1)
                continue

            # Weekend check
            is_weekend = current_date.weekday() >= 5
            if is_weekend and rng.random() > params["weekend_activity_prob"]:
                current_date += timedelta(days=1)
                continue

            # Does the student have a session today?
            # Convert weekly rate to daily probability
            daily_prob = params["sessions_per_week"][0] / 5.0  # out of weekdays
            daily_prob = min(daily_prob, 1.0)

            # Add noise
            daily_prob += rng.normal(0, 0.1)
            daily_prob = np.clip(daily_prob, 0.05, 0.98)

            if is_weekend:
                daily_prob *= 0.5

            if rng.random() > daily_prob:
                current_date += timedelta(days=1)
                continue

            # --- Generate session ---
            session_counter += 1
            session_id = f"{student['student_id']}-S{session_counter:05d}"

            # Study hour with archetype-specific distribution
            hour = rng.normal(params["study_hour_center"],
                              params["study_hour_spread"])
            hour = int(np.clip(hour, 6, 23))
            minute = int(rng.integers(0, 60))
            session_start = current_date.replace(hour=hour, minute=minute, second=0)

            subject = rng.choice(subjects)

            actions = generate_session_actions(rng, params, session_start, subject)

            for action in actions:
                all_logs.append({
                    "student_id": student["student_id"],
                    "timestamp": action["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id": session_id,
                    "action_type": action["action_type"],
                    "resource_category": RESOURCE_CATEGORIES[action["action_type"]],
                    "subject": subject,
                    "duration_seconds": action["duration_seconds"],
                })

            # Possible second session in the same day (multi-session students)
            if rng.random() < 0.15 and params["sessions_per_week"][0] >= 4:
                session_counter += 1
                session_id2 = f"{student['student_id']}-S{session_counter:05d}"
                later_hour = min(23, hour + int(rng.integers(3, 6)))
                session_start2 = current_date.replace(
                    hour=later_hour, minute=int(rng.integers(0, 60)), second=0
                )
                subject2 = rng.choice(subjects)
                actions2 = generate_session_actions(rng, params, session_start2, subject2)
                for action in actions2:
                    all_logs.append({
                        "student_id": student["student_id"],
                        "timestamp": action["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "session_id": session_id2,
                        "action_type": action["action_type"],
                        "resource_category": RESOURCE_CATEGORIES[action["action_type"]],
                        "subject": subject2,
                        "duration_seconds": action["duration_seconds"],
                    })

            current_date += timedelta(days=1)

    df = pd.DataFrame(all_logs)
    df.sort_values(["student_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_interactions(rng: np.random.Generator,
                          students_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table C: Interaction Metadata (no message content)."""
    total = len(students_df)
    all_interactions = []

    for idx, student in students_df.iterrows():
        if idx % 1000 == 0:
            print(f"  Interactions: student {idx + 1}/{total}...")

        params = ARCHETYPES[student["_archetype"]]
        enroll_start = datetime.strptime(student["enrollment_start"], "%Y-%m-%d")
        enroll_end = datetime.strptime(student["enrollment_end"], "%Y-%m-%d")
        enrolled_weeks = max(1, (enroll_end - enroll_start).days // 7)

        # Total interactions over the enrollment period
        mean_per_week, std_per_week = params["interaction_per_week"]
        total_interactions = max(0, int(
            rng.normal(mean_per_week * enrolled_weeks,
                       std_per_week * np.sqrt(enrolled_weeks))
        ))

        for _ in range(total_interactions):
            # Random date within enrollment
            day_offset = int(rng.integers(0, max(1, (enroll_end - enroll_start).days)))
            interaction_date = enroll_start + timedelta(days=day_offset)

            if is_break(interaction_date):
                continue

            hour = int(np.clip(rng.normal(12, 4), 7, 22))
            minute = int(rng.integers(0, 60))
            ts = interaction_date.replace(hour=hour, minute=minute,
                                          second=int(rng.integers(0, 60)))

            # Interaction type — feedback is more common for engaged students
            if rng.random() < params["feedback_freq"] * 2:
                itype = "Feedback_Received"
            elif rng.random() < params["sync_ratio"]:
                itype = "Live_Question_Asked"
            else:
                itype = rng.choice(INTERACTION_TYPES)

            # Sender (student or teacher/peer)
            if itype in ("Message_Sent_To_Teacher", "Forum_Post", "Live_Question_Asked"):
                sender = "Student"
            else:
                sender = "Teacher" if "Teacher" in itype or "Feedback" in itype else "Peer"

            all_interactions.append({
                "student_id": student["student_id"],
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "interaction_type": itype,
                "sender": sender,
            })

    df = pd.DataFrame(all_interactions)
    if not df.empty:
        df.sort_values(["student_id", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def generate_academic_outcomes(rng: np.random.Generator,
                               students_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table D: Academic Outcomes."""
    print("  Generating academic outcomes...")
    rows = []

    for _, student in students_df.iterrows():
        params = ARCHETYPES[student["_archetype"]]
        subjects = student["subjects"].split(";")
        dropped = student["_withdrew_early"]

        # Base academic ability (student-level random effect)
        ability_offset = rng.normal(0, 5)

        for subject in subjects:
            # Final grade
            grade = rng.normal(params["grade_mean"] + ability_offset,
                               params["grade_std"])
            # At-risk students have an additional penalty
            if student["at_risk"]:
                grade -= rng.uniform(3, 10)
            grade = np.clip(grade, 0, 100)

            # Course completion percentage
            if dropped:
                weeks_enrolled = max(1, (
                    datetime.strptime(student["enrollment_end"], "%Y-%m-%d") -
                    datetime.strptime(student["enrollment_start"], "%Y-%m-%d")
                ).days // 7)
                total_weeks = (ACADEMIC_YEAR_END - ACADEMIC_YEAR_START).days // 7
                completion_pct = round((weeks_enrolled / total_weeks) * 100, 1)
                completion_pct = min(completion_pct, rng.uniform(30, 85))
                passed = False
            else:
                completed = rng.random() < params["completion_prob"]
                completion_pct = round(rng.uniform(85, 100) if completed
                                       else rng.uniform(40, 84), 1)
                passed = grade >= 60 and completion_pct >= 70

            # Assignment scores (5–15 assignments per subject)
            n_assignments = int(rng.integers(5, 16))
            assignment_scores = np.clip(
                rng.normal(grade, params["grade_std"] * 0.7, size=n_assignments),
                0, 100
            ).round(1).tolist()

            # Quiz scores (3–8 quizzes per subject)
            n_quizzes = int(rng.integers(3, 9))
            quiz_scores = np.clip(
                rng.normal(grade - 2, params["grade_std"] * 0.8, size=n_quizzes),
                0, 100
            ).round(1).tolist()

            # Standardized test score (~60% of students have one)
            has_std_test = rng.random() < 0.60
            std_test_score = None
            if has_std_test:
                # Scale: roughly 200–800
                std_test_score = int(np.clip(
                    rng.normal(350 + grade * 3.5 + ability_offset * 2,
                               50),
                    200, 800
                ))

            rows.append({
                "student_id": student["student_id"],
                "subject": subject,
                "final_grade": round(grade, 1),
                "assignment_scores": ";".join(map(str, assignment_scores)),
                "quiz_scores": ";".join(map(str, quiz_scores)),
                "standardized_test_score": std_test_score,
                "completion_percentage": completion_pct,
                "passed": passed,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# DATA DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════

def generate_data_dictionary(output_dir: str):
    """Create a human-readable data dictionary (Section 3 of the spec)."""
    content = """# Data Dictionary — Synthetic Virtual Choreographies Dataset
# =============================================================
# Generated by generate_synthetic_data.py
# Matches the Data Specification for Master's Thesis (FEUP/INESC TEC)

## students.csv
| Column             | Type     | Description                                              |
|--------------------|----------|----------------------------------------------------------|
| student_id         | string   | Anonymized unique ID (e.g., STU-000042)                  |
| grade_level        | int      | 0 = Kindergarten, 1–12                                   |
| school             | string   | School identifier (e.g., CA-FL)                          |
| enrollment_start   | date     | First day of enrollment (YYYY-MM-DD)                     |
| enrollment_end     | date     | Last day of enrollment (withdrawal date if applicable)   |
| at_risk            | bool     | At-risk status indicator                                 |
| iep_504_plan       | bool     | Has IEP or 504 accommodation plan                       |
| enrollment_reason  | string   | Reason for choosing online school                        |
| subjects           | string   | Semicolon-separated list of enrolled subjects            |

## activity_logs.csv
| Column             | Type     | Description                                              |
|--------------------|----------|----------------------------------------------------------|
| student_id         | string   | Links to students.csv                                    |
| timestamp          | datetime | Exact date and time (YYYY-MM-DD HH:MM:SS)               |
| session_id         | string   | Groups actions into a single study session               |
| action_type        | string   | Semantic action (see list below)                         |
| resource_category  | string   | Category of content accessed                             |
| subject            | string   | Course/subject for this session                          |
| duration_seconds   | int      | Time spent on this action (0 for instantaneous events)   |

### Action Types
| Action              | Resource Category    | Description                         |
|---------------------|----------------------|-------------------------------------|
| Login               | System               | Session start                       |
| Logout              | System               | Session end                         |
| Content_Study       | Asynchronous Content | Reading text/content pages          |
| Video_Watch         | Asynchronous Content | Watching instructional video        |
| Assessment_Start    | Evaluation           | Started a quiz/test                 |
| Assessment_Submit   | Evaluation           | Submitted a quiz/test               |
| Assignment_Start    | Evaluation           | Started working on an assignment    |
| Assignment_Submit   | Evaluation           | Submitted an assignment             |
| Synchronous_Join    | Live Classroom       | Joined a live lesson                |
| Synchronous_Leave   | Live Classroom       | Left a live lesson                  |
| Forum_View          | Social               | Viewed discussion forum             |
| Forum_Post          | Social               | Posted in discussion forum          |
| Feedback_Review     | Metacognition        | Reviewed teacher feedback           |
| Dashboard_View      | Navigation           | Viewed student dashboard            |
| Grade_Check         | Navigation           | Checked grades                      |
| Resource_Download   | Asynchronous Content | Downloaded a resource/file          |
| Page_Navigation     | Navigation           | General page navigation             |

## interactions.csv
| Column             | Type     | Description                                              |
|--------------------|----------|----------------------------------------------------------|
| student_id         | string   | Links to students.csv                                    |
| timestamp          | datetime | When the interaction occurred                            |
| interaction_type   | string   | Type of interaction (see list below)                     |
| sender             | string   | Who initiated: Student, Teacher, or Peer                 |

### Interaction Types
Message_Sent_To_Teacher, Message_Received_From_Teacher,
Feedback_Received, Forum_Post, Forum_Reply_Received, Live_Question_Asked

## academic_outcomes.csv
| Column                    | Type   | Description                                      |
|---------------------------|--------|--------------------------------------------------|
| student_id                | string | Links to students.csv                            |
| subject                   | string | Course/subject name                              |
| final_grade               | float  | Final course grade (0–100)                       |
| assignment_scores         | string | Semicolon-separated individual scores (0–100)    |
| quiz_scores               | string | Semicolon-separated individual scores (0–100)    |
| standardized_test_score   | int    | Standardized test (200–800 scale), null if N/A   |
| completion_percentage     | float  | Percentage of course completed                   |
| passed                    | bool   | True if grade >= 60 and completion >= 70%        |
"""
    path = os.path.join(output_dir, "DATA_DICTIONARY.md")
    with open(path, "w") as f:
        f.write(content)
    print(f"  Data dictionary → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for Virtual Choreographies thesis"
    )
    parser.add_argument("--output-dir", default="./data",
                        help="Directory for output CSV files (default: ./data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num-students", type=int, default=2000,
                        help="Number of students to generate (default: 2000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("Virtual Choreographies — Synthetic Data Generator")
    print("=" * 60)
    print(f"  Students:   {args.num_students}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    print()

    # A. Students
    print("[1/4] Students & demographics...")
    students_df = generate_students(rng, args.num_students)
    students_path = os.path.join(args.output_dir, "students.csv")
    # Drop internal columns before saving
    save_cols = [c for c in students_df.columns if not c.startswith("_")]
    students_df[save_cols].to_csv(students_path, index=False)
    print(f"  ✓ {len(students_df)} students → {students_path}")

    # Print summary
    print(f"  At-risk students: {students_df['at_risk'].sum()} "
          f"({students_df['at_risk'].mean()*100:.1f}%)")
    print()

    # B. Activity logs
    print("[2/4] Activity logs (this takes a while for 2000 students)...")
    logs_df = generate_activity_logs(rng, students_df)
    logs_path = os.path.join(args.output_dir, "activity_logs.csv")
    logs_df.to_csv(logs_path, index=False)
    print(f"  ✓ {len(logs_df):,} events → {logs_path}")
    print()

    # C. Interactions
    print("[3/4] Interaction metadata...")
    interactions_df = generate_interactions(rng, students_df)
    interactions_path = os.path.join(args.output_dir, "interactions.csv")
    interactions_df.to_csv(interactions_path, index=False)
    print(f"  ✓ {len(interactions_df):,} interactions → {interactions_path}")
    print()

    # D. Academic outcomes
    print("[4/4] Academic outcomes...")
    outcomes_df = generate_academic_outcomes(rng, students_df)
    outcomes_path = os.path.join(args.output_dir, "academic_outcomes.csv")
    outcomes_df.to_csv(outcomes_path, index=False)
    print(f"  ✓ {len(outcomes_df):,} outcome records → {outcomes_path}")
    print()

    # Data dictionary
    generate_data_dictionary(args.output_dir)

    # Summary
    print()
    print("=" * 60)
    print("DONE! Files generated:")
    print("=" * 60)
    for fname in ["students.csv", "activity_logs.csv",
                   "interactions.csv", "academic_outcomes.csv",
                   "DATA_DICTIONARY.md"]:
        fpath = os.path.join(args.output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {fname:30s} {size_mb:8.1f} MB")
    print()
    print("Done! You can now inspect the data with inspect_data.py")
    print()


if __name__ == "__main__":
    main()