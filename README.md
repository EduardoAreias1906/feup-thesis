# Virtual Choreographies of Learning — Analysis Pipeline

Master's thesis — Eduardo Salé Areias (FEUP / INESC TEC)
*Virtual Choreographies of Learning: AI-Driven Discovery in K–12 Online American Schools*

This repository contains the full data-analysis pipeline for the dissertation:
from synthetic data generation through to the five research questions
(behavioral pattern discovery, stability, outcome association, early prediction,
and actionable interpretation).

---

## ⚠️ Note on the data

The analysis runs on a **synthetic dataset** generated locally. The real
Connections Academy / Pearson dataset was not available, so a synthetic dataset
is used as the study's dataset (its synthetic nature is stated in the
methodology). Behavioral *archetypes* are baked into the generator to create
realistic variation; the pipeline rediscovers them without supervision, which
validates the methodology.

The `data/` folder is **not** version-controlled (large files). Regenerate it
locally with the generator (step 1 below). A fixed random seed (42) makes the
output reproducible.

---

## Requirements

```bash
pip install pandas numpy scikit-learn umap-learn matplotlib seaborn scipy
# optional (RQ5 SHAP bonus):
pip install shap
```

Python 3.10+ recommended.

---

## How to run (full pipeline, in order)

Run every command from the project root (the folder that contains these scripts
and the `data/` folder).

| # | Command | Produces | Answers |
|---|---------|----------|---------|
| 1 | `python database-generator.py --num-students 5000` | `data/*.csv` (4 files) | — |
| 2 | `python build_weekly_features.py` | `data/student_week_features.csv` | feature engineering |
| 3 | `python run_clustering.py --k 4` | `results/student_week_clusters.csv`, k-selection & UMAP plots | **RQ1** |
| 4 | `python validate_clusters.py` | `results/cluster_vs_archetype.csv` | **RQ1** (validation) |
| 5 | `python temporal_stability.py` | `results/temporal_ari.png`, `transition_matrix.*` | **RQ2 (a)** |
| 6 | `python context_stability.py` | `results/profile_by_*.png`, `context_distribution.csv` | **RQ2 (b)** |
| 7 | `python outcome_association.py` | `results/outcomes_by_cluster.csv`, grade plots | **RQ3** |
| 8 | `python early_prediction.py` | `results/rq4_results.csv`, `rq4_comparison.png` | **RQ4** |
| 9 | `python interpretability.py` | `results/rq5_importance.*` | **RQ5** |

Tip: for a quick test, generate a smaller sample first
(`python database-generator.py --num-students 200`).

---

## The data files (`data/`)

| File | Content |
|------|---------|
| `demographics.csv` | Student_ID, Grade_Level, School_ID, State, enrolment dates, At_Risk_Status, IEP_Status, Enrollment_Reason, Subjects |
| `activity_logs.csv` | Event-level logs: Student_ID, Timestamp, Session_ID, Action_Type, Subject, Resource_Category, Time_Of_Day, Duration_Seconds (+ Archetype, ground-truth label) |
| `interaction_metadata.csv` | Teacher/peer interaction metadata (type, sender, time) |
| `academic_outcomes.csv` | Student_ID, Subject, Final_Grade, Assignment_Average, Quiz_Average, Standardized_Test_Score, Course_Completion_Percentage, Pass_Fail |

> Note: `Archetype` (in `activity_logs.csv`) is ground truth used ONLY for
> validation (step 4). It is never used as an input to clustering or prediction.

---

## The scripts

| Script | Role |
|--------|------|
| `database-generator.py` | Generates the synthetic dataset (5 archetypes, fixed seed). |
| `build_weekly_features.py` | Aggregates events into one row per student-week: action-mix (`act__*`) + time-of-day (`time__*`) + volume metadata. Login/Logout excluded. |
| `run_clustering.py` | k-means (shape-normalized) + UMAP. Picks k via elbow/silhouette or `--k`. |
| `validate_clusters.py` | Cross-tabs discovered clusters vs true archetypes; ARI / NMI. |
| `temporal_stability.py` | Week-over-week ARI + cluster transition matrix. |
| `context_stability.py` | Profile distribution by grade/school/state + signature consistency. |
| `outcome_association.py` | Grades/pass-rate by choreography; ANOVA; at-risk control. |
| `early_prediction.py` | Random Forest, first-N-weeks (no leakage); baseline vs +choreography. |
| `interpretability.py` | RF feature importance + direction of effect (+ optional SHAP). |

---

## Research questions → results

- **RQ1 — Identification.** 4 choreographies discovered (Steady_Worker,
  Balanced_Engager, Low_Engagement, Night_Crammer). Validated against archetypes
  (ARI ≈ 0.53).
- **RQ2 — Robustness.** Temporally stable (week-over-week ARI ≈ 0.80, retention
  ≈ 91%) and context-robust (near-identical profile distribution across grades,
  schools, states; cross-grade signature similarity ≈ 1.0).
- **RQ3 — Outcomes.** Clear success hierarchy (Balanced ≈ Steady > Night_Crammer
  > Low_Engagement); ANOVA significant, large effect (η² ≈ 0.35). At-risk status
  has its own ~10-pt effect within the same choreography.
- **RQ4 — Prediction.** Success is predictable early (AUC ≈ 0.83 from 4 weeks),
  but choreography adds no value over volume here (the two are coupled by design).
- **RQ5 — Actionability.** Assignment activity, live sessions and session
  regularity signal success; grade/dashboard checking without work, and
  night-only study, signal risk.

---

## Reproducibility

All randomness is seeded (default 42). Re-running the pipeline from step 1 on the
same machine reproduces all results, tables and figures in `results/`.