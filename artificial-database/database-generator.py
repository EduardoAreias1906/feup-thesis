import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

NUM_STUDENTS = 5000
SCHOOL_START = datetime(2025, 8, 15)
SCHOOL_END   = datetime(2026, 6, 10)

# 32 Connections Academy states (one school per state)
STATES = [
    ('Alabama', 'CA-AL'), ('Arizona', 'CA-AZ'), ('Arkansas', 'CA-AR'),
    ('California', 'CA-CA'), ('Colorado', 'CA-CO'), ('Connecticut', 'CA-CT'),
    ('Florida', 'CA-FL'), ('Georgia', 'CA-GA'), ('Idaho', 'CA-ID'),
    ('Illinois', 'CA-IL'), ('Indiana', 'CA-IN'), ('Kansas', 'CA-KS'),
    ('Kentucky', 'CA-KY'), ('Louisiana', 'CA-LA'), ('Maryland', 'CA-MD'),
    ('Michigan', 'CA-MI'), ('Minnesota', 'CA-MN'), ('Missouri', 'CA-MO'),
    ('Nevada', 'CA-NV'), ('New Jersey', 'CA-NJ'), ('New Mexico', 'CA-NM'),
    ('New York', 'CA-NY'), ('North Carolina', 'CA-NC'), ('Ohio', 'CA-OH'),
    ('Oklahoma', 'CA-OK'), ('Oregon', 'CA-OR'), ('Pennsylvania', 'CA-PA'),
    ('South Carolina', 'CA-SC'), ('Tennessee', 'CA-TN'), ('Texas', 'CA-TX'),
    ('Virginia', 'CA-VA'), ('Washington', 'CA-WA'),
]
STATE_NAMES  = [s[0] for s in STATES]
SCHOOL_IDS   = [s[1] for s in STATES]

# Subjects per grade band
ELEMENTARY_SUBJECTS = ['Mathematics', 'English Language Arts', 'Science', 'Social Studies', 'Art', 'Physical Education']
MIDDLE_SUBJECTS     = ['Mathematics', 'English Language Arts', 'Science', 'Social Studies', 'Foreign Language', 'Physical Education']
HIGH_SUBJECTS       = ['Mathematics', 'English Language Arts', 'Science', 'Social Studies', 'Foreign Language', 'Elective']

# Semantic action types aligned with the Virtual Choreographies framework
ACTION_TYPES = [
    'Login', 'Logout',
    'Content_Study', 'Video_Watch',
    'Assessment_Start', 'Assessment_Submit',
    'Assignment_Start', 'Assignment_Submit',
    'Synchronous_Join', 'Synchronous_Leave',
    'Forum_View', 'Forum_Post',
    'Feedback_Review',
    'Dashboard_View', 'Grade_Check',
    'Resource_Download', 'Page_Navigation',
]

RESOURCE_MAP = {
    'Login':               'System',
    'Logout':              'System',
    'Content_Study':       'Asynchronous Content',
    'Video_Watch':         'Asynchronous Content',
    'Assessment_Start':    'Assessment Platform',
    'Assessment_Submit':   'Assessment Platform',
    'Assignment_Start':    'Assessment Platform',
    'Assignment_Submit':   'Assessment Platform',
    'Synchronous_Join':    'Live Classroom',
    'Synchronous_Leave':   'Live Classroom',
    'Forum_View':          'Discussion Board',
    'Forum_Post':          'Discussion Board',
    'Feedback_Review':     'Asynchronous Content',
    'Dashboard_View':      'Navigation',
    'Grade_Check':         'Navigation',
    'Resource_Download':   'Asynchronous Content',
    'Page_Navigation':     'Navigation',
}

# Actions that carry duration
DURATION_ACTIONS = {
    'Content_Study', 'Video_Watch', 'Assignment_Start',
    'Synchronous_Join', 'Resource_Download',
}

INTERACTION_TYPES = [
    'Message_Sent_To_Teacher', 'Message_Received_From_Teacher',
    'Feedback_Received', 'Forum_Post', 'Forum_Reply_Received',
    'Live_Question_Asked', 'Announcement_Viewed',
]

# ==========================================
# Helpers
# ==========================================
def get_school_days(start, end):
    """Return a list of school days (Mon–Fri) excluding major US holidays."""
    holidays = {
        datetime(2025, 9,  1),  # Labor Day
        datetime(2025, 11, 11), # Veterans Day
        datetime(2025, 11, 26), # Thanksgiving
        datetime(2025, 11, 27), # Day after Thanksgiving
        datetime(2025, 12, 24), # Christmas Eve
        datetime(2025, 12, 25), # Christmas
        datetime(2025, 12, 26), # Christmas break
        datetime(2025, 12, 29), # Christmas break
        datetime(2025, 12, 30), # Christmas break
        datetime(2025, 12, 31), # New Year's Eve
        datetime(2026,  1,  1), # New Year's Day
        datetime(2026,  1, 19), # MLK Day
        datetime(2026,  2, 16), # Presidents' Day
        datetime(2026,  3, 30), # Spring break
        datetime(2026,  3, 31), # Spring break
        datetime(2026,  4,  1), # Spring break
        datetime(2026,  4,  2), # Spring break
        datetime(2026,  4,  3), # Spring break
        datetime(2026,  5, 25), # Memorial Day
    }
    days, current = [], start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            days.append(current)
        current += timedelta(days=1)
    return days


def get_time_bin(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Evening'


def subjects_for_grade(grade_label):
    """Return semicolon-separated subjects appropriate for a grade level."""
    if grade_label == 'K' or grade_label in [f'Grade {i}' for i in range(1, 6)]:
        pool = ELEMENTARY_SUBJECTS
        n = random.randint(4, 6)
    elif grade_label in [f'Grade {i}' for i in range(6, 9)]:
        pool = MIDDLE_SUBJECTS
        n = random.randint(4, 6)
    else:
        pool = HIGH_SUBJECTS
        n = random.randint(4, 6)
    return ';'.join(random.sample(pool, min(n, len(pool))))


SCHOOL_DAYS = get_school_days(SCHOOL_START, SCHOOL_END)


# ==========================================
# 1. Student Demographics & Context
# ==========================================
def generate_demographics(num_students):
    print("Generating demographics...")
    student_ids = [f"STU_{str(i).zfill(4)}" for i in range(num_students)]
    grades      = np.random.choice([f"Grade {i}" for i in range(1, 13)] + ['K'], num_students)

    state_idx   = np.random.randint(0, len(STATES), num_students)
    states      = [STATE_NAMES[i]  for i in state_idx]
    schools     = [SCHOOL_IDS[i]   for i in state_idx]

    at_risk     = np.random.choice([True, False], num_students, p=[0.30, 0.70])

    reasons = ['Medical', 'Prior Academic Struggles', 'Social/Bullying']
    enrollment_reasons = [
        np.random.choice(reasons) if risk else 'None' for risk in at_risk
    ]

    # IEP more likely for at-risk students (~35%) than general (~5%)
    iep_status = [
        bool(np.random.choice([True, False], p=[0.35, 0.65])) if risk
        else bool(np.random.choice([True, False], p=[0.05, 0.95]))
        for risk in at_risk
    ]

    start_dates = [SCHOOL_START + timedelta(days=random.randint(0, 30))
                   for _ in range(num_students)]

    # ~10 % of students withdraw before end of year
    withdrawn = np.random.choice([True, False], num_students, p=[0.10, 0.90])
    end_dates = [
        s + timedelta(days=random.randint(30, 150)) if w else SCHOOL_END
        for s, w in zip(start_dates, withdrawn)
    ]

    subjects_list = [subjects_for_grade(g) for g in grades]

    df = pd.DataFrame({
        'Student_ID':        student_ids,
        'Grade_Level':       grades,
        'School_ID':         schools,
        'State':             states,
        'Enrollment_Start':  start_dates,
        'Enrollment_End':    end_dates,
        'Withdrawn':         withdrawn,
        'At_Risk_Status':    at_risk,
        'IEP_Status':        iep_status,
        'Enrollment_Reason': enrollment_reasons,
        'Subjects':          subjects_list,
    })
    print(f"  data/demographics.csv ({len(df):,} rows)")
    return df


# ==========================================
# 2. Activity Logs (Time-Series Data)
# ==========================================
def generate_activity_logs(students_df):
    print("Generating activity logs...")
    all_logs = []

    for idx, (_, student) in enumerate(students_df.iterrows()):
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1:,}/{len(students_df):,} students processed…")

        is_at_risk   = student['At_Risk_Status']
        enroll_start = student['Enrollment_Start']
        enroll_end   = student['Enrollment_End']
        subjects     = student['Subjects'].split(';')

        # Behavioural profile
        if is_at_risk:
            chronotype  = np.random.choice(['morning', 'afternoon', 'evening'], p=[0.20, 0.30, 0.50])
            att_rate    = np.random.uniform(0.50, 0.82)
            events_mean = np.random.randint(3, 10)
        else:
            chronotype  = np.random.choice(['morning', 'afternoon', 'evening'], p=[0.40, 0.45, 0.15])
            att_rate    = np.random.uniform(0.78, 0.97)
            events_mean = np.random.randint(7, 18)

        active_days = [
            d for d in SCHOOL_DAYS
            if enroll_start.date() <= d.date() <= enroll_end.date() and random.random() < att_rate
        ]

        for day in active_days:
            if chronotype == 'morning':
                start_hour = random.randint(7, 10)
            elif chronotype == 'afternoon':
                start_hour = random.randint(11, 14)
            else:
                start_hour = random.randint(16, 20)

            current_time = day.replace(hour=start_hour, minute=random.randint(0, 59))
            session_id   = f"SESS_{random.randint(10000, 99999)}"
            subject      = random.choice(subjects)
            num_events   = max(1, events_mean + random.randint(-3, 3))

            for i in range(num_events):
                # Occasionally start a new session or switch subject
                if i > 0 and random.random() > 0.85:
                    session_id    = f"SESS_{random.randint(10000, 99999)}"
                    current_time += timedelta(minutes=random.randint(30, 120))
                    if random.random() > 0.60:
                        subject = random.choice(subjects)

                current_time += timedelta(minutes=random.randint(2, 40))

                if current_time.hour >= 23 or current_time.date() > enroll_end.date():
                    break

                action   = random.choice(ACTION_TYPES)
                resource = RESOURCE_MAP[action]
                duration = random.randint(30, 1800) if action in DURATION_ACTIONS else 0

                all_logs.append({
                    'Student_ID':        student['Student_ID'],
                    'Timestamp':         current_time,
                    'Session_ID':        session_id,
                    'Action_Type':       action,
                    'Subject':           subject,
                    'Resource_Category': resource,
                    'Time_Of_Day':       get_time_bin(current_time.hour),
                    'Duration_Seconds':  duration,
                })

    df = pd.DataFrame(all_logs)
    print(f"  data/activity_logs.csv ({len(df):,} rows)")
    return df


# ==========================================
# 3. Interaction Metadata
# ==========================================
def generate_interaction_metadata(students_df):
    print("Generating interaction metadata...")
    interactions = []

    for _, student in students_df.iterrows():
        is_at_risk   = student['At_Risk_Status']
        enroll_start = student['Enrollment_Start']
        enroll_end   = student['Enrollment_End']
        days_enrolled = max(1, (enroll_end - enroll_start).days)

        # At-risk students receive more teacher outreach
        num_interactions = random.randint(15, 40) if is_at_risk else random.randint(5, 20)

        for _ in range(num_interactions):
            int_time         = enroll_start + timedelta(
                days=random.randint(1, days_enrolled - 1),
                hours=random.randint(8, 20),
                minutes=random.randint(0, 59),
            )
            interaction_type = random.choice(INTERACTION_TYPES)

            # Derive sender role from interaction type
            if interaction_type in ['Message_Received_From_Teacher', 'Feedback_Received', 'Announcement_Viewed']:
                sender_role = 'Teacher'
            elif interaction_type in ['Message_Sent_To_Teacher', 'Live_Question_Asked']:
                sender_role = 'Student'
            elif interaction_type == 'Forum_Post':
                sender_role = np.random.choice(['Student', 'Teacher'], p=[0.80, 0.20])
            else:
                sender_role = 'Student'

            interactions.append({
                'Student_ID':       student['Student_ID'],
                'Interaction_Type': interaction_type,
                'Timestamp':        int_time,
                'Sender_Role':      sender_role,
            })

    df = pd.DataFrame(interactions)
    print(f"  data/interaction_metadata.csv ({len(df):,} rows)")
    return df


# ==========================================
# 4. Academic Outcomes (per student per subject)
# ==========================================
def generate_academic_outcomes(students_df):
    print("Generating academic outcomes...")
    outcomes = []

    for _, student in students_df.iterrows():
        is_at_risk = student['At_Risk_Status']
        has_iep    = student['IEP_Status']
        subjects   = student['Subjects'].split(';')

        # Score penalty based on risk factors
        penalty = 0
        if is_at_risk:
            penalty += random.randint(5, 15)
        if has_iep:
            penalty += random.randint(3, 10)

        for subject in subjects:
            # Some subjects harder than others for at-risk students
            subject_penalty = random.randint(0, 5) if subject in ['Mathematics', 'Science'] else 0
            base = max(30, min(100, random.randint(55, 95) - penalty - subject_penalty))

            final_grade  = max(0, min(100, base + random.randint(-5,  5)))
            assign_avg   = max(0, min(100, base + random.randint(-8,  8)))
            quiz_avg     = max(0, min(100, base + random.randint(-10, 10)))
            std_test     = max(0, min(100, base + random.randint(-12, 12)))
            completion   = random.randint(75, 100) if base >= 60 else random.randint(20, 79)
            passed       = final_grade >= 60 and completion >= 70

            outcomes.append({
                'Student_ID':                   student['Student_ID'],
                'Subject':                      subject,
                'Final_Grade':                  final_grade,
                'Assignment_Average':           assign_avg,
                'Quiz_Average':                 quiz_avg,
                'Standardized_Test_Score':      std_test,
                'Course_Completion_Percentage': completion,
                'Pass_Fail':                    'Pass' if passed else 'Fail',
            })

    df = pd.DataFrame(outcomes)
    print(f"  data/academic_outcomes.csv ({len(df):,} rows)")
    return df


# ==========================================
# Execute
# ==========================================
os.makedirs('data', exist_ok=True)

df_demographics = generate_demographics(NUM_STUDENTS)
df_demographics.to_csv('data/demographics.csv', index=False)

df_activity = generate_activity_logs(df_demographics)
df_activity.to_csv('data/activity_logs.csv', index=False)

df_interactions = generate_interaction_metadata(df_demographics)
df_interactions.to_csv('data/interaction_metadata.csv', index=False)

df_outcomes = generate_academic_outcomes(df_demographics)
df_outcomes.to_csv('data/academic_outcomes.csv', index=False)

print("\nDatabase generated successfully.")
