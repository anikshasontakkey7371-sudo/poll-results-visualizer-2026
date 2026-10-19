"""
Synthetic Poll Data Generator
=============================
Generates realistic, reproducible survey data for the Poll Results Visualizer.
Supports injecting controlled data anomalies (missing values, inconsistent casing,
whitespaces, and duplicates) to test and showcase the data cleaning pipeline.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


# Survey Domain Specifications
DEFAULT_OPTIONS = [
    "Remote-First",
    "Hybrid (2-3 Days Office)",
    "Office-Centric",
    "Flexible Asynchronous",
]

OPTION_PROBABILITIES = [0.42, 0.36, 0.12, 0.10]

REGIONS = [
    "North America",
    "Europe",
    "Asia-Pacific",
    "Latin America",
    "Middle East & Africa",
]

REGION_PROBABILITIES = [0.35, 0.30, 0.20, 0.08, 0.07]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_PROBABILITIES = [0.15, 0.45, 0.25, 0.10, 0.05]

INDUSTRIES = [
    "Technology & Software",
    "Finance & Banking",
    "Healthcare & Life Sciences",
    "Education & Research",
    "Creative & Media",
    "Other",
]

AI_ADOPTION_LEVELS = [
    "Daily Power User",
    "Occasional User",
    "Exploring / Learning",
    "Not Using AI",
]

# Realistic feedback templates mapped by satisfaction rating level
FEEDBACK_BY_RATING = {
    5: [
        "Remote work has drastically boosted my productivity, mental health, and family time.",
        "The autonomy and flexibility we have now is the best benefit a company can provide.",
        "AI code assistance has removed tedious boilerplate and accelerated our sprint velocity.",
        "Hybrid collaboration on Tuesdays and Thursdays keeps team camaraderie high.",
        "Flexible asynchronous hours allow me to work during my personal peak focus hours.",
        "Exceptional work-life balance and zero time wasted on miserable daily rush-hour commutes.",
        "Modern cloud tools and asynchronous documentation make remote delivery seamless.",
    ],
    4: [
        "Overall very positive, though cross-time-zone communication can occasionally lag.",
        "Hybrid setup works well for our sprint planning, though meeting rooms are often crowded.",
        "Good flexibility, but we need clearer guidelines around after-hours Slack notifications.",
        "AI tools are helpful for drafting documentation and finding syntax errors.",
        "I enjoy working from home, but I do miss occasional spontaneous whiteboarding sessions.",
        "Satisfied with current policy, but our home office equipment stipend should be updated.",
    ],
    3: [
        "Neutral experience. Communication can be fragmented when half the team is in-office.",
        "Hybrid policy feels somewhat arbitrary with mandatory minimum badge-swipe days.",
        "Productivity is decent, but video meeting fatigue is real after 4 consecutive hours.",
        "Some managers still expect immediate replies regardless of declared focus hours.",
        "The tools are fine, but company culture hasn't fully adapted to distributed teams.",
    ],
    2: [
        "Mandatory 4-day office mandates feel like a massive step backwards for trust.",
        "Open-plan office is extremely noisy and counterproductive for deep technical work.",
        "Struggling with work-life boundary blur; working longer hours than ever before.",
        "Leadership says they value flexibility, but promotion bias clearly favors in-office peers.",
        "Commuting two hours round-trip just to sit on Zoom calls in a cubicle makes no sense.",
    ],
    1: [
        "Extremely dissatisfied with sudden rigid return-to-office mandates without valid reasoning.",
        "Severe burnout due to micromanagement and constant surveillance tools.",
        "Zero consideration for working parents or long-distance commuters.",
        "Disorganized communication, conflicting expectations, and non-existent team cohesion.",
    ],
}


def generate_poll_data(
    n_samples: int = 1500,
    random_seed: int = 42,
    inject_noise: bool = True,
    noise_ratio: float = 0.08,
) -> pd.DataFrame:
    """
    Generate synthetic poll responses with realistic correlations and distributions.

    Parameters
    ----------
    n_samples : int
        Number of initial respondents to simulate (default 1500).
    random_seed : int
        Seed for reproducibility across random and numpy modules.
    inject_noise : bool
        If True, injects dirty data (casing, whitespace, duplicates, nulls, out-of-bounds ratings).
    noise_ratio : float
        Fraction of records to corrupt with anomalies if inject_noise is True.

    Returns
    -------
    pd.DataFrame
        Synthesized poll dataset containing respondent demographic, voting, rating, and feedback fields.
    """
    # Set seeds for deterministic generation
    random.seed(random_seed)
    np.random.seed(random_seed)

    records = []
    base_start_date = datetime(2026, 9, 1, 8, 0, 0)
    survey_duration_days = 30

    for i in range(1, n_samples + 1):
        resp_id = f"RESP-{i:05d}"

        # Assign demographic variables
        region = np.random.choice(REGIONS, p=REGION_PROBABILITIES)
        age_group = np.random.choice(AGE_GROUPS, p=AGE_PROBABILITIES)
        industry = random.choice(INDUSTRIES)
        ai_adoption = random.choice(AI_ADOPTION_LEVELS)

        # Realistic conditional probability for preferred work model based on age group
        # Younger cohorts slightly prefer Remote/Flex, older cohorts slightly more Hybrid/Office
        if age_group in ["18-24", "25-34"]:
            opt_probs = [0.48, 0.35, 0.07, 0.10]
        elif age_group == "35-44":
            opt_probs = [0.42, 0.38, 0.11, 0.09]
        else:
            opt_probs = [0.30, 0.40, 0.20, 0.10]

        preferred_option = np.random.choice(DEFAULT_OPTIONS, p=opt_probs)

        # Satisfaction rating is correlated with preferred option
        # Remote & Hybrid tend to have higher satisfaction than forced office
        if preferred_option == "Remote-First":
            sat_probs = [0.03, 0.05, 0.12, 0.35, 0.45]
        elif preferred_option == "Hybrid (2-3 Days Office)":
            sat_probs = [0.05, 0.08, 0.22, 0.40, 0.25]
        elif preferred_option == "Flexible Asynchronous":
            sat_probs = [0.04, 0.06, 0.15, 0.40, 0.35]
        else:  # Office-Centric
            sat_probs = [0.22, 0.28, 0.25, 0.15, 0.10]

        satisfaction = int(np.random.choice([1, 2, 3, 4, 5], p=sat_probs))

        # Sample a feedback comment matching their satisfaction rating
        feedback_pool = FEEDBACK_BY_RATING.get(satisfaction, ["No comments provided."])
        feedback = random.choice(feedback_pool)

        # Timestamp distributed realistically across 30 days with daytime clustering
        day_offset = random.uniform(0, survey_duration_days)
        # Hour weight: more responses during 09:00 - 18:00
        hour_offset = int(np.clip(np.random.normal(13, 3.5), 8, 22))
        minute_offset = random.randint(0, 59)
        second_offset = random.randint(0, 59)

        timestamp = base_start_date + timedelta(
            days=int(day_offset),
            hours=hour_offset,
            minutes=minute_offset,
            seconds=second_offset,
        )

        records.append({
            "respondent_id": resp_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "preferred_option": preferred_option,
            "satisfaction_rating": satisfaction,
            "region": region,
            "age_group": age_group,
            "industry": industry,
            "ai_adoption_level": ai_adoption,
            "feedback": feedback,
            "is_synthetic": True,
        })

    df = pd.DataFrame(records)

    # Controlled noise injection for pedagogical and testing realism
    if inject_noise:
        df = _inject_data_anomalies(df, noise_ratio, random_seed)

    return df


def _inject_data_anomalies(df: pd.DataFrame, noise_ratio: float, seed: int) -> pd.DataFrame:
    """
    Inject realistic messy anomalies into the dataframe:
    - Inconsistent casing in categorical columns
    - Padding whitespaces
    - Missing values (NaNs)
    - Duplicate response rows
    - Out-of-bounds ratings
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)
    noise_count = int(n * noise_ratio)

    # 1. Inconsistent casing and whitespace in preferred_option
    noisy_indices_option = rng.choice(n, size=noise_count, replace=False)
    for idx in noisy_indices_option:
        val = str(df.at[idx, "preferred_option"])
        transform_type = rng.choice(["lower", "upper", "space_pad", "mixed"])
        if transform_type == "lower":
            df.at[idx, "preferred_option"] = val.lower()
        elif transform_type == "upper":
            df.at[idx, "preferred_option"] = val.upper()
        elif transform_type == "space_pad":
            df.at[idx, "preferred_option"] = f"  {val}  "
        else:
            df.at[idx, "preferred_option"] = val.title() + " "

    # 2. Inconsistent casing and whitespace in region
    noisy_indices_region = rng.choice(n, size=noise_count, replace=False)
    for idx in noisy_indices_region:
        val = str(df.at[idx, "region"])
        if rng.random() > 0.5:
            df.at[idx, "region"] = f" {val.lower()} "
        else:
            df.at[idx, "region"] = val.upper()

    # 3. Missing values in satisfaction_rating, feedback, and age_group
    null_sat_indices = rng.choice(n, size=int(noise_count * 0.7), replace=False)
    df.loc[null_sat_indices, "satisfaction_rating"] = np.nan

    null_feedback_indices = rng.choice(n, size=int(noise_count * 1.2), replace=False)
    df.loc[null_feedback_indices, "feedback"] = np.nan

    null_age_indices = rng.choice(n, size=int(noise_count * 0.5), replace=False)
    df.loc[null_age_indices, "age_group"] = np.nan

    # 4. Out-of-bounds ratings (e.g. 0, 7)
    oob_indices = rng.choice(n, size=max(3, int(noise_count * 0.15)), replace=False)
    for idx in oob_indices:
        df.at[idx, "satisfaction_rating"] = rng.choice([0, 6, 7, -1])

    # 5. Duplicate submissions (e.g., impatient respondents clicking submit twice)
    dup_indices = rng.choice(n, size=max(5, int(noise_count * 0.35)), replace=False)
    duplicates = df.iloc[dup_indices].copy()
    # Slightly modify timestamp or keep identical
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle slightly so duplicates aren't all at the very end
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """Save the dataset to a CSV file, ensuring parent directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[Dataset Generator] Successfully saved {len(df)} records to: {filepath}")


if __name__ == "__main__":
    output_path = os.path.join("data", "raw_poll_responses.csv")
    print("Generating synthetic poll responses with intentional anomalies for cleaning...")
    raw_df = generate_poll_data(n_samples=1500, random_seed=42, inject_noise=True)
    save_dataset(raw_df, output_path)
    print("Sample generated rows:")
    print(raw_df.head())
    print("\nMissing values count:")
    print(raw_df.isnull().sum())
