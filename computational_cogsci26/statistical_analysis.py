#!/usr/bin/env python3
"""
Statistical Analysis for CogSci 2026 Paper:
"How the Teaching Style and Interpretation Type of State Interventions Shape Multi-Agent Coordination"

This script conducts statistical analyses to support the qualitative claims in the paper.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Try to import statsmodels for ANOVA
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Using scipy for basic tests.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Mappings
TEACHING_STYLES = {
    1: "Undoing",
    2: "Correcting",
    3: "Exploration-encouraging Correcting",
    4: "Restart",
}
INTERPRETATION_TYPES = {
    0: "Suggestion",
    1: "Reset",
    2: "Interrupt",
    3: "Transition",
    4: "Disrupt",
    5: "Impede",
}
INTERVENTION_RATES = [0, 0.25, 0.5, 0.75, 1.0]

# Data directories
TWO_AGENT_DIR = "world3_size_12"
THREE_AGENT_DIR = "world3_size_12_3agent"

# Number of runs per condition
N_RUNS = 50

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def print_progress(current, total, prefix="", suffix="", bar_length=40):
    """Print a progress bar to stderr."""
    import sys

    percent = current / total if total > 0 else 0
    filled_length = int(bar_length * percent)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stderr.write(f"\r{prefix} |{bar}| {current}/{total} {suffix}")
    sys.stderr.flush()
    if current == total:
        sys.stderr.write("\n")


def load_run_data(filepath):
    """Load a single run CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception:
        return None


def compute_policy_divergence(df):
    """
    Compute Policy Divergence for each agent.
    Policy Divergence = center1_q - center2_q (region preference)
    We then compute the absolute difference between agents' preferences.
    """
    # Get final timestep data for each agent
    final_step = df.groupby("agentid").last().reset_index()

    if len(final_step) >= 2:
        agent0_pref = (
            final_step[final_step["agentid"] == 0]["center1_q"].values[0]
            - final_step[final_step["agentid"] == 0]["center2_q"].values[0]
        )
        agent1_pref = (
            final_step[final_step["agentid"] == 1]["center1_q"].values[0]
            - final_step[final_step["agentid"] == 1]["center2_q"].values[0]
        )
        # Policy divergence = difference in preferences (larger = more coordinated/specialized)
        return abs(agent0_pref - agent1_pref)
    return np.nan


def compute_system_equilibrium(df):
    """
    Compute System Equilibrium.
    System Equilibrium = |CumulativeReward_agent0 - CumulativeReward_agent1|
    Smaller values = more balanced = better equilibrium.
    """
    final_step = df.groupby("agentid").last().reset_index()

    if len(final_step) >= 2:
        reward0 = final_step[final_step["agentid"] == 0]["CumulativeReward"].values[0]
        reward1 = final_step[final_step["agentid"] == 1]["CumulativeReward"].values[0]
        return abs(reward0 - reward1)
    return np.nan


def load_visit_stats(filepath):
    """Load visit statistics file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception:
        return None


def compute_region_attachment_3agent(
    visit_df: pd.DataFrame | None,
) -> tuple[int, int, int] | tuple[None, None, None]:
    """
    For 3-agent case: compute visits to each agent's 'home' region.
    Based on the paper, regions are:
    - Region 1: around (2,2) - center1
    - Region 2: around (9,9) - center2
    - Region 3: Need to identify from data

    Returns visit counts to closest region for each agent as tuple of ints,
    or (None, None, None) if data is invalid.
    """
    if visit_df is None or "agent_2" not in visit_df.columns:
        return (None, None, None)

    # Define region centers (based on the 2-agent data showing centers at ~(2,2) and ~(9,9))
    # For 3 regions, we'll identify them from the grid coordinates
    region1_coords = [(x, y) for x in range(0, 5) for y in range(0, 5)]
    region2_coords = [(x, y) for x in range(7, 12) for y in range(7, 12)]
    # Region 3 might be in another corner - let's check
    region3_coords = [
        (x, y) for x in range(7, 12) for y in range(0, 5)
    ]  # bottom-right?

    def get_region_visits(df, region_coords, agent_col):
        total = 0
        for x, y in region_coords:
            row = df[(df["grid_x"] == x) & (df["grid_y"] == y)]
            if len(row) > 0:
                total += row[agent_col].values[0]
        return total

    # For simplicity, compute total visits per agent to their "home" region
    # We'll use the maximum visited region as each agent's home
    agent0_visits = {
        "r1": get_region_visits(visit_df, region1_coords, "agent_0"),
        "r2": get_region_visits(visit_df, region2_coords, "agent_0"),
        "r3": get_region_visits(visit_df, region3_coords, "agent_0"),
    }
    agent1_visits = {
        "r1": get_region_visits(visit_df, region1_coords, "agent_1"),
        "r2": get_region_visits(visit_df, region2_coords, "agent_1"),
        "r3": get_region_visits(visit_df, region3_coords, "agent_1"),
    }
    agent2_visits = {
        "r1": get_region_visits(visit_df, region1_coords, "agent_2"),
        "r2": get_region_visits(visit_df, region2_coords, "agent_2"),
        "r3": get_region_visits(visit_df, region3_coords, "agent_2"),
    }

    # Return max visits (to their primary region) for each agent
    return (
        max(agent0_visits.values()),
        max(agent1_visits.values()),
        max(agent2_visits.values()),
    )


def load_all_two_agent_data():
    """Load all two-agent experiment data."""
    results = []

    # Count total conditions for progress
    total_conditions = 4 + (6 * 4 * 4)  # baseline + intervention conditions
    current = 0

    # Baseline conditions (rate=0)
    for mode in range(1, 5):
        current += 1
        print_progress(current, total_conditions, "Loading 2-agent data")
        dir_path = os.path.join(TWO_AGENT_DIR, f"rate_0_mode_{mode}")
        if os.path.exists(dir_path):
            for run in range(N_RUNS):
                run_file = os.path.join(dir_path, f"run_{run}.csv")
                df = load_run_data(run_file)
                if df is not None:
                    results.append(
                        {
                            "teaching_style": TEACHING_STYLES[mode],
                            "teaching_style_code": mode,
                            "interpretation_type": "Baseline",
                            "interpretation_type_code": -1,
                            "intervention_rate": 0.0,
                            "run": run,
                            "policy_divergence": compute_policy_divergence(df),
                            "system_equilibrium": compute_system_equilibrium(df),
                        }
                    )

    # Intervention conditions
    for interp_type in range(6):
        for rate in [0.25, 0.5, 0.75, 1.0]:
            for mode in range(1, 5):
                current += 1
                print_progress(current, total_conditions, "Loading 2-agent data")
                dir_path = os.path.join(
                    TWO_AGENT_DIR, f"type_{interp_type}_rate_{rate}_mode_{mode}"
                )
                if os.path.exists(dir_path):
                    for run in range(N_RUNS):
                        run_file = os.path.join(dir_path, f"run_{run}.csv")
                        df = load_run_data(run_file)
                        if df is not None:
                            results.append(
                                {
                                    "teaching_style": TEACHING_STYLES[mode],
                                    "teaching_style_code": mode,
                                    "interpretation_type": INTERPRETATION_TYPES[
                                        interp_type
                                    ],
                                    "interpretation_type_code": interp_type,
                                    "intervention_rate": rate,
                                    "run": run,
                                    "policy_divergence": compute_policy_divergence(df),
                                    "system_equilibrium": compute_system_equilibrium(
                                        df
                                    ),
                                }
                            )

    return pd.DataFrame(results)


def load_all_three_agent_data():
    """Load all three-agent experiment data."""
    results = []

    # Count total conditions for progress
    total_conditions = 4 + (6 * 4 * 4)  # baseline + intervention conditions
    current = 0

    # Baseline conditions (rate=0)
    for mode in range(1, 5):
        current += 1
        print_progress(current, total_conditions, "Loading 3-agent data")
        dir_path = os.path.join(THREE_AGENT_DIR, f"rate_0_mode_{mode}")
        if os.path.exists(dir_path):
            for run in range(N_RUNS):
                visit_file = os.path.join(dir_path, f"visit_stats_{run}.csv")
                visit_df = load_visit_stats(visit_file)
                if visit_df is not None:
                    result = compute_region_attachment_3agent(visit_df)
                    if result[0] is not None:
                        a0, a1, a2 = result[0], result[1], result[2]
                        # Average region attachment across agents
                        avg_attachment = (a0 + a1 + a2) / 3
                        results.append(
                            {
                                "teaching_style": TEACHING_STYLES[mode],
                                "teaching_style_code": mode,
                                "interpretation_type": "Baseline",
                                "interpretation_type_code": -1,
                                "intervention_rate": 0.0,
                                "run": run,
                                "agent0_attachment": a0,
                                "agent1_attachment": a1,
                                "agent2_attachment": a2,
                                "avg_attachment": avg_attachment,
                            }
                        )

    # Intervention conditions
    for interp_type in range(6):
        for rate in [0.25, 0.5, 0.75, 1.0]:
            for mode in range(1, 5):
                current += 1
                print_progress(current, total_conditions, "Loading 3-agent data")
                dir_path = os.path.join(
                    THREE_AGENT_DIR, f"type_{interp_type}_rate_{rate}_mode_{mode}"
                )
                if os.path.exists(dir_path):
                    for run in range(N_RUNS):
                        visit_file = os.path.join(dir_path, f"visit_stats_{run}.csv")
                        visit_df = load_visit_stats(visit_file)
                        if visit_df is not None:
                            result = compute_region_attachment_3agent(visit_df)
                            if result[0] is not None:
                                a0, a1, a2 = result[0], result[1], result[2]
                                avg_attachment = (a0 + a1 + a2) / 3
                                results.append(
                                    {
                                        "teaching_style": TEACHING_STYLES[mode],
                                        "teaching_style_code": mode,
                                        "interpretation_type": INTERPRETATION_TYPES[
                                            interp_type
                                        ],
                                        "interpretation_type_code": interp_type,
                                        "intervention_rate": rate,
                                        "run": run,
                                        "agent0_attachment": a0,
                                        "agent1_attachment": a1,
                                        "agent2_attachment": a2,
                                        "avg_attachment": avg_attachment,
                                    }
                                )

    return pd.DataFrame(results)


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================


def run_factorial_anova(df, dv, factors):
    """Run factorial ANOVA using statsmodels."""
    if not HAS_STATSMODELS:
        print("statsmodels required for factorial ANOVA")
        return None

    # Build formula
    formula = f"{dv} ~ " + " * ".join([f"C({f})" for f in factors])

    try:
        model = ols(formula, data=df).fit()  # type: ignore[possibly-undefined]
        anova_table = sm.stats.anova_lm(model, typ=2)  # type: ignore[possibly-undefined]
        return anova_table
    except Exception as e:
        print(f"ANOVA error: {e}")
        return None


def run_posthoc_tukey(df, dv, factor):
    """Run Tukey HSD post-hoc test."""
    if not HAS_STATSMODELS:
        return None

    try:
        tukey = pairwise_tukeyhsd(df[dv].dropna(), df[factor].dropna())  # type: ignore[possibly-undefined]
        return tukey
    except Exception as e:
        print(f"Tukey error: {e}")
        return None


def welch_ttest(group1, group2) -> tuple[float, float]:
    """Run Welch's t-test (unequal variances)."""
    group1 = group1.dropna()
    group2 = group2.dropna()
    if len(group1) < 2 or len(group2) < 2:
        return float("nan"), float("nan")
    result = stats.ttest_ind(group1, group2, equal_var=False)
    return float(result[0]), float(result[1])  # type: ignore[arg-type]


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    group1 = group1.dropna()
    group2 = group2.dropna()
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def linear_trend_test(df, dv, rate_col="intervention_rate") -> tuple[float, float]:
    """Test for linear trend with intervention rate."""
    df_clean = df[[rate_col, dv]].dropna()
    if len(df_clean) < 3:
        return float("nan"), float("nan")

    # pyright has incomplete scipy type stubs
    result = stats.pearsonr(df_clean[rate_col], df_clean[dv])
    correlation = float(result[0])  # type: ignore[arg-type]
    p_value = float(result[1])  # type: ignore[arg-type]
    return correlation, p_value


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def analyze_two_agent_claims(df):
    """Analyze claims for two-agent experiments."""
    print("\n" + "=" * 80)
    print("TWO-AGENT EXPERIMENT STATISTICAL ANALYSES")
    print("=" * 80)

    # Filter to intervention conditions only (exclude baseline for main analyses)
    df_intervention = df[df["interpretation_type"] != "Baseline"].copy()
    df_intervention = df_intervention.dropna(
        subset=["policy_divergence", "system_equilibrium"]
    )

    print(f"\nTotal observations (intervention conditions): {len(df_intervention)}")
    print(
        f"Unique conditions: {df_intervention[['teaching_style', 'interpretation_type', 'intervention_rate']].drop_duplicates().shape[0]}"
    )

    # -------------------------------------------------------------------------
    # 1. FACTORIAL ANOVA for Policy Divergence
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(
        "1. FACTORIAL ANOVA: Policy Divergence ~ Teaching Style × Interpretation Type × Rate"
    )
    print("-" * 80)

    if HAS_STATSMODELS:
        anova_pd = run_factorial_anova(
            df_intervention,
            "policy_divergence",
            ["teaching_style", "interpretation_type", "intervention_rate"],
        )
        if anova_pd is not None:
            print(anova_pd.to_string())
    else:
        # Fallback: Kruskal-Wallis for main effects
        print("\nKruskal-Wallis tests (non-parametric alternative):")
        for factor in ["teaching_style", "intervention_rate"]:
            groups = [
                group["policy_divergence"].dropna().values
                for name, group in df_intervention.groupby(factor)
            ]
            if all(len(g) > 0 for g in groups):
                h_stat, p_val = stats.kruskal(*groups)
                print(f"  {factor}: H = {h_stat:.3f}, p = {p_val:.4f}")

    # -------------------------------------------------------------------------
    # 2. FACTORIAL ANOVA for System Equilibrium
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(
        "2. FACTORIAL ANOVA: System Equilibrium ~ Teaching Style × Interpretation Type × Rate"
    )
    print("-" * 80)

    if HAS_STATSMODELS:
        anova_se = run_factorial_anova(
            df_intervention,
            "system_equilibrium",
            ["teaching_style", "interpretation_type", "intervention_rate"],
        )
        if anova_se is not None:
            print(anova_se.to_string())

    # -------------------------------------------------------------------------
    # 3. CLAIM: Undoing is most robust for policy divergence
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("3. CLAIM: 'Undoing' style is most robust for policy divergence")
    print("-" * 80)

    # Compare Undoing vs each other style
    undoing_pd = df_intervention[df_intervention["teaching_style"] == "Undoing"][
        "policy_divergence"
    ]

    print("\nPairwise comparisons (Undoing vs others) for Policy Divergence:")
    print(
        f"{'Comparison':<50} {'t-stat':>10} {'p-value':>12} {"Cohen's d":>12} {'Result':>15}"
    )
    print("-" * 99)

    for style in ["Correcting", "Exploration-encouraging Correcting", "Restart"]:
        other_pd = df_intervention[df_intervention["teaching_style"] == style][
            "policy_divergence"
        ]
        t_stat, p_val = welch_ttest(undoing_pd, other_pd)
        d = cohens_d(undoing_pd, other_pd)
        sig = "*" if p_val < 0.05 else "" if not np.isnan(p_val) else "N/A"
        sig += "*" if p_val < 0.01 else ""
        sig += "*" if p_val < 0.001 else ""
        print(
            f"Undoing vs {style:<36} {t_stat:>10.3f} {p_val:>12.4f} {d:>12.3f} {sig:>15}"
        )

    print("\nMean Policy Divergence by Teaching Style:")
    means = df_intervention.groupby("teaching_style")["policy_divergence"].agg(
        ["mean", "std", "count"]
    )
    print(means.to_string())

    # -------------------------------------------------------------------------
    # 4. CLAIM: Correcting requires high intervention rates (≥0.75)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("4. CLAIM: 'Correcting' requires high intervention rates (≥0.75)")
    print("-" * 80)

    correcting_df = df_intervention[df_intervention["teaching_style"] == "Correcting"]

    # Compare low rates (<0.75) vs high rates (≥0.75)
    low_rate = correcting_df[correcting_df["intervention_rate"] < 0.75][
        "policy_divergence"
    ]
    high_rate = correcting_df[correcting_df["intervention_rate"] >= 0.75][
        "policy_divergence"
    ]

    t_stat, p_val = welch_ttest(high_rate, low_rate)
    d = cohens_d(high_rate, low_rate)

    print("\nComparing Correcting at high (≥0.75) vs low (<0.75) intervention rates:")
    print(
        f"  Policy Divergence: t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}"
    )
    print(
        f"  High rate mean: {high_rate.mean():.3f} (SD={high_rate.std():.3f}, n={len(high_rate)})"
    )
    print(
        f"  Low rate mean: {low_rate.mean():.3f} (SD={low_rate.std():.3f}, n={len(low_rate)})"
    )

    # Same for System Equilibrium
    low_rate_se = correcting_df[correcting_df["intervention_rate"] < 0.75][
        "system_equilibrium"
    ]
    high_rate_se = correcting_df[correcting_df["intervention_rate"] >= 0.75][
        "system_equilibrium"
    ]

    t_stat_se, p_val_se = welch_ttest(
        low_rate_se, high_rate_se
    )  # Note: reversed for equilibrium (smaller is better)
    d_se = cohens_d(low_rate_se, high_rate_se)

    print(
        f"\n  System Equilibrium: t = {t_stat_se:.3f}, p = {p_val_se:.4f}, Cohen's d = {d_se:.3f}"
    )
    print(f"  High rate mean: {high_rate_se.mean():.3f} (SD={high_rate_se.std():.3f})")
    print(f"  Low rate mean: {low_rate_se.mean():.3f} (SD={low_rate_se.std():.3f})")

    # -------------------------------------------------------------------------
    # 5. CLAIM: Restart discourages system equilibrium
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5. CLAIM: 'Restart' discourages system equilibrium")
    print("-" * 80)

    restart_se = df_intervention[df_intervention["teaching_style"] == "Restart"][
        "system_equilibrium"
    ]
    other_se = df_intervention[df_intervention["teaching_style"] != "Restart"][
        "system_equilibrium"
    ]

    t_stat, p_val = welch_ttest(restart_se, other_se)
    d = cohens_d(restart_se, other_se)

    print("\nComparing Restart vs all other styles for System Equilibrium:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
    print(f"  Restart mean: {restart_se.mean():.3f} (SD={restart_se.std():.3f})")
    print(f"  Others mean: {other_se.mean():.3f} (SD={other_se.std():.3f})")
    print("  (Higher System Equilibrium = larger score gap = worse balance)")

    # -------------------------------------------------------------------------
    # 6. CLAIM: Reset interpretation performs poorly
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("6. CLAIM: 'Reset' interpretation performs poorly across teaching styles")
    print("-" * 80)

    reset_pd = df_intervention[df_intervention["interpretation_type"] == "Reset"][
        "policy_divergence"
    ]
    other_pd = df_intervention[df_intervention["interpretation_type"] != "Reset"][
        "policy_divergence"
    ]

    t_stat, p_val = welch_ttest(reset_pd, other_pd)
    d = cohens_d(reset_pd, other_pd)

    print("\nComparing Reset vs all other interpretation types for Policy Divergence:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
    print(f"  Reset mean: {reset_pd.mean():.3f} (SD={reset_pd.std():.3f})")
    print(f"  Others mean: {other_pd.mean():.3f} (SD={other_pd.std():.3f})")

    # Mean by interpretation type
    print("\nMean Policy Divergence by Interpretation Type:")
    means = df_intervention.groupby("interpretation_type")["policy_divergence"].agg(
        ["mean", "std", "count"]
    )
    print(means.sort_values("mean", ascending=False).to_string())

    # -------------------------------------------------------------------------
    # 7. CLAIM: Disrupt × Correcting is incompatible
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("7. CLAIM: 'Disrupt' interpretation is incompatible with 'Correcting' style")
    print("-" * 80)

    # Disrupt + Correcting vs Disrupt + Others
    disrupt_correcting = df_intervention[
        (df_intervention["interpretation_type"] == "Disrupt")
        & (df_intervention["teaching_style"] == "Correcting")
    ]["policy_divergence"]
    disrupt_others = df_intervention[
        (df_intervention["interpretation_type"] == "Disrupt")
        & (df_intervention["teaching_style"] != "Correcting")
    ]["policy_divergence"]

    t_stat, p_val = welch_ttest(disrupt_correcting, disrupt_others)
    d = cohens_d(disrupt_correcting, disrupt_others)

    print("\nComparing Disrupt+Correcting vs Disrupt+Other styles:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
    print(
        f"  Disrupt+Correcting mean: {disrupt_correcting.mean():.3f} (SD={disrupt_correcting.std():.3f})"
    )
    print(
        f"  Disrupt+Others mean: {disrupt_others.mean():.3f} (SD={disrupt_others.std():.3f})"
    )

    # Interaction effect: Compare the Disrupt disadvantage for Correcting vs other styles
    print("\nMean Policy Divergence for Disrupt by Teaching Style:")
    disrupt_df = df_intervention[df_intervention["interpretation_type"] == "Disrupt"]
    print(
        disrupt_df.groupby("teaching_style")["policy_divergence"]
        .agg(["mean", "std", "count"])
        .to_string()
    )

    return df_intervention


def analyze_three_agent_claims(df):
    """Analyze claims for three-agent experiments."""
    print("\n" + "=" * 80)
    print("THREE-AGENT EXPERIMENT STATISTICAL ANALYSES")
    print("=" * 80)

    # Filter to intervention conditions
    df_intervention = df[df["interpretation_type"] != "Baseline"].copy()
    df_intervention = df_intervention.dropna(subset=["avg_attachment"])

    print(f"\nTotal observations (intervention conditions): {len(df_intervention)}")

    # -------------------------------------------------------------------------
    # 1. FACTORIAL ANOVA for Region Attachment
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(
        "1. FACTORIAL ANOVA: Region Attachment ~ Teaching Style × Interpretation Type × Rate"
    )
    print("-" * 80)

    if HAS_STATSMODELS:
        anova_att = run_factorial_anova(
            df_intervention,
            "avg_attachment",
            ["teaching_style", "interpretation_type", "intervention_rate"],
        )
        if anova_att is not None:
            print(anova_att.to_string())

    # -------------------------------------------------------------------------
    # 2. CLAIM: Undoing and Correcting amplify spatial attachment
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("2. CLAIM: 'Undoing' and 'Correcting' amplify spatial attachment")
    print("-" * 80)

    # Individual style attachments (not directly used but computed for reference)
    other_att = df_intervention[
        ~df_intervention["teaching_style"].isin(["Undoing", "Correcting"])
    ]["avg_attachment"]

    # Combined Undoing+Correcting vs Others
    undoing_correcting_att = df_intervention[
        df_intervention["teaching_style"].isin(["Undoing", "Correcting"])
    ]["avg_attachment"]

    t_stat, p_val = welch_ttest(undoing_correcting_att, other_att)
    d = cohens_d(undoing_correcting_att, other_att)

    print("\nComparing Undoing+Correcting vs other styles:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
    print(
        f"  Undoing+Correcting mean: {undoing_correcting_att.mean():.3f} (SD={undoing_correcting_att.std():.3f})"
    )
    print(f"  Others mean: {other_att.mean():.3f} (SD={other_att.std():.3f})")

    print("\nMean Region Attachment by Teaching Style:")
    print(
        df_intervention.groupby("teaching_style")["avg_attachment"]
        .agg(["mean", "std", "count"])
        .to_string()
    )

    # -------------------------------------------------------------------------
    # 3. CLAIM: Linear strengthening with intervention rate
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("3. CLAIM: Linear strengthening of niche persistence with intervention rate")
    print("-" * 80)

    # Overall correlation
    r, p = linear_trend_test(df_intervention, "avg_attachment")
    print("\nOverall linear trend (Rate → Attachment):")
    print(f"  Pearson r = {r:.3f}, p = {p:.4f}")

    # Per teaching style
    print("\nLinear trend by Teaching Style:")
    print(f"{'Style':<40} {'r':>10} {'p-value':>12} {'Significant':>15}")
    print("-" * 77)
    for style in TEACHING_STYLES.values():
        style_df = df_intervention[df_intervention["teaching_style"] == style]
        r, p = linear_trend_test(style_df, "avg_attachment")
        sig = (
            "Yes***"
            if p < 0.001
            else "Yes**"
            if p < 0.01
            else "Yes*"
            if p < 0.05
            else "No"
        )
        print(f"{style:<40} {r:>10.3f} {p:>12.4f} {sig:>15}")

    # -------------------------------------------------------------------------
    # 4. CLAIM: Restart is detrimental
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("4. CLAIM: 'Restart' is detrimental to spatial coordination")
    print("-" * 80)

    restart_att = df_intervention[df_intervention["teaching_style"] == "Restart"][
        "avg_attachment"
    ]
    other_att = df_intervention[df_intervention["teaching_style"] != "Restart"][
        "avg_attachment"
    ]

    t_stat, p_val = welch_ttest(restart_att, other_att)
    d = cohens_d(restart_att, other_att)

    print("\nComparing Restart vs all other styles:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
    print(f"  Restart mean: {restart_att.mean():.3f} (SD={restart_att.std():.3f})")
    print(f"  Others mean: {other_att.mean():.3f} (SD={other_att.std():.3f})")

    return df_intervention


def main():
    """Main analysis function."""
    print("\n" + "#" * 80)
    print("# STATISTICAL ANALYSIS FOR COGSCI 2026 PAPER")
    print("# Multi-Agent Coordination Under Pedagogical Intervention")
    print("#" * 80)

    # Load data
    print("\nLoading two-agent data...")
    df_two_agent = load_all_two_agent_data()
    print(f"Loaded {len(df_two_agent)} observations")

    print("\nLoading three-agent data...")
    df_three_agent = load_all_three_agent_data()
    print(f"Loaded {len(df_three_agent)} observations")

    # Run analyses
    if len(df_two_agent) > 0:
        analyze_two_agent_claims(df_two_agent)
    else:
        print("\nWARNING: No two-agent data loaded!")

    if len(df_three_agent) > 0:
        analyze_three_agent_claims(df_three_agent)
    else:
        print("\nWARNING: No three-agent data loaded!")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)
    print("""
    Two-Agent Experiments:
    - See detailed ANOVA tables and pairwise comparisons above
    - Key tests for: Undoing superiority, Correcting rate dependency, 
      Restart disadvantage, Reset poor performance, Disrupt×Correcting mismatch
    
    Three-Agent Experiments:
    - See detailed ANOVA tables and trend analyses above
    - Key tests for: Undoing+Correcting advantage, linear rate effect, 
      Restart detrimental effect
    
    Note: * p < .05, ** p < .01, *** p < .001
    """)

    return df_two_agent, df_three_agent


if __name__ == "__main__":
    df_two, df_three = main()
