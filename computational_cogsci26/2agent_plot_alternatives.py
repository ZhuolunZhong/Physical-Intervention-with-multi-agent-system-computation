"""Three alternative variability depictions for 2-agent figures."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess


def identify_patches(world_size=10):
    return [(x, y) for x in range(1, 4) for y in range(1, 4)]


def load_and_process_data_with_variability(
    folder_path, intervention_type, intervention_mode, world_size=10
):
    """Load data and return mean, std, and count for SE calculation."""
    patch1 = identify_patches(world_size)
    data_dict = {}

    def process_condition(cond_dir, rate):
        if not cond_dir.exists():
            return None

        runs_data = []
        for visit_csv in cond_dir.glob("visit_stats_*.csv"):
            run_id = visit_csv.stem.split("_")[-1]
            run_csv = cond_dir / f"run_{run_id}.csv"
            if not run_csv.exists():
                continue

            visit_df = pd.read_csv(visit_csv)
            a0, a1 = 0, 0
            for _, row in visit_df.iterrows():
                if (row["grid_x"], row["grid_y"]) in patch1:
                    a0 += row["agent_0"]
                    a1 += row["agent_1"]
            g1_id, g2_id = (0, 1) if a0 > a1 else (1, 0)

            run_df = pd.read_csv(run_csv)
            if "center1_q" not in run_df.columns or "center2_q" not in run_df.columns:
                continue

            g1 = run_df[run_df["agentid"] == g1_id]
            g2 = run_df[run_df["agentid"] == g2_id]
            m = pd.merge(g1, g2, on="step", suffixes=("_g1", "_g2"))
            m["reward_diff"] = np.abs(
                m["CumulativeReward_g1"] - m["CumulativeReward_g2"]
            )
            m["q_diff_diff"] = (m["center1_q_g1"] - m["center2_q_g1"]) - (
                m["center1_q_g2"] - m["center2_q_g2"]
            )
            runs_data.append(m[["step", "reward_diff", "q_diff_diff"]])

        if not runs_data:
            return None

        all_runs = pd.concat(runs_data)
        stats = all_runs.groupby("step").agg(["mean", "std", "count"]).reset_index()
        stats.columns = [
            "step",
            "reward_diff_mean",
            "reward_diff_std",
            "reward_diff_count",
            "q_diff_diff_mean",
            "q_diff_diff_std",
            "q_diff_diff_count",
        ]
        stats["rate"] = rate
        return stats

    for rate in [0.25, 0.5, 0.75, 1.0]:
        cond_dir = (
            folder_path
            / f"type_{intervention_type}_rate_{rate}_mode_{intervention_mode}"
        )
        result = process_condition(cond_dir, rate)
        if result is not None:
            data_dict[rate] = result

    zero_dir = folder_path / f"rate_0_mode_{intervention_mode}"
    result = process_condition(zero_dir, 0)
    if result is not None:
        data_dict[0] = result

    return data_dict


# Color scheme
COLORS = {
    0: "#2d2d2d",
    0.25: "#fdae61",
    0.5: "#d7191c",
    0.75: "#abd9e9",
    1.0: "#2c7bb6",
}

RATE_LABELS = {0: "0%", 0.25: "25%", 0.5: "50%", 0.75: "75%", 1.0: "100%"}

MODE_MAP = {
    1: "Undoing",
    2: "Correcting",
    3: "Exploration-\nencouraging",
    4: "Restart",
}

TYPE_MAP = {
    0: "Suggestion",
    1: "Reset",
    2: "Interrupt",
    3: "Transition",
    4: "Disrupt",
    5: "Impede",
}


def smooth_data(x, y, frac=0.1):
    """Apply LOWESS smoothing."""
    sort_idx = np.argsort(x)
    x_sorted = np.array(x)[sort_idx]
    y_sorted = np.array(y)[sort_idx]
    smoothed = lowess(y_sorted, x_sorted, frac=frac)
    return smoothed[:, 0], smoothed[:, 1]


# =============================================================================
# ALTERNATIVE 1: Standard Error bands (SE = SD/sqrt(n))
# =============================================================================
def plot_alternative1_se_bands(ax, df, plot_type, color, linewidth=2.0):
    """Plot with Standard Error bands (much narrower than SD)."""
    mean_col = f"{plot_type}_mean"
    std_col = f"{plot_type}_std"
    count_col = f"{plot_type}_count"

    df = df.copy()
    df[std_col] = df[std_col].fillna(0)
    df[count_col] = df[count_col].fillna(1).replace(0, 1)

    # Calculate Standard Error
    se = df[std_col] / np.sqrt(df[count_col])

    x = df["step"].values
    y = df[mean_col].values
    y_upper = y + se.values
    y_lower = y - se.values

    # Smooth all curves
    x_smooth, y_smooth = smooth_data(x, y)
    _, upper_smooth = smooth_data(x, y_upper)
    _, lower_smooth = smooth_data(x, y_lower)

    # Plot SE band (thin and subtle)
    ax.fill_between(x_smooth, lower_smooth, upper_smooth, color=color, alpha=0.15)
    ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth, alpha=0.9)


# =============================================================================
# ALTERNATIVE 2: Endpoint error bars only
# =============================================================================
def plot_alternative2_endpoint_bars(ax, df, plot_type, color, linewidth=2.0):
    """Plot lines with error bars only at the final timepoint."""
    mean_col = f"{plot_type}_mean"
    std_col = f"{plot_type}_std"
    count_col = f"{plot_type}_count"

    df = df.copy()
    df[std_col] = df[std_col].fillna(0)
    df[count_col] = df[count_col].fillna(1).replace(0, 1)

    x = df["step"].values
    y = df[mean_col].values

    # Smooth the line
    x_smooth, y_smooth = smooth_data(x, y)
    ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth, alpha=0.9)

    # Get final point stats (use SE for error bar)
    final_idx = df["step"].idxmax()
    final_x = df.loc[final_idx, "step"]
    final_y = df.loc[final_idx, mean_col]
    final_se = df.loc[final_idx, std_col] / np.sqrt(df.loc[final_idx, count_col])

    # Plot error bar at endpoint
    ax.errorbar(
        final_x,
        final_y,
        yerr=final_se,
        fmt="none",
        ecolor=color,
        elinewidth=1.5,
        capsize=3,
        capthick=1.5,
        alpha=0.8,
    )


# =============================================================================
# ALTERNATIVE 3: Thin outline bounds (dotted lines)
# =============================================================================
def plot_alternative3_outline_bounds(ax, df, plot_type, color, linewidth=2.0):
    """Plot with thin dotted lines showing ±SE bounds."""
    mean_col = f"{plot_type}_mean"
    std_col = f"{plot_type}_std"
    count_col = f"{plot_type}_count"

    df = df.copy()
    df[std_col] = df[std_col].fillna(0)
    df[count_col] = df[count_col].fillna(1).replace(0, 1)

    # Calculate Standard Error
    se = df[std_col] / np.sqrt(df[count_col])

    x = df["step"].values
    y = df[mean_col].values
    y_upper = y + se.values
    y_lower = y - se.values

    # Smooth all curves
    x_smooth, y_smooth = smooth_data(x, y)
    _, upper_smooth = smooth_data(x, y_upper)
    _, lower_smooth = smooth_data(x, y_lower)

    # Plot main line
    ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth, alpha=0.9)

    # Plot thin dotted bounds
    ax.plot(
        x_smooth,
        upper_smooth,
        color=color,
        linewidth=0.8,
        linestyle=":",
        alpha=0.5,
    )
    ax.plot(
        x_smooth,
        lower_smooth,
        color=color,
        linewidth=0.8,
        linestyle=":",
        alpha=0.5,
    )


def create_figure(root_path, plot_func, suffix, description, show_legend="both"):
    """Create figures using the specified plotting function.

    Args:
        show_legend: "both", "reward_diff", "q_diff_diff", or "none"
                     Controls which figure gets the legend.
    """
    world_dirs = [
        d
        for d in root_path.glob("world3_size_*")
        if d.is_dir() and "3agent" not in d.name
    ]
    if not world_dirs:
        print("No 2-agent world folders found.")
        return

    for plot_type in ["reward_diff", "q_diff_diff"]:
        if plot_type == "reward_diff":
            ylabel_shared = "Reward Difference"
        else:
            ylabel_shared = "Expected Reward Difference"

        for wdir in world_dirs:
            print(f"Processing {wdir.name} for {plot_type} ({description})...")

            fig, axes = plt.subplots(6, 4, figsize=(14, 16), sharex=True, sharey=True)
            plt.subplots_adjust(
                left=0.12, right=0.95, top=0.92, bottom=0.08, hspace=0.15, wspace=0.08
            )

            for type_idx, int_type in enumerate([0, 1, 2, 3, 4, 5]):
                for mode_idx, mode in enumerate([1, 2, 3, 4]):
                    ax = axes[type_idx, mode_idx]
                    data = load_and_process_data_with_variability(wdir, int_type, mode)

                    for rate in [0, 0.25, 0.5, 0.75, 1.0]:
                        if rate not in data or data[rate] is None:
                            continue
                        plot_func(ax, data[rate], plot_type, COLORS[rate])

                    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
                    ax.tick_params(axis="both", labelsize=10, width=1)
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.yaxis.set_major_locator(MaxNLocator(4))

                    if type_idx == 0:
                        ax.set_title(
                            MODE_MAP[mode], fontsize=14, fontweight="bold", pad=8
                        )
                    if mode_idx == 0:
                        ax.set_ylabel(
                            TYPE_MAP[int_type], fontsize=12, fontweight="bold"
                        )
                    if type_idx == 5:
                        ax.set_xlabel("Step", fontsize=12, fontweight="bold")

            fig.text(
                0.02,
                0.5,
                ylabel_shared,
                va="center",
                rotation="vertical",
                fontsize=14,
                fontweight="bold",
            )

            # Conditionally add legend based on show_legend parameter
            include_legend = show_legend == "both" or show_legend == plot_type

            if include_legend:
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        color=COLORS[rate],
                        linewidth=3,
                        label=f"{RATE_LABELS[rate]}",
                    )
                    for rate in [0, 0.25, 0.5, 0.75, 1.0]
                ]
                fig.legend(
                    handles=legend_elements,
                    loc="upper right",
                    ncol=1,
                    fontsize=11,
                    frameon=True,
                    bbox_to_anchor=(1.0, 0.98),
                    title="Intervention Rate",
                    title_fontsize=11,
                )

            out = root_path / f"alt{suffix}_{plot_type}_{wdir.name}.png"
            plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
            plt.close()
            print(f"Saved -> {out}")


if __name__ == "__main__":
    root = Path(__file__).parent

    print("\n=== Alternative 1: Standard Error bands ===")
    # Legend only on reward_diff (top figure when vertically stacked)
    create_figure(
        root,
        plot_alternative1_se_bands,
        "1_se_bands",
        "SE bands",
        show_legend="reward_diff",
    )

    print("\n=== Alternative 2: Endpoint error bars ===")
    create_figure(
        root,
        plot_alternative2_endpoint_bars,
        "2_endpoint_bars",
        "endpoint bars",
        show_legend="reward_diff",
    )

    print("\n=== Alternative 3: Outline bounds ===")
    create_figure(
        root,
        plot_alternative3_outline_bounds,
        "3_outline_bounds",
        "outline bounds",
        show_legend="reward_diff",
    )

    print("\nAll alternatives generated!")
