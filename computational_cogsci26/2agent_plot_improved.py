"""Improved 2-agent plotting script with better legibility and variability depiction."""

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
    """Load data and return both mean and individual runs for variability."""
    patch1 = identify_patches(world_size)
    data_dict = {}

    def process_condition(cond_dir, rate):
        """Process a single condition folder."""
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
            if (
                "center1_q" not in run_df.columns
                or "center2_q" not in run_df.columns
            ):
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

        # Compute mean and std for each step
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

    # Process rates 0.25~1.0
    for rate in [0.25, 0.5, 0.75, 1.0]:
        cond_dir = (
            folder_path
            / f"type_{intervention_type}_rate_{rate}_mode_{intervention_mode}"
        )
        result = process_condition(cond_dir, rate)
        if result is not None:
            data_dict[rate] = result

    # Process rate 0
    zero_dir = folder_path / f"rate_0_mode_{intervention_mode}"
    result = process_condition(zero_dir, 0)
    if result is not None:
        data_dict[0] = result

    return data_dict


def plot_lowess_with_band(ax, x, y, y_std, color, alpha=0.8, linewidth=2.5):
    """Plot LOWESS smoothed line with variability band."""
    # Sort by x for proper plotting
    sort_idx = np.argsort(x)
    x_sorted = x.iloc[sort_idx].values
    y_sorted = y.iloc[sort_idx].values
    std_sorted = y_std.iloc[sort_idx].values

    # LOWESS smoothing for mean
    smoothed = lowess(y_sorted, x_sorted, frac=0.1)
    x_smooth = smoothed[:, 0]
    y_smooth = smoothed[:, 1]

    # LOWESS smoothing for upper and lower bounds
    y_upper = y_sorted + std_sorted
    y_lower = y_sorted - std_sorted
    upper_smooth = lowess(y_upper, x_sorted, frac=0.1)[:, 1]
    lower_smooth = lowess(y_lower, x_sorted, frac=0.1)[:, 1]

    # Plot shaded band
    ax.fill_between(x_smooth, lower_smooth, upper_smooth, color=color, alpha=0.2)

    # Plot mean line
    ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth, alpha=alpha)


def create_improved_plots(root_path):
    """Create improved figures with better legibility."""
    # Color scheme - more distinct colors
    colors = {
        0: "#2d2d2d",  # Dark gray/black for baseline
        0.25: "#fdae61",  # Orange (light)
        0.5: "#d7191c",  # Red (dark)
        0.75: "#abd9e9",  # Blue (light)
        1.0: "#2c7bb6",  # Blue (dark)
    }

    rate_labels = {
        0: "0%",
        0.25: "25%",
        0.5: "50%",
        0.75: "75%",
        1.0: "100%",
    }

    # Intervention mode labels (column headers)
    mode_map = {
        1: "Undoing",
        2: "Correcting",
        3: "Exploration-\nencouraging",
        4: "Restart",
    }

    # Intervention type labels (row labels) - SHORTENED
    type_map = {
        0: "Suggestion",
        1: "Reset",
        2: "Interrupt",
        3: "Transition",
        4: "Disrupt",
        5: "Impede",
    }

    # Only process 2-agent world folders
    world_dirs = [
        d
        for d in root_path.glob("world3_size_*")
        if d.is_dir() and "3agent" not in d.name
    ]
    if not world_dirs:
        print("No 2-agent world folders found.")
        return

    for plot_type in ["reward_diff", "q_diff_diff"]:
        # Shortened Y-axis label
        if plot_type == "reward_diff":
            ylabel_shared = "Reward Difference"
            filename_suffix = "reward_diff"
        else:
            ylabel_shared = "Expected Reward Difference"
            filename_suffix = "q_diff_diff"

        for wdir in world_dirs:
            print(f"Processing {wdir.name} for {plot_type}...")

            # Larger figure for better readability
            fig, axes = plt.subplots(
                6, 4, figsize=(14, 16), sharex=True, sharey=True
            )

            # Adjust spacing
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
                        df = data[rate]
                        mean_col = f"{plot_type}_mean"
                        std_col = f"{plot_type}_std"

                        # Fill NaN std with 0
                        df[std_col] = df[std_col].fillna(0)

                        plot_lowess_with_band(
                            ax,
                            df["step"],
                            df[mean_col],
                            df[std_col],
                            color=colors[rate],
                            alpha=0.9,
                            linewidth=2.0,
                        )

                    # Styling
                    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
                    ax.tick_params(axis="both", labelsize=10, width=1)

                    # Reduce number of x-ticks
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.yaxis.set_major_locator(MaxNLocator(4))

                    # Column titles (intervention modes) - only top row
                    if type_idx == 0:
                        ax.set_title(
                            mode_map[mode], fontsize=14, fontweight="bold", pad=8
                        )

                    # Row labels (intervention types) - only first column
                    if mode_idx == 0:
                        ax.set_ylabel(
                            type_map[int_type], fontsize=12, fontweight="bold"
                        )

                    # X-axis label - only bottom row
                    if type_idx == 5:
                        ax.set_xlabel("Step", fontsize=12, fontweight="bold")

            # Add shared Y-axis label on the left side
            fig.text(
                0.02,
                0.5,
                ylabel_shared,
                va="center",
                rotation="vertical",
                fontsize=14,
                fontweight="bold",
            )

            # Create legend at the top
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color=colors[rate],
                    linewidth=3,
                    label=f"{rate_labels[rate]} intervention",
                )
                for rate in [0, 0.25, 0.5, 0.75, 1.0]
            ]
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                ncol=5,
                fontsize=11,
                frameon=True,
                bbox_to_anchor=(0.55, 0.98),
            )

            # Save
            out = root_path / f"improved_{filename_suffix}_{wdir.name}.png"
            plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
            plt.close()
            print(f"Saved -> {out}")


if __name__ == "__main__":
    create_improved_plots(Path(__file__).parent)
