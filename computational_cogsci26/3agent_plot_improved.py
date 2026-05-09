"""Improved 3-agent plotting script with error bars and better legibility."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def identify_patches(world_size, selected_grids, mode3_centers):
    """Identify patches based on world.py logic."""
    centers = np.array(mode3_centers)
    patches = [[] for _ in centers]

    for grid in selected_grids:
        grid_center = np.array([grid[0] + 0.5, grid[1] + 0.5])
        distances = np.linalg.norm(centers - grid_center, axis=1)
        patch_idx = np.argmin(distances)
        patches[patch_idx].append(grid)

    return patches


def load_and_process_visit_data_with_variability(
    folder_path, intervention_mode, world_size
):
    """Load and process visit statistics data with variability tracking."""
    mode3_centers = [[2.5, 2.5], [9.5, 2.5], [5.5, 9.5]]
    random_grid = 18

    all_grids = [(x, y) for x in range(world_size) for y in range(world_size)]
    centers = np.array(mode3_centers)
    num_centers = len(centers)
    grids_per_center = random_grid // num_centers
    remaining = random_grid % num_centers

    selected_grids = []
    for i, center in enumerate(centers):
        count = grids_per_center + (1 if i < remaining else 0)
        if count == 0:
            continue

        distances = []
        for x, y in all_grids:
            grid_center = np.array([x + 0.5, y + 0.5])
            dist = np.linalg.norm(grid_center - center)
            distances.append((dist, (x, y)))

        distances.sort()
        selected = []
        for dist, grid in distances:
            if grid not in selected_grids:
                selected.append(grid)
                if len(selected) == count:
                    break
        selected_grids.extend(selected)

    patches = identify_patches(world_size, selected_grids, mode3_centers)

    # Store all individual run data for computing std
    data_dict = {}

    # Process all intervention types (merged)
    for intervention_type in [0, 1, 2, 3, 4, 5]:
        for rate in [0.25, 0.5, 0.75, 1.0]:
            condition_folder = (
                folder_path
                / f"type_{intervention_type}_rate_{rate}_mode_{intervention_mode}"
            )

            if not condition_folder.exists():
                continue

            for visit_file in condition_folder.glob("visit_stats_*.csv"):
                df = pd.read_csv(visit_file)

                # Calculate each agent's visits to each patch
                patch_visits = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}

                for _, row in df.iterrows():
                    x, y = int(row["grid_x"]), int(row["grid_y"])
                    for i, patch in enumerate(patches):
                        if (x, y) in patch:
                            for agent_id in [0, 1, 2]:
                                patch_visits[agent_id][i] += row[f"agent_{agent_id}"]
                            break

                # Close Agent group: agent0-patch1, agent1-patch2, agent2-patch3 average
                close_visits = (
                    patch_visits[0][0] + patch_visits[1][1] + patch_visits[2][2]
                ) / 3

                # Other Agents group
                other_visits = 0
                for agent_id in [0, 1, 2]:
                    if agent_id == 0:
                        other_visits += (patch_visits[0][1] + patch_visits[0][2]) / 2
                    elif agent_id == 1:
                        other_visits += (patch_visits[1][0] + patch_visits[1][2]) / 2
                    else:
                        other_visits += (patch_visits[2][0] + patch_visits[2][1]) / 2

                other_visits /= 3

                # Store individual run data
                if rate not in data_dict:
                    data_dict[rate] = {"close": [], "other": []}
                data_dict[rate]["close"].append(close_visits)
                data_dict[rate]["other"].append(other_visits)

    # Process rate 0 data
    zero_rate_folder = folder_path / f"rate_0_mode_{intervention_mode}"
    if zero_rate_folder.exists():
        for visit_file in zero_rate_folder.glob("visit_stats_*.csv"):
            df = pd.read_csv(visit_file)

            patch_visits = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}

            for _, row in df.iterrows():
                x, y = int(row["grid_x"]), int(row["grid_y"])
                for i, patch in enumerate(patches):
                    if (x, y) in patch:
                        for agent_id in [0, 1, 2]:
                            patch_visits[agent_id][i] += row[f"agent_{agent_id}"]
                        break

            close_visits_zero = (
                patch_visits[0][0] + patch_visits[1][1] + patch_visits[2][2]
            ) / 3

            other_visits_zero = 0
            for agent_id in [0, 1, 2]:
                if agent_id == 0:
                    other_visits_zero += (patch_visits[0][1] + patch_visits[0][2]) / 2
                elif agent_id == 1:
                    other_visits_zero += (patch_visits[1][0] + patch_visits[1][2]) / 2
                else:
                    other_visits_zero += (patch_visits[2][0] + patch_visits[2][1]) / 2

            other_visits_zero /= 3

            if 0 not in data_dict:
                data_dict[0] = {"close": [], "other": []}
            data_dict[0]["close"].append(close_visits_zero)
            data_dict[0]["other"].append(other_visits_zero)

    # Calculate final stats (mean, std, sem) across all runs
    final_data = {}
    for rate in [0, 0.25, 0.5, 0.75, 1.0]:
        if rate in data_dict and data_dict[rate]["close"] and data_dict[rate]["other"]:
            close_arr = np.array(data_dict[rate]["close"])
            other_arr = np.array(data_dict[rate]["other"])
            n_close = len(close_arr)
            n_other = len(other_arr)

            final_data[rate] = {
                "close_mean": np.mean(close_arr),
                "close_std": np.std(close_arr, ddof=1) if n_close > 1 else 0,
                "close_sem": np.std(close_arr, ddof=1) / np.sqrt(n_close)
                if n_close > 1
                else 0,
                "close_n": n_close,
                "other_mean": np.mean(other_arr),
                "other_std": np.std(other_arr, ddof=1) if n_other > 1 else 0,
                "other_sem": np.std(other_arr, ddof=1) / np.sqrt(n_other)
                if n_other > 1
                else 0,
                "other_n": n_other,
            }

    return final_data


def create_improved_3agent_plots(root_path):
    """Create improved 1 row 4 column patch visit frequency plots with error bars."""
    # Improved color scheme matching figure 2 style
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

    # Intervention mode mapping
    intervention_mode_map = {
        1: "Undoing",
        2: "Correcting",
        3: "Exploration-\nencouraging",
        4: "Restart",
    }

    intervention_modes = list(intervention_mode_map.keys())
    rates = [0, 0.25, 0.5, 0.75, 1.0]

    # Get only 3-agent world folders
    world_folders = [f for f in root_path.glob("world3_size_*_3agent") if f.is_dir()]

    if not world_folders:
        print("Error: No world3_size_*_3agent folders found in the directory.")
        return

    for world_folder in world_folders:
        # Extract world size from folder name like "world3_size_12_3agent"
        parts = world_folder.name.split("_")
        size_idx = parts.index("size") + 1
        world_size = int(parts[size_idx])

        print(f"Processing {world_folder.name} with world_size={world_size}")

        # Create 1x4 figure with better sizing
        fig, axes = plt.subplots(
            1, len(intervention_modes), figsize=(16, 5), sharey=True
        )

        if len(intervention_modes) == 1:
            axes = np.array([axes])

        # Adjust spacing - lower top value to prevent legend overlap with titles
        plt.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.15, wspace=0.08)

        # Iterate through all intervention modes (columns)
        for i, mode in enumerate(intervention_modes):
            ax = axes[i]

            # Load data with variability info
            data_dict = load_and_process_visit_data_with_variability(
                world_folder, mode, world_size
            )

            # Prepare data
            rates = [0, 0.25, 0.5, 0.75, 1.0]
            close_means = []
            close_sems = []
            other_means = []
            other_sems = []

            for rate in rates:
                if rate in data_dict:
                    close_means.append(data_dict[rate]["close_mean"])
                    close_sems.append(data_dict[rate]["close_sem"])
                    other_means.append(data_dict[rate]["other_mean"])
                    other_sems.append(data_dict[rate]["other_sem"])
                else:
                    close_means.append(0)
                    close_sems.append(0)
                    other_means.append(0)
                    other_sems.append(0)

            # Set x-axis positions
            groups = ["Close\nAgent", "Other\nAgents"]
            x_pos = np.arange(len(groups))
            bar_width = 0.15

            # Draw bar chart with error bars
            for idx, rate in enumerate(rates):
                # Close Agent bars
                ax.bar(
                    x_pos[0] + idx * bar_width,
                    close_means[idx],
                    bar_width,
                    yerr=close_sems[idx],
                    color=colors[rate],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                    capsize=2,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2, "alpha": 0.8},
                    label=f"{rate_labels[rate]} intervention" if i == 0 else "",
                )
                # Other Agents bars
                ax.bar(
                    x_pos[1] + idx * bar_width,
                    other_means[idx],
                    bar_width,
                    yerr=other_sems[idx],
                    color=colors[rate],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                    capsize=2,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2, "alpha": 0.8},
                )

            # Styling improvements
            ax.set_xlabel("Agent Group", fontsize=13, fontweight="bold")
            if i == 0:
                ax.set_ylabel("Visit Count", fontsize=13, fontweight="bold")

            ax.tick_params(axis="both", which="major", labelsize=11)
            ax.set_xticks(x_pos + bar_width * 2)
            ax.set_xticklabels(groups, fontsize=11)
            ax.grid(True, alpha=0.2, axis="y", linestyle="--", linewidth=0.5)

            # Set subplot title with improved styling
            ax.set_title(
                intervention_mode_map[mode], pad=12, fontsize=14, fontweight="bold"
            )

            # Add subtle spines
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#666666")

        # Create legend at the top
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=colors[rate],
                linewidth=8,
                solid_capstyle="butt",
                label=f"{rate_labels[rate]} intervention",
            )
            for rate in rates
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=5,
            fontsize=11,
            frameon=True,
            bbox_to_anchor=(0.53, 0.98),
            handlelength=1.5,
            columnspacing=1.0,
        )

        # Save figure
        output_file = root_path / "improved_3agent_figure.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()
        print(f"Saved improved 3-agent plot to {output_file}")


if __name__ == "__main__":
    current_folder = Path(__file__).parent
    create_improved_3agent_plots(current_folder)
