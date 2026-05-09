import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def load_and_process_visit_data(folder_path, intervention_mode, world_size):
    """Load and process visit statistics data."""
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

            close_agent_data = []
            other_agents_data = []

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

                close_agent_data.append(close_visits)
                other_agents_data.append(other_visits)

            if close_agent_data and other_agents_data:
                if rate not in data_dict:
                    data_dict[rate] = {"close": [], "other": []}
                data_dict[rate]["close"].append(np.mean(close_agent_data))
                data_dict[rate]["other"].append(np.mean(other_agents_data))

    # Process rate 0 data
    zero_rate_folder = folder_path / f"rate_0_mode_{intervention_mode}"
    if zero_rate_folder.exists():
        close_agent_data_zero = []
        other_agents_data_zero = []

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

            close_agent_data_zero.append(close_visits_zero)
            other_agents_data_zero.append(other_visits_zero)

        if close_agent_data_zero and other_agents_data_zero:
            if 0 not in data_dict:
                data_dict[0] = {"close": [], "other": []}
            data_dict[0]["close"].append(np.mean(close_agent_data_zero))
            data_dict[0]["other"].append(np.mean(other_agents_data_zero))

    # Calculate final averages across all intervention types
    final_data = {}
    for rate in [0, 0.25, 0.5, 0.75, 1.0]:
        if (
            rate in data_dict
            and data_dict[rate]["close"]
            and data_dict[rate]["other"]
        ):
            final_data[rate] = {
                "close": np.mean(data_dict[rate]["close"]),
                "other": np.mean(data_dict[rate]["other"]),
            }

    return final_data


def create_1x4_patch_plots(root_path):
    """Create 1 row 4 column patch visit frequency plots."""
    # Color definitions
    colors = {
        0.25: (1.0, 0.7, 0.7),  # light red
        0.5: (0.8, 0.2, 0.2),  # dark red
        0.75: (0.7, 0.7, 1.0),  # light blue
        1.0: (0.2, 0.2, 0.8),  # dark blue
        0: (0, 0, 0),  # black
    }

    # Intervention mode mapping
    intervention_mode_map = {
        1: "Undoing",
        2: "Correcting",
        3: "Exploration-encouraging\nCorrecting",
        4: "Restart",
    }

    intervention_modes = list(intervention_mode_map.keys())

    # Get only 3-agent world folders
    world_folders = [
        f for f in root_path.glob("world3_size_*_3agent") if f.is_dir()
    ]

    if not world_folders:
        print("Error: No world3_size_*_3agent folders found in the directory.")
        exit(1)

    for world_folder in world_folders:
        # Extract world size from folder name like "world3_size_12_3agent"
        parts = world_folder.name.split("_")
        # Find the part after "size"
        size_idx = parts.index("size") + 1
        world_size = int(parts[size_idx])

        print(f"Processing {world_folder.name} with world_size={world_size}")

        # Create 1x4 figure
        fig, axes = plt.subplots(
            1, len(intervention_modes), figsize=(20, 6), sharey=True
        )

        if len(intervention_modes) == 1:
            axes = np.array([axes])

        # Iterate through all intervention modes (columns)
        for i, mode in enumerate(intervention_modes):
            ax = axes[i]

            # Load data (merged from all intervention types)
            data_dict = load_and_process_visit_data(world_folder, mode, world_size)

            # Prepare data
            rates = [0, 0.25, 0.5, 0.75, 1.0]
            close_values = []
            other_values = []

            for rate in rates:
                if rate in data_dict:
                    close_values.append(data_dict[rate]["close"])
                    other_values.append(data_dict[rate]["other"])
                else:
                    close_values.append(0)
                    other_values.append(0)

            # Set x-axis positions
            groups = ["Close Agent", "Other Agents"]
            x_pos = np.arange(len(groups))
            bar_width = 0.15

            # Draw bar chart
            for idx, rate in enumerate(rates):
                ax.bar(
                    x_pos[0] + idx * bar_width,
                    close_values[idx],
                    bar_width,
                    color=colors[rate],
                    alpha=0.8,
                    label=f"Rate {rate}" if i == 0 else "",
                )
                ax.bar(
                    x_pos[1] + idx * bar_width,
                    other_values[idx],
                    bar_width,
                    color=colors[rate],
                    alpha=0.8,
                )

            # Set subplot titles and labels
            ax.set_xlabel("Agent Group", fontsize=24, fontweight="bold")
            if i == 0:
                ax.set_ylabel("Visit Count", fontsize=24, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=20)
            ax.set_xticks(x_pos + bar_width * 2)
            ax.set_xticklabels(groups)
            ax.grid(True, alpha=0.3, axis="y")

            # Set subplot title
            ax.set_title(
                intervention_mode_map[mode], pad=10, fontsize=24, fontweight="bold"
            )

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        # Save figure
        output_file = root_path / "reproduced_3agent_figure.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved 1x4 plot to {output_file}")


if __name__ == "__main__":
    current_folder = Path(__file__).parent
    create_1x4_patch_plots(current_folder)
