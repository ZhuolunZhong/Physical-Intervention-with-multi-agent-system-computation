import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

def identify_patches(world_size=10):
    return [(x, y) for x in range(1, 4) for y in range(1, 4)]

def load_and_process_data(folder_path, intervention_type, intervention_mode, world_size=10):
    patch1 = identify_patches(world_size)
    data_dict = {}

    # Process rates 0.25~1.0
    for rate in [0.25, 0.5, 0.75, 1.0]:
        cond_dir = folder_path / f'type_{intervention_type}_rate_{rate}_mode_{intervention_mode}'
        if not cond_dir.exists():
            continue
        runs = []
        for visit_csv in cond_dir.glob('visit_stats_*.csv'):
            run_id = visit_csv.stem.split('_')[-1]
            run_csv = cond_dir / f'run_{run_id}.csv'
            if not run_csv.exists():
                continue

            visit_df = pd.read_csv(visit_csv)
            a0, a1 = 0, 0
            for _, row in visit_df.iterrows():
                if (row['grid_x'], row['grid_y']) in patch1:
                    a0 += row['agent_0']
                    a1 += row['agent_1']
            g1_id, g2_id = (0, 1) if a0 > a1 else (1, 0)

            run_df = pd.read_csv(run_csv)
            
            # Check if required columns exist
            if 'center1_q' not in run_df.columns or 'center2_q' not in run_df.columns:
                continue
                
            g1 = run_df[run_df['agentid'] == g1_id]
            g2 = run_df[run_df['agentid'] == g2_id]
            m = pd.merge(g1, g2, on='step', suffixes=('_g1', '_g2'))
            m['reward_diff'] = np.abs(m['CumulativeReward_g1'] - m['CumulativeReward_g2'])
            m['q_diff_diff'] = (m['center1_q_g1'] - m['center2_q_g1']) - (m['center1_q_g2'] - m['center2_q_g2'])
            runs.append(m[['step', 'reward_diff', 'q_diff_diff']])

        if runs:
            avg = pd.concat(runs).groupby('step').mean().reset_index()
            avg['rate'] = rate
            data_dict[rate] = avg

    # Process rate 0
    zero_dir = folder_path / f'rate_0_mode_{intervention_mode}'
    if zero_dir.exists():
        runs = []
        for visit_csv in zero_dir.glob('visit_stats_*.csv'):
            run_id = visit_csv.stem.split('_')[-1]
            run_csv = zero_dir / f'run_{run_id}.csv'
            if not run_csv.exists():
                continue
            visit_df = pd.read_csv(visit_csv)
            a0, a1 = 0, 0
            for _, row in visit_df.iterrows():
                if (row['grid_x'], row['grid_y']) in patch1:
                    a0 += row['agent_0']
                    a1 += row['agent_1']
            g1_id, g2_id = (0, 1) if a0 > a1 else (1, 0)
            run_df = pd.read_csv(run_csv)
            
            # Check if required columns exist
            if 'center1_q' not in run_df.columns or 'center2_q' not in run_df.columns:
                continue
                
            g1 = run_df[run_df['agentid'] == g1_id]
            g2 = run_df[run_df['agentid'] == g2_id]
            m = pd.merge(g1, g2, on='step', suffixes=('_g1', '_g2'))
            m['reward_diff'] = np.abs(m['CumulativeReward_g1'] - m['CumulativeReward_g2'])
            m['q_diff_diff'] = (m['center1_q_g1'] - m['center2_q_g1']) - (m['center1_q_g2'] - m['center2_q_g2'])
            runs.append(m[['step', 'reward_diff', 'q_diff_diff']])
        if runs:
            avg = pd.concat(runs).groupby('step').mean().reset_index()
            avg['rate'] = 0
            data_dict[0] = avg

    return data_dict

def plot_lowess(ax, x, y, color, linestyle='-', alpha=0.8):
    smoothed = lowess(y, x, frac=0.1)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linestyle=linestyle, alpha=alpha, linewidth=5)

def create_plots(root_path):
    colors = {0.25: (1.0, 0.7, 0.7), 0.5: (0.8, 0.2, 0.2),
              0.75: (0.7, 0.7, 1.0), 1.0: (0.2, 0.2, 0.8), 0: (0, 0, 0)}
    
    mode_map = {1: 'Undoing', 2: 'Correcting', 3: 'Exploration-encouraging\nCorrecting', 4: 'Restart'}
    type_map = {0: 'Suggestion', 1: 'Reset', 2: 'Interrupt', 
                3: 'Transition', 4: 'Disrupt', 5: 'Impede'}

    # Only process 2-agent world folders (exclude 3agent)
    world_dirs = [d for d in root_path.glob('world3_size_*') if d.is_dir() and '3agent' not in d.name]
    if not world_dirs:
        print('No 2-agent world folders found.')
        return

    for plot_type in ['reward_diff', 'q_diff_diff']:
        ylabel_base = 'Reward Difference' if plot_type == 'reward_diff' else 'Expected Reward Difference'
        
        for wdir in world_dirs:
            print(f'Processing {wdir.name} for {plot_type}...')
            fig, axes = plt.subplots(6, 4, figsize=(16, 18), sharex=True, sharey=True)
            
            for type_idx, int_type in enumerate([0, 1, 2, 3, 4, 5]):
                for mode_idx, mode in enumerate([1, 2, 3, 4]):
                    ax = axes[type_idx, mode_idx]
                    data = load_and_process_data(wdir, int_type, mode)
                    
                    for rate in [0, 0.25, 0.5, 0.75, 1.0]:
                        if rate not in data or data[rate] is None:
                            continue
                        plot_lowess(ax, data[rate]['step'], data[rate][plot_type],
                                   color=colors[rate], linestyle='-', alpha=0.8)
                    
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=12)
                    
                    if type_idx == 0:
                        ax.set_title(mode_map[mode], fontsize=13, fontweight='bold')
                    
                    if mode_idx == 0:
                        type_name = type_map[int_type]
                        full_ylabel = f'{type_name}\n{ylabel_base}'
                        ax.set_ylabel(full_ylabel, fontsize=12, fontweight='bold')
                    
                    if type_idx == 5:
                        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            out = root_path / f'reproduced_{plot_type}_{wdir.name}.png'
            plt.savefig(out, bbox_inches='tight', dpi=300)
            plt.close()
            print(f'Saved -> {out}')

if __name__ == '__main__':
    create_plots(Path(__file__).parent)
