import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def identify_patches(world_size, selected_grids, mode3_centers):
    """根据world.py的逻辑识别三个patch的网格坐标"""
    centers = np.array(mode3_centers)
    patches = [[] for _ in centers]
    
    for grid in selected_grids:
        grid_center = np.array([grid[0] + 0.5, grid[1] + 0.5])
        distances = np.linalg.norm(centers - grid_center, axis=1)
        patch_idx = np.argmin(distances)
        patches[patch_idx].append(grid)
    
    return patches

def load_and_process_visit_data(folder_path, intervention_mode, world_size):
    """加载并处理访问统计数据 - 合并所有干预类型的数据"""
    mode3_centers = [[2.5,2.5],[9.5,2.5],[5.5,9.5]]
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
    
    # 处理所有干预类型的数据（合并处理）
    for intervention_type in [0, 1, 2, 3, 4, 5]:
        for rate in [0.25, 0.5, 0.75, 1.0]:
            condition_folder = folder_path / f"type_{intervention_type}_rate_{rate}_mode_{intervention_mode}"
            
            if not condition_folder.exists():
                continue
                
            close_agent_data = []
            other_agents_data = []
            
            for visit_file in condition_folder.glob("visit_stats_*.csv"):
                df = pd.read_csv(visit_file)
                
                # 计算每个agent对每个patch的访问次数
                patch_visits = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
                
                for _, row in df.iterrows():
                    x, y = int(row['grid_x']), int(row['grid_y'])
                    for i, patch in enumerate(patches):
                        if (x, y) in patch:
                            for agent_id in [0, 1, 2]:
                                patch_visits[agent_id][i] += row[f'agent_{agent_id}']
                            break
                
                # Close Agent组: agent0-patch1, agent1-patch2, agent2-patch3 的平均值
                close_visits = (patch_visits[0][0] + patch_visits[1][1] + patch_visits[2][2]) / 3
                
                # Other Agents组: 每个agent访问其他两个patch的平均值，再对所有agent和patch取平均
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
            
            # 计算平均值并累加到数据字典
            if close_agent_data and other_agents_data:
                if rate not in data_dict:
                    data_dict[rate] = {'close': [], 'other': []}
                data_dict[rate]['close'].append(np.mean(close_agent_data))
                data_dict[rate]['other'].append(np.mean(other_agents_data))
    
    # 处理干预率为0的数据（同样合并所有类型）
    zero_rate_folder = folder_path / f"rate_0_mode_{intervention_mode}"
    if zero_rate_folder.exists():
        close_agent_data_zero = []
        other_agents_data_zero = []
        
        for visit_file in zero_rate_folder.glob("visit_stats_*.csv"):
            df = pd.read_csv(visit_file)
            
            patch_visits = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
            
            for _, row in df.iterrows():
                x, y = int(row['grid_x']), int(row['grid_y'])
                for i, patch in enumerate(patches):
                    if (x, y) in patch:
                        for agent_id in [0, 1, 2]:
                            patch_visits[agent_id][i] += row[f'agent_{agent_id}']
                        break
            
            close_visits_zero = (patch_visits[0][0] + patch_visits[1][1] + patch_visits[2][2]) / 3
            
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
                data_dict[0] = {'close': [], 'other': []}
            data_dict[0]['close'].append(np.mean(close_agent_data_zero))
            data_dict[0]['other'].append(np.mean(other_agents_data_zero))
    
    # 计算所有干预类型的最终平均值
    final_data = {}
    for rate in [0, 0.25, 0.5, 0.75, 1.0]:
        if rate in data_dict and data_dict[rate]['close'] and data_dict[rate]['other']:
            final_data[rate] = {
                'close': np.mean(data_dict[rate]['close']),
                'other': np.mean(data_dict[rate]['other'])
            }
    
    return final_data

def create_1x4_patch_plots(root_path):
    """创建1行4列的patch访问频率图表"""
    # 颜色定义
    colors = {
        0.25: (1.0, 0.7, 0.7),  # 淡红
        0.5: (0.8, 0.2, 0.2),   # 深红
        0.75: (0.7, 0.7, 1.0),  # 淡蓝
        1.0: (0.2, 0.2, 0.8),   # 深蓝
        0: (0, 0, 0)            # 黑色
    }
    
    # 干预模式映射
    intervention_mode_map = {
        1: "Undoing",
        2: "Correcting", 
        3: "Exploration-encouraging\nCorrecting",
        4: "Restart"
    }
    
    intervention_modes = list(intervention_mode_map.keys())
    
    # 获取所有world3文件夹
    world_folders = [f for f in root_path.glob("world3_size_*") if f.is_dir()]
    
    if not world_folders:
        print("Error: No world3 folders found in the directory.")
        exit(1)
    
    for world_folder in world_folders:
        world_size = int(world_folder.name.split('_')[-1])
        
        # 创建1行4列的大图
        fig, axes = plt.subplots(1, len(intervention_modes), figsize=(20, 6), sharey=True)
        
        if len(intervention_modes) == 1:
            axes = np.array([axes])
        
        # 遍历所有干预模式（列）
        for i, mode in enumerate(intervention_modes):
            ax = axes[i]
            
            # 加载数据（合并所有干预类型）
            data_dict = load_and_process_visit_data(world_folder, mode, world_size)
            
            # 准备数据
            rates = [0, 0.25, 0.5, 0.75, 1.0]
            close_values = []
            other_values = []
            
            for rate in rates:
                if rate in data_dict:
                    close_values.append(data_dict[rate]['close'])
                    other_values.append(data_dict[rate]['other'])
                else:
                    close_values.append(0)
                    other_values.append(0)
            
            # 设置x轴位置 - 组在x轴上，干预率在每组内
            groups = ['Close Agent', 'Other Agents']
            x_pos = np.arange(len(groups))
            bar_width = 0.15
            
            # 绘制柱状图 - 每个组内显示所有干预率
            for idx, rate in enumerate(rates):
                # Close Agent组的干预率
                ax.bar(x_pos[0] + idx * bar_width, close_values[idx], bar_width, 
                      color=colors[rate], alpha=0.8, label=f'Rate {rate}' if i == 0 else '')
                
                # Other Agents组的干预率
                ax.bar(x_pos[1] + idx * bar_width, other_values[idx], bar_width, 
                      color=colors[rate], alpha=0.8)
            
            # 设置子图标题和标签
            ax.set_xlabel('Agent Group', fontsize=24, fontweight='bold')
            if i == 0:
                ax.set_ylabel('Visit Count', fontsize=24, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_xticks(x_pos + bar_width * 2)
            ax.set_xticklabels(groups)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 设置子图标题
            ax.set_title(intervention_mode_map[mode], pad=10, fontsize=24, fontweight='bold')
        
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        output_file = root_path / f"6.2.1.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved 1x4 plot to {output_file}")

if __name__ == "__main__":
    current_folder = Path(__file__).parent
    create_1x4_patch_plots(current_folder)