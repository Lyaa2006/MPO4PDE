import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata

plt.rcParams['font.family']='SimHei'
class CylinderFlowMPODataset(Dataset):
    """
    圆柱绕流数据集 - 专门用于MPO-Transformer训练
    输出格式: (空间特征序列, 时间编码, 查询坐标, 目标值)
    """
    def __init__(self, file_path: str, 
                 trajectory_indices: Optional[List[int]] = None,
                 time_start: int = 0, 
                 time_used: int = 200,
                 seq_len: int = 10,
                 num_sequences: int = 1000,
                 grid_size: Tuple[int, int] = (64, 64),
                 normalize: bool = True):
        """
        Args:
            file_path: HDF5文件路径
            trajectory_indices: 使用的轨迹索引
            time_start: 起始时间步
            time_used: 使用的时间步数  
            seq_len: 序列长度
            num_sequences: 序列数量
            grid_size: 插值网格大小 (height, width)
            normalize: 是否归一化
        """
        self.file_path = file_path
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.grid_size = grid_size
        self.normalize = normalize
        
        # 加载原始数据
        self.raw_data = self._load_raw_data(trajectory_indices, time_start, time_used)
        
        # 插值到规则网格
        self.grid_data = self._interpolate_to_grid()
        
        # 准备MPO-Transformer训练数据
        self.sequences = self._prepare_sequences()
        
        print(f"数据加载完成: {len(self.sequences)} 个序列")
        print(f"网格形状: {self.grid_data.shape}")
        
    def _load_raw_data(self, trajectory_indices, time_start, time_used):
        """加载原始HDF5数据"""
        with h5py.File(self.file_path, 'r') as f:
            # 读取几何信息
            pos = f['pos'][:]  # [n_nodes, 2]
            mesh = f['mesh'][:]  # [n_triangles, 3]
            
            # 读取物理参数
            cylinder_center = (f.attrs['x_c'], f.attrs['y_c'])
            cylinder_radius = f.attrs['r']
            domain_bounds = {
                'x_l': f.attrs['x_l'], 'x_r': f.attrs['x_r'],
                'y_b': f.attrs['y_b'], 'y_t': f.attrs['y_t']
            }
            mu = f.attrs['mu']
            rho = f.attrs['rho']
            
            # 读取节点类型
            node_type = f['node_type']
            node_indices = {
                'inlet': node_type['inlet'][:],
                'cylinder': node_type['cylinder'][:], 
                'outlet': node_type['outlet'][:],
                'inner': node_type['inner'][:]
            }
            
            # 确定使用的轨迹
            if trajectory_indices is None:
                available_keys = [k for k in f.keys() if k not in ['pos', 'node_type', 'mesh']]
                trajectory_indices = [int(k) for k in available_keys]
            
            # 收集速度数据
            all_velocity_data = []
            trajectory_info = []
            
            for idx in trajectory_indices:
                traj_key = str(idx)
                if traj_key in f:
                    traj_group = f[traj_key]
                    U = traj_group['U'][time_start:time_start + time_used]  # [time_used, n_nodes, 2]
                    
                    # 无量纲化处理 (参考专业做法)
                    u_m = traj_group.attrs.get('u_m', 1.0)  # 进口速度
                    U = U / u_m  # 无量纲化速度
                    
                    all_velocity_data.append(U)
                    trajectory_info.append({
                        'u_m': u_m,
                        'dt': traj_group.attrs.get('dt', 1.0),
                        'trajectory_idx': idx
                    })
            
            if not all_velocity_data:
                raise ValueError("没有找到可用的轨迹数据")
                
            # 合并所有轨迹
            velocity_data = np.concatenate(all_velocity_data, axis=0)  # [total_time, n_nodes, 2]
            
            return {
                'positions': pos,
                'mesh': mesh,
                'velocity_data': velocity_data,
                'cylinder_center': cylinder_center,
                'cylinder_radius': cylinder_radius,
                'domain_bounds': domain_bounds,
                'node_indices': node_indices,
                'trajectory_info': trajectory_info,
                'physical_params': {'mu': mu, 'rho': rho}
            }
    
    def _interpolate_to_grid(self):
        """将不规则网格数据插值到规则网格"""
        positions = self.raw_data['positions']
        velocity_data = self.raw_data['velocity_data']  # [T, n_nodes, 2]
        
        # 创建规则网格
        grid_y, grid_x = self.grid_size
        x_range = (positions[:, 0].min(), positions[:, 0].max())
        y_range = (positions[:, 1].min(), positions[:, 1].max())
        
        grid_x_vals = np.linspace(x_range[0], x_range[1], grid_x)
        grid_y_vals = np.linspace(y_range[0], y_range[1], grid_y)
        grid_xx, grid_yy = np.meshgrid(grid_x_vals, grid_y_vals)
        grid_points = np.stack([grid_xx.flatten(), grid_yy.flatten()], axis=1)
        
        # 插值所有时间步
        grid_data = []
        for t in range(velocity_data.shape[0]):
            velocities_t = velocity_data[t]  # [n_nodes, 2]
            
            # 分别插值u和v分量
            u_interp = griddata(positions, velocities_t[:, 0], 
                               grid_points, method='linear', fill_value=0.0)
            v_interp = griddata(positions, velocities_t[:, 1], 
                               grid_points, method='linear', fill_value=0.0)
            
            # 重塑为网格形状 [grid_y, grid_x, 2]
            u_grid = u_interp.reshape(grid_y, grid_x)
            v_grid = v_interp.reshape(grid_y, grid_x)
            uv_grid = np.stack([u_grid, v_grid], axis=-1)
            
            grid_data.append(uv_grid)
        
        grid_data = np.array(grid_data)  # [T, grid_y, grid_x, 2]
        
        # 保存网格信息
        self.grid_shape = (grid_y, grid_x, 2)
        self.grid_points = grid_points
        self.grid_x_vals = grid_x_vals
        self.grid_y_vals = grid_y_vals
        
        return grid_data
    
    def _prepare_sequences(self):
        """准备MPO-Transformer训练序列"""
        T, grid_y, grid_x, _ = self.grid_data.shape
        sequences = []
        
        for _ in range(self.num_sequences):
            # 随机选择起始时间
            start_t = np.random.randint(0, T - self.seq_len)
            
            # 随机选择空间位置序列
            spatial_coords = []
            time_encoding = []
            query_coords = []
            targets = []
            
            for i in range(self.seq_len):
                t = start_t + i
                # 随机选择网格位置
                y_idx = np.random.randint(0, grid_y)
                x_idx = np.random.randint(0, grid_x)
                
                # 空间坐标 (归一化)
                spatial_coord = np.array([
                    y_idx / (grid_y - 1),  # y坐标
                    x_idx / (grid_x - 1)   # x坐标
                ])
                
                # 时间编码 (归一化)
                time_enc = np.array([t / (T - 1)])
                
                # 查询坐标 (t, y, x) 归一化
                query_coord = np.array([
                    t / (T - 1),          # 时间
                    y_idx / (grid_y - 1), # y坐标  
                    x_idx / (grid_x - 1)  # x坐标
                ])
                
                # 目标值 (u, v)
                target = self.grid_data[t, y_idx, x_idx]  # [2]
                
                spatial_coords.append(spatial_coord)
                time_encoding.append(time_enc)
                query_coords.append(query_coord)
                targets.append(target)
            
            sequence = {
                'spatial_coords': np.array(spatial_coords),      # [seq_len, 2]
                'time_encoding': np.array(time_encoding),        # [seq_len, 1]  
                'query_coords': np.array(query_coords),          # [seq_len, 3]
                'targets': np.array(targets)                     # [seq_len, 2]
            }
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """返回一个训练样本"""
        sequence = self.sequences[idx]
        
        return {
            'spatial_coords': torch.FloatTensor(sequence['spatial_coords']),
            'time_encoding': torch.FloatTensor(sequence['time_encoding']),
            'query_coords': torch.FloatTensor(sequence['query_coords']),
            'targets': torch.FloatTensor(sequence['targets'])
        }
    
    def get_grid_info(self):
        """获取网格信息用于MPO模型初始化"""
        return {
            'grid_shape': self.grid_shape,  # (Y, X, U)
            'grid_size': (self.grid_shape[0], self.grid_shape[1]),
            'time_steps': self.grid_data.shape[0]
        }
    
    def visualize_sample(self, sequence_idx=0, show_grid=False):
        """可视化样本数据"""
        sequence = self.sequences[sequence_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 空间坐标分布
        spatial_coords = sequence['spatial_coords']
        axes[0, 0].scatter(spatial_coords[:, 1], spatial_coords[:, 0], 
                          c=range(len(spatial_coords)), cmap='viridis')
        axes[0, 0].set_title('空间坐标序列')
        axes[0, 0].set_xlabel('x坐标')
        axes[0, 0].set_ylabel('y坐标')
        
        # 时间编码
        time_encoding = sequence['time_encoding']
        axes[0, 1].plot(time_encoding.flatten(), 'o-')
        axes[0, 1].set_title('时间编码序列')
        axes[0, 1].set_xlabel('序列步长')
        axes[0, 1].set_ylabel('归一化时间')
        
        # 目标值u分量
        targets = sequence['targets']
        axes[1, 0].plot(targets[:, 0], 'o-', label='u分量')
        axes[1, 0].set_title('u分量目标值')
        axes[1, 0].set_xlabel('序列步长')
        axes[1, 0].set_ylabel('u值')
        
        # 目标值v分量  
        axes[1, 1].plot(targets[:, 1], 'o-', label='v分量')
        axes[1, 1].set_title('v分量目标值')
        axes[1, 1].set_xlabel('序列步长')
        axes[1, 1].set_ylabel('v值')
        
        plt.tight_layout()
        plt.show()
        
        if show_grid:
            # 显示整个流场
            self._visualize_flow_field(time_step=0)
    
    def _visualize_flow_field(self, time_step=0):
        """可视化流场"""
        flow_field = self.grid_data[time_step]  # [Y, X, 2]
        speed = np.sqrt(flow_field[:, :, 0]**2 + flow_field[:, :, 1]**2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        im1 = axes[0].imshow(speed, cmap='viridis', aspect='auto')
        axes[0].set_title(f'速度大小 (t={time_step})')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(flow_field[:, :, 0], cmap='RdBu_r', aspect='auto')
        axes[1].set_title(f'u分量 (t={time_step})')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(flow_field[:, :, 1], cmap='RdBu_r', aspect='auto')
        axes[2].set_title(f'v分量 (t={time_step})')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

# 使用示例
def create_mpo_transformer_dataloader(config):
    """创建MPO-Transformer数据加载器"""
    
    # 训练数据集
    train_dataset = CylinderFlowMPODataset(
        file_path=config.train_data_path,
        trajectory_indices=config.train_trajectories,
        time_start=config.time_start,
        time_used=config.time_used,
        seq_len=config.seq_len,
        num_sequences=config.num_train_sequences,
        grid_size=config.grid_size
    )
    
    # 测试数据集
    test_dataset = CylinderFlowMPODataset(
        file_path=config.test_data_path,
        trajectory_indices=config.test_trajectories, 
        time_start=config.time_start,
        time_used=config.time_used,
        seq_len=config.seq_len,
        num_sequences=config.num_test_sequences,
        grid_size=config.grid_size
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 获取网格信息用于MPO模型初始化
    grid_info = train_dataset.get_grid_info()
    
    return train_loader, test_loader, grid_info

def main_simple():
    file_path = "train_cf_4x2000x1598x2.h5"
    
    print("加载圆柱绕流数据...")
    
    try:
        # 简单配置
        dataset = CylinderFlowMPODataset(
            file_path=file_path,
            trajectory_indices=[0, 1],  # 使用前2个轨迹
            time_start=0,
            time_used=200,  # 200个时间步
            seq_len=10,     # 序列长度
            num_sequences=500,  # 500个序列
            grid_size=(64, 64)  # 64x64网格
        )
        
        print("数据加载成功!")
        print(f"数据集大小: {len(dataset)}")
        
        # 查看第一个样本
        sample = dataset[0]
        print(f"\n样本数据形状:")
        for key, value in sample.items():
            print(f"  {key}: {value.shape}")
        
        # 可视化
        dataset.visualize_sample(0)
        
        return dataset
        
    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == "__main__":
    
    dataset = main_simple()