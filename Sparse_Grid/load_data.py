# load_data.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata

plt.rcParams['font.family'] = 'SimHei'

class CylinderFlowMPODataset(Dataset):
    """
    圆柱绕流数据集 - 专门用于MPO训练
    输出原始的不规则网格数据
    """
    
    def __init__(self, file_path: str, 
                 trajectory_indices: Optional[List[int]] = None,
                 time_start: int = 0, 
                 time_used: int = 200,
                 normalize: bool = True):
        """
        Args:
            file_path: HDF5文件路径
            trajectory_indices: 使用的轨迹索引
            time_start: 起始时间步
            time_used: 使用的时间步数  
            normalize: 是否归一化
        """
        self.file_path = file_path
        self.normalize = normalize
        self.time_used = time_used
        
        # 加载原始数据
        self.raw_data = self._load_raw_data(trajectory_indices, time_start, time_used)
        
        print(f"数据加载完成: {self.raw_data['velocity_data'].shape}")
        print(f"轨迹数: {len(self.raw_data['trajectory_info'])}")
        print(f"时间步数: {self.raw_data['velocity_data'].shape[1]}")
        print(f"网格点数: {self.raw_data['velocity_data'].shape[2]}")
        
    def _load_raw_data(self, trajectory_indices, time_start, time_used):
        """加载原始HDF5数据"""
        with h5py.File(self.file_path, 'r') as f:
            # 创建节点选择掩码 - 只选择前1584个节点
            node_slice = slice(0, 1584)
            
            # 读取几何信息
            pos = f['pos'][node_slice]  # [1584, 2]
            mesh = f['mesh'][node_slice] if 'mesh' in f else None
            
            # 读取物理参数
            cylinder_center = (f.attrs['x_c'], f.attrs['y_c'])
            cylinder_radius = f.attrs['r']
            domain_bounds = {
                'x_l': f.attrs['x_l'], 'x_r': f.attrs['x_r'],
                'y_b': f.attrs['y_b'], 'y_t': f.attrs['y_t']
            }
            mu = f.attrs['mu']
            rho = f.attrs['rho']
            
            # 读取节点类型并应用相同切片
            node_type = f['node_type']
            node_indices = {
                'inlet': node_type['inlet'][:][node_slice],
                'cylinder': node_type['cylinder'][:][node_slice], 
                'outlet': node_type['outlet'][:][node_slice],
                'inner': node_type['inner'][:][node_slice]
            }
            
            # 确定使用的轨迹
            if trajectory_indices is None:
                available_keys = [k for k in f.keys() if k not in ['pos', 'node_type', 'mesh']]
                trajectory_indices = [int(k) for k in available_keys]
            
            # 收集速度数据 - 按轨迹组织
            all_velocity_data = []
            trajectory_info = []
            
            for idx in trajectory_indices:
                traj_key = str(idx)
                if traj_key in f:
                    traj_group = f[traj_key]
                    # 应用时间掩码和节点掩码 - 只取前1584个节点
                    U = traj_group['U'][time_start:time_start + time_used, node_slice, :]  # [time_used, 1584, 2]
                    
                    # 无量纲化处理
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
                
            # 合并所有轨迹 [n_trajectories, time_used, 1584, 2]
            velocity_data = np.stack(all_velocity_data, axis=0)
            
            return {
                'positions': pos,  # [1584, 2]
                'mesh': mesh,
                'velocity_data': velocity_data,  # [n_trajectories, time_used, 1584, 2]
                'cylinder_center': cylinder_center,
                'cylinder_radius': cylinder_radius,
                'domain_bounds': domain_bounds,
                'node_indices': node_indices,
                'trajectory_info': trajectory_info,
                'physical_params': {'mu': mu, 'rho': rho}
            }
    
    def __len__(self):
        return self.raw_data['velocity_data'].shape[0]  # 轨迹数
    
    def __getitem__(self, idx):
        """返回一个轨迹的所有时间步数据"""
        trajectory_data = self.raw_data['velocity_data'][idx]  # [time_used, 1584, 2]
        
        return {
            'trajectory_data': torch.FloatTensor(trajectory_data),  # [time_used, 1584, 2]
            'trajectory_idx': torch.tensor(idx, dtype=torch.long),
            'positions': torch.FloatTensor(self.raw_data['positions'])  # [1584, 2]
        }
    
    def get_trajectory_info(self, idx):
        """获取指定轨迹的信息"""
        if idx < len(self.raw_data['trajectory_info']):
            return self.raw_data['trajectory_info'][idx]
        return None
    
    def get_dataset_info(self):
        """获取数据集信息"""
        velocity_data = self.raw_data['velocity_data']
        return {
            'num_trajectories': velocity_data.shape[0],
            'time_steps': velocity_data.shape[1],
            'num_nodes': velocity_data.shape[2],
            'positions_shape': self.raw_data['positions'].shape,
            'trajectory_indices': [info['trajectory_idx'] for info in self.raw_data['trajectory_info']]
        }


class SimpleMPODataset(Dataset):
    """
    简化的MPO数据集 - 直接从HDF5文件读取已处理的数据
    假设数据已经是规则网格格式
    """
    
    def __init__(self, file_path: str, 
                 trajectory_indices: List[int] = None,
                 time_start: int = 0,
                 time_used: int = 200):
        """
        Args:
            file_path: HDF5文件路径
            trajectory_indices: 轨迹索引列表
            time_start: 起始时间步
            time_used: 使用的时间步数
        """
        self.file_path = file_path
        self.trajectory_indices = trajectory_indices or [0, 1, 2, 3]
        self.time_start = time_start
        self.time_used = time_used
        
        # 加载数据
        self.data, self.positions, self.trajectory_info = self._load_data()
        
        print(f"数据加载完成: {self.data.shape}")
        print(f"轨迹数: {self.data.shape[0]}")
        print(f"时间步数: {self.data.shape[1]}")
        print(f"网格点数: {self.data.shape[2]}")
    
    def _load_data(self):
        """从HDF5文件加载数据"""
        with h5py.File(self.file_path, 'r') as f:
            # 创建节点选择掩码 - 只选择前1584个节点
            node_slice = slice(0, 1584)
            
            # 读取位置信息
            positions = f['pos'][node_slice]  # [1584, 2]
            
            all_data = []
            trajectory_info = []
            
            for traj_idx in self.trajectory_indices:
                traj_key = str(traj_idx)
                if traj_key in f:
                    # 读取轨迹数据并应用节点掩码
                    traj_group = f[traj_key]
                    traj_data = traj_group['U'][self.time_start:self.time_start + self.time_used, node_slice, :]  # [time_used, 1584, 2]
                    
                    # 无量纲化处理
                    u_m = traj_group.attrs.get('u_m', 1.0)
                    traj_data = traj_data / u_m
                    
                    all_data.append(traj_data)
                    trajectory_info.append({
                        'trajectory_idx': traj_idx,
                        'u_m': u_m,
                        'dt': traj_group.attrs.get('dt', 1.0)
                    })
            
            if not all_data:
                raise ValueError("没有找到可用的轨迹数据")
            
            # 合并所有轨迹 [n_trajectories, time_used, 1584, 2]
            data = np.stack(all_data, axis=0)
            
            return data, positions, trajectory_info
    
    def __len__(self):
        return len(self.data)  # 轨迹数
    
    def __getitem__(self, idx):
        """返回单个轨迹的所有时间步数据"""
        return {
            'trajectory_data': torch.FloatTensor(self.data[idx]),  # [time_used, 1584, 2]
            'trajectory_idx': torch.tensor(idx, dtype=torch.long),
            'positions': torch.FloatTensor(self.positions)  # [1584, 2]
        }
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'num_trajectories': self.data.shape[0],
            'time_steps': self.data.shape[1],
            'num_nodes': self.data.shape[2],
            'positions_shape': self.positions.shape,
            'trajectory_indices': [info['trajectory_idx'] for info in self.trajectory_info]
        }


def create_mpo_dataloader(config: Dict):
    """
    创建MPO数据加载器
    Args:
        config: 配置字典，包含:
            - data_path: 数据文件路径
            - trajectories: 轨迹索引列表
            - time_start: 起始时间步
            - time_used: 使用的时间步数
            - batch_size: 批次大小 (默认=1)
    Returns:
        dataloader: 数据加载器
        dataset_info: 数据集信息
    """
    data_path = config['data_path']
    trajectories = config.get('trajectories', [0, 1, 2, 3])
    time_start = config.get('time_start', 0)
    time_used = config.get('time_used', 200)
    batch_size = config.get('batch_size', 1)
    
    # 使用原始不规则网格数据集
    dataset = CylinderFlowMPODataset(
        file_path=data_path,
        trajectory_indices=trajectories,
        time_start=time_start,
        time_used=time_used
    )
    dataset_info = dataset.get_dataset_info()
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    return dataloader, dataset_info


def create_train_test_dataloaders(config: Dict):
    """
    创建训练和测试数据加载器
    Args:
        config: 配置字典，包含训练和测试配置
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器  
        dataset_info: 数据集信息
    """
    train_config = {
        'data_path': config['train_data_path'],
        'trajectories': config.get('train_trajectories', [0, 1, 2]),
        'time_start': config.get('time_start', 0),
        'time_used': config.get('time_used', 200),
        'batch_size': config.get('batch_size', 1)
    }
    
    test_config = {
        'data_path': config.get('test_data_path', config['train_data_path']),
        'trajectories': config.get('test_trajectories', [3]),
        'time_start': config.get('time_start', 0),
        'time_used': config.get('time_used', 200),
        'batch_size': config.get('batch_size', 1)
    }
    
    train_loader, dataset_info = create_mpo_dataloader(train_config)
    test_loader, _ = create_mpo_dataloader(test_config)
    
    return train_loader, test_loader, dataset_info


def test_data_loading():
    """测试数据加载功能"""
    config = {
        'train_data_path': "train_cf_4x2000x1598x2.h5",
        'train_trajectories': [0, 1, 2],
        'test_trajectories': [3],
        'time_start': 0,
        'time_used': 200,
        'batch_size': 1
    }
    
    print("测试数据加载...")
    
    try:
        train_loader, test_loader, dataset_info = create_train_test_dataloaders(config)
        
        print("数据加载成功!")
        print(f"训练集轨迹数: {len(train_loader.dataset)}")
        print(f"测试集轨迹数: {len(test_loader.dataset)}")
        print(f"数据集信息: {dataset_info}")
        
        # 查看第一个样本
        sample = next(iter(train_loader))
        print(f"\n样本数据:")
        for key, value in sample.items():
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        
        # 验证数据形状
        trajectory_data = sample['trajectory_data']  # [batch, time_steps, num_nodes, 2]
        positions = sample['positions']  # [batch, num_nodes, 2]
        
        print(f"\n数据形状验证:")
        print(f"轨迹数据: {trajectory_data.shape}")
        print(f"位置坐标: {positions.shape}")
        
        return True
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False


if __name__ == "__main__":
    # 测试数据加载
    success = test_data_loading()
    
    if success:
        print("\n数据加载测试通过!")
    else:
        print("\n数据加载测试失败!")