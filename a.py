import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class TopoMPO:
    def __init__(self, data: np.ndarray, bond_dims: Dict[str, int] = None, 
                 tolerance: float = 1e-6, max_iter: int = 100):
        """
        拓扑MPO分解器 - 专为Burgers方程设计
        
        参数:
        data: 4维张量 [t, x, y, 2] (时间, x空间, y空间, 变量u,v)
        bond_dims: 键维度配置
        tolerance: 收敛容差
        max_iter: 最大迭代次数
        """
        self.data = data
        self.t_dim, self.x_dim, self.y_dim, self.var_dim = data.shape
        
        if bond_dims is None:
            bond_dims = {
                'd_tx': min(self.t_dim, self.x_dim, 16),  # t-x连接
                'd_ty': min(self.t_dim, self.y_dim, 16),  # t-y连接  
                'd_xout': min(self.x_dim, self.var_dim, 8),  # x-输出连接
                'd_yout': min(self.y_dim, self.var_dim, 8)   # y-输出连接
            }
        self.bond_dims = bond_dims
        self.tol = tolerance
        self.max_iter = max_iter
        
        # 初始化核心张量
        self.initialize_tensors()
        
    def initialize_tensors(self):
        """初始化拓扑结构核心张量"""
        # T: 时间张量 [t, d_tx, d_ty]
        self.T = np.random.randn(self.t_dim, self.bond_dims['d_tx'], self.bond_dims['d_ty'])
        
        # X: x-空间张量 [x, d_tx, d_xout] 
        self.X = np.random.randn(self.x_dim, self.bond_dims['d_tx'], self.bond_dims['d_xout'])
        print(self.X.shape)
        # Y: y-空间张量 [y, d_ty, d_yout]
        self.Y = np.random.randn(self.y_dim, self.bond_dims['d_ty'], self.bond_dims['d_yout'])
        
        # O: 输出张量 [d_xout, d_yout, var]
        self.O = np.random.randn(self.bond_dims['d_xout'], self.bond_dims['d_yout'], self.var_dim)
        
        # 归一化
        for tensor in [self.T, self.X, self.Y, self.O]:
            tensor /= np.linalg.norm(tensor)
    
    def reconstruct(self) -> np.ndarray:
        """从MPO重建完整张量"""
        # 使用简洁的下标命名：
        # T: [t, a, d] 其中 a=d_tx, d=d_ty
        # X: [x, a, b] 其中 a=d_tx, b=d_xout
        # Y: [y, d, c] 其中 d=d_ty, c=d_yout
        # O: [b, c, v] 其中 b=d_xout, c=d_yout, v=var
        
        # T × X: [t, a, d] × [x, a, b] → [t, x, d, b]
        temp1 = np.einsum('tad,xab->txdb', self.T, self.X)
        
        # temp1 × Y: [t, x, d, b] × [y, d, c] → [t, x, y, b, c]
        temp2 = np.einsum('txdb,ydc->txybc', temp1, self.Y)
        
        # temp2 × O: [t, x, y, b, c] × [b, c, v] → [t, x, y, v]
        reconstructed = np.einsum('txybc,bcv->txyv', temp2, self.O)
        
        return reconstructed
    
    def loss_function(self) -> float:
        """计算重建误差"""
        reconstructed = self.reconstruct()
        # F范数（忽略矩阵的位置，只考虑各个数的大小）
        return np.linalg.norm(self.data - reconstructed) ** 2 / np.linalg.norm(self.data) ** 2
    
    def update_tensor_T(self) -> float:
        """固定X,Y,O，更新时间张量T"""
        # 获取当前实际维度
        x_dim, a_dim, b_dim = self.X.shape  # X: [x, a, b]
        _, c_dim, v_dim = self.O.shape      # O: [b, c, v]
        y_dim, d_dim, c_dim_y = self.Y.shape # Y: [y, d, c]
        
        # 检查维度一致性
        assert b_dim == self.O.shape[0], f"X的b维度({b_dim})与O的b维度({self.O.shape[0]})不匹配"
        assert c_dim == c_dim_y, f"O的c维度({c_dim})与Y的c维度({c_dim_y})不匹配"
        
        # X × O: [x, a, b] × [b, c, v] -> [x, a, c, v]
        XO = np.einsum('xab,bcv->xacv', self.X, self.O)
        
        # XO × Y: [x, a, c, v] × [y, d, c] -> [x, y, a, d, v]
        XOY = np.einsum('xacv,ydc->xyadv', XO, self.Y)
        
        # 重塑为目标形式 [x, y, v, a, d]
        M = XOY.transpose(0, 1, 4, 2, 3)
        M_flat = M.reshape(-1, a_dim * d_dim)
        
        # 重塑数据 [x, y, v, t]
        target = self.data.transpose(1, 2, 3, 0)
        target_flat = target.reshape(-1, self.t_dim)
        
        # 最小二乘求解
        T_flat, residuals, _, _ = np.linalg.lstsq(M_flat, target_flat, rcond=None)
        self.T = T_flat.T.reshape(self.t_dim, a_dim, d_dim)
        
        return np.mean(residuals) if len(residuals) > 0 else 0

    def update_tensor_X(self) -> float:
        """固定T,Y,O，更新x-空间张量X"""
        # 获取当前实际维度
        t_dim, a_dim, d_dim = self.T.shape  # T: [t, a, d]
        y_dim, d_dim_y, c_dim = self.Y.shape # Y: [y, d, c]
        b_dim, c_dim_o, v_dim = self.O.shape # O: [b, c, v]
        
        # 检查维度一致性
        assert d_dim == d_dim_y, f"T的d维度({d_dim})与Y的d维度({d_dim_y})不匹配"
        assert c_dim == c_dim_o, f"Y的c维度({c_dim})与O的c维度({c_dim_o})不匹配"
        
        # T × Y: [t, a, d] × [y, d, c] -> [t, y, a, c]
        TY = np.einsum('tad,ydc->tyac', self.T, self.Y)
        
        # TY × O: [t, y, a, c] × [b, c, v] -> [t, y, a, b, v]
        TYO = np.einsum('tyac,bcv->tyabv', TY, self.O)
        
        # 重塑为目标形式 [t, y, v, a, b]
        M = TYO.transpose(0, 1, 4, 2, 3)
        M_flat = M.reshape(-1, a_dim * b_dim)
        
        # 重塑数据 [t, y, v, x]
        target = self.data.transpose(0, 2, 3, 1)
        target_flat = target.reshape(-1, self.x_dim)
        
        # 最小二乘求解
        X_flat, residuals, _, _ = np.linalg.lstsq(M_flat, target_flat, rcond=None)
        self.X = X_flat.T.reshape(self.x_dim, a_dim, b_dim)
        
        return np.mean(residuals) if len(residuals) > 0 else 0

    def update_tensor_Y(self) -> float:
        """固定T,X,O，更新y-空间张量Y"""
        # 获取当前实际维度
        t_dim, a_dim, d_dim = self.T.shape  # T: [t, a, d]
        x_dim, a_dim_x, b_dim = self.X.shape # X: [x, a, b]
        b_dim_o, c_dim, v_dim = self.O.shape # O: [b, c, v]
        
        # 检查维度一致性
        assert a_dim == a_dim_x, f"T的a维度({a_dim})与X的a维度({a_dim_x})不匹配"
        assert b_dim == b_dim_o, f"X的b维度({b_dim})与O的b维度({b_dim_o})不匹配"
        
        # T × X: [t, a, d] × [x, a, b] -> [t, x, d, b]
        TX = np.einsum('tad,xab->txdb', self.T, self.X)
        
        # TX × O: [t, x, d, b] × [b, c, v] -> [t, x, d, c, v]
        TXO = np.einsum('txdb,bcv->txdcv', TX, self.O)
        
        # 重塑为目标形式 [t, x, v, d, c]
        M = TXO.transpose(0, 1, 4, 2, 3)
        M_flat = M.reshape(-1, d_dim * c_dim)
        
        # 重塑数据 [t, x, v, y]
        target = self.data.transpose(0, 1, 3, 2)
        target_flat = target.reshape(-1, self.y_dim)
        
        # 最小二乘求解
        Y_flat, residuals, _, _ = np.linalg.lstsq(M_flat, target_flat, rcond=None)
        self.Y = Y_flat.T.reshape(self.y_dim, d_dim, c_dim)
        
        return np.mean(residuals) if len(residuals) > 0 else 0
    
    def update_tensor_O(self) -> float:
        """固定T,X,Y，更新输出张量O"""
        # 获取当前实际维度
        t_dim, a_dim, d_dim = self.T.shape      # T: [t, a, d]
        x_dim, a_dim_x, b_dim = self.X.shape    # X: [x, a, b]
        y_dim, d_dim_y, c_dim = self.Y.shape    # Y: [y, d, c]
        
        # 检查维度一致性
        assert a_dim == a_dim_x, f"T的a维度({a_dim})与X的a维度({a_dim_x})不匹配"
        assert d_dim == d_dim_y, f"T的d维度({d_dim})与Y的d维度({d_dim_y})不匹配"
        
        # T × X: [t, a, d] × [x, a, b] -> [t, x, d, b]
        TX = np.einsum('tad,xab->txdb', self.T, self.X)
        
        # TX × Y: [t, x, d, b] × [y, d, c] -> [t, x, y, b, c]
        TXY = np.einsum('txdb,ydc->txybc', TX, self.Y)
        
        # 重塑为 [t*x*y, b*c]
        M_flat = TXY.reshape(-1, b_dim * c_dim)
        
        # 数据重塑为 [t*x*y, v]
        target_flat = self.data.reshape(-1, self.var_dim)
        
        # 最小二乘求解
        O_flat, residuals, _, _ = np.linalg.lstsq(M_flat, target_flat, rcond=None)
        
        # 重塑回输出张量形状 [b, c, v]
        self.O = O_flat.T.reshape(b_dim, c_dim, self.var_dim)
        
        return np.mean(residuals) if len(residuals) > 0 else 0
    
    def truncate_bond(self, tensor: np.ndarray, bond_name: str, threshold: float = 1e-3) -> Tuple[np.ndarray, int]:
        """对称的截断函数，包含对O张量的特殊处理"""
        
        # 对于O张量，需要特殊处理
        if tensor is self.O:
            return self.truncate_O_bond(bond_name, threshold)
        
        # 其他张量的截断逻辑
        trunc_dim_map = {
            'd_tx': 1,      # T张量的第1个维度 (a维度)
            'd_ty': 2,      # T张量的第2个维度 (d维度)
            'd_xout': 2,    # X张量的第2个维度 (b维度)
            'd_yout': 2,    # Y张量的第2个维度 (c维度)
        }
        
        dim_to_trunc = trunc_dim_map[bond_name]
        
        # 将目标维度移到最后
        if dim_to_trunc == 0:
            mat = tensor.reshape(tensor.shape[0], -1)
        elif dim_to_trunc == 1:
            mat = tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
        else:  # dim_to_trunc == 2
            mat = tensor.reshape(-1, tensor.shape[2])
        
        # SVD截断
        U, s, Vh = svd(mat, full_matrices=False)
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        trunc_idx = np.sum(cumulative_energy < (1 - threshold**2))
        trunc_idx = max(1, min(trunc_idx, tensor.shape[dim_to_trunc]))
        
        U_trunc = U[:, :trunc_idx]
        s_trunc = s[:trunc_idx]
        Vh_trunc = Vh[:trunc_idx, :]
        
        # 完整重建
        new_tensor_flat = U_trunc @ np.diag(s_trunc) @ Vh_trunc
        
        # 重塑回原始结构
        if dim_to_trunc == 0:
            new_tensor = new_tensor_flat.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2])
        elif dim_to_trunc == 1:
            new_tensor = new_tensor_flat.reshape(tensor.shape[1], tensor.shape[0], tensor.shape[2])
            new_tensor = new_tensor.transpose(1, 0, 2)
        else:  # dim_to_trunc == 2
            new_tensor = new_tensor_flat.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2])
        print(trunc_idx)
        return new_tensor, trunc_idx

    def truncate_O_bond(self, bond_name: str, threshold: float = 1e-3) -> Tuple[np.ndarray, int]:
        """专门处理O张量的截断"""
        if bond_name == 'd_xout':
            # 截断第一个维度 (b维度)
            mat = self.O.reshape(self.O.shape[0], -1)
            U, s, Vh = svd(mat, full_matrices=False)
            
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            trunc_idx = np.sum(cumulative_energy < (1 - threshold**2))
            trunc_idx = max(1, min(trunc_idx, self.O.shape[0]))
            
            U_trunc = U[:, :trunc_idx]
            s_trunc = s[:trunc_idx]
            Vh_trunc = Vh[:trunc_idx, :]
            new_tensor_flat = U_trunc @ np.diag(s_trunc) @ Vh_trunc
            
            new_tensor = new_tensor_flat.reshape(self.O.shape[0], self.O.shape[1], self.O.shape[2])
            
        else:  # 'd_yout'
            # 截断第二个维度 (c维度)
            mat = self.O.transpose(1, 0, 2).reshape(self.O.shape[1], -1)
            U, s, Vh = svd(mat, full_matrices=False)
            
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            trunc_idx = np.sum(cumulative_energy < (1 - threshold**2))
            trunc_idx = max(1, min(trunc_idx, self.O.shape[1]))
            
            U_trunc = U[:, :trunc_idx]
            s_trunc = s[:trunc_idx]
            Vh_trunc = Vh[:trunc_idx, :]
            new_tensor_flat = U_trunc @ np.diag(s_trunc) @ Vh_trunc
            
            new_tensor = new_tensor_flat.reshape(self.O.shape[1], self.O.shape[0], self.O.shape[2])
            new_tensor = new_tensor.transpose(1, 0, 2)
        
        return new_tensor, trunc_idx
    
    def fit(self) -> List[float]:
        """执行交替优化训练"""
        losses = []
        
        for iteration in range(self.max_iter):
            # 交替更新各个张量
            loss_T = self.update_tensor_T()
            loss_X = self.update_tensor_X()
            loss_Y = self.update_tensor_Y() 
            loss_O = self.update_tensor_O()
            
            current_loss = self.loss_function()
            losses.append(current_loss)
            
            print(f"Iteration {iteration+1:3d}, Loss: {current_loss:.6e}")
            
            # 检查收敛
            if iteration > 0 and abs(losses[-1] - losses[-2]) < self.tol:
                print(f"收敛于第 {iteration+1} 次迭代")
                break
            
            # 定期截断
            if (iteration + 1) % 20 == 0:
                self.adaptive_truncation()
                
        return losses
    
    def adaptive_truncation(self, threshold: float = 1e-3):
        """自适应键维度截断"""
        print("执行自适应截断...")
        
        # 先截断X和Y
        self.X, new_d_xout = self.truncate_bond(self.X, 'd_xout', threshold)
        self.Y, new_d_yout = self.truncate_bond(self.Y, 'd_yout', threshold)
        
        # 然后截断O，使用X和Y截断后的维度作为参考
        # 确保O的维度与X和Y匹配
        target_b_dim = self.X.shape[2]  # X的b维度
        target_c_dim = self.Y.shape[2]  # Y的c维度
        
        # 调整O张量以匹配目标维度
        if self.O.shape[0] != target_b_dim or self.O.shape[1] != target_c_dim:
            new_O = np.zeros((target_b_dim, target_c_dim, self.O.shape[2]))
            min_b = min(self.O.shape[0], target_b_dim)
            min_c = min(self.O.shape[1], target_c_dim)
            new_O[:min_b, :min_c, :] = self.O[:min_b, :min_c, :]
            self.O = new_O
        
        # 更新键维度
        self.bond_dims.update({
            'd_tx': self.X.shape[1],
            'd_ty': self.Y.shape[1],
            'd_xout': target_b_dim,
            'd_yout': target_c_dim
        })
    
    def visualize_topology(self):
        """可视化拓扑结构"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # 绘制拓扑节点
        nodes = {
            'T': (0.5, 0.7, '时间', 'red'),
            'X': (0.3, 0.3, 'X空间', 'blue'), 
            'Y': (0.7, 0.3, 'Y空间', 'green'),
            'O': (0.5, 0.1, '输出(u,v)', 'purple')
        }
        
        # 绘制连接
        connections = [
            ('T', 'X', 'd_tx'),
            ('T', 'Y', 'd_ty'),
            ('X', 'O', 'd_xout'),
            ('Y', 'O', 'd_yout')
        ]
        
        for start, end, bond in connections:
            x1, y1, _, _ = nodes[start]
            x2, y2, _, _ = nodes[end]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=2)
            ax.text((x1+x2)/2, (y1+y2)/2+0.02, bond, ha='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        # 绘制节点
        for node, (x, y, label, color) in nodes.items():
            ax.scatter(x, y, s=300, c=color, alpha=0.8, edgecolors='black')
            ax.text(x, y, label, ha='center', va='center', fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        ax.set_title('MPO拓扑结构: t→x, t→y, x→输出, y→输出')
        ax.axis('off')
        
        return fig

    def get_tensors(self) -> Dict[str, np.ndarray]:
        """获取所有张量"""
        return {
            'T': self.T,
            'X': self.X, 
            'Y': self.Y,
            'O': self.O
        }


class MPOLinear(nn.Module):
    """基于预训练MPO分解的线性层"""
    
    def __init__(self, in_features: int, out_features: int, mpo_model: TopoMPO, 
                 mode: str = "spatial", use_bias: bool = False):
        """
        参数:
        in_features: 输入特征维度
        out_features: 输出特征维度  
        mpo_model: 预训练的TopoMPO模型
        mode: 使用模式 - "spatial"(空间特征) 或 "temporal"(时间特征)
        use_bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mpo_model = mpo_model
        self.mode = mode
        self.use_bias = use_bias
        
        # 根据模式选择不同的MPO张量组合方式
        if mode == "spatial":
            # 使用X和Y空间张量构建权重
            self._init_spatial_weights()
        elif mode == "temporal":
            # 使用T时间张量构建权重  
            self._init_temporal_weights()
        else:
            raise ValueError("mode必须是 'spatial' 或 'temporal'")
            
        # 初始化偏置
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def _init_spatial_weights(self):
        """使用空间张量X和Y初始化适配器参数"""
        # X: [x_dim, d_tx, d_xout], Y: [y_dim, d_ty, d_yout]
        x_dim, d_tx, d_xout = self.mpo_model.X.shape
        y_dim, d_ty, d_yout = self.mpo_model.Y.shape
        
        # 创建适配器矩阵，将MPO空间特征映射到线性层权重
        self.adapter_in = nn.Parameter(torch.randn(self.in_features, x_dim + y_dim))
        self.adapter_out = nn.Parameter(torch.randn(d_xout * d_yout, self.out_features))
        
        # 初始化适配器参数
        nn.init.xavier_uniform_(self.adapter_in)
        nn.init.xavier_uniform_(self.adapter_out)
    
    def _init_temporal_weights(self):
        """使用时间张量T初始化适配器参数"""
        # T: [t_dim, d_tx, d_ty]
        t_dim, d_tx, d_ty = self.mpo_model.T.shape
        
        # 创建适配器矩阵，将MPO时间特征映射到线性层权重
        self.adapter_in = nn.Parameter(torch.randn(self.in_features, t_dim))
        self.adapter_out = nn.Parameter(torch.randn(d_tx * d_ty, self.out_features))
        
        # 初始化适配器参数
        nn.init.xavier_uniform_(self.adapter_in)
        nn.init.xavier_uniform_(self.adapter_out)
    
    def _get_spatial_features(self) -> torch.Tensor:
        """从MPO模型提取空间特征"""
        # 获取X和Y张量的特征化表示
        X_features = self.mpo_model.X.reshape(self.mpo_model.X.shape[0], -1)  # [x_dim, d_tx*d_xout]
        Y_features = self.mpo_model.Y.reshape(self.mpo_model.Y.shape[0], -1)  # [y_dim, d_ty*d_yout]
        
        # 组合空间特征
        spatial_features = torch.cat([X_features, Y_features], dim=0)  # [x_dim+y_dim, features]
        return torch.tensor(spatial_features, dtype=torch.float32)
    
    def _get_temporal_features(self) -> torch.Tensor:
        """从MPO模型提取时间特征"""
        # 获取T张量的特征化表示
        T_features = self.mpo_model.T.reshape(self.mpo_model.T.shape[0], -1)  # [t_dim, d_tx*d_ty]
        return torch.tensor(T_features, dtype=torch.float32)
    
    def _get_output_features(self) -> torch.Tensor:
        """从MPO模型提取输出特征"""
        # 使用O张量或其他组合来构建输出特征
        if self.mode == "spatial":
            # 对于空间模式，使用O张量的特征
            O_features = self.mpo_model.O.reshape(-1, self.mpo_model.O.shape[-1])  # [d_xout*d_yout, var_dim]
            return torch.tensor(O_features, dtype=torch.float32)
        else:
            # 对于时间模式，使用T张量的展平特征
            T_flat = self.mpo_model.T.reshape(self.mpo_model.T.shape[0], -1)  # [t_dim, d_tx*d_ty]
            return torch.tensor(T_flat.T, dtype=torch.float32)  # [d_tx*d_ty, t_dim]
    
    def reconstruct_weight(self) -> torch.Tensor:
        """从MPO特征重建权重矩阵"""
        if self.mode == "spatial":
            # 获取空间特征
            spatial_features = self._get_spatial_features()  # [x_dim+y_dim, features]
            
            # 通过适配器生成权重
            # adapter_in: [in_features, x_dim+y_dim]
            # spatial_features: [x_dim+y_dim, features]  
            # 中间特征: [in_features, features]
            intermediate = self.adapter_in @ spatial_features.T
            
            # 获取输出特征并应用输出适配器
            output_features = self._get_output_features()  # [d_xout*d_yout, var_dim]
            weight = intermediate @ output_features @ self.adapter_out
            
        else:  # temporal mode
            # 获取时间特征
            temporal_features = self._get_temporal_features()  # [t_dim, d_tx*d_ty]
            
            # 通过适配器生成权重
            # adapter_in: [in_features, t_dim]
            # temporal_features: [t_dim, d_tx*d_ty]
            intermediate = self.adapter_in @ temporal_features.T
            
            # 应用输出适配器
            weight = intermediate @ self.adapter_out
            
        return weight  # [in_features, out_features]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        weight = self.reconstruct_weight()  # [in_features, out_features]
        
        if self.bias is not None:
            return F.linear(x, weight.T, self.bias)
        else:
            return F.linear(x, weight.T)
    
    def get_parameter_stats(self) -> Dict[str, float]:
        """获取参数统计信息"""
        # 计算MPO张量的参数数量
        mpo_tensors = self.mpo_model.get_tensors()
        mpo_params = sum(tensor.size for tensor in mpo_tensors.values())
        
        adapter_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        original_params = self.in_features * self.out_features
        
        return {
            'original_params': original_params,
            'mpo_params': mpo_params,
            'adapter_params': adapter_params,
            'total_params': mpo_params + adapter_params,
            'expansion_ratio': (mpo_params + adapter_params) / original_params
        }


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#相当于10000^(2i/d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)#放到缓冲区，因为这一层的参数不用训练
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MPOTransformerEncoderLayer(nn.Module):
    """使用MPO的Transformer编码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, pretrained_mpo: TopoMPO = None, use_mpo: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pretrained_mpo = pretrained_mpo
        
        # 前馈网络 - 使用MPO或普通线性层
        if use_mpo and pretrained_mpo is not None:
            self.linear1 = MPOLinear(d_model, dim_feedforward, mpo_model=self.pretrained_mpo, mode="spatial", use_bias=False)
            self.linear2 = MPOLinear(dim_feedforward, d_model, mpo_model=self.pretrained_mpo, mode="spatial", use_bias=False)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src


class BurgersTransformer(nn.Module):
    """用于Burgers方程的Transformer模型"""
    
    def __init__(self, input_dim: int = 3, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 512, 
                 dropout: float = 0.1, use_mpo: bool = True, data_path: str = None):
        super().__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 加载预训练的MPO模型
        if use_mpo and data_path:
            print("加载并训练MPO模型...")
            mpo_data = load_burgers_data4MPO(data_path)
            mpo = TopoMPO(mpo_data, bond_dims={
                'd_tx': min(12, mpo_data.shape[0], mpo_data.shape[1]),
                'd_ty': min(12, mpo_data.shape[0], mpo_data.shape[2]),
                'd_xout': min(8, mpo_data.shape[1], mpo_data.shape[3]),
                'd_yout': min(8, mpo_data.shape[2], mpo_data.shape[3])
            })
            losses = mpo.fit()
            print("MPO训练完成")
        else:
            mpo = None
        
        # Transformer编码器层
        self.layers = nn.ModuleList([
            MPOTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, mpo, use_mpo)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, 2)  # 输出u, v
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_dim] 
            input_dim: [t, x, y] 或 [x, y, t] 根据数据格式
        """
        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 输出投影 - 对每个时间步预测u,v
        output = self.output_projection(x)
        
        return output


class BurgersDataset(torch.utils.data.Dataset):
    """Burgers方程数据集"""
    
    def __init__(self, data_tensor: torch.Tensor, sequence_length: int = 10):
        self.data = data_tensor  # [batch, time, x, y, features]
        self.sequence_length = sequence_length
        
    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.sequence_length)
    
    def __getitem__(self, idx):
        batch_idx = idx // (self.data.shape[1] - self.sequence_length)
        time_idx = idx % (self.data.shape[1] - self.sequence_length)
        
        # 输入: 当前时刻的时空坐标 [x, y, t]
        # 输出: 下一时刻的物理场 [u, v]
        input_seq = self.data[batch_idx, time_idx:time_idx+self.sequence_length, :, :, :3]  # [x, y, t]
        target_seq = self.data[batch_idx, time_idx+1:time_idx+self.sequence_length+1, :, :, 3:]  # [u, v]
        
        return input_seq, target_seq


def load_burgers_data4MPO(data_path: str) -> np.ndarray:
    """加载Burgers数据用于MPO训练"""
    # 这里应该是你的数据加载逻辑
    # 返回形状为 [t, x, y, 2] 的numpy数组
    # 示例: 创建随机数据
    t_dim, x_dim, y_dim = 50, 32, 32
    data = np.random.randn(t_dim, x_dim, y_dim, 2)
    return data


def load_burgers_data(data_path: str) -> torch.Tensor:
    """加载Burgers数据用于Transformer训练"""
    # 这里应该是你的数据加载逻辑
    # 返回形状为 [batch, time, x, y, features] 的tensor
    # features: [x, y, t, u, v]
    batch_size, time_steps, x_dim, y_dim = 10, 100, 32, 32
    data = torch.randn(batch_size, time_steps, x_dim, y_dim, 5)
    return data


def train_model():
    """训练模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_path = "path/to/your/data"  # 替换为实际路径
    data = load_burgers_data(data_path)
    
    # 创建数据集和数据加载器
    dataset = BurgersDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    mpo_model = BurgersTransformer(
        input_dim=4,  # [u, v, x_norm, y_norm]
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        use_mpo=True,
        data_path=data_path
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(mpo_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    criterion = nn.MSELoss()
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        mpo_model.train()
        total_loss = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # 重塑输入序列
            batch_size, seq_len, x_dim, y_dim, input_features = input_seq.shape
            input_seq = input_seq.reshape(batch_size, seq_len * x_dim * y_dim, input_features)
            
            # 重塑目标序列
            _, _, _, _, output_features = target_seq.shape
            target_seq = target_seq.reshape(batch_size, seq_len * x_dim * y_dim, output_features)
            
            optimizer.zero_grad()
            output = mpo_model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch:03d} | Batch: {batch_idx:03d} | Loss: {loss.item():.6f}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch:03d} | Average Loss: {avg_loss:.6f}')


if __name__ == "__main__":
    train_model()