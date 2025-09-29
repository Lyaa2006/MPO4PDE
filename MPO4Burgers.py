import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pyvista as pv
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
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
        #F范数（忽略矩阵的位置，只考虑各个数的大小）
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
        #M.shape为（101，101，2，12，12）；M_flat.shape为（20402，144）
    # 重塑数据 [x, y, v, t]
        target = self.data.transpose(1, 2, 3, 0)
        target_flat = target.reshape(-1, self.t_dim)
    
    # 最小二乘求解
        T_flat, residuals, _, _ = np.linalg.lstsq(M_flat, target_flat, rcond=None)
        self.T = T_flat.T.reshape(self.t_dim, a_dim, d_dim)
    #residual.shape为target_flat.shape[1]
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
        TX = np.einsum('tad,xab->txdb', self.T, self.X)#爱因斯坦契合，字符串暗示维度转换，自动做矩阵乘法
    
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
        U, s, Vh = svd(mat, full_matrices=False)#返回值：左奇异向量、奇异值、右奇异向量的共轭转置
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)#数组累计求和，eg.ans[2]=matrix[0]+matrix[1]+matrix[2]
        trunc_idx = np.sum(cumulative_energy < (1 - threshold**2))
        trunc_idx = max(1, min(trunc_idx, tensor.shape[dim_to_trunc]))
    
        U_trunc = U[:, :trunc_idx]
        s_trunc = s[:trunc_idx]
        Vh_trunc = Vh[:trunc_idx, :]  # 使用Vh

# 完整重建
        new_tensor_flat = U_trunc @ np.diag(s_trunc) @ Vh_trunc#不改变矩阵的形状大小，但是实现了降秩，减小计算开支，相当于去噪
    
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
            Vh_trunc=Vh[:trunc_idx,:]
            new_tensor_flat = U_trunc @ np.diag(s_trunc)@Vh_trunc
        
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
            Vh_trunc=Vh[:trunc_idx,:]
            new_tensor_flat = U_trunc @ np.diag(s_trunc)@Vh_trunc
        
            new_tensor = new_tensor_flat.reshape(self.O.shape[1], self.O.shape[0], self.O.shape[2])
            new_tensor = new_tensor.transpose(1, 0, 2)
    
        return new_tensor, trunc_idx
    
    def fit(self) -> List[float]:
        """执行交替优化训练"""
        losses = []
        
        for iteration in range(self.max_iter):
            # 交替更新各个张量
            #print(iteration)
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


class VTKGraphBuilder:
    def __init__(self, vtk_file_path: str, mpo_model):
        """
        从VTK文件构建图结构
        
        Args:
            vtk_file_path: VTK文件路径
            mpo_model: MPO模型用于提取物理特征
        """
        self.vtk_file_path = vtk_file_path
        self.mpo_model = mpo_model
        self.mesh = None
        self.load_vtk_file()
    
    def load_vtk_file(self):
        """加载VTK文件并提取网格信息"""
        if not os.path.exists(self.vtk_file_path):
            raise FileNotFoundError(f"VTK文件不存在: {self.vtk_file_path}")
        
        self.mesh = pv.read(self.vtk_file_path)
        print(f"成功加载VTK文件: {self.vtk_file_path}")
        print(f"节点数量: {self.mesh.n_points}")
        print(f"单元数量: {self.mesh.n_cells}")
    
    def extract_mesh_topology(self) -> Tuple[np.ndarray, np.ndarray]:
        """提取网格拓扑结构（节点坐标和连接关系）"""
        if self.mesh is None:
            raise ValueError("未加载VTK文件")
        
        # 获取节点坐标
        points = self.mesh.points  # (n_points, 3)
        
        # 获取三角形连接性
        faces = self.mesh.faces
        triangles = self._extract_triangles(faces)
        
        print(f"提取到 {len(triangles)} 个三角形单元")
        return points, triangles
    
    def _extract_triangles(self, faces: np.ndarray) -> np.ndarray:
        """从面数据中提取三角形连接性"""
        triangles = []
        i = 0
        while i < len(faces):
            n_vertices = faces[i]
            if n_vertices == 3:  # 三角形
                triangles.append(faces[i+1:i+4])
            i += n_vertices + 1
        
        return np.array(triangles)
    
    def build_edges_from_triangles(self, triangles: np.ndarray) -> np.ndarray:
        """从三角形连接性构建边索引"""
        edges = set()
        
        for triangle in triangles:
            # 添加三角形的三条边（无向图，确保每个边只添加一次）
            edges.add(tuple(sorted([triangle[0], triangle[1]])))
            edges.add(tuple(sorted([triangle[1], triangle[2]])))
            edges.add(tuple(sorted([triangle[2], triangle[0]])))
        
        # 转换为numpy数组并添加双向边
        edge_list = []
        for edge in edges:
            edge_list.append([edge[0], edge[1]])
            edge_list.append([edge[1], edge[0]])  # 无向图需要双向连接
        
        return np.array(edge_list).T  # (2, n_edges)
    
    def extract_physical_features(self, coordinates: np.ndarray) -> np.ndarray:
        """基于MPO模型提取物理特征"""
        mpo_features = []
        
        for coord in coordinates:
            t, x, y = coord[:3]  # 假设坐标是(t, x, y)格式
            features = self._extract_mpo_features_single(t, x, y)
            mpo_features.append(features)
        
        return np.array(mpo_features)
    
    def _extract_mpo_features_single(self, t: float, x: float, y: float) -> np.ndarray:
        """提取单个节点的MPO特征"""
        t_idx = np.argmin(np.abs(np.linspace(0, 1, self.mpo_model.t_dim) - t))
        x_idx = np.argmin(np.abs(np.linspace(0, 1, self.mpo_model.x_dim) - x))
        y_idx = np.argmin(np.abs(np.linspace(0, 1, self.mpo_model.y_dim) - y))
        
        t_feat = self.mpo_model.T[t_idx, :, :]
        x_feat = self.mpo_model.X[x_idx, :, :]
        y_feat = self.mpo_model.Y[y_idx, :, :]
        
        # 计算张量积特征
        tem1 = np.einsum('ad,ab->abd', t_feat, x_feat)
        tem2 = np.einsum('abd,dc->abcd', tem1, y_feat)
        
        return tem2.flatten()
    
    def build_graph_data(self, targets: Optional[np.ndarray] = None) -> Data:
        """构建完整的图数据对象"""
        # 提取网格拓扑
        points, triangles = self.extract_mesh_topology()
        
        # 构建边索引
        edge_index = self.build_edges_from_triangles(triangles)
        
        # 提取物理特征
        physical_features = self.extract_physical_features(points)
        
        # 组合节点特征：坐标 + 物理特征
        node_features = np.concatenate([points, physical_features], axis=1)
        
        # 创建PyG Data对象
        if targets is not None:
            if len(targets) != len(points):
                raise ValueError("目标值数量与节点数量不匹配")
            y_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            y_tensor = None
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            y=y_tensor,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(points, dtype=torch.float32)  # 保存节点位置信息
        )

class MPOGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GCN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 输出层 - 输出每个节点的u, v
        self.node_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 每个节点输出u, v
        )
        
        # 图级别输出（如果需要）
        self.graph_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 逐层GCN处理
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:  # 不是最后一层
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 节点级别预测
        node_predictions = self.node_output(x)
        
        # 如果是图级别任务，添加全局池化
        if batch is not None and torch.unique(batch).shape[0] > 1:
            graph_embeddings = global_mean_pool(x, batch)
            graph_predictions = self.graph_output(graph_embeddings)
            return node_predictions, graph_predictions
        
        return node_predictions

class MPOGNNTrainer:
    def __init__(self, model, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            if isinstance(predictions, tuple):  # 如果有图级别预测
                node_pred, graph_pred = predictions
                loss = self.criterion(node_pred, batch.y)
            else:
                loss = self.criterion(predictions, batch.y)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch:03d}, Train Loss: {avg_loss:.6f}')
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                predictions = self.model(batch)
                
                if isinstance(predictions, tuple):
                    node_pred, graph_pred = predictions
                    loss = self.criterion(node_pred, batch.y)
                else:
                    loss = self.criterion(predictions, batch.y)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_loss:.6f}')
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs: int = 100, patience: int = 20):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_vtk_mpo_gnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

# 使用示例和工具函数
def create_vtk_graph_builder(vtk_file_path: str, mpo_model) -> VTKGraphBuilder:
    """创建VTK图构建器"""
    return VTKGraphBuilder(vtk_file_path, mpo_model)

def create_mpo_gnn_model(input_dim: int, hidden_dim: int = 128, 
                        num_layers: int = 3, dropout: float = 0.1) -> MPOGNN:
    """创建MPO-GNN模型"""
    return MPOGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

def load_multiple_vtk_graphs(vtk_files: List[str], mpo_model, targets_list: List[np.ndarray]) -> List[Data]:
    """批量加载多个VTK文件构建图数据集"""
    graphs = []
    for vtk_file, targets in zip(vtk_files, targets_list):
        builder = VTKGraphBuilder(vtk_file, mpo_model)
        graph_data = builder.build_graph_data(targets)
        graphs.append(graph_data)
    return graphs

# 示例使用代码
def example_usage():
    """使用示例"""
    # 假设您已经有MPO模型和VTK文件
    mpo_model = ...  # 您的MPO模型
    vtk_file = "path/to/your/file.vtk"
    
    # 创建图构建器
    graph_builder = create_vtk_graph_builder(vtk_file, mpo_model)
    
    # 构建图数据（假设您有目标值）
    targets = np.random.randn(graph_builder.mesh.n_points, 2)  # 示例目标值
    graph_data = graph_builder.build_graph_data(targets)
    
    print(f"图数据信息:")
    print(f"节点特征维度: {graph_data.x.shape}")
    print(f"边数量: {graph_data.edge_index.shape[1]}")
    print(f"目标值维度: {graph_data.y.shape}")
    
    # 创建模型
    input_dim = graph_data.x.shape[1]
    model = create_mpo_gnn_model(input_dim)
    
    # 创建数据加载器
    train_loader = DataLoader([graph_data], batch_size=1, shuffle=True)
    
    # 创建训练器
    trainer = MPOGNNTrainer(model)
    
    # 训练模型
    # trainer.train(train_loader, train_loader, epochs=50)

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from torch_geometric.data import DataLoader
    
    # 读取数据
    df = pd.read_csv("D:\Comsol建模\Burgers数据集.csv", comment='%', header=None)

    # 获取所有列名（实际上是时间步的重复结构）
    columns_per_timestep = 5  # 每个时间步有5列：t, x, y, u, v
    num_timesteps = (df.shape[1] - 2) // columns_per_timestep  # 前两列是X,Y，后面每5列一个时间步

    # 提取X和Y坐标
    X = df.iloc[:, 0].values
    Y = df.iloc[:, 1].values

    # 获取唯一的X和Y值，用于构建网格
    x_unique = np.sort(np.unique(X))
    y_unique = np.sort(np.unique(Y))
    nx = len(x_unique)
    ny = len(y_unique)
    
    # 初始化结果数组：形状为 (num_timesteps, nx, ny, 2)
    result = np.zeros((num_timesteps, nx, ny, 2))

    # 构建坐标到索引的映射
    x_to_idx = {x_val: idx for idx, x_val in enumerate(x_unique)}
    y_to_idx = {y_val: idx for idx, y_val in enumerate(y_unique)}
    
    coordinates = []
    targets = []
    
    # 遍历每个时间步
    for t_idx in range(num_timesteps):
        # 计算当前时间步的列起始位置
        start_col = 2 + t_idx * columns_per_timestep
        # 提取当前时间步的5列数据
        t_data = df.iloc[:, start_col:start_col+5].values
        # 遍历每一行（即每个空间点）
        for row in t_data:
            x_val, y_val, u, v = row[1], row[2], row[3], row[4]  # 第0列是t，忽略
            i = x_to_idx[x_val]
            j = y_to_idx[y_val]
            result[t_idx, i, j, 0] = u
            result[t_idx, i, j, 1] = v
            coordinates.append([row[0], row[1], row[2]])  # t, x, y
            targets.append([u, v])
    
    coordinates = np.array(coordinates)
    targets = np.array(targets)
    
    # MPO预训练过程（保持不变）
    # 初始化MPO分解器
    mpo = TopoMPO(result, bond_dims={
        'd_tx': 12, 'd_ty': 12, 'd_xout': 8, 'd_yout': 8
    })
    
    # 可视化拓扑结构
    fig = mpo.visualize_topology()
    plt.show()
    
    # 执行分解
    print("开始MPO分解...")
    losses = mpo.fit()
    
    # 分析结果
    final_loss = losses[-1]
    print(f"\n结果:")
    print(f"最终损失: {final_loss:.6e}")
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogy(losses)
    plt.title('训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('相对误差')
    plt.grid(True, alpha=0.3)
    
    # 比较原始和重建数据
    time_idx = 20
    original_u = result[time_idx, :, :, 0]
    reconstructed = mpo.reconstruct()
    reconstructed_u = reconstructed[time_idx, :, :, 0]
    
    plt.subplot(1, 3, 2)
    plt.imshow(original_u, cmap='RdBu_r', aspect='auto')
    plt.title('原始 u')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_u, cmap='RdBu_r', aspect='auto')
    plt.title('重建 u')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # ========== 修改部分：使用VTK网格构建图 ==========
    
    # 首先需要从COMSOL导出VTK文件，这里假设您已经导出了VTK文件
    vtk_file_path = "D:\Comsol建模\不规则网格下的二维Burgers.vtu"  # 请修改为实际的VTK文件路径
    
    # 检查VTK文件是否存在，如果不存在则创建示例网格
    if not os.path.exists(vtk_file_path):
        print(f"VTK文件不存在: {vtk_file_path}")
        print("创建示例矩形网格...")
        
        # 创建示例网格（基于您的坐标数据）
        import pyvista as pv
        
        # 创建网格点
        xx, yy = np.meshgrid(x_unique, y_unique, indexing='ij')
        points = np.column_stack([xx.flatten(), yy.flatten(), np.zeros_like(xx.flatten())])
        
        # 创建三角形网格
        n_x, n_y = len(x_unique), len(y_unique)
        faces = []
        for i in range(n_x - 1):
            for j in range(n_y - 1):
                # 第一个三角形
                v0 = i * n_y + j
                v1 = (i + 1) * n_y + j
                v2 = i * n_y + j + 1
                faces.extend([3, v0, v1, v2])
                
                # 第二个三角形
                v0 = (i + 1) * n_y + j
                v1 = (i + 1) * n_y + j + 1
                v2 = i * n_y + j + 1
                faces.extend([3, v0, v1, v2])
        
        faces = np.array(faces, dtype=np.int32)
        
        # 创建PyVista网格
        mesh = pv.PolyData(points, faces)
        
        # 保存为VTK文件
        mesh.save(vtk_file_path)
        print(f"已创建示例网格并保存到: {vtk_file_path}")
    
    # 使用VTKGraphBuilder构建图
    print("使用VTK网格构建图结构...")
    graph_builder = VTKGraphBuilder(vtk_file_path, mpo)
    graph_data = graph_builder.build_graph_data(targets)
    
    print(f"图数据信息:")
    print(f"节点数: {graph_data.num_nodes}")
    print(f"边数: {graph_data.num_edges}")
    print(f"节点特征维度: {graph_data.num_node_features}")
    print(f"边索引形状: {graph_data.edge_index.shape}")
    
    # 可视化图结构（可选）
    if graph_data.num_nodes <= 1000:  # 避免节点太多导致可视化困难
        try:
            import networkx as nx
            from torch_geometric.utils import to_networkx
            
            G = to_networkx(graph_data, to_undirected=True)
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, node_size=20, width=0.5, alpha=0.8)
            plt.title("图结构可视化")
            plt.show()
        except ImportError:
            print("NetworkX未安装，跳过图可视化")
    
    # 创建模型 - 使用VTKGraphBuilder构建的特征维度
    input_dim = graph_data.num_node_features
    model = create_mpo_gnn_model(input_dim, hidden_dim=128, num_layers=3, dropout=0.1)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    
    # 由于是单个图，我们需要创建多个图实例用于训练验证划分
    # 这里简单地将节点划分为训练集和验证集
    train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    
    indices = np.arange(graph_data.num_nodes)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    
    # 创建数据加载器
    train_loader = DataLoader([graph_data], batch_size=1, shuffle=True)
    val_loader = DataLoader([graph_data], batch_size=1, shuffle=False)
    
    # 训练模型
    print("开始训练MPO-GNN模型...")
    trainer = MPOGNNTrainer(model)
    trainer.train(train_loader, val_loader, epochs=100, patience=20)
    
    # 预测和评估
    model.eval()
    with torch.no_grad():
        predictions = model(graph_data)
        
        # 计算训练集和验证集的误差
        train_pred = predictions[graph_data.train_mask]
        train_true = graph_data.y[graph_data.train_mask]
        train_rmse = torch.sqrt(torch.mean((train_pred - train_true) ** 2))
        
        val_pred = predictions[graph_data.val_mask]
        val_true = graph_data.y[graph_data.val_mask]
        val_rmse = torch.sqrt(torch.mean((val_pred - val_true) ** 2))
        
        print(f"训练集RMSE: {train_rmse:.6f}")
        print(f"验证集RMSE: {val_rmse:.6f}")
        
        # 可视化部分预测结果
        plt.figure(figsize=(15, 5))
        
        # 原始目标值
        plt.subplot(1, 3, 1)
        plt.scatter(graph_data.pos[graph_data.train_mask, 0].numpy(), 
                   graph_data.pos[graph_data.train_mask, 1].numpy(), 
                   c=graph_data.y[graph_data.train_mask, 0].numpy(), 
                   cmap='RdBu_r', s=10)
        plt.colorbar()
        plt.title('训练集真实值 (u)')
        
        # 预测值
        plt.subplot(1, 3, 2)
        plt.scatter(graph_data.pos[graph_data.train_mask, 0].numpy(), 
                   graph_data.pos[graph_data.train_mask, 1].numpy(), 
                   c=predictions[graph_data.train_mask, 0].numpy(), 
                   cmap='RdBu_r', s=10)
        plt.colorbar()
        plt.title('训练集预测值 (u)')
        
        # 误差
        plt.subplot(1, 3, 3)
        error = (predictions[graph_data.train_mask, 0] - graph_data.y[graph_data.train_mask, 0]).abs().numpy()
        plt.scatter(graph_data.pos[graph_data.train_mask, 0].numpy(), 
                   graph_data.pos[graph_data.train_mask, 1].numpy(), 
                   c=error, cmap='Reds', s=10)
        plt.colorbar()
        plt.title('预测误差 (u)')
        
        plt.tight_layout()
        plt.show()