import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import os


class TopoMPO:
    def __init__(self, data: np.ndarray, bond_dims: Dict[str, int] = None, 
                 tolerance: float = 1e-6, max_iter: int = 10000):
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


def load_burgers_data(csv_file_path: str) -> np.ndarray:
    """从CSV文件加载Burgers方程数据"""
    df = pd.read_csv(csv_file_path, comment='%', header=None)

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
    
    return result


def main():
    """主函数：演示MPO模型的使用"""
    # 加载数据
    print("加载Burgers方程数据...")
    csv_file_path = "D:\Comsol建模\Burgers数据集.csv"
    data = load_burgers_data(csv_file_path)
    
    print(f"数据形状: {data.shape}")
    print(f"时间步数: {data.shape[0]}")
    print(f"空间网格: {data.shape[1]} x {data.shape[2]}")
    print(f"变量数: {data.shape[3]}")
    
    # 初始化MPO分解器
    mpo = TopoMPO(data, bond_dims={
        'd_tx': 12, 'd_ty': 12, 'd_xout': 8, 'd_yout': 8
    })
    
    # 可视化拓扑结构
    print("可视化MPO拓扑结构...")
    fig = mpo.visualize_topology()
    plt.show()
    
    # 执行分解
    print("开始MPO分解...")
    losses = mpo.fit()
    
    # 分析结果
    final_loss = losses[-1]
    print(f"\n结果:")
    print(f"最终损失: {final_loss:.6e}")
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogy(losses)
    plt.title('训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('相对误差')
    plt.grid(True, alpha=0.3)
    
    # 比较原始和重建数据
    time_idx = 20
    original_u = data[time_idx, :, :, 0]
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
    
    # 保存模型结果
    '''print("保存MPO模型结果...")
    np.savez('mpo_model_results.npz',
             T=mpo.T, X=mpo.X, Y=mpo.Y, O=mpo.O,
             bond_dims=mpo.bond_dims, losses=losses)'''
    
    print("MPO模型训练完成！")


if __name__ == "__main__":
    main()