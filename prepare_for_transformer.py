import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

class MPODecomposition4D(nn.Module):
    """
    四维MPO分解: (T, X, Y, U) 
    其中U=2代表u/v两个物理量
    """
    def __init__(self, T, X, Y, U=2, rank_time=8, rank_space=12, rank_physics=4):
        super().__init__()
        self.T, self.X, self.Y, self.U = T, X, Y, U
        self.rank_t, self.rank_s, self.rank_p = rank_time, rank_space, rank_physics
        
        # MPO核心张量 - 使用Xavier初始化
        self.core_time = nn.Parameter(torch.randn(1, T, self.rank_t) * 0.1)           # [1, T, R_t]
        self.core_space_x = nn.Parameter(torch.randn(self.rank_t, X, self.rank_s) * 0.1)  # [R_t, X, R_s]
        self.core_space_y = nn.Parameter(torch.randn(self.rank_s, Y, self.rank_p) * 0.1)  # [R_s, Y, R_p]
        self.core_physics = nn.Parameter(torch.randn(self.rank_p, U, 1) * 0.1)        # [R_p, U, 1]
        
        # 参数数量统计
        self.num_params = (1 * T * rank_time + 
                          rank_time * X * rank_space + 
                          rank_space * Y * rank_physics + 
                          rank_physics * U * 1)
        
    def forward(self, indices=None):
        """前向传播"""
        if indices is not None:
            return self._evaluate_at_indices(indices)
        else:
            return self._reconstruct_full()
    
    def _reconstruct_full(self):
        """重建完整四维张量 [T, X, Y, U]"""
        # 逐步收缩核心张量
        # [1,T,R_t] × [R_t,X,R_s] -> [1,T,X,R_s]
        temp1 = torch.einsum('atb,bxc->atxc', self.core_time, self.core_space_x)
        
        # [1,T,X,R_s] × [R_s,Y,R_p] -> [1,T,X,Y,R_p]  
        temp2 = torch.einsum('atxd,dye->atxye', temp1, self.core_space_y)
        
        # [1,T,X,Y,R_p] × [R_p,U,1] -> [1,T,X,Y,U,1]
        result = torch.einsum('atxyf,fug->atxyug', temp2, self.core_physics)
        
        return result.squeeze(0).squeeze(-1)  # [T, X, Y, U]
    
    def _evaluate_at_indices(self, indices):
        """在特定位置评估 - 用于稀疏训练"""
        # indices: [batch_size, 4] - [t_idx, x_idx, y_idx, u_idx]
        t_idx = indices[:, 0].long()
        x_idx = indices[:, 1].long()  
        y_idx = indices[:, 2].long()
        u_idx = indices[:, 3].long()
        
        # 获取对应位置的核心值
        time_vals = self.core_time[0, t_idx, :]  # [batch, R_t]
        space_x_vals = self.core_space_x[:, x_idx, :]  # [R_t, batch, R_s]
        space_x_vals = space_x_vals.permute(1, 0, 2)  # [batch, R_t, R_s]
        space_y_vals = self.core_space_y[:, y_idx, :]  # [R_s, batch, R_p]
        space_y_vals = space_y_vals.permute(1, 0, 2)  # [batch, R_s, R_p]
        physics_vals = self.core_physics[:, u_idx, :]  # [R_p, batch, 1]
        physics_vals = physics_vals.permute(1, 0, 2)  # [batch, R_p, 1]
        
        # 收缩计算
        temp1 = torch.einsum('bi,bij->bj', time_vals, space_x_vals)  # [batch, R_s]
        temp2 = torch.einsum('bj,bjk->bk', temp1, space_y_vals)      # [batch, R_p]
        result = torch.einsum('bk,bkl->bl', temp2, physics_vals)     # [batch, 1]
        
        return result.squeeze(-1)
    
    def get_compression_ratio(self, original_elements):
        """计算压缩比"""
        return original_elements / self.num_params
    
    def get_core_tensors(self):
        """获取所有核心张量"""
        return {
            'core_time': self.core_time.data,      # [1, T, R_t]
            'core_space_x': self.core_space_x.data, # [R_t, X, R_s]
            'core_space_y': self.core_space_y.data, # [R_s, Y, R_p]
            'core_physics': self.core_physics.data  # [R_p, U, 1]
        }

class MPOTrainer:
    """MPO训练器"""
    def __init__(self, model, learning_rate=0.01, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)#优化学习率
        self.loss_history = []
        self.criterion = nn.MSELoss()
        
    def train_full_tensor(self, target_tensor, epochs=2000, print_interval=100):
        """完整张量训练"""
        # 确保target_tensor是torch张量且是float类型
        if isinstance(target_tensor, np.ndarray):
            target_tensor = torch.from_numpy(target_tensor).float()
        
        print(f"开始MPO训练...")
        print(f"目标张量形状: {target_tensor.shape}")
        print(f"目标张量类型: {target_tensor.dtype}")
        print(f"MPO参数数量: {self.model.num_params}")
        print(f"压缩比: {self.model.get_compression_ratio(target_tensor.numel()):.2f}x")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 重建完整张量
            reconstructed = self.model()
            
            # 计算MSE损失
            loss = self.criterion(reconstructed, target_tensor)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            self.loss_history.append(loss.item())
            
            if epoch % print_interval == 0:
                print(f'Epoch {epoch:04d}, Loss: {loss.item():.8f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        return self.loss_history
    
    def train_sparse(self, indices, values, epochs=1000, batch_size=256, print_interval=100):
        """稀疏观测训练"""
        dataset = TensorDataset(indices, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"开始稀疏MPO训练...")
        print(f"观测点数量: {len(indices)}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_indices, batch_values in dataloader:
                self.optimizer.zero_grad()
                
                # 在观测位置重建
                pred_values = self.model(batch_indices)
                loss = self.criterion(pred_values, batch_values)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            self.loss_history.append(epoch_loss)
            self.scheduler.step(epoch_loss)
            
            if epoch % print_interval == 0:
                print(f'Epoch {epoch:04d}, Loss: {epoch_loss:.8f}')

def prepare_transformer_input(mpo_model):
    """准备Transformer输入序列 - 用于PDE时空预测"""
    cores = mpo_model.get_core_tensors()
    
    core_time = cores['core_time'].squeeze(0)        # [T, R_t]
    core_space_x = cores['core_space_x']             # [R_t, X, R_s]
    core_space_y = cores['core_space_y']             # [R_s, Y, R_p]
    core_physics = cores['core_physics'].squeeze(-1) # [R_p, U]
    
    # 获取空间网格信息
    X, Y = core_space_x.shape[1], core_space_y.shape[1]
    T = core_time.shape[0]
    
    # 创建输入序列：每个样本是 (x, y, t) 坐标 + 上下文特征
    input_sequences = []
    target_values = []
    
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                # 输入特征：坐标信息 + 局部上下文特征
                coord_features = torch.tensor([x/X, y/Y, t/T], dtype=torch.float32)  # 归一化坐标
                
                # 从MPO核心提取局部特征
                time_feat = core_time[t]  # [R_t] 当前时间步特征
                space_x_feat = core_space_x[:, x, :]  # [R_t, R_s] x位置特征  
                space_y_feat = core_space_y[:, y, :]  # [R_s, R_p] y位置特征
                physics_feat = core_physics  # [R_p, U] 物理量特征
                
                # 构建上下文特征（可选）
                context_feat = torch.cat([
                    time_feat,
                    space_x_feat.flatten(),
                    space_y_feat.flatten(), 
                    physics_feat.flatten()
                ])#MPO自己的参数包含全局信息
                
                # 完整输入特征：坐标 + 上下文
                input_feat = torch.cat([coord_features, context_feat])
                
                # 目标：该位置的u,v值
                # 从MPO重建该点的物理场值
                with torch.no_grad():
                    u_value = torch.einsum('i,ij,jk,k->', 
                                         core_time[t], 
                                         core_space_x[:, x, :],
                                         core_space_y[:, y, :], 
                                         core_physics[:, 0])
                    v_value = torch.einsum('i,ij,jk,k->',
                                         core_time[t],
                                         core_space_x[:, x, :], 
                                         core_space_y[:, y, :],
                                         core_physics[:, 1])
                    target = torch.tensor([u_value.item(), v_value.item()])
                
                input_sequences.append(input_feat)
                target_values.append(target)
    
    # 返回形状：
    # inputs: [T*X*Y, feature_dim] - 每个时空点的特征
    # targets: [T*X*Y, 2] - 每个点的u,v值
    return torch.stack(input_sequences), torch.stack(target_values)

def plot_training_results(loss_history, original, reconstructed):
    """绘制训练结果"""
    # 确保使用detach()来分离梯度
    if isinstance(original, torch.Tensor):
        original_np = original.detach().numpy()
    else:
        original_np = original
        
    if isinstance(reconstructed, torch.Tensor):
        reconstructed_np = reconstructed.detach().numpy()
    else:
        reconstructed_np = reconstructed
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 训练损失
    axes[0,0].semilogy(loss_history)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    
    # 原始u场 (第一个时间步)
    im1 = axes[0,1].imshow(original_np[0, :, :, 0], cmap='RdBu_r')
    axes[0,1].set_title('Original u field (t=0)')
    plt.colorbar(im1, ax=axes[0,1])
    
    # 重建u场 (第一个时间步)  
    im2 = axes[1,0].imshow(reconstructed_np[0, :, :, 0], cmap='RdBu_r')
    axes[1,0].set_title('Reconstructed u field (t=0)')
    plt.colorbar(im2, ax=axes[1,0])
    
    # 重建误差
    error = np.abs(original_np - reconstructed_np).mean(axis=-1)
    im3 = axes[1,1].imshow(error[0], cmap='hot')
    axes[1,1].set_title('Reconstruction Error (t=0)')
    plt.colorbar(im3, ax=axes[1,1])
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
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
    print(f"数据张量形状: {result.shape}")
    
    # 将numpy数组转换为torch张量
    result_tensor = torch.from_numpy(result).float()
    print(f"转换后的张量形状: {result_tensor.shape}")
    print(f"张量数据类型: {result_tensor.dtype}")
    
    # 初始化MPO模型
    mpo_model = MPODecomposition4D(
        T=num_timesteps, X=nx, Y=ny, U=2,
        rank_time=6,   # 时间秩
        rank_space=8,  # 空间秩  
        rank_physics=3 # 物理量秩
    )
    
    # 训练MPO分解
    trainer = MPOTrainer(mpo_model, learning_rate=0.02)
    loss_history = trainer.train_full_tensor(result_tensor, epochs=1500, print_interval=150)
    
    # 评估最终结果
    with torch.no_grad():
        final_reconstruction = mpo_model()
        final_loss = nn.MSELoss()(final_reconstruction, result_tensor)
        print(f"\n最终重建MSE: {final_loss.item():.8f}")
        
        # 准备Transformer输入
        transformer_input = prepare_transformer_input(mpo_model)
        print(f"Transformer输入形状: {transformer_input[0].shape}")
        
        # 获取核心张量
        cores = mpo_model.get_core_tensors()
        for name, core in cores.items():
            print(f"{name}形状: {core.shape}")
    
    # 绘制结果
    plot_training_results(loss_history, result_tensor, final_reconstruction)
    
    print("\nMPO分解完成！现在可以用transformer_input训练Transformer模型了。")