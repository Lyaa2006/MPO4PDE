import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split

plt.rcParams['font.family'] = ['SimHei']  # 黑体

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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)
        self.loss_history = []
        self.criterion = nn.MSELoss()
        
    def train_full_tensor(self, target_tensor, epochs=2000, print_interval=100):
        """完整张量训练"""
        if isinstance(target_tensor, np.ndarray):
            target_tensor = torch.from_numpy(target_tensor).float()
        
        print(f"开始MPO训练...")
        print(f"目标张量形状: {target_tensor.shape}")
        print(f"目标张量类型: {target_tensor.dtype}")
        print(f"MPO参数数量: {self.model.num_params}")
        print(f"压缩比: {self.model.get_compression_ratio(target_tensor.numel()):.2f}x")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            reconstructed = self.model()
            loss = self.criterion(reconstructed, target_tensor)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            self.loss_history.append(loss.item())
            
            if epoch % print_interval == 0:
                print(f'Epoch {epoch:04d}, Loss: {loss.item():.8f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        return self.loss_history

def prepare_transformer_input(mpo_model):
    """准备Transformer输入序列 - 用于PDE时空预测"""
    cores = mpo_model.get_core_tensors()
    
    core_time = cores['core_time'].squeeze(0)        # [T, R_t]
    core_space_x = cores['core_space_x']             # [R_t, X, R_s]
    core_space_y = cores['core_space_y']             # [R_s, Y, R_p]
    core_physics = cores['core_physics'].squeeze(-1) # [R_p, U]
    
    X, Y = core_space_x.shape[1], core_space_y.shape[1]
    T = core_time.shape[0]
    
    input_sequences = []
    target_values = []
    
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                coord_features = torch.tensor([x/X, y/Y, t/T], dtype=torch.float32)
                
                time_feat = core_time[t]
                space_x_feat = core_space_x[:, x, :]
                space_y_feat = core_space_y[:, y, :]
                physics_feat = core_physics
                
                context_feat = torch.cat([
                    time_feat,
                    space_x_feat.flatten(),
                    space_y_feat.flatten(), 
                    physics_feat.flatten()
                ])
                
                input_feat = torch.cat([coord_features, context_feat])
                
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
    
    return torch.stack(input_sequences), torch.stack(target_values)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def generate_causal_mask(seq_len):
    """
    生成因果掩码（下三角矩阵）
    Args:
        seq_len: 序列长度
    Returns:
        mask: [seq_len, seq_len] 下三角矩阵，右上角为-inf，左下角为0
    """
    # 创建下三角矩阵（包含对角线）
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # 将0转换为-inf，1转换为0
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class MPOTransformer(nn.Module):
    """基于MPO分解的Transformer模型（带因果掩码）"""
    def __init__(self, input_dim=87, d_model=128, nhead=8, num_layers=6, 
                 dim_feedforward=512, dropout=0.1, output_dim=2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
        
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: [batch_size, seq_len, input_dim] 输入序列
            src_mask: 如果不提供，会自动生成因果掩码
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 如果没有提供mask，自动生成因果掩码
        if src_mask is None:
            seq_len = x.size(1)
            src_mask = generate_causal_mask(seq_len).to(x.device)
        
        encoded = self.transformer_encoder(x, mask=src_mask, 
                                         src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(encoded)
        return output

class MPOTransformerTrainer:
    """Transformer训练器（带因果掩码）"""
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=20, factor=0.5)
        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.val_loss_history = []
    
    def train(self, train_loader, val_loader=None, epochs=100, print_interval=10):
        """训练模型（使用因果掩码）"""
        print("开始Transformer训练（使用因果掩码）...")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_inputs, batch_targets in train_loader:
                self.optimizer.zero_grad()
                
                # 自动生成因果掩码
                seq_len = batch_inputs.size(1)
                causal_mask = generate_causal_mask(seq_len).to(batch_inputs.device)
                
                # 前向传播（传入因果掩码）
                outputs = self.model(batch_inputs, src_mask=causal_mask)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            self.loss_history.append(epoch_loss)
            self.scheduler.step(epoch_loss)
            
            if epoch % print_interval == 0:
                val_loss = None
                if val_loader is not None:
                    val_loss = self.validate(val_loader)
                    self.val_loss_history.append(val_loss)
                    print(f'Epoch {epoch:04d}, Train Loss: {epoch_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                else:
                    print(f'Epoch {epoch:04d}, Train Loss: {epoch_loss:.8f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
    
    def validate(self, val_loader):
        """验证模型（使用因果掩码）"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                # 验证时同样使用因果掩码
                seq_len = batch_inputs.size(1)
                causal_mask = generate_causal_mask(seq_len).to(batch_inputs.device)
                
                outputs = self.model(batch_inputs, src_mask=causal_mask)
                loss = self.criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def test(self, test_loader):
        """测试模型并返回预测结果（使用因果掩码）"""
        self.model.eval()
        test_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                # 测试时同样使用因果掩码
                seq_len = batch_inputs.size(1)
                causal_mask = generate_causal_mask(seq_len).to(batch_inputs.device)
                
                outputs = self.model(batch_inputs, src_mask=causal_mask)
                loss = self.criterion(outputs, batch_targets)
                test_loss += loss.item()
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        test_loss /= len(test_loader)
        print(f'测试集损失: {test_loss:.8f}')
        
        return all_predictions, all_targets, test_loss
    
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

def create_sequences_for_transformer(input_data, target_data, seq_len=10):
    """
    为Transformer创建序列数据
    """
    sequences = []
    targets = []
    
    total_points = input_data.shape[0]
    
    for i in range(total_points - seq_len):
        seq_input = input_data[i:i+seq_len]
        seq_target = target_data[i+1:i+seq_len+1]
        
        sequences.append(seq_input)
        targets.append(seq_target)
    
    return torch.stack(sequences), torch.stack(targets)

def plot_training_curves(train_losses, val_losses=None):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='训练损失', alpha=0.7)
    if val_losses is not None:
        val_x = np.linspace(0, len(train_losses)-1, len(val_losses))
        plt.semilogy(val_x, val_losses, label='验证损失', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练和验证损失曲线（使用因果掩码）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_test_predictions(predictions, targets, num_samples=5, seq_len=20):
    """可视化测试集预测结果"""
    num_sequences = predictions.shape[0]
    sample_indices = np.random.choice(num_sequences, min(num_samples, num_sequences), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(sample_indices):
        pred_seq = predictions[sample_idx]
        target_seq = targets[sample_idx]
        time_steps = np.arange(seq_len)
        
        # u分量
        axes[idx, 0].plot(time_steps, target_seq[:, 0], 'b-', label='真实值', linewidth=2, alpha=0.8)
        axes[idx, 0].plot(time_steps, pred_seq[:, 0], 'r--', label='预测值', linewidth=2, alpha=0.8)
        axes[idx, 0].set_title(f'样本 {sample_idx} - u分量')
        axes[idx, 0].set_xlabel('时间步')
        axes[idx, 0].set_ylabel('u值')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # v分量
        axes[idx, 1].plot(time_steps, target_seq[:, 1], 'g-', label='真实值', linewidth=2, alpha=0.8)
        axes[idx, 1].plot(time_steps, pred_seq[:, 1], 'orange', linestyle='--', label='预测值', linewidth=2, alpha=0.8)
        axes[idx, 1].set_title(f'样本 {sample_idx} - v分量')
        axes[idx, 1].set_xlabel('时间步')
        axes[idx, 1].set_ylabel('v值')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        
        # 误差
        u_error = np.abs(pred_seq[:, 0] - target_seq[:, 0])
        v_error = np.abs(pred_seq[:, 1] - target_seq[:, 1])
        axes[idx, 2].plot(time_steps, u_error, 'r-', label='u误差', alpha=0.7)
        axes[idx, 2].plot(time_steps, v_error, 'b-', label='v误差', alpha=0.7)
        axes[idx, 2].set_title(f'样本 {sample_idx} - 预测误差')
        axes[idx, 2].set_xlabel('时间步')
        axes[idx, 2].set_ylabel('绝对误差')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(predictions, targets):
    """绘制误差分布"""
    u_errors = np.abs(predictions[:, :, 0] - targets[:, :, 0]).flatten()
    v_errors = np.abs(predictions[:, :, 1] - targets[:, :, 1]).flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(u_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.set_xlabel('u分量绝对误差')
    ax1.set_ylabel('频数')
    ax1.set_title(f'u误差分布 (均值: {np.mean(u_errors):.6f})')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(v_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('v分量绝对误差')
    ax2.set_ylabel('频数')
    ax2.set_title(f'v误差分布 (均值: {np.mean(v_errors):.6f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_causal_mask(seq_len=10):
    """可视化因果掩码"""
    mask = generate_causal_mask(seq_len)
    plt.figure(figsize=(8, 6))
    plt.imshow(mask.numpy(), cmap='coolwarm', aspect='equal')
    plt.colorbar(label='掩码值 (-inf 或 0)')
    plt.title(f'因果掩码可视化 (序列长度: {seq_len})')
    plt.xlabel('Key位置')
    plt.ylabel('Query位置')
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 可视化因果掩码
    print("=== 因果掩码可视化 ===")
    visualize_causal_mask(seq_len=10)
    
    # 读取数据
    df = pd.read_csv("D:\Comsol建模\Burgers数据集.csv", comment='%', header=None)

    # 数据预处理
    columns_per_timestep = 5
    num_timesteps = (df.shape[1] - 2) // columns_per_timestep

    X = df.iloc[:, 0].values
    Y = df.iloc[:, 1].values

    x_unique = np.sort(np.unique(X))
    y_unique = np.sort(np.unique(Y))
    nx = len(x_unique)
    ny = len(y_unique)
    
    result = np.zeros((num_timesteps, nx, ny, 2))

    x_to_idx = {x_val: idx for idx, x_val in enumerate(x_unique)}
    y_to_idx = {y_val: idx for idx, y_val in enumerate(y_unique)}
    
    for t_idx in range(num_timesteps):
        start_col = 2 + t_idx * columns_per_timestep
        t_data = df.iloc[:, start_col:start_col+5].values
        for row in t_data:
            x_val, y_val, u, v = row[1], row[2], row[3], row[4]
            i = x_to_idx[x_val]
            j = y_to_idx[y_val]
            result[t_idx, i, j, 0] = u
            result[t_idx, i, j, 1] = v
    
    result_tensor = torch.from_numpy(result).float()
    print(f"数据张量形状: {result_tensor.shape}")
    
    # 初始化MPO模型
    mpo_model = MPODecomposition4D(
        T=num_timesteps, X=nx, Y=ny, U=2,
        rank_time=6, rank_space=8, rank_physics=3
    )
    
    # 训练MPO分解
    trainer = MPOTrainer(mpo_model, learning_rate=0.02)
    loss_history = trainer.train_full_tensor(result_tensor, epochs=1500, print_interval=150)
    
    # 评估MPO结果
    with torch.no_grad():
        final_reconstruction = mpo_model()
        final_loss = nn.MSELoss()(final_reconstruction, result_tensor)
        print(f"\nMPO最终重建MSE: {final_loss.item():.8f}")
    
    # 准备Transformer输入
    transformer_inputs, transformer_targets = prepare_transformer_input(mpo_model)
    print(f"Transformer输入形状: {transformer_inputs.shape}")
    print(f"Transformer目标形状: {transformer_targets.shape}")
    
    plot_training_results(loss_history, result_tensor, final_reconstruction)
    # 创建序列数据
    seq_inputs, seq_targets = create_sequences_for_transformer(
        transformer_inputs, transformer_targets, seq_len=20
    )
    print(f"序列输入形状: {seq_inputs.shape}")
    print(f"序列目标形状: {seq_targets.shape}")
    
    # 数据集分割 (7:3)
    dataset_size = len(seq_inputs)
    train_size = int(0.7 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(
        TensorDataset(seq_inputs, seq_targets), 
        [train_size, test_size]
    )
    
    # 从训练集中再分割出验证集
    train_size_final = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size_final
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size_final, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化Transformer模型
    transformer_model = MPOTransformer(
        input_dim=transformer_inputs.shape[1],
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=2
    )
    
    # 训练Transformer（使用因果掩码）
    transformer_trainer = MPOTransformerTrainer(transformer_model, learning_rate=1e-4)
    transformer_trainer.train(train_loader, val_loader, epochs=10, print_interval=20)
    
    # 绘制训练曲线
    plot_training_curves(transformer_trainer.loss_history, transformer_trainer.val_loss_history)
    
    # 测试模型
    print("\n开始测试...")
    test_predictions, test_targets, test_loss = transformer_trainer.test(test_loader)
    
    # 可视化测试结果
    print("\n可视化测试结果...")
    plot_test_predictions(test_predictions, test_targets, num_samples=5, seq_len=20)
    plot_error_distribution(test_predictions, test_targets)
    
    # 输出统计信息
    u_errors = np.abs(test_predictions[:, :, 0] - test_targets[:, :, 0])
    v_errors = np.abs(test_predictions[:, :, 1] - test_targets[:, :, 1])
    
    print(f"\n=== 测试结果统计（使用因果掩码）===")
    print(f"测试集MSE损失: {test_loss:.8f}")
    print(f"u分量平均绝对误差: {np.mean(u_errors):.6f}")
    print(f"v分量平均绝对误差: {np.mean(v_errors):.6f}")
    print(f"u分量最大绝对误差: {np.max(u_errors):.6f}")
    print(f"v分量最大绝对误差: {np.max(v_errors):.6f}")
    print(f"u分量误差标准差: {np.std(u_errors):.6f}")
    print(f"v分量误差标准差: {np.std(v_errors):.6f}")
    
    print("\n所有训练完成！")