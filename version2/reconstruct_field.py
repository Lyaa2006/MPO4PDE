import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import pandas as pd
import math
#from sklearn.model_selection import train_test_split

# ===============================
# 你的MPO代码（完全不变）
# ===============================
class MPODecomposition4D(nn.Module):
    """四维MPO分解: (T, X, Y, U)"""
    def __init__(self, T, X, Y, U=2, rank_time=8, rank_space=12, rank_physics=4):
        super().__init__()
        self.T, self.X, self.Y, self.U = T, X, Y, U
        self.rank_t, self.rank_s, self.rank_p = rank_time, rank_space, rank_physics
        
        self.core_tensor = nn.Parameter(torch.randn(rank_time, rank_space, rank_physics) * 0.1)
        self.edge_time = nn.Parameter(torch.randn(T, rank_time) * 0.1)
        self.edge_space_x = nn.Parameter(torch.randn(X, rank_space) * 0.1)
        self.edge_space_y = nn.Parameter(torch.randn(Y, rank_space) * 0.1)
        self.edge_physics = nn.Parameter(torch.randn(U, rank_physics) * 0.1)
        
        self.num_params = (rank_time * rank_space * rank_physics +
                          T * rank_time + 
                          X * rank_space + 
                          Y * rank_space + 
                          U * rank_physics)
        
    def forward(self, indices=None):
        if indices is not None:
            return self._evaluate_at_indices(indices)
        else:
            return self._reconstruct_full()
    
    def _reconstruct_full(self):
        spatial_interaction = torch.einsum('xi,yj->xyij', self.edge_space_x, self.edge_space_y)
        spatial_interaction = spatial_interaction.sum(dim=-1)
        time_evolved_core = torch.einsum('tr,rsp->tsp', self.edge_time, self.core_tensor)
        spacetime_field = torch.einsum('tsp,xys->txyp', time_evolved_core, spatial_interaction)
        result = torch.einsum('txyp,up->txyu', spacetime_field, self.edge_physics)
        return result
    
    def _evaluate_at_indices(self, indices):
        t_idx = indices[:, 0].long()
        x_idx = indices[:, 1].long()  
        y_idx = indices[:, 2].long()
        u_idx = indices[:, 3].long()
        
        time_feat = self.edge_time[t_idx, :]
        space_x_feat = self.edge_space_x[x_idx, :]
        space_y_feat = self.edge_space_y[y_idx, :]
        physics_feat = self.edge_physics[u_idx, :]
        
        spatial_feat = space_x_feat * space_y_feat
        time_evolved = torch.einsum('bi,ijk->bjk', time_feat, self.core_tensor)
        spacetime_feat = torch.einsum('bjk,bj->bk', time_evolved, spatial_feat)
        result = torch.einsum('bk,bk->b', spacetime_feat, physics_feat)
        return result
    
    def analyze_evolution_patterns(self):
        with torch.no_grad():
            core_reshaped = self.core_tensor.reshape(self.rank_t, -1)
            U, S, V = torch.svd(core_reshaped)
            time_modes = self.edge_time @ U[:, :3]
            spatial_modes_x = self.edge_space_x @ self.core_tensor.mean(dim=[0,2])
            spatial_modes_y = self.edge_space_y @ self.core_tensor.mean(dim=[0,2])
        return {
            'singular_values': S,
            'time_evolution_modes': time_modes,
            'spatial_modes_x': spatial_modes_x,
            'spatial_modes_y': spatial_modes_y,
            'core_tensor_norm': torch.norm(self.core_tensor)
        }
    
    def get_compression_ratio(self, original_elements):
        return original_elements / self.num_params
    
    def get_core_and_edges(self):
        return {
            'core_tensor': self.core_tensor.data,
            'edge_time': self.edge_time.data,
            'edge_space_x': self.edge_space_x.data,
            'edge_space_y': self.edge_space_y.data,
            'edge_physics': self.edge_physics.data
        }

class MPOTrainer:
    def __init__(self, model, learning_rate=0.01, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)
        self.loss_history = []
        self.criterion = nn.MSELoss()
        
    def train_full_tensor(self, target_tensor, epochs=2000, print_interval=100):
        if isinstance(target_tensor, np.ndarray):
            target_tensor = torch.from_numpy(target_tensor).float()
        
        print(f"开始MPO训练...")
        print(f"目标张量形状: {target_tensor.shape}")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            reconstructed = self.model()
            loss = self.criterion(reconstructed, target_tensor)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            self.loss_history.append(loss.item())
            
            if epoch % print_interval == 0:
                print(f'Epoch {epoch:04d}, Loss: {loss.item():.8f}')
        
        return self.loss_history

# ===============================
# Transformer物理场重建模型
# ===============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 动态计算位置编码
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return x + pe.unsqueeze(0)

class FieldReconstructionTransformer(nn.Module):
    """
    使用PyTorch内置Transformer的物理场重建模型
    """
    def __init__(self, coord_dim=3, value_dim=2, d_model=256, nhead=8, 
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, 
                 dropout=0.1, output_grid_size=(32, 32), output_dim=2):
        super().__init__()
        
        self.output_grid_size = output_grid_size
        self.output_dim = output_dim
        self.d_model = d_model
        H, W = output_grid_size
        self.num_queries = H * W
        
        # 输入编码层
        self.coord_projection = nn.Linear(coord_dim, d_model // 2)
        self.value_projection = nn.Linear(value_dim, d_model // 2)
        self.input_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器 - 处理观测序列
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer解码器 - 处理查询序列与观测序列的交叉注意力
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出解码器
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim)
        )
        
    def _create_query_coordinates(self, batch_size, device):
        """创建查询坐标网格"""
        H, W = self.output_grid_size
        x_coords = torch.linspace(0, 1, W, device=device)
        y_coords = torch.linspace(0, 1, H, device=device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        query_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [H*W, 2]
        
        # 扩展到批次维度
        query_coords = query_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, H*W, 2]
        return query_coords
    
    def _get_query_times(self, observed_coords, batch_size, device):
        """获取查询时间"""
        # 使用观测时间的平均值作为查询时间
        query_times = observed_coords[:, :, 0].mean(dim=1, keepdim=True)  # [batch, 1]
        query_times = query_times.unsqueeze(1).expand(batch_size, self.num_queries, 1)  # [batch, H*W, 1]
        return query_times
    
    def _prepare_encoder_input(self, observed_coords, observed_values):
        """准备编码器输入序列"""
        batch_size, num_obs, _ = observed_coords.shape
        
        # 编码观测坐标和值
        coord_feat = self.coord_projection(observed_coords)  # [batch, num_obs, d_model//2]
        value_feat = self.value_projection(observed_values)  # [batch, num_obs, d_model//2]
        
        # 合并特征
        encoder_input = torch.cat([coord_feat, value_feat], dim=-1)  # [batch, num_obs, d_model]
        encoder_input = self.input_norm(encoder_input)
        
        # 位置编码
        encoder_input = self.pos_encoder(encoder_input)
        encoder_input = self.dropout(encoder_input)
        
        return encoder_input
    
    def _prepare_decoder_input(self, observed_coords, batch_size, device):
        """准备解码器输入序列（查询序列）"""
        # 创建查询坐标
        query_coords = self._create_query_coordinates(batch_size, device)  # [batch, H*W, 2]
        
        # 获取查询时间
        query_times = self._get_query_times(observed_coords, batch_size, device)  # [batch, H*W, 1]
        
        # 合并时间和空间坐标
        query_coords_full = torch.cat([query_times, query_coords], dim=-1)  # [batch, H*W, 3]
        
        # 编码查询坐标（只有坐标信息，没有观测值）
        query_coord_feat = self.coord_projection(query_coords_full)  # [batch, H*W, d_model//2]
        query_value_feat = torch.zeros(batch_size, self.num_queries, self.d_model // 2, 
                                     device=device)
        decoder_input = torch.cat([query_coord_feat, query_value_feat], dim=-1)
        decoder_input = self.input_norm(decoder_input)
        
        # 位置编码
        decoder_input = self.pos_encoder(decoder_input)
        decoder_input = self.dropout(decoder_input)
        
        return decoder_input
    
    def forward(self, observed_coords, observed_values):
        """
        Args:
            observed_coords: [batch_size, num_obs, 3] - (t, x, y) 观测坐标
            observed_values: [batch_size, num_obs, value_dim] - 观测值
        Returns:
            reconstructed_field: [batch_size, H, W, output_dim] - 重建的完整场
        """
        batch_size, num_obs, _ = observed_coords.shape
        device = observed_coords.device
        H, W = self.output_grid_size
        
        # 1. 准备编码器输入并编码观测序列
        encoder_input = self._prepare_encoder_input(observed_coords, observed_values)
        memory = self.transformer_encoder(encoder_input)  # [batch, num_obs, d_model]
        
        # 2. 准备解码器输入（查询序列）
        decoder_input = self._prepare_decoder_input(observed_coords, batch_size, device)
        
        # 3. 使用Transformer解码器进行交叉注意力
        # decoder_input: [batch, num_queries, d_model] - 查询序列
        # memory: [batch, num_obs, d_model] - 观测序列（编码器输出）
        query_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory
        )  # [batch, num_queries, d_model]
        
        # 4. 解码输出
        field_flat = self.output_decoder(query_output)  # [batch, H*W, output_dim]
        
        # 5. 重塑为场格式
        reconstructed_field = field_flat.reshape(batch_size, H, W, self.output_dim)
        
        return reconstructed_field
    
class FieldReconstructionTrainer:
    """场重建训练器"""
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, val_loader=None, epochs=100, print_interval=10):
        print("开始场重建Transformer训练...")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_obs_coords, batch_obs_values, batch_full_fields in train_loader:
                self.optimizer.zero_grad()
                
                # 前向传播
                reconstructed = self.model(batch_obs_coords, batch_obs_values)
                
                # 计算损失
                loss = self.criterion(reconstructed, batch_full_fields)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            self.train_losses.append(epoch_loss)
            
            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(epoch_loss)
            
            
            if val_loader is not None:
                    print(f'Epoch {epoch:04d}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                    print(f'Epoch {epoch:04d}, Train Loss: {epoch_loss:.6f}')
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_obs_coords, batch_obs_values, batch_full_fields in val_loader:
                reconstructed = self.model(batch_obs_coords, batch_obs_values)
                loss = self.criterion(reconstructed, batch_full_fields)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

def create_sparse_observation_dataset(full_fields, observation_ratio=0.1):
    """
    创建稀疏观测数据集
    Args:
        full_fields: [num_samples, T, X, Y, U] 完整场数据
        observation_ratio: 观测点比例
    Returns:
        Dataset: (观测坐标, 观测值, 完整场)
    """
    if isinstance(full_fields, np.ndarray):
        full_fields = torch.from_numpy(full_fields).float()
    
    num_samples, T, X, Y, U = full_fields.shape
    
    all_obs_coords = []
    all_obs_values = []
    all_full_fields = []
    
    for sample_idx in range(num_samples):
        for t in range(T):
            # 创建完整场
            full_field = full_fields[sample_idx, t]  # [X, Y, U]
            
            # 随机选择观测点
            total_points = X * Y
            num_obs = max(1, int(total_points * observation_ratio))
            
            # 生成随机观测位置
            obs_indices = np.random.choice(total_points, num_obs, replace=False)
            obs_x = obs_indices // Y
            obs_y = obs_indices % Y
            
            # 创建观测坐标 (t, x, y)
            obs_coords = []
            obs_values = []
            
            for i in range(num_obs):
                x_idx, y_idx = obs_x[i], obs_y[i]
                # 归一化坐标
                t_norm = t / T
                x_norm = x_idx / X
                y_norm = y_idx / Y
                
                obs_coords.append([t_norm, x_norm, y_norm])
                obs_values.append(full_field[x_idx, y_idx].numpy())
            
            obs_coords = torch.tensor(obs_coords, dtype=torch.float32)
            obs_values = torch.tensor(obs_values, dtype=torch.float32)
            
            all_obs_coords.append(obs_coords)
            all_obs_values.append(obs_values)
            all_full_fields.append(full_field)
    
    return all_obs_coords, all_obs_values, all_full_fields

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    obs_coords, obs_values, full_fields = zip(*batch)
    
    # 找到最大观测点数
    max_obs = max(coords.shape[0] for coords in obs_coords)
    batch_size = len(obs_coords)
    coord_dim = obs_coords[0].shape[1]
    value_dim = obs_values[0].shape[1]
    field_shape = full_fields[0].shape
    
    # 填充批次
    padded_obs_coords = torch.zeros(batch_size, max_obs, coord_dim)
    padded_obs_values = torch.zeros(batch_size, max_obs, value_dim)
    mask = torch.zeros(batch_size, max_obs, dtype=torch.bool)
    
    for i, (coords, values) in enumerate(zip(obs_coords, obs_values)):
        num_obs = coords.shape[0]
        padded_obs_coords[i, :num_obs] = coords
        padded_obs_values[i, :num_obs] = values
        mask[i, :num_obs] = True
    
    full_fields_batch = torch.stack(full_fields)
    
    return padded_obs_coords, padded_obs_values, full_fields_batch

def visualize_reconstruction_results(original, reconstructed, obs_coords=None, title="场重建结果"):
    """可视化重建结果"""
    if isinstance(original, torch.Tensor):
        original = original.detach().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始场
    im1 = axes[0, 0].imshow(original[:, :, 0], cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('原始场 - u分量')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(original[:, :, 1], cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('原始场 - v分量')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 重建场
    im3 = axes[1, 0].imshow(reconstructed[:, :, 0], cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('重建场 - u分量')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(reconstructed[:, :, 1], cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('重建场 - v分量')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 误差
    error_u = np.abs(original[:, :, 0] - reconstructed[:, :, 0])
    error_v = np.abs(original[:, :, 1] - reconstructed[:, :, 1])
    
    im5 = axes[0, 2].imshow(error_u, cmap='hot', aspect='auto')
    axes[0, 2].set_title('u分量重建误差')
    plt.colorbar(im5, ax=axes[0, 2])
    
    im6 = axes[1, 2].imshow(error_v, cmap='hot', aspect='auto')
    axes[1, 2].set_title('v分量重建误差')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # 标记观测点
    if obs_coords is not None:
        for ax in [axes[1, 0], axes[1, 1]]:
            X, Y = original.shape[:2]
            for coord in obs_coords:
                t, x, y = coord
                x_idx = int(x * X)
                y_idx = int(y * Y)
                ax.plot(y_idx, x_idx, 'ro', markersize=3, alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.savefig("result.png")
    plt.close()
    
    print(f"重建统计:")
    print(f"u分量MAE: {error_u.mean():.6f}, Max: {error_u.max():.6f}")
    print(f"v分量MAE: {error_v.mean():.6f}, Max: {error_v.max():.6f}")

# 主程序示例
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 读取数据（使用你的数据读取代码）
    df = pd.read_csv("Burgers_data1.csv", comment='%', header=None)
    
    # 数据预处理（使用你的预处理代码）
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
    
    # 1. 首先进行MPO分解
    print("\n=== 步骤1: MPO分解 ===")
    mpo_model = MPODecomposition4D(
        T=num_timesteps, X=nx, Y=ny, U=2,
        rank_time=6, rank_space=8, rank_physics=3
    )
    
    mpo_trainer = MPOTrainer(mpo_model, learning_rate=0.02)
    mpo_loss_history = mpo_trainer.train_full_tensor(result_tensor, epochs=1000, print_interval=100)
    
    # 2. 创建稀疏观测数据集
    print("\n=== 步骤2: 创建稀疏观测数据集 ===")
    # 将MPO重建结果作为真实场（或者使用原始数据）
    with torch.no_grad():
        mpo_reconstruction = mpo_model()
    
    # 创建训练数据
    obs_coords_list, obs_values_list, full_fields_list = create_sparse_observation_dataset(
        mpo_reconstruction.unsqueeze(0),  # 添加批次维度
        observation_ratio=0.1  # 10%的观测点
    )
    
    # 创建数据集
    dataset = list(zip(obs_coords_list, obs_values_list, full_fields_list))
    
    # 数据集分割
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # 3. 训练场重建Transformer
    print("\n=== 步骤3: 训练场重建Transformer ===")
    reconstruction_model = FieldReconstructionTransformer(
        coord_dim=3,
        value_dim=2,
        d_model=256,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        output_grid_size=(nx, ny),
        output_dim=2
    )
    
    reconstruction_trainer = FieldReconstructionTrainer(reconstruction_model, learning_rate=1e-4)
    reconstruction_trainer.train(train_loader, val_loader, epochs=10, print_interval=10)
    
    # 4. 测试和可视化
    print("\n=== 步骤4: 测试重建效果 ===")
    reconstruction_model.eval()
    
    with torch.no_grad():
        # 取一个测试样本
        test_batch = next(iter(test_loader))
        test_obs_coords, test_obs_values, test_full_fields = test_batch
        
        # 重建
        reconstructed = reconstruction_model(test_obs_coords, test_obs_values)
        
        # 可视化第一个样本
        sample_idx = 0
        original_field = test_full_fields[sample_idx].numpy()
        reconstructed_field = reconstructed[sample_idx].numpy()
        obs_coords_sample = test_obs_coords[sample_idx][test_obs_coords[sample_idx].sum(dim=1) != 0].numpy()
        
        visualize_reconstruction_results(
            original_field, reconstructed_field, obs_coords_sample,
            title="Transformer物理场重建结果"
        )
    
    print("\n=== 训练完成 ===")

if __name__ == "__main__":
    main()