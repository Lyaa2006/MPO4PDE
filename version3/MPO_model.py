import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Callable

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MPODecomposition(nn.Module):
    """
    MPO分解模型
    学习数据的低秩MPO表示，核心张量作为特征
    """
    def __init__(self, data_shape, ranks):
        super(MPODecomposition, self).__init__()
        self.T, self.X, self.Y, self.U = data_shape
        self.rank_t, self.rank_x, self.rank_y, self.rank_p = ranks
        
        # 边缘向量参数
        self.edge_time = nn.Parameter(torch.randn(self.T, self.rank_t) * 0.1)
        self.edge_space_x = nn.Parameter(torch.randn(self.X, self.rank_x) * 0.1)
        self.edge_space_y = nn.Parameter(torch.randn(self.Y, self.rank_y) * 0.1)
        self.edge_physics = nn.Parameter(torch.randn(self.U, self.rank_p) * 0.1)
        
        # 核心张量 - 这就是我们要的特征！
        self.core_tensor = nn.Parameter(
            torch.randn(self.rank_t, self.rank_x, self.rank_y, self.rank_p) * 0.1
        )
        
    def forward(self, indices=None):
        """
        MPO重建: T × core × X × Y × P
        """
        if indices is not None:
            return self._evaluate_at_indices(indices)
        else:
            return self._reconstruct_full()
    
    def _reconstruct_full(self):
        """重建完整张量"""
        # 使用批量重建
        T, X, Y, U = self.T, self.X, self.Y, self.U
        indices = []
        
        for t in range(T):
            for x in range(X):
                for y in range(Y):
                    indices.append([t, x, y])
        
        indices = torch.tensor(indices, dtype=torch.long)
        reconstructed = self._evaluate_at_indices(indices)
        return reconstructed.reshape(T, X, Y, U)
    
    def _evaluate_at_indices(self, indices):
        """在特定位置评估"""
        t_idx = indices[:, 0].long()
        x_idx = indices[:, 1].long()
        y_idx = indices[:, 2].long()
        
        time_feat = self.edge_time[t_idx]  # [batch, rank_t]
        space_x_feat = self.edge_space_x[x_idx]  # [batch, rank_x]
        space_y_feat = self.edge_space_y[y_idx]  # [batch, rank_y]
        
        # 与核心张量相互作用 - 修正的einsum操作
        # core_tensor: [rank_t, rank_x, rank_y, rank_p]
        # time_feat: [batch, rank_t] -> [batch, rank_x, rank_y, rank_p]
        temp = torch.einsum('bi,ijkp->bjkp', time_feat, self.core_tensor)
        # space_x_feat: [batch, rank_x] -> [batch, rank_y, rank_p]
        temp = torch.einsum('bj,bjkp->bkp', space_x_feat, temp)
        # space_y_feat: [batch, rank_y] -> [batch, rank_p]
        temp = torch.einsum('bk,bkp->bp', space_y_feat, temp)  # [batch, rank_p]
        
        # 物理变量
        result = torch.einsum('bp,up->bu', temp, self.edge_physics)  # [batch, U]
        return result
    
    def get_core_features(self):
        """获取核心张量特征 - 用于Transformer输入"""
        # 展平核心张量作为全局特征
        return self.core_tensor.flatten().detach()
    
    def get_compressed_features(self, coordinates):
        """
        获取坐标对应的压缩特征
        Args:
            coordinates: [batch_size, 3] - (t, x, y) 归一化坐标
        Returns:
            features: [batch_size, rank_p] - 压缩特征
        """
        batch_size = coordinates.shape[0]
        
        # 转换为索引
        t_idx = (coordinates[:, 0] * (self.T - 1)).long().clamp(0, self.T-1)
        x_idx = (coordinates[:, 1] * (self.X - 1)).long().clamp(0, self.X-1)
        y_idx = (coordinates[:, 2] * (self.Y - 1)).long().clamp(0, self.Y-1)
        
        # 获取边缘向量
        time_feat = self.edge_time[t_idx]  # [batch, rank_t]
        space_x_feat = self.edge_space_x[x_idx]  # [batch, rank_x]
        space_y_feat = self.edge_space_y[y_idx]  # [batch, rank_y]
        
        # 与核心张量相互作用得到特征（不经过物理边缘）
        features = torch.einsum('bi,ijkp->bjkp', time_feat, self.core_tensor)
        features = torch.einsum('bj,bjkp->bkp', space_x_feat, features)
        features = torch.einsum('bk,bkp->bp', space_y_feat, features)  # [batch, rank_p]
        
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return x + pe.unsqueeze(0)

class MPOEncoderDecoderTransformer(nn.Module):
    """
    Encoder-Decoder Transformer
    编码器: 处理MPO核心特征序列
    解码器: 处理坐标查询序列，与编码器特征进行交叉注意力
    """
    def __init__(self, core_feature_dim, coord_dim=3, d_model=256, nhead=8, 
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(MPOEncoderDecoderTransformer, self).__init__()
        
        self.d_model = d_model
        self.core_feature_dim = core_feature_dim
        self.coord_dim = coord_dim
        
        # 编码器部分 - 处理MPO核心特征
        self.core_projection = nn.Linear(core_feature_dim, d_model)
        self.core_norm = nn.LayerNorm(d_model)
        
        # 解码器部分 - 处理坐标查询
        self.coord_projection = nn.Linear(coord_dim, d_model)
        self.coord_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 2)  # 输出u,v
        )
        
    def forward(self, core_features, query_coordinates):
        """
        Args:
            core_features: [batch_size, enc_seq_len, core_feature_dim] - MPO核心特征序列
            query_coordinates: [batch_size, dec_seq_len, 3] - 查询坐标序列
        Returns:
            predictions: [batch_size, dec_seq_len, 2] - u,v预测值
        """
        batch_size = core_features.shape[0]
        
        # 1. 编码器处理MPO核心特征
        core_encoded = self.core_projection(core_features)  # [batch, enc_seq_len, d_model]
        core_encoded = self.core_norm(core_encoded)
        core_encoded = self.pos_encoder(core_encoded)
        core_encoded = self.dropout(core_encoded)
        
        # Transformer编码
        memory = self.encoder(core_encoded)  # [batch, enc_seq_len, d_model]
        
        # 2. 解码器处理坐标查询
        query_encoded = self.coord_projection(query_coordinates)  # [batch, dec_seq_len, d_model]
        query_encoded = self.coord_norm(query_encoded)
        query_encoded = self.pos_encoder(query_encoded)
        query_encoded = self.dropout(query_encoded)
        
        # Transformer解码 (交叉注意力)
        decoder_output = self.decoder(
            tgt=query_encoded,  # 查询序列
            memory=memory       # 编码器输出的MPO特征
        )  # [batch, dec_seq_len, d_model]
        
        # 3. 输出预测
        predictions = self.output_layer(decoder_output)  # [batch, dec_seq_len, 2]
        
        return predictions

def load_burgers_data(csv_path):
    """加载Burgers数据"""
    import pandas as pd
    
    df = pd.read_csv(csv_path, comment='%', header=None)
    
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
    
    return torch.from_numpy(result).float(), (num_timesteps, nx, ny, 2)