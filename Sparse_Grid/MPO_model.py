# new_MPO_model.py
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Tuple, Optional
import torch.nn.functional as F

class MPODecomposition(nn.Module):
    """
    MPO分解模型 - 支持不规则网格数据
    输入形状: (轨迹数, 时间步数, 节点数, 2, 1)
    节点数 = 1598, 2 = 速度分量, 1 = 扩展维度
    灵活的子张量数量设计
    """
    
    def __init__(self, num_nodes: int = 1598,
                 num_trajectories: int = 4,
                 time_steps_per_traj: int = 200,
                 bond_scale: float = 1.5,
                 num_tensors: int = 6):  # 可配置的子张量数量
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_trajectories = num_trajectories
        self.time_steps_per_traj = time_steps_per_traj
        self.total_time_steps = num_trajectories * time_steps_per_traj
        self.bond_scale = bond_scale
        self.num_tensors = num_tensors
        
        # 计算分解因子和键维数
        self.i_factors, self.j_factors, self.bond_dims = self._compute_factors_and_bonds()
        
        print(f"轨迹数: {num_trajectories}, 每轨迹时间步: {time_steps_per_traj}")
        print(f"节点数: {num_nodes}")
        print(f"子张量数量: {num_tensors}")
        print(f"空间分解: {self.i_factors}")
        print(f"速度分量分解: {self.j_factors}")
        print(f"键维数: {self.bond_dims}")
        
        # 初始化MPO张量
        self.tensors = self._initialize_tensors()
        
    def _compute_factors_and_bonds(self):
        """计算分解因子和键维数 - 灵活分解1598×2×1"""
        # 分解节点数1598和速度分量2
        i_factors=[4,4,33,3]
        j_factors=[1,1,1,2]
        # 计算键维数
        bond_dims = []
        for k in range(len(i_factors) - 1):
            left_size = np.prod(i_factors[:k+1]) * np.prod(j_factors[:k+1])
            right_size = np.prod(i_factors[k+1:]) * np.prod(j_factors[k+1:])
            bond_dim = int(self.bond_scale * min(left_size, right_size))
            bond_dims.append(max(8, bond_dim))
        
        return i_factors, j_factors, bond_dims
    
    def _initialize_tensors(self):
        """初始化MPO张量"""
        tensors = nn.ParameterList()
        
        # 第一个张量: [轨迹数, 时间步, i1, j1, bond1]
        tensor1 = nn.Parameter(torch.randn(
            self.num_trajectories, 
            self.i_factors[0], self.j_factors[0], self.bond_dims[0]
        ) * 0.1)
        tensors.append(tensor1)
        
        tensor2=nn.Parameter(torch.randn(
            self.num_trajectories, self.bond_dims[0],
            self.i_factors[1], self.j_factors[1], self.bond_dims[1]
        ) * 0.1)
        tensors.append(tensor2)
        
        tensor3=nn.Parameter(torch.randn(
            self.num_trajectories, self.time_steps_per_traj,self.bond_dims[1],
            self.i_factors[2], self.j_factors[2], self.bond_dims[2]
        ) * 0.1)
        tensors.append(tensor3)
        
        tensor4=nn.Parameter(torch.randn(
            self.num_trajectories, self.bond_dims[2],
            self.i_factors[3], self.j_factors[3]
        ) * 0.1)
        tensors.append(tensor4)
        for i in tensors:
            print(i.shape)
        
        return tensors
    
    def forward(self, time_indices: torch.Tensor):
        """前向传播 - 完整的张量收缩，支持批量处理"""
        batch_size = time_indices.shape[0]
    
        # 将全局时间索引转换为 (轨迹索引, 时间索引)
        traj_indices = time_indices // self.time_steps_per_traj
        time_in_traj = time_indices % self.time_steps_per_traj
    
        # 批量获取张量
        T1 = self.tensors[0][traj_indices]  # [batch_size, i1, j1, bond1]
        T2 = self.tensors[1][traj_indices]  # [batch_size, bond1, i2, j2, bond2]
        T3 = self.tensors[2][traj_indices, time_in_traj]  # [batch_size, bond2, i3, j3, bond3]
        T4 = self.tensors[3][traj_indices]  # [batch_size, bond3, i4, j4]
        
        # 第一步收缩: T1 × T2 → [batch_size, i1, j1, i2, j2, bond2]
        temp1 = torch.einsum('b i j d, b d p q w -> b i j p q w', T1, T2)
        new_i1 = temp1.shape[1] * temp1.shape[3]  # i1 * i2
        new_j1 = temp1.shape[2] * temp1.shape[4]  # j1 * j2
        temp1_reshaped = temp1.reshape(batch_size, new_i1, new_j1, self.bond_dims[1])
    
        # 第二步收缩: temp1_reshaped × T3 → [batch_size, new_i1, new_j1, i3, j3, bond3]
        temp2 = torch.einsum('b i j d, b d p q w -> b i j p q w', temp1_reshaped, T3)
        new_i2 = temp2.shape[1] * temp2.shape[3]  # new_i1 * i3
        new_j2 = temp2.shape[2] * temp2.shape[4]  # new_j1 * j3
        temp2_reshaped = temp2.reshape(batch_size, new_i2, new_j2, self.bond_dims[2])
    
        # 第三步收缩: temp2_reshaped × T4 → [batch_size, new_i2, new_j2, i4, j4]
        temp3 = torch.einsum('b i j d, b d p q -> b i j p q', temp2_reshaped, T4)
        
        # 合并空间维度: [batch_size, total_nodes, total_components, 1]
        total_nodes = temp3.shape[1] * temp3.shape[3]  # new_i2 * i4
        total_components = temp3.shape[2] * temp3.shape[4]  # new_j2 * j4
        final_tensor = temp3.reshape(batch_size, 1, total_nodes, total_components, 1)
    
        return final_tensor
    
    def get_time_features(self, time_indices: torch.Tensor):
        """获取时间序列特征 - 返回T3特征，支持批量"""
        traj_indices = time_indices // self.time_steps_per_traj
        time_in_traj = time_indices % self.time_steps_per_traj
        
        # 获取T3特征: [batch_size, bond_dims[1], i_factors[2], j_factors[2], bond_dims[2]]
        T3_features = self.tensors[2][traj_indices, time_in_traj]
        batch_size = T3_features.shape[0]
        flattened_features = T3_features.reshape(batch_size, -1)
        return flattened_features
    
    def get_all_T1_features(self):
        """获取所有训练好的T3特征 - 用于Transformer训练"""
        # 返回所有轨迹和时间的T3特征 [num_trajectories * time_steps_per_traj, T3_dim]
        all_T3 = self.tensors[2].reshape(-1, *self.tensors[2].shape[2:])
        flattened_T3 = all_T3.reshape(all_T3.shape[0], -1)
        return flattened_T3
    
    def get_time_series_features(self):
        """
        获取时间序列特征 - 轨迹间平均
        返回: [time_steps_per_traj, T3_dim] 每个时间步的轨迹平均特征
        """
        # T3张量形状: [num_trajectories, time_steps_per_traj, bond2, i3, j3, bond3]
        T3_tensor = self.tensors[2]  # [num_trajectories, time_steps_per_traj, ...]
        
        # 在轨迹维度求平均
        avg_T3 = T3_tensor.mean(dim=0)  # [time_steps_per_traj, bond2, i3, j3, bond3]
        
        # 展平特征维度
        flattened_avg_T3 = avg_T3.reshape(avg_T3.shape[0], -1)  # [time_steps_per_traj, T3_dim]
        
        return flattened_avg_T3
    
    def update_time_features(self, time_indices: torch.Tensor, new_features: torch.Tensor):
        """更新时间特征 - 更新T3特征"""
        traj_indices = time_indices // self.time_steps_per_traj
        time_in_traj = time_indices % self.time_steps_per_traj
        feature_shape = self.tensors[2].shape[2:]  # [bond_dims[1], i_factors[2], j_factors[2], bond_dims[2]]
        expected_feature_dim = np.prod(feature_shape)
        
        if new_features.shape[1] != expected_feature_dim:
            raise ValueError(f"特征维度不匹配: 期望{expected_feature_dim}, 得到{new_features.shape[1]}")
        
        reshaped_features = new_features.reshape(-1, *feature_shape)
        self.tensors[2].data[traj_indices, time_in_traj] = reshaped_features
    
    def reconstruct_from_T1(self, T1_features: torch.Tensor):
        """
        从T3特征重建物理场 - 使用与forward一致的张量收缩逻辑
        Args:
            T1_features: [batch_size, bond_dims[1], i_factors[2], j_factors[2], bond_dims[2]] 或 [batch_size, T3_dim]
        Returns:
            output: [batch_size, total_nodes, 2, 1]
        """
        batch_size = T1_features.shape[0]
    
        # 如果输入是展平的，需要重塑为T3特征形状
        if len(T1_features.shape) == 2:
            T3_features = T1_features.reshape(batch_size, *self.tensors[2].shape[2:])
        else:
            T3_features = T1_features
    
        # 从T3特征开始重建
        # 使用平均的T1和T2特征作为基准
        base_traj_idx = torch.zeros(batch_size, dtype=torch.long, device=T1_features.device)
        
        T1_base = self.tensors[0][base_traj_idx]  # [batch_size, i1, j1, bond1]
        T2_base = self.tensors[1][base_traj_idx]  # [batch_size, bond1, i2, j2, bond2]
        T4_base = self.tensors[3][base_traj_idx]  # [batch_size, bond2, i4, j4]
        
        # 第一步收缩: T1_base × T2_base
        temp1 = torch.einsum('b i j d, b d p q w -> b i j p q w', T1_base, T2_base)
        new_i1 = temp1.shape[1] * temp1.shape[3]  # i1 * i2
        new_j1 = temp1.shape[2] * temp1.shape[4]  # j1 * j2
        temp1_reshaped = temp1.reshape(batch_size, new_i1, new_j1, self.bond_dims[1])
        
        # 第二步收缩: temp1_reshaped × T3_features
        temp2 = torch.einsum('b i j d, b d p q w -> b i j p q w', temp1_reshaped, T3_features)
        new_i2 = temp2.shape[1] * temp2.shape[3]  # new_i1 * i3
        new_j2 = temp2.shape[2] * temp2.shape[4]  # new_j1 * j3
        temp2_reshaped = temp2.reshape(batch_size, new_i2, new_j2, self.bond_dims[2])
        
        # 第三步收缩: temp2_reshaped × T4_base
        temp3 = torch.einsum('b i j d, b d p q -> b i j p q', temp2_reshaped, T4_base)
        total_nodes = temp3.shape[1] * temp3.shape[3]  # new_i2 * i4
        total_components = temp3.shape[2] * temp3.shape[4]  # new_j2 * j4
        final_tensor = temp3.reshape(batch_size, total_nodes, total_components, 1)
        
        return final_tensor
    
    def get_parameter_info(self):
        """获取参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        original_size = self.num_nodes * 2 * 1 * self.total_time_steps
        
        return {
            'total_params': total_params,
            'original_size': original_size,
            'compression_ratio': original_size / total_params,
            'i_factors': self.i_factors,
            'j_factors': self.j_factors,
            'bond_dims': self.bond_dims,
            'num_tensors': self.num_tensors,
            'num_trajectories': self.num_trajectories,
            'time_steps_per_traj': self.time_steps_per_traj
        }


# 保持原有的Transformer组件不变
class FixedTimeEncoding(nn.Module):
    """不可训练时间编码"""
    def __init__(self, d_model, max_len=5000):
        super(FixedTimeEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

def generate_causal_mask(sz: int, device=None):
    """生成上三角的因果mask"""
    mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
    mask = mask.transpose(0, 1)
    mask = ~mask
    float_mask = torch.zeros(sz, sz, device=device)
    float_mask[mask] = float('-inf')
    return float_mask

class T1Transformer(nn.Module):
    """
    T3特征时间序列Transformer - 预测T3特征的时间演化
    注意：类名保持为T1Transformer以保持接口一致，但实际处理的是T3特征
    """
    def __init__(self, T1_dim: int, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super(T1Transformer, self).__init__()
        
        self.d_model = d_model
        self.T1_dim = T1_dim

        # 输入投影
        self.input_projection = nn.Linear(self.T1_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # 位置编码
        self.pos_encoder_enc = FixedTimeEncoding(d_model)
        self.pos_encoder_dec = FixedTimeEncoding(d_model)
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
            nn.Linear(dim_feedforward, self.T1_dim)
        )

    def forward(self, src_T1, tgt_T1):
        device = src_T1.device
        batch_size, enc_seq_len, _ = src_T1.shape
        dec_seq_len = tgt_T1.shape[1]

        # 编码器
        encoder_input = self.input_norm(self.input_projection(src_T1))
        encoder_input = self.pos_encoder_enc(encoder_input)
        encoder_input = self.dropout(encoder_input)
        memory = self.encoder(encoder_input)

        # 解码器
        decoder_input = self.input_norm(self.input_projection(tgt_T1))
        decoder_input = self.pos_encoder_dec(decoder_input)
        decoder_input = self.dropout(decoder_input)

        tgt_mask = generate_causal_mask(dec_seq_len, device=device)
        decoder_output = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)

        predictions = self.output_layer(decoder_output)
        return predictions

    def predict_future(self, src_T1, pred_len):
        batch_size, src_seq_len, T1_dim = src_T1.shape
        device = src_T1.device
        
        last_T1 = src_T1[:, -1:, :]
        decoder_input = last_T1.repeat(1, pred_len, 1)
        
        encoder_input = self.input_norm(self.input_projection(src_T1))
        encoder_input = self.pos_encoder_enc(encoder_input)
        memory = self.encoder(encoder_input)
        
        tgt_mask = generate_causal_mask(pred_len, device=device)
        decoder_input_proj = self.input_norm(self.input_projection(decoder_input))
        decoder_input_pe = self.pos_encoder_dec(decoder_input_proj)
        
        decoder_output = self.decoder(decoder_input_pe, memory, tgt_mask=tgt_mask)
        future_T1 = self.output_layer(decoder_output)
        return future_T1


class EnhancedMPOTransformer(nn.Module):
    """
    增强的MPO-Transformer模型
    结合MPO分解和Transformer时间序列预测
    """
    def __init__(self, num_nodes: int = 1598,
                 num_trajectories: int = 4,
                 time_steps_per_traj: int = 200,
                 bond_scale: float = 1.5,
                 num_tensors: int = 6,
                 d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4,
                 dropout: float = 0.1):
        super(EnhancedMPOTransformer, self).__init__()
        
        # MPO分解模块
        self.mpo_decomposition = MPODecomposition(
            num_nodes=num_nodes,
            num_trajectories=num_trajectories,
            time_steps_per_traj=time_steps_per_traj,
            bond_scale=bond_scale,
            num_tensors=num_tensors
        )
        
        # T3特征维度（保持接口名称为T1_dim，但实际是T3的维度）
        T1_dim = self.mpo_decomposition.tensors[2].shape[2:].numel()
        
        # Transformer时间序列预测模块
        self.T1_transformer = T1Transformer(
            T1_dim=T1_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        )
        
        self.num_nodes = num_nodes
        self.num_trajectories = num_trajectories
        self.time_steps_per_traj = time_steps_per_traj
        self.T1_dim = T1_dim
    
    def forward(self, time_indices: torch.Tensor, 
                src_T1: Optional[torch.Tensor] = None,
                tgt_T1: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            time_indices: [batch_size] 时间索引
            src_T1: [batch, seq_len, T1_dim] Transformer编码器输入（T3特征）
            tgt_T1: [batch, seq_len, T1_dim] Transformer解码器输入（T3特征）
        Returns:
            T1_predictions: T3特征预测
            spatial_reconstruction: 空间场重建
        """
        # MPO分解获取当前T3特征和重建
        current_reconstruction = self.mpo_decomposition(time_indices)
        current_T1_features = self.mpo_decomposition.get_time_features(time_indices)
        
        # Transformer预测（如果提供序列数据）
        T1_predictions = None
        if src_T1 is not None and tgt_T1 is not None:
            T1_predictions = self.T1_transformer(src_T1, tgt_T1)
        
        return T1_predictions, current_reconstruction
    
    def predict_future(self, historical_time_indices: torch.Tensor, pred_steps: int) -> torch.Tensor:
        """
        预测未来多个时间步
        Args:
            historical_time_indices: [hist_T] 历史时间索引
            pred_steps: 预测步数
        Returns:
            future_fields: [pred_steps, 节点数, 2, 1] 预测的未来场
        """
        hist_T = historical_time_indices.shape[0]
        
        # 对历史数据获取T3特征
        historical_T1 = self.mpo_decomposition.get_time_features(historical_time_indices)
        
        # 使用Transformer预测未来T3特征
        src_T1 = historical_T1.unsqueeze(0)
        future_T1 = self.T1_transformer.predict_future(src_T1, pred_steps)
        future_T1 = future_T1.squeeze(0)
        
        # 从预测的T3特征重建空间场
        future_fields = self.mpo_decomposition.reconstruct_from_T1(future_T1)
        
        return future_fields