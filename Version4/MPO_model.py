# MPO_model.py
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Callable, Tuple

class TemporalMPODecomposition(nn.Module):
    """
    MPO分解模型 - 输出形状 (T, X, Y, 2)
    - 输入: (X, Y, T) 的时空数据
    - 通过矩阵A(rank_x, X), B(Y, rank_y)压缩空间维度
    - 与核心张量 (T, rank_t) 相互作用
    - 通过逆矩阵重建空间，输出 (T, X, Y, 2)
    """
    def __init__(self, spatial_shape: Tuple[int, int], time_steps: int, 
                 ranks: Tuple[int, int, int] = (24, 24, 48), output_dim: int = 2):
        super(TemporalMPODecomposition, self).__init__()
        self.X, self.Y = spatial_shape  # 空间维度
        self.T = time_steps  # 时间步数
        self.rank_x, self.rank_y, self.rank_t = ranks
        self.output_dim = output_dim
        
        # 空间压缩矩阵
        self.matrix_A = nn.Parameter(torch.randn(self.rank_x, self.X,dtype=torch.float64) * 0.1)  # [rank_x, X]
        self.matrix_B = nn.Parameter(torch.randn(self.Y, self.rank_y,dtype=torch.float64) * 0.1)  # [Y, rank_y]
        
        # 空间重建矩阵 (A和B的伪逆)
        self.matrix_A_inv = nn.Parameter(torch.randn(self.X, self.rank_x,dtype=torch.float64) * 0.1)  # [X, rank_x]
        self.matrix_B_inv = nn.Parameter(torch.randn(self.rank_y, self.Y,dtype=torch.float64) * 0.1)  # [rank_y, Y]
        
        # 时间核心张量
        self.core_tensor = nn.Parameter(torch.randn(self.T, self.rank_t,dtype=torch.float64) * 0.1)  # [T, rank_t]
        
        # 输出映射矩阵 - 只映射时间特征到输出
        self.matrix_C = nn.Parameter(torch.randn(self.rank_t, self.output_dim,dtype=torch.float64) * 0.1)  # [rank_t, 2]
        
        # 交互权重
        self.interaction_weights = nn.Parameter(torch.randn(self.rank_t, self.rank_x, self.rank_y,dtype=torch.float64) * 0.1)
        
        
        
    def forward(self, input_data: torch.Tensor, custom_core: torch.Tensor = None):
        """
        前向传播 - 输出形状 (T, X, Y, 2)
        Args:
            input_data: [X, Y, T] 输入数据
            custom_core: 预测时传入的core_tensor [T, rank_t] 或 [rank_t]
        Returns:
            output: [T, X, Y, 2] 预测值
        """
        if custom_core is not None:
            # 预测模式：使用传入的core_tensor
            return self._predict_with_custom_core(input_data, custom_core)
        else:
            # 训练模式：使用自学习的core_tensor
            return self._train_forward(input_data)
    
    def _train_forward(self, input_data: torch.Tensor):
        """训练模式前向传播"""
        # input_data: [X, Y, T]
        
        # 空间压缩: [X, Y, T] -> [rank_x, rank_y, T]
        # 使用einsum进行批量矩阵乘法
        compressed = torch.einsum('ix,xyt,yj->ijt', self.matrix_A, input_data, self.matrix_B)
        
        # 时空特征融合
        temporal_output = self._fuse_features(compressed, self.core_tensor)  # [rank_x, rank_y, T, 2]
        
        # 空间重建: [rank_x, rank_y, T, 2] -> [X, Y, T, 2]
        print(self.matrix_A_inv.shape,temporal_output.shape,self.matrix_B_inv.shape)
        reconstructed = torch.einsum('xi,ijtk,jy->xytk', self.matrix_A_inv, temporal_output, self.matrix_B_inv)
        
        # 转置为 [T, X, Y, 2]
        output = reconstructed.permute(2, 0, 1, 3)
        
        return output
    
    def _predict_with_custom_core(self, input_data: torch.Tensor, custom_core: torch.Tensor):
        """
        预测模式：使用自定义core_tensor
        Args:
            input_data: [X, Y, 1] 单时间步空间数据
            custom_core: [rank_t] 预测的core_tensor
        Returns:
            output: [1, X, Y, 2] 预测值 (单时间步)
        """
        # input_data: [X, Y, 1]
        # 扩展为 [X, Y, T=1] 以保持维度一致性
        spatial_data = input_data  # [X, Y, 1]
        
        # 空间压缩: [X, Y, 1] -> [rank_x, rank_y, 1]
        compressed = torch.einsum('ix,xya,yj->ija', self.matrix_A, spatial_data, self.matrix_B)
        
        # 将custom_core扩展为 [1, rank_t]
        if len(custom_core.shape) == 1:
            custom_core = custom_core.unsqueeze(0)  # [1, rank_t]
        
        # 时空特征融合
        temporal_output = self._fuse_features(compressed, custom_core)  # [rank_x, rank_y, 1, 2]
        
        # 空间重建: [rank_x, rank_y, 1, 2] -> [X, Y, 1, 2]
        reconstructed = torch.einsum('xi,ijak,jy->xyak', self.matrix_A_inv, temporal_output, self.matrix_B_inv)
        
        # 转置为 [1, X, Y, 2]
        output = reconstructed.permute(2, 0, 1, 3)
        
        return output
    
    def _fuse_features(self, spatial_compressed: torch.Tensor, time_core: torch.Tensor):
        """
        时空特征融合
        Args:
            spatial_compressed: [rank_x, rank_y, T] 空间压缩特征
            time_core: [T, rank_t] 时间核心特征
        Returns:
            fused: [rank_x, rank_y, T, 2] 融合后的特征
        """
        T = spatial_compressed.shape[2]
        
        # 扩展空间特征: [rank_x, rank_y, T] -> [rank_x, rank_y, T, 1]
        spatial_expanded = spatial_compressed.unsqueeze(-1)
        
        # 扩展时间特征: [T, rank_t] -> [1, 1, T, rank_t]
        time_expanded = time_core.permute(1, 0).unsqueeze(0).unsqueeze(0)  # [1, 1, rank_t, T] -> [1, 1, T, rank_t]
        time_expanded = time_expanded.permute(0, 1, 3, 2)  # [1, 1, T, rank_t]
        
        # 特征交互: [rank_x, rank_y, T, 1] × [1, 1, T, rank_t] -> [rank_x, rank_y, T, rank_t]
        # 使用广播机制
        interacted = spatial_expanded * time_expanded
        
        # 应用交互权重: [rank_x, rank_y, T, rank_t] × [rank_t, rank_x, rank_y] -> [rank_x, rank_y, T, rank_t]
        # 这里简化处理，直接使用element-wise乘法
        weight_expanded = self.interaction_weights.permute(1, 2, 0).unsqueeze(2)  # [rank_x, rank_y, 1, rank_t]
        weighted = interacted * weight_expanded
        
        # 时间特征映射到输出: [rank_x, rank_y, T, rank_t] × [rank_t, 2] -> [rank_x, rank_y, T, 2]
        output = torch.einsum('ijtk,kl->ijtl', weighted, self.matrix_C)
        
        return output
    
    def get_core_tensor(self):
        """获取核心张量 [T, rank_t]"""
        return self.core_tensor

# 以下保持原有的Transformer代码不变
class FixedTimeEncoding(nn.Module):
    """不可训练时间编码"""
    def __init__(self, d_model, max_len=5000):
        super(FixedTimeEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

def generate_causal_mask(sz: int, device=None):
    """生成上三角的因果 mask"""
    mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
    mask = mask.transpose(0, 1)
    mask = ~mask
    float_mask = torch.zeros(sz, sz, device=device)
    float_mask[mask] = float('-inf')
    return float_mask

class CoreTensorTransformer(nn.Module):
    """
    核心张量时间序列Transformer - 预测核心张量的时间演化
    """
    def __init__(self, core_dim: int, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super(CoreTensorTransformer, self).__init__()
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        
        self.d_model = d_model
        self.core_dim = core_dim

        # 输入投影
        self.input_projection = nn.Linear(self.core_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # 位置编码
        self.pos_encoder_enc = FixedTimeEncoding(d_model)
        self.pos_encoder_dec = FixedTimeEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出层 - 预测核心张量
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.core_dim)
        )
        torch.set_default_dtype(original_dtype)

    def forward(self, src_cores, tgt_cores):
        """
        Args:
            src_cores: [batch_size, enc_seq_len, core_dim] - 编码器输入核心张量
            tgt_cores: [batch_size, dec_seq_len, core_dim] - 解码器输入核心张量
        Returns:
            predictions: [batch_size, dec_seq_len, core_dim] - 预测的核心张量
        """
        device = src_cores.device
        batch_size, enc_seq_len, _ = src_cores.shape
        dec_seq_len = tgt_cores.shape[1]

        # --- 编码器 ---
        encoder_input = self.input_norm(self.input_projection(src_cores))
        encoder_input = self.pos_encoder_enc(encoder_input)
        encoder_input = self.dropout(encoder_input)

        memory = self.encoder(encoder_input)  # [batch, enc_seq_len, d_model]

        # --- 解码器 ---
        decoder_input = self.input_norm(self.input_projection(tgt_cores))
        decoder_input = self.pos_encoder_dec(decoder_input)
        decoder_input = self.dropout(decoder_input)

        # 生成 decoder 因果 mask
        tgt_mask = generate_causal_mask(dec_seq_len, device=device)

        # Transformer Decoder
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [batch, dec_seq_len, d_model]

        # --- 输出 ---
        predictions = self.output_layer(decoder_output)  # [batch, dec_seq_len, core_dim]
        return predictions

    def predict_future(self, src_cores, pred_len):
        """
        预测未来多个时间步的核心张量
        Args:
            src_cores: [batch_size, src_seq_len, core_dim] - 输入核心张量序列
            pred_len: 要预测的未来时间步数
        Returns:
            future_cores: [batch_size, pred_len, core_dim] - 预测的未来核心张量
        """
        batch_size, src_seq_len, core_dim = src_cores.shape
        device = src_cores.device
        
        # 使用最后一个时间步作为解码器初始输入
        last_core = src_cores[:, -1:, :]  # [batch_size, 1, core_dim]
        
        # 复制作为解码器输入 (自回归的起始点)
        decoder_input = last_core.repeat(1, pred_len, 1)  # [batch_size, pred_len, core_dim]
        
        # 编码器输入
        encoder_input = self.input_norm(self.input_projection(src_cores))
        encoder_input = self.pos_encoder_enc(encoder_input)
        memory = self.encoder(encoder_input)
        
        # 解码器掩码
        tgt_mask = generate_causal_mask(pred_len, device=device)
        
        # 解码器前向传播
        decoder_input_proj = self.input_norm(self.input_projection(decoder_input))
        decoder_input_pe = self.pos_encoder_dec(decoder_input_proj)
        
        decoder_output = self.decoder(
            tgt=decoder_input_pe,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        future_cores = self.output_layer(decoder_output)
        return future_cores