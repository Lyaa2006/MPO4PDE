# new_MPO_training.py
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from MPO_model import MPODecomposition, T1Transformer, EnhancedMPOTransformer
from load_data import create_mpo_dataloader
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.rcParams['font.family'] = 'SimHei'

def loss_fn(pred, gt):
    """MSE损失 - 每个轨迹先和自己的真实数据比较，再求平均"""
    batch_size = pred.shape[0]
    trajectory_losses = []
    
    # 对每个轨迹单独计算损失
    for i in range(batch_size):
        # 单个轨迹的MSE损失
        single_loss = F.mse_loss(pred[i], gt[i])
        trajectory_losses.append(single_loss)
    
    # 对所有轨迹的损失求平均
    return torch.stack(trajectory_losses).mean()

def loss_fn2(pred, gt):
    """MAE损失 - 每个轨迹先和自己的真实数据比较，再求平均"""
    batch_size = pred.shape[0]
    trajectory_losses = []
    
    # 对每个轨迹单独计算损失
    for i in range(batch_size):
        # 单个轨迹的MAE损失
        single_loss = F.l1_loss(pred[i], gt[i])
        trajectory_losses.append(single_loss)
    
    # 对所有轨迹的损失求平均
    return torch.stack(trajectory_losses).mean()

def train_mpo_decomposition(config, device, data_config, model_name="train_mpo"):
    """训练MPO分解模型 - 优化批量处理版本"""
    print(f"=== 训练MPO分解模型 ({model_name}) ===")
    
    # 加载数据
    dataloader, dataset_info = create_mpo_dataloader(data_config)
    
    # 获取轨迹数和节点数
    num_trajectories = len(data_config['trajectories'])
    num_nodes = dataset_info['num_nodes']
    
    # 创建MPO模型
    model = MPODecomposition(
        num_nodes=num_nodes,
        num_trajectories=num_trajectories,
        time_steps_per_traj=config.time_used,
        bond_scale=config.bond_scale,
        num_tensors=config.num_tensors
    )
    model.to(device)
    
    # 显示模型信息
    param_info = model.get_parameter_info()
    print(f"\n模型信息 ({model_name}):")
    print(f"轨迹数: {num_trajectories}")
    print(f"节点数: {num_nodes}")
    print(f"总参数量: {param_info['total_params']:,}")
    print(f"原始数据量: {param_info['original_size']:,}")
    print(f"压缩比: {param_info['compression_ratio']:.2f}x")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.mpo_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # 训练循环 - 优化批量处理
    best_loss = float('inf')
    train_losses = []
    
    for epoch in tqdm(range(config.mpo_epochs)):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # 获取轨迹数据 [batch, time_steps, num_nodes, 2]
            trajectory_data = batch['trajectory_data'].to(device)
            positions = batch['positions'].to(device)
            
            batch_size, time_steps, num_nodes, _ = trajectory_data.shape
            
            # 创建所有时间步的索引 - 批量处理
            all_time_indices = []
            all_target_data = []
            
            for t in range(time_steps):
                # 为每个时间步创建索引
                time_indices = torch.tensor([t] * batch_size, dtype=torch.long).to(device)
                target_data = trajectory_data[:, t].unsqueeze(-1)  # [batch, num_nodes, 2, 1]
                
                all_time_indices.append(time_indices)
                all_target_data.append(target_data)
            
            # 批量处理所有时间步
            batch_loss = 0
            for t in range(time_steps):
                time_indices = all_time_indices[t]
                target_data = all_target_data[t]
                
                # 前向传播 - 批量处理
                reconstructed = model(time_indices)  # [batch, 1, num_nodes, 2, 1]
                
                # 计算重建损失 - 使用改进的损失函数
                loss = loss_fn(reconstructed, target_data)
                batch_loss += loss
            
            avg_batch_loss = batch_loss / time_steps
            epoch_loss += avg_batch_loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_epoch_loss)
        
        # 评估
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_batches = 0
                for batch in dataloader:
                    trajectory_data = batch['trajectory_data'].to(device)
                    batch_size, time_steps, num_nodes, _ = trajectory_data.shape
                    
                    batch_loss = 0
                    for t in range(time_steps):
                        target_data = trajectory_data[:, t].unsqueeze(-1)
                        time_indices = torch.tensor([t] * batch_size, dtype=torch.long).to(device)
                        reconstructed = model(time_indices)
                        batch_loss += loss_fn(reconstructed, target_data)
                    
                    eval_loss += (batch_loss / time_steps).item()
                    eval_batches += 1
                
                avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0
                
                print(f'MPO {model_name} Epoch {epoch:04d}, Train Loss: {avg_epoch_loss:.6f}, '
                      f'Eval Loss: {avg_eval_loss:.6f}')
                
                scheduler.step(avg_eval_loss)
                
                if avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'param_info': param_info
                    }, f'{config.output_dir}/{model_name}_best.pth')
    
    print(f"MPO分解训练完成! 最佳Loss: {best_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.title(f'MPO training loss ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{config.output_dir}/{model_name}_training_loss.png')
    plt.show()
    
    return model, param_info

def create_T1_sequence_data_averaged(mpo_model, seq_len, pred_len, device):
    """
    创建轨迹间平均的T1特征序列数据
    返回: [num_sequences, seq_len, T1_dim] 和 [num_sequences, pred_len, T1_dim]
    """
    mpo_model.eval()
    
    # 获取轨迹间平均的时间序列特征
    avg_time_series = mpo_model.get_time_series_features()  # [time_steps_per_traj, T1_dim]
    time_steps_per_traj = avg_time_series.shape[0]
    
    sequences = []
    targets = []
    
    # 创建序列-目标对 - 使用平均后的时间序列
    for i in range(time_steps_per_traj - seq_len - pred_len + 1):
        src_seq = avg_time_series[i:i+seq_len]  # [seq_len, T1_dim]
        tgt_seq = avg_time_series[i+seq_len:i+seq_len+pred_len]  # [pred_len, T1_dim]
        
        sequences.append(src_seq)
        targets.append(tgt_seq)
    
    sequences = torch.stack(sequences)  # [num_sequences, seq_len, T1_dim]
    targets = torch.stack(targets)      # [num_sequences, pred_len, T1_dim]
    
    print(f"轨迹平均T1特征序列数据: {len(sequences)} 个序列")
    print(f"输入序列形状: {sequences.shape}, 目标序列形状: {targets.shape}")
    
    return sequences, targets

class TimeAwareTransformerTrainer:
    """时间感知的Transformer训练器"""
    
    def __init__(self, model, device, seq_len, pred_len):
        self.model = model
        self.device = device
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def apply_time_masking(self, batch_src, mask_ratio=0.15):
        """应用时间masking增强时间序列预测能力"""
        batch_size, seq_len, feature_dim = batch_src.shape
        
        # 创建随机mask
        mask = torch.rand(batch_size, seq_len, device=self.device) > mask_ratio
        masked_batch = batch_src.clone()
        
        # 对masked位置添加噪声
        noise_std = 0.1
        for i in range(batch_size):
            for t in range(seq_len):
                if not mask[i, t]:
                    # 用前后时间步的均值加噪声来mask
                    if t == 0:
                        # 第一个时间步，用下一个时间步
                        masked_value = batch_src[i, t+1] if seq_len > 1 else batch_src[i, t]
                    elif t == seq_len - 1:
                        # 最后一个时间步，用前一个时间步
                        masked_value = batch_src[i, t-1]
                    else:
                        # 中间时间步，用前后均值
                        masked_value = (batch_src[i, t-1] + batch_src[i, t+1]) / 2
                    
                    noise = torch.randn_like(masked_value) * noise_std
                    masked_batch[i, t] = masked_value + noise
        
        return masked_batch, mask

def train_T1_transformer(config, mpo_model, device):
    """训练T1特征Transformer模型 - 使用轨迹间平均的时间序列"""
    print("\n=== 阶段2: 训练T1特征Transformer模型 (轨迹间平均) ===")
    
    # 创建轨迹间平均的T1特征序列数据
    sequences, targets = create_T1_sequence_data_averaged(
        mpo_model, config.seq_len, config.pred_len, device
    )
    
    # 创建数据集
    train_dataset = TensorDataset(sequences, targets)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"Transformer训练数据: {len(train_dataset)} 序列")
    
    # 创建T1 Transformer模型
    T1_dim = sequences.shape[2]  # T1特征维度
    transformer = T1Transformer(
        T1_dim=T1_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout
    ).to(device)
    
    # 创建时间感知训练器
    time_trainer = TimeAwareTransformerTrainer(transformer, device, config.seq_len, config.pred_len)
    
    # 优化器
    optimizer = optim.AdamW(transformer.parameters(), lr=config.transformer_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    for epoch in tqdm(range(config.transformer_epochs)):
        transformer.train()
        
        epoch_loss = 0
        
        for batch_src, batch_tgt in train_loader:
            optimizer.zero_grad()
            
            batch_src = batch_src.to(device)  # [batch, seq_len, T1_dim]
            batch_tgt = batch_tgt.to(device)  # [batch, pred_len, T1_dim]
            
            # 应用时间masking增强（每5个epoch应用一次）
            if epoch % 5 == 0:
                masked_src, _ = time_trainer.apply_time_masking(batch_src, mask_ratio=0.15)
            else:
                masked_src = batch_src
            
            # Transformer前向传播
            predictions = transformer(masked_src, batch_tgt)  # [batch, pred_len, T1_dim]
            
            loss = F.mse_loss(predictions, batch_tgt)  # 直接使用MSE，因为已经是平均后的序列
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # 评估
        if epoch % 20 == 0:
            transformer.eval()
            with torch.no_grad():
                # 使用训练数据计算验证损失
                val_loss = 0
                for batch_src, batch_tgt in train_loader:
                    batch_src = batch_src.to(device)
                    batch_tgt = batch_tgt.to(device)
                    
                    predictions = transformer(batch_src, batch_tgt)
                    val_loss += F.mse_loss(predictions, batch_tgt).item()
                
                avg_val_loss = val_loss / len(train_loader)
                
                print(f'T1 Transformer Epoch {epoch:04d}, Train Loss: {avg_epoch_loss:.6f}, '
                      f'Val Loss: {avg_val_loss:.6f}')
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save({
                        'model_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'T1_dim': T1_dim
                    }, f'{config.output_dir}/T1_transformer_best.pth')
    
    print(f"T1特征Transformer训练完成! 最佳Loss: {best_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.title('T1 Transformer training loss (轨迹间平均)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{config.output_dir}/transformer_training_loss.png')
    plt.show()
    
    return transformer, best_loss

def interpolate_to_grid(pred_data, target_data, positions, grid_size=(64, 64)):
    """将不规则网格数据插值到规则网格"""
    # 确保输入数据没有batch维度
    if len(pred_data.shape) == 3:  # [batch, num_nodes, 2]
        pred_data = pred_data[0]   # 取第一个样本 [num_nodes, 2]
    if len(target_data.shape) == 3:
        target_data = target_data[0]
    if len(positions.shape) == 3:  # [batch, num_nodes, 2]
        positions = positions[0]   # 取第一个样本 [num_nodes, 2]
    
    # 提取坐标和值
    x_coords = positions[:, 0].cpu().numpy()  # [num_nodes]
    y_coords = positions[:, 1].cpu().numpy()  # [num_nodes]
    
    # 创建规则网格
    grid_x = np.linspace(x_coords.min(), x_coords.max(), grid_size[0])
    grid_y = np.linspace(y_coords.min(), y_coords.max(), grid_size[1])
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    
    # 分别处理u和v分量
    pred_u = pred_data[:, 0].cpu().numpy()  # [num_nodes]
    pred_v = pred_data[:, 1].cpu().numpy()  # [num_nodes]
    target_u = target_data[:, 0].cpu().numpy()
    target_v = target_data[:, 1].cpu().numpy()
    
    # 准备插值点
    points = np.column_stack((x_coords, y_coords))  # [num_nodes, 2]
    
    # 插值
    pred_u_grid = griddata(points, pred_u, (grid_xx, grid_yy), method='linear', fill_value=0)
    pred_v_grid = griddata(points, pred_v, (grid_xx, grid_yy), method='linear', fill_value=0)
    target_u_grid = griddata(points, target_u, (grid_xx, grid_yy), method='linear', fill_value=0)
    target_v_grid = griddata(points, target_v, (grid_xx, grid_yy), method='linear', fill_value=0)
    
    # 合并分量
    pred_grid = np.stack([pred_u_grid, pred_v_grid], axis=-1)  # [grid_size[0], grid_size[1], 2]
    target_grid = np.stack([target_u_grid, target_v_grid], axis=-1)
    
    return pred_grid, target_grid, (grid_x, grid_y)

def test_full_model_with_test_data(config, transformer, device):
    """使用独立的测试集评估完整模型 - 重新训练MPO"""
    print("\n=== 阶段3: 使用测试集评估完整模型 ===")
    
    # 检查测试数据路径是否存在
    if not hasattr(config, 'test_data_path') or config.test_data_path is None:
        print("警告: 未指定测试数据路径，跳过测试阶段")
        return 0, 0, 0
    
    # 测试数据配置 - 使用测试轨迹
    test_data_config = {
        'data_path': config.test_data_path,
        'trajectories': config.test_trajectories if hasattr(config, 'test_trajectories') else [0],
        'time_start': config.test_time_start if hasattr(config, 'test_time_start') else 0,
        'time_used': config.test_time_used if hasattr(config, 'test_time_used') else config.time_used,
    }
    
    # 重新为测试数据训练MPO模型
    test_mpo_model, test_param_info = train_mpo_decomposition(
        config, device, test_data_config, model_name="test_mpo"
    )
    
    transformer.eval()
    test_mpo_model.eval()
    
    # 加载测试数据用于评估
    test_dataloader, test_dataset_info = create_mpo_dataloader(test_data_config)
    
    print(f"测试数据信息: {test_dataset_info}")
    
    total_physical_loss = 0
    total_physical_mae = 0
    total_T1_loss = 0
    
    predictions_list = []
    targets_list = []
    positions_list = []
    time_indices_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            trajectory_data = batch['trajectory_data'].to(device)  # [batch, time_steps, num_nodes, 2]
            positions = batch['positions'].to(device)  # [batch, num_nodes, 2]
            
            batch_size, time_steps, num_nodes, _ = trajectory_data.shape
            
            for t in range(time_steps):
                # 构建历史序列
                hist_start = max(0, t - config.seq_len)
                hist_indices = list(range(hist_start, t))
                
                if len(hist_indices) < config.seq_len:
                    continue
                
                # 获取历史T1特征 - 使用测试MPO模型
                hist_time_tensor = torch.tensor(hist_indices, dtype=torch.long).to(device)
                historical_T1 = test_mpo_model.get_time_features(hist_time_tensor)  # [seq_len, T1_dim]
                
                # 使用Transformer预测下一个时间步的T1特征
                src_T1 = historical_T1.unsqueeze(0)  # [1, seq_len, T1_dim]
                pred_T1 = transformer.predict_future(src_T1, 1)  # [1, 1, T1_dim]
                pred_T1_single = pred_T1[0, 0]  # [T1_dim]
                
                # 从预测的T1特征重建物理场 - 使用测试MPO模型
                pred_field = test_mpo_model.reconstruct_from_T1(pred_T1_single.unsqueeze(0))  # [1, num_nodes, 2, 1]
                pred_field = pred_field.squeeze(-1).squeeze(0)  # [num_nodes, 2]
                
                # 修复：直接从测试数据获取真实值，而不是通过MPO模型
                true_field = trajectory_data[0, t]  # [num_nodes, 2] - 直接从原始数据获取
                
                # 获取真实的T1特征用于T1损失计算
                true_time_idx = torch.tensor([t]).to(device)
                true_T1_features = test_mpo_model.get_time_features(true_time_idx)  # [1, T1_dim]
                
                # 计算损失
                physical_loss = loss_fn(pred_field.unsqueeze(0), true_field.unsqueeze(0))
                physical_mae = loss_fn2(pred_field.unsqueeze(0), true_field.unsqueeze(0))
                T1_loss = F.mse_loss(pred_T1_single.unsqueeze(0), true_T1_features)
                
                total_physical_loss += physical_loss.item()
                total_physical_mae += physical_mae.item()
                total_T1_loss += T1_loss.item()
                
                # 保存用于可视化
                if len(predictions_list) < 10 and t % 10 == 0:  # 只保存少量样本用于可视化
                    predictions_list.append(pred_field.cpu())
                    targets_list.append(true_field.cpu())
                    positions_list.append(positions.cpu())
                    time_indices_list.append(t)
                
                if batch_idx % 10 == 0 and t % 20 == 0:
                    current_samples = batch_idx * time_steps + t + 1
                    current_physical_loss = total_physical_loss / current_samples
                    current_physical_mae = total_physical_mae / current_samples
                    current_T1_loss = total_T1_loss / current_samples
                    print(f'测试样本 {current_samples:03d}: '
                          f'Physical Loss={current_physical_loss:.6f}, '
                          f'Physical MAE={current_physical_mae:.6f}, '
                          f'T1 Loss={current_T1_loss:.6f}')
    
    num_eval_steps = len(test_dataloader.dataset) * time_steps
    avg_physical_loss = total_physical_loss / num_eval_steps if num_eval_steps > 0 else 0
    avg_physical_mae = total_physical_mae / num_eval_steps if num_eval_steps > 0 else 0
    avg_T1_loss = total_T1_loss / num_eval_steps if num_eval_steps > 0 else 0
    
    print(f"\n测试结果:")
    print(f"测试样本数: {num_eval_steps}")
    print(f"测试轨迹: {config.test_trajectories}")
    print(f"平均物理场MSE损失: {avg_physical_loss:.6f}")
    print(f"平均物理场MAE损失: {avg_physical_mae:.6f}")
    print(f"平均T1特征MSE损失: {avg_T1_loss:.6f}")
    
    # 可视化一些预测结果
    if len(predictions_list) > 0:
        visualize_predictions(predictions_list, targets_list, positions_list, time_indices_list, config.output_dir)
    
    return avg_physical_loss, avg_physical_mae, avg_T1_loss

def visualize_predictions(predictions, targets, positions_list, time_indices, output_dir, grid_size=(64, 64)):
    """可视化预测结果 - u/v分量分开绘图"""
    n_samples = min(2, len(predictions))
    
    # 改为4行（u预测、u真实、v预测、v真实），3列（预测、真实、误差）
    fig, axes = plt.subplots(n_samples * 2, 3, figsize=(15, 8*n_samples))
    if n_samples == 1:
        axes = axes.reshape(2, -1)
    
    for i in range(n_samples):
        pred = predictions[i]  # [num_nodes, 2]
        target = targets[i]    # [num_nodes, 2]
        positions = positions_list[i]  # [num_nodes, 2]
        time_idx = time_indices[i]
        
        # 插值到规则网格
        pred_grid, target_grid, (grid_x, grid_y) = interpolate_to_grid(
            pred, target, positions, grid_size
        )
        
        # 提取u分量和v分量
        pred_u = pred_grid[:, :, 0]  # u分量
        pred_v = pred_grid[:, :, 1]  # v分量
        target_u = target_grid[:, :, 0]
        target_v = target_grid[:, :, 1]
        
        # 计算误差
        u_error = np.abs(pred_u - target_u)
        v_error = np.abs(pred_v - target_v)
        
        # u分量的行索引
        u_row = i * 2
        # v分量的行索引
        v_row = i * 2 + 1
        
        # u分量预测图
        im1 = axes[u_row, 0].imshow(pred_u, cmap='viridis', aspect='auto',
                                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], origin='lower')
        axes[u_row, 0].set_title(f'预测u分量 (时间{time_idx})')
        plt.colorbar(im1, ax=axes[u_row, 0])
        
        # u分量真实图
        im2 = axes[u_row, 1].imshow(target_u, cmap='viridis', aspect='auto',
                                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], origin='lower')
        axes[u_row, 1].set_title(f'真实u分量 (时间{time_idx})')
        plt.colorbar(im2, ax=axes[u_row, 1])
        
        # u分量误差图
        im3 = axes[u_row, 2].imshow(u_error, cmap='hot', aspect='auto',
                                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], origin='lower')
        axes[u_row, 2].set_title(f'u分量误差 (时间{time_idx})')
        plt.colorbar(im3, ax=axes[u_row, 2])
        
        # v分量预测图
        im4 = axes[v_row, 0].imshow(pred_v, cmap='viridis', aspect='auto',
                                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], origin='lower')
        axes[v_row, 0].set_title(f'预测v分量 (时间{time_idx})')
        plt.colorbar(im4, ax=axes[v_row, 0])
        
        # v分量真实图
        im5 = axes[v_row, 1].imshow(target_v, cmap='viridis', aspect='auto',
                                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], origin='lower')
        axes[v_row, 1].set_title(f'真实v分量 (时间{time_idx})')
        plt.colorbar(im5, ax=axes[v_row, 1])
        
        # v分量误差图
        im6 = axes[v_row, 2].imshow(v_error, cmap='hot', aspect='auto',
                                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], origin='lower')
        axes[v_row, 2].set_title(f'v分量误差 (时间{time_idx})')
        plt.colorbar(im6, ax=axes[v_row, 2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='T1-MPO-Transformer训练脚本')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, default='train_cf_4x2000x1598x2.h5', help='训练数据路径')
    parser.add_argument('--test_data_path', type=str, default='test_cf_9x2000x1598x2.h5', help='测试数据路径')
    parser.add_argument('--train_trajectories', type=int, nargs='+', default=[0,1,2,3], help='训练轨迹')
    parser.add_argument('--test_trajectories', type=int, nargs='+', default=[3], help='测试轨迹')
    parser.add_argument('--time_start', type=int, default=0, help='起始时间步')
    parser.add_argument('--time_used', type=int, default=200, help='使用的时间步数')
    parser.add_argument('--test_time_start', type=int, default=0, help='测试数据起始时间步')
    parser.add_argument('--test_time_used', type=int, default=200, help='测试数据使用的时间步数')
    
    # MPO参数
    parser.add_argument('--bond_scale', type=float, default=1, help='MPO键维数缩放因子')
    parser.add_argument('--num_tensors', type=int, default=6, help='MPO子张量数量')
    parser.add_argument('--mpo_lr', type=float, default=1e-3, help='MPO学习率')
    parser.add_argument('--mpo_epochs', type=int, default=10, help='MPO训练轮数')
    
    # Transformer参数
    parser.add_argument('--seq_len', type=int, default=10, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=5, help='预测序列长度')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer隐藏维度')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer头数')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='解码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--transformer_lr', type=float, default=1e-4, help='Transformer学习率')
    parser.add_argument('--transformer_epochs', type=int, default=200, help='Transformer训练轮数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    try:
        # 阶段1: 训练MPO分解 (训练数据)
        train_data_config = {
            'data_path': args.train_data_path,
            'trajectories': args.train_trajectories,
            'time_start': args.time_start,
            'time_used': args.time_used,
        }
        train_mpo_model, train_param_info = train_mpo_decomposition(args, device, train_data_config, "train_mpo")
        
        # 阶段2: 训练T1特征Transformer - 使用轨迹间平均的时间序列
        transformer, best_transformer_loss = train_T1_transformer(args, train_mpo_model, device)
        
        # 阶段3: 使用独立测试集评估完整模型 - 重新训练测试数据的MPO模型
        if args.test_data_path is not None:
            test_loss, test_mae, test_T1_loss = test_full_model_with_test_data(args, transformer, device)
        else:
            test_loss, test_mae, test_T1_loss = 0, 0, 0
            print("未提供测试数据路径，跳过测试阶段")
        
        print(f"\n训练完成!")
        print(f"训练轨迹: {args.train_trajectories}")
        print(f"测试轨迹: {args.test_trajectories}")
        print(f"训练MPO压缩比: {train_param_info['compression_ratio']:.2f}x")
        print(f"Transformer最佳损失: {best_transformer_loss:.6f}")
        if args.test_data_path is not None:
            print(f"最终测试 - 物理场MSE: {test_loss:.6f}, MAE: {test_mae:.6f}, T1损失: {test_T1_loss:.6f}")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()