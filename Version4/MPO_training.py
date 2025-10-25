# MPO_training.py
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from MPO_model import TemporalMPODecomposition, CoreTensorTransformer
from load_cylinder_flow import CylinderFlowMPODataset
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

def loss_fn(pred, gt):
    """MSE损失"""
    return torch.mean((pred - gt) ** 2)

def loss_fn2(pred, gt):
    """MAE损失"""
    return torch.mean(torch.abs(pred - gt))

def train_mpo_model(config, device):
    """训练MPO分解模型"""
    print("=== 阶段1: 训练MPO分解模型 ===")
    
    # 加载圆柱绕流数据
    train_dataset = CylinderFlowMPODataset(
        file_path=config.train_data_path,
        trajectory_indices=config.train_trajectories,
        time_start=config.time_start,
        time_used=config.time_used,
        grid_size=config.grid_size,
        normalize=True
    )
    
    # 获取网格信息
    grid_info = train_dataset.get_grid_info()
    Y, X, U = grid_info['grid_shape']
    T = train_dataset.grid_data.shape[0]  # 时间步数
    
    print(f"网格形状: T={T}, Y={Y}, X={X}, U={U}")
    print(f"原始数据形状: {train_dataset.grid_data.shape}")  # [T, Y, X, 2]
    
    # 重塑数据为 [X, Y, T, 2] 格式
    grid_data = torch.from_numpy(train_dataset.grid_data)  # [T, Y, X, 2]
    input_data = grid_data.permute(2, 1, 0, 3)  # [X, Y, T, 2]
    
    print(f"MPO输入数据形状: {input_data.shape}")
    
    # 创建MPO模型
    spatial_shape = (X, Y)
    ranks = (config.rank_space_x, config.rank_space_y, config.rank_time)
    mpo_model = TemporalMPODecomposition(spatial_shape, T, ranks, output_dim=U).to(device)
    
    print(f"MPO模型参数:")
    print(f"  空间形状: {spatial_shape}")
    print(f"  时间步数: {T}")
    print(f"  秩: {ranks}")
    print(f"  输出维度: {U}")
    
    # 准备训练数据 - 使用所有时间步的所有空间点
    # 输入: [X, Y, T] 的空间场 (取u/v平均值)
    # 目标: [T, X, Y, 2] 的真实值
    train_input = input_data[:, :, :, :].mean(dim=-1)  # [X, Y, T] - 取u/v平均值作为输入
    train_target = grid_data.permute(0, 2, 1, 3)  # [T, X, Y, 2] - 真实目标
    
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    
    print(f"训练输入形状: {train_input.shape}")
    print(f"训练目标形状: {train_target.shape}")
    
    # 优化器
    optimizer = optim.AdamW(mpo_model.parameters(), lr=config.mpo_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    for epoch in tqdm(range(config.mpo_epochs)):
        mpo_model.train()
        
        # 前向传播 - 一次性处理所有时间步
        predictions = mpo_model(train_input)  # [T, X, Y, 2]
        
        # 计算损失
        loss = loss_fn(predictions, train_target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mpo_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 评估
        if epoch % 50 == 0:
            mpo_model.eval()
            with torch.no_grad():
                # 在训练数据上评估
                eval_predictions = mpo_model(train_input)
                eval_loss = loss_fn(eval_predictions, train_target)
                eval_mae = loss_fn2(eval_predictions, train_target)
                
                print(f'MPO Epoch {epoch:04d}, Train Loss: {loss.item():.6f}, '
                      f'Eval Loss: {eval_loss.item():.6f}, Eval MAE: {eval_mae.item():.6f}')
                
                scheduler.step(eval_loss)
                
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save({
                        'model_state_dict': mpo_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'grid_info': grid_info
                    }, f'{config.output_dir}/temporal_mpo_best.pth')
    
    print(f"MPO训练完成! 最佳Loss: {best_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.title('MPO训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return mpo_model, grid_info, input_data

def create_core_transformer_data(mpo_model, input_data, seq_len, pred_len):
    """创建核心张量Transformer训练数据"""
    mpo_model.eval()
    
    # 获取训练好的核心张量
    with torch.no_grad():
        core_tensor = mpo_model.get_core_tensor()  # [T, rank_t]
    
    T, rank_t = core_tensor.shape
    
    sequences = []
    targets = []
    
    # 创建序列-目标对
    for i in range(T - seq_len - pred_len + 1):
        src_seq = core_tensor[i:i+seq_len]  # [seq_len, rank_t]
        tgt_seq = core_tensor[i+seq_len:i+seq_len+pred_len]  # [pred_len, rank_t]
        
        sequences.append(src_seq)
        targets.append(tgt_seq)
    
    sequences = torch.stack(sequences)  # [num_sequences, seq_len, rank_t]
    targets = torch.stack(targets)      # [num_sequences, pred_len, rank_t]
    
    print(f"核心张量序列数据: {len(sequences)} 个序列")
    print(f"输入序列形状: {sequences.shape}, 目标序列形状: {targets.shape}")
    
    return sequences, targets

def train_core_transformer(config, mpo_model, grid_info, input_data, device):
    """训练核心张量Transformer模型"""
    print("\n=== 阶段2: 训练核心张量Transformer模型 ===")
    
    # 创建核心张量序列数据
    core_sequences, core_targets = create_core_transformer_data(
        mpo_model, input_data, config.seq_len, config.pred_len
    )
    
    # 分割训练测试集
    dataset = TensorDataset(core_sequences, core_targets)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset_trans, test_dataset_trans = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset_trans, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_trans, batch_size=config.batch_size, shuffle=False)
    
    print(f"Transformer训练数据: {len(train_dataset_trans)} 序列")
    print(f"Transformer测试数据: {len(test_dataset_trans)} 序列")
    
    # 创建核心张量Transformer模型
    core_dim = mpo_model.rank_t
    transformer = CoreTensorTransformer(
        core_dim=core_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout
    ).to(device)
    
    # 优化器
    optimizer = optim.AdamW(transformer.parameters(), lr=config.transformer_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # 训练循环
    best_rmse = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(config.transformer_epochs)):
        transformer.train()
        
        epoch_loss = 0
        
        for batch_src, batch_tgt in train_loader:
            optimizer.zero_grad()
            
            batch_src = batch_src.to(device)  # [batch, seq_len, core_dim]
            batch_tgt = batch_tgt.to(device)  # [batch, pred_len, core_dim]
            
            # Transformer前向传播
            predictions = transformer(batch_src, batch_tgt)  # [batch, pred_len, core_dim]
            
            loss = loss_fn(predictions, batch_tgt)
            
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
                test_loss = 0
                for batch_src, batch_tgt in test_loader:
                    batch_src = batch_src.to(device)
                    batch_tgt = batch_tgt.to(device)
                    
                    predictions = transformer(batch_src, batch_tgt)
                    test_loss += loss_fn(predictions, batch_tgt).item()
                
                test_rmse = test_loss / len(test_loader)
                test_losses.append(test_rmse)
                
                print(f'Core Transformer Epoch {epoch:04d}, Train Loss: {avg_epoch_loss:.6f}, '
                      f'Test RMSE: {test_rmse:.6f}')
                
                scheduler.step(test_rmse)
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    torch.save({
                        'model_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_rmse': best_rmse
                    }, f'{config.output_dir}/core_transformer_best.pth')
    
    print(f"核心张量Transformer训练完成! 最佳RMSE: {best_rmse:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Core Transformer训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_losses)
    plt.title('Core Transformer测试RMSE')
    plt.xlabel('Epoch (x20)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return transformer, best_rmse

def test_model(config, mpo_model, transformer, test_dataset, device):
    """在测试集上评估模型"""
    print("\n=== 阶段3: 测试模型 ===")
    
    # 准备测试数据
    test_grid_data = torch.from_numpy(test_dataset.grid_data)  # [T_test, Y, X, 2]
    test_input = test_grid_data.permute(2, 1, 0, 3)  # [X, Y, T_test, 2]
    X, Y, T_test, U = test_input.shape
    
    mpo_model.eval()
    transformer.eval()
    
    # 获取训练的核心张量作为历史信息
    train_cores = mpo_model.get_core_tensor().detach()  # [train_T, rank_t]
    
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        # 对每个测试时间步进行预测
        for t in range(T_test):
            # 使用Transformer预测当前时间步的核心张量
            if t < config.seq_len:
                # 对于前seq_len个时间步，使用训练数据的最后几个core作为输入
                src_cores = train_cores[-config.seq_len:].unsqueeze(0)  # [1, seq_len, rank_t]
            else:
                # 使用前几个测试时间步预测的core
                src_cores = train_cores[-config.seq_len:].unsqueeze(0)
            
            # 预测当前时间步的core_tensor
            pred_cores = transformer.predict_future(src_cores.to(device), pred_len=1)  # [1, 1, rank_t]
            pred_core = pred_cores[0, 0]  # [rank_t]
            
            # 当前时间步的空间数据 [X, Y, 1]
            spatial_data = test_input[:, :, t, :].mean(dim=-1).unsqueeze(-1)  # [X, Y, 1]
            spatial_data = spatial_data.to(device)
            
            # 使用MPO模型和预测的core进行预测
            pred = mpo_model(spatial_data, custom_core=pred_core)  # [1, X, Y, 2]
            target = test_input[:, :, t, :].unsqueeze(0)# [1, 2, X, Y]
           
            target = target.to(device)
            
            # 计算损失
            #(pred.shape,target.shape)
            loss = loss_fn(pred, target)
            mae = loss_fn2(pred, target)
            
            total_loss += loss.item()
            total_mae += mae.item()
            
            if t % 10 == 0:
                current_loss = total_loss / (t + 1)
                current_mae = total_mae / (t + 1)
                print(f'时间步 {t:03d}/{T_test}: Loss={current_loss:.6f}, MAE={current_mae:.6f}')
    
    avg_loss = total_loss / T_test
    avg_mae = total_mae / T_test
    
    print(f"\n测试结果:")
    print(f"覆盖空间点: {X}×{Y} = {X*Y}个点")
    print(f"时间步数: {T_test}")
    print(f"平均Loss: {avg_loss:.6f}")
    print(f"平均MAE: {avg_mae:.6f}")
    
    return avg_loss, avg_mae

def main():
    parser = argparse.ArgumentParser(description='MPO分解和核心张量Transformer训练')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, default='train_cf_4x2000x1598x2.h5', help='训练数据路径')
    parser.add_argument('--test_data_path', type=str, default='test_cf_9x2000x1598x2.h5', help='测试数据路径')
    parser.add_argument('--train_trajectories', type=int, nargs='+', default=[0, 1, 2], help='训练轨迹索引')
    parser.add_argument('--test_trajectories', type=int, nargs='+', default=[3], help='测试轨迹索引')
    parser.add_argument('--time_start', type=int, default=0, help='起始时间步')
    parser.add_argument('--time_used', type=int, default=200, help='使用的时间步数')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[50, 100], help='网格大小 [Y, X]')
    
    # MPO模型参数
    parser.add_argument('--rank_space_x', type=int, default=24, help='空间X维度的秩')
    parser.add_argument('--rank_space_y', type=int, default=24, help='空间Y维度的秩')
    parser.add_argument('--rank_time', type=int, default=48, help='时间维度的秩')
    parser.add_argument('--mpo_lr', type=float, default=1e-3, help='MPO学习率')
    parser.add_argument('--mpo_epochs', type=int, default=1000, help='MPO训练轮数')
    
    # Transformer参数
    parser.add_argument('--seq_len', type=int, default=50, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=1, help='预测序列长度')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer头数')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='解码器层数')
    parser.add_argument('--transformer_lr', type=float, default=1e-4, help='Transformer学习率')
    parser.add_argument('--transformer_epochs', type=int, default=100, help='Transformer训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    config = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"使用设备: {config.device}")
    
    # 训练MPO模型
    mpo_model, grid_info, input_data = train_mpo_model(config, config.device)
    
    # 训练核心张量Transformer
    transformer, best_rmse = train_core_transformer(config, mpo_model, grid_info, input_data, config.device)
    
    # 在测试集上评估
    test_dataset = CylinderFlowMPODataset(
        file_path=config.test_data_path,
        trajectory_indices=config.test_trajectories,
        time_start=config.time_start,
        time_used=config.time_used,
        grid_size=config.grid_size,
        normalize=True
    )
    
    test_loss, test_mae = test_model(config, mpo_model, transformer, test_dataset, config.device)
    
    print(f"\n=== 训练完成 ===")
    print(f"核心张量预测RMSE: {best_rmse:.6f}")
    print(f"测试集物理场预测Loss: {test_loss:.6f}")
    print(f"测试集物理场预测MAE: {test_mae:.6f}")

if __name__ == '__main__':
    main()