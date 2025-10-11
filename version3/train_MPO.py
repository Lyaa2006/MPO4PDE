import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from MPO_model import MPODecomposition, MPOEncoderDecoderTransformer, load_burgers_data
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family']='SimHei'

def loss_fn(pred, gt):
    """RMSE损失"""
    return torch.sqrt(torch.mean((pred - gt) ** 2))

def loss_fn2(pred, gt):
    """MAE损失"""
    return torch.mean(torch.abs(pred - gt))

def create_mpo_training_data(data_tensor):
    """创建MPO训练数据"""
    T, X, Y, U = data_tensor.shape
    
    indices = []
    values = []
    
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                indices.append([t, x, y])
                values.append(data_tensor[t, x, y].numpy())  # 转换为numpy数组
    
    indices = torch.tensor(indices, dtype=torch.long)
    values = torch.tensor(np.array(values), dtype=torch.float32)  # 从numpy数组创建tensor
    
    return indices, values

def create_transformer_training_data(mpo_model, data_tensor, num_sequences=1000, seq_len=10):
    """创建Transformer训练数据"""
    T, X, Y, U = data_tensor.shape
    
    all_core_features = []
    all_query_coords = []
    all_targets = []
    
    for _ in range(num_sequences):
        # 随机选择序列点
        t_indices = torch.randint(0, T, (seq_len,))
        x_indices = torch.randint(0, X, (seq_len,))
        y_indices = torch.randint(0, Y, (seq_len,))
        
        # 创建归一化坐标
        query_coords = torch.stack([
            t_indices.float() / (T - 1),
            x_indices.float() / (X - 1),
            y_indices.float() / (Y - 1)
        ], dim=1)
        
        # 获取MPO核心特征
        with torch.no_grad():
            core_features = mpo_model.get_compressed_features(query_coords)  # [seq_len, rank_p]
        
        # 获取真实目标值
        targets = torch.stack([
            data_tensor[t_indices[i], x_indices[i], y_indices[i]] 
            for i in range(seq_len)
        ])  # [seq_len, 2]
        
        all_core_features.append(core_features)
        all_query_coords.append(query_coords)
        all_targets.append(targets)
    
    return (torch.stack(all_core_features), 
            torch.stack(all_query_coords), 
            torch.stack(all_targets))

def train_mpo_model(config, device):
    """训练MPO模型"""
    print("=== 阶段1: 训练MPO模型 ===")
    
    # 加载数据
    data_tensor, data_shape = load_burgers_data(config.data_path)
    T, X, Y, U = data_shape
    print(f"数据形状: {data_shape}")
    
    # 创建MPO模型
    ranks = (config.rank_time, config.rank_space_x, config.rank_space_y, config.rank_physics)
    mpo_model = MPODecomposition(data_shape, ranks).to(device)
    
    # 创建训练数据
    indices, values = create_mpo_training_data(data_tensor)
    dataset = TensorDataset(indices, values)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 优化器
    optimizer = optim.AdamW(mpo_model.parameters(), lr=config.mpo_lr)
    
    # 训练循环
    best_rmse = float('inf')
    
    for epoch in tqdm(range(config.mpo_epochs)):
        mpo_model.train()
        epoch_loss = 0
        
        for batch_indices, batch_values in train_loader:
            optimizer.zero_grad()
            
            batch_indices = batch_indices.to(device)
            batch_values = batch_values.to(device)
            
            reconstructed = mpo_model(batch_indices)  # [batch_size, U]
            
            loss = loss_fn(reconstructed, batch_values)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 评估
        if epoch % 100 == 0:
            mpo_model.eval()
            with torch.no_grad():
                test_loss = 0
                for test_indices, test_values in test_loader:
                    test_indices = test_indices.to(device)
                    test_values = test_values.to(device)
                    
                    test_reconstructed = mpo_model(test_indices)
                    test_loss += loss_fn(test_reconstructed, test_values).item()
                
                test_rmse = test_loss / len(test_loader)
                test_mae = 0
                for test_indices, test_values in test_loader:
                    test_indices = test_indices.to(device)
                    test_values = test_values.to(device)
                    test_reconstructed = mpo_model(test_indices)
                    test_mae += loss_fn2(test_reconstructed, test_values).item()
                test_mae = test_mae / len(test_loader)
                
                print(f'MPO Epoch {epoch:04d}, Loss: {epoch_loss/len(train_loader):.6f}, '
                      f'Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}')
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    torch.save(mpo_model.state_dict(), f'{config.output_dir}/mpo_best.pth')
    
    print(f"MPO训练完成! 最佳RMSE: {best_rmse:.6f}")
    
    # 可视化MPO重建结果
    visualize_mpo_reconstruction(mpo_model, data_tensor, config.output_dir)
    
    return mpo_model

def train_transformer_model(config, mpo_model, device):
    """训练Transformer模型"""
    print("\n=== 阶段2: 训练Transformer模型 ===")
    
    # 加载数据
    data_tensor, data_shape = load_burgers_data(config.data_path)
    
    # 创建Transformer训练数据
    core_features, query_coords, targets = create_transformer_training_data(
        mpo_model, data_tensor, 
        num_sequences=config.num_sequences,
        seq_len=config.seq_len
    )
    
    print(f"Transformer数据形状: core_features {core_features.shape}, "
          f"query_coords {query_coords.shape}, targets {targets.shape}")
    
    # 创建数据集
    dataset = TensorDataset(core_features, query_coords, targets)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 创建Transformer模型
    transformer = MPOEncoderDecoderTransformer(
        core_feature_dim=config.rank_physics,  # 使用MPO的物理维度秩
        coord_dim=3,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout
    ).to(device)
    
    # 优化器
    optimizer = optim.AdamW(transformer.parameters(), lr=config.transformer_lr)
    
    # 训练循环
    best_rmse = float('inf')
    
    for epoch in tqdm(range(config.transformer_epochs)):
        transformer.train()
        epoch_loss = 0
        
        for batch_core_features, batch_query_coords, batch_targets in train_loader:
            optimizer.zero_grad()
            
            batch_core_features = batch_core_features.to(device)
            batch_query_coords = batch_query_coords.to(device)
            batch_targets = batch_targets.to(device)
            
            predictions = transformer(batch_core_features, batch_query_coords)
            loss = loss_fn(predictions, batch_targets)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 评估
        if epoch % 50 == 0:
            transformer.eval()
            with torch.no_grad():
                test_loss = 0
                for batch_core_features, batch_query_coords, batch_targets in test_loader:
                    batch_core_features = batch_core_features.to(device)
                    batch_query_coords = batch_query_coords.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    predictions = transformer(batch_core_features, batch_query_coords)
                    test_loss += loss_fn(predictions, batch_targets).item()
                
                test_rmse = test_loss / len(test_loader)
                print(f'Transformer Epoch {epoch:04d}, Train Loss: {epoch_loss/len(train_loader):.6f}, '
                      f'Test RMSE: {test_rmse:.6f}')
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    torch.save(transformer.state_dict(), f'{config.output_dir}/transformer_best.pth')
    
    print(f"Transformer训练完成! 最佳RMSE: {best_rmse:.6f}")
    print("\n=== Transformer训练结果检验 ===")
    transformer.eval()
    mpo_model.eval()
    
    with torch.no_grad():
        # 1. 随机选择一些测试样本进行可视化
        test_core_features, test_query_coords, test_targets = next(iter(test_loader))
        test_core_features = test_core_features.to(device)
        test_query_coords = test_query_coords.to(device)
        test_targets = test_targets.to(device)
        
        predictions = transformer(test_core_features, test_query_coords)
        
        # 计算批次统计
        batch_rmse = loss_fn(predictions, test_targets).item()
        batch_mae = loss_fn2(predictions, test_targets).item()
        
        print(f"测试批次统计:")
        print(f"  - RMSE: {batch_rmse:.6f}")
        print(f"  - MAE: {batch_mae:.6f}")
        
        # 2. 单个序列详细分析
        sample_idx = 0  # 选择第一个样本
        sample_pred = predictions[sample_idx].cpu().numpy()  # [seq_len, 2]
        sample_true = test_targets[sample_idx].cpu().numpy()  # [seq_len, 2]
        
        print(f"\n单个序列详细分析 (样本 {sample_idx}):")
        for i in range(min(5, config.seq_len)):  # 显示前5个时间步
            pred_u, pred_v = sample_pred[i]
            true_u, true_v = sample_true[i]
            error_u = abs(pred_u - true_u)
            error_v = abs(pred_v - true_v)
            
            print(f"  步长 {i}: u={pred_u:.4f}(真值{true_u:.4f}, 误差{error_u:.4f}), "
                  f"v={pred_v:.4f}(真值{true_v:.4f}, 误差{error_v:.4f})")
        
        # 3. 整体统计
        all_predictions = []
        all_targets = []
        
        for batch_core_features, batch_query_coords, batch_targets in test_loader:
            batch_core_features = batch_core_features.to(device)
            batch_query_coords = batch_query_coords.to(device)
            batch_targets = batch_targets.to(device)
            
            batch_predictions = transformer(batch_core_features, batch_query_coords)
            all_predictions.append(batch_predictions.cpu())
            all_targets.append(batch_targets.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算整体统计
        overall_rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
        overall_mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
        # 分量统计
        rmse_u = torch.sqrt(torch.mean((all_predictions[:, :, 0] - all_targets[:, :, 0]) ** 2)).item()
        rmse_v = torch.sqrt(torch.mean((all_predictions[:, :, 1] - all_targets[:, :, 1]) ** 2)).item()
        mae_u = torch.mean(torch.abs(all_predictions[:, :, 0] - all_targets[:, :, 0])).item()
        mae_v = torch.mean(torch.abs(all_predictions[:, :, 1] - all_targets[:, :, 1])).item()
        
        print(f"\n整体测试集统计:")
        print(f"  - 总体RMSE: {overall_rmse:.6f}")
        print(f"  - 总体MAE: {overall_mae:.6f}")
        print(f"  - u分量RMSE: {rmse_u:.6f}, MAE: {mae_u:.6f}")
        print(f"  - v分量RMSE: {rmse_v:.6f}, MAE: {mae_v:.6f}")
        
        # 4. 可视化一些预测结果
        visualize_transformer_results(transformer, mpo_model, test_loader, config.output_dir, device)
    
    return transformer, best_rmse

def visualize_mpo_reconstruction(mpo_model, data_tensor, output_dir):
    """可视化MPO重建结果"""
    mpo_model.eval()
    
    with torch.no_grad():
        # 重建完整场
        reconstructed = mpo_model().cpu().numpy()
        original = data_tensor.numpy()
    
    # 选择第一个时间步进行对比
    t_idx = 0
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始场
    im1 = axes[0, 0].imshow(original[t_idx, :, :, 0], cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Original - u')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(original[t_idx, :, :, 1], cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Original - v')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 重建场
    im3 = axes[1, 0].imshow(reconstructed[t_idx, :, :, 0], cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('MPO Reconstruction - u')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(reconstructed[t_idx, :, :, 1], cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('MPO Reconstruction - v')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 误差
    error_u = np.abs(original[t_idx, :, :, 0] - reconstructed[t_idx, :, :, 0])
    error_v = np.abs(original[t_idx, :, :, 1] - reconstructed[t_idx, :, :, 1])
    
    im5 = axes[0, 2].imshow(error_u, cmap='hot', aspect='auto')
    axes[0, 2].set_title('Error - u')
    plt.colorbar(im5, ax=axes[0, 2])
    
    im6 = axes[1, 2].imshow(error_v, cmap='hot', aspect='auto')
    axes[1, 2].set_title('Error - v')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle('MPO Reconstruction vs Original Field')
    plt.tight_layout()
    #plt.savefig(f'{output_dir}/mpo_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"MPO重建误差 - u: MAE={error_u.mean():.6f}, v: MAE={error_v.mean():.6f}")
    
def visualize_transformer_results(transformer, mpo_model, test_loader, output_dir, device):
    """可视化Transformer预测结果"""
    transformer.eval()
    mpo_model.eval()
    
    with torch.no_grad():
        # 获取一个测试批次
        batch_core_features, batch_query_coords, batch_targets = next(iter(test_loader))
        batch_core_features = batch_core_features.to(device)
        batch_query_coords = batch_query_coords.to(device)
        batch_targets = batch_targets.to(device)
        
        predictions = transformer(batch_core_features, batch_query_coords)
        
        # 选择第一个序列进行可视化
        seq_idx = 0
        pred_seq = predictions[seq_idx].cpu().numpy()  # [seq_len, 2]
        true_seq = batch_targets[seq_idx].cpu().numpy()  # [seq_len, 2]
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # u分量
        time_steps = range(len(pred_seq))
        axes[0, 0].plot(time_steps, pred_seq[:, 0], 'b-', label='预测 u', linewidth=2)
        axes[0, 0].plot(time_steps, true_seq[:, 0], 'r--', label='真实 u', linewidth=2)
        axes[0, 0].set_title('u分量预测 vs 真实值')
        axes[0, 0].set_xlabel('序列步长')
        axes[0, 0].set_ylabel('u值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # v分量
        axes[0, 1].plot(time_steps, pred_seq[:, 1], 'g-', label='预测 v', linewidth=2)
        axes[0, 1].plot(time_steps, true_seq[:, 1], 'm--', label='真实 v', linewidth=2)
        axes[0, 1].set_title('v分量预测 vs 真实值')
        axes[0, 1].set_xlabel('序列步长')
        axes[0, 1].set_ylabel('v值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 误差
        error_u = np.abs(pred_seq[:, 0] - true_seq[:, 0])
        error_v = np.abs(pred_seq[:, 1] - true_seq[:, 1])
        
        axes[1, 0].plot(time_steps, error_u, 'orange', label='u误差', linewidth=2)
        axes[1, 0].set_title('u分量绝对误差')
        axes[1, 0].set_xlabel('序列步长')
        axes[1, 0].set_ylabel('绝对误差')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_steps, error_v, 'purple', label='v误差', linewidth=2)
        axes[1, 1].set_title('v分量绝对误差')
        axes[1, 1].set_xlabel('序列步长')
        axes[1, 1].set_ylabel('绝对误差')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Transformer预测结果分析')
        plt.tight_layout()
        plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="D:\Comsol建模\不规则网格下的二维Burgers.csv")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # MPO参数
    parser.add_argument("--rank_time", type=int, default=4)
    parser.add_argument("--rank_space_x", type=int, default=8)
    parser.add_argument("--rank_space_y", type=int, default=8)
    parser.add_argument("--rank_physics", type=int, default=2)
    parser.add_argument("--mpo_lr", type=float, default=1e-3)
    parser.add_argument("--mpo_epochs", type=int, default=1)
    
    # Transformer参数
    parser.add_argument("--transformer_lr", type=float, default=1e-4)
    parser.add_argument("--transformer_epochs", type=int, default=1)
    parser.add_argument("--num_sequences", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 通用参数
    parser.add_argument("--batch_size", type=int, default=32)
    
    config = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 阶段1: 训练MPO模型
    mpo_model = train_mpo_model(config, device)
    
    # 阶段2: 训练Transformer模型
    transformer, final_rmse = train_transformer_model(config, mpo_model, device)
    
    print(f"\n=== 训练完成 ===")
    print(f"Transformer最终预测RMSE: {final_rmse:.6f}")

if __name__ == "__main__":
    main()