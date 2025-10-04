# MPO4PDE
## Step1：MPO矩阵分解
将每个数据点分解为t/x/y/phy四个向量相乘的形式，通过预训练MPO模型，使得矩阵分解符合数值.

## Step2：Transformer
将t/x/y的向量cat为输入（由于MPO是用全局训练得到的，所以得通过包含全局信息）
Encoder-Only的Transformer，损失函数由模型输出和原有target合成.