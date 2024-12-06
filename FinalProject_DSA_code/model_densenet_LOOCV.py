import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_definition import DenseNet1DModel  # 导入你定义的 DenseNet1DModel
import numpy as np
import matplotlib.pyplot as plt

# 定义多任务回归模型
class MultiTaskDenseNetModel(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(MultiTaskDenseNetModel, self).__init__()
        self.densenet_model = DenseNet1DModel(input_dim=input_dim, output_dims=output_dims)
        
    def forward(self, x):
        outputs = self.densenet_model(x)
        return outputs

# 加载数据
data = torch.load('combined_protein_features.pt')  # 假设数据是一个tensor，形状为 [802, 39]
X = data[:, :33]  # 前33列是蛋白质序列特征
y = data[:, 33:]  # 后6列是蛋白质性质特征

# 使用 CPU
device = torch.device("cpu")
print(f'Using device: {device}')

# 定义损失函数和优化器
criterion = nn.MSELoss()  

# LOOCV 初始化
num_samples = X.shape[0]
epochs = 10  # 设置训练轮次
batch_size = 64

# 用于存储结果
y_true = []
y_pred = []
errors = []
train_loss_per_batch = []  # 存储每个批次的训练损失（最后一轮）
test_loss_per_epoch = []

for i in range(num_samples):
    # 初始化模型并将模型放到设备上
    model = MultiTaskDenseNetModel(input_dim=33, output_dims=[1, 1, 1, 1, 1, 1])  # 6个输出任务
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X_train = torch.cat((X[:i], X[i+1:]), dim=0)  
    y_train = torch.cat((y[:i], y[i+1:]), dim=0)  
    X_test = X[i:i+1]  
    y_test = y[i:i+1]  
    
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    final_epoch_losses = []  # 用于记录最后一轮的每个 epoch 的每个 batch 损失
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_losses = []  # 用于记录每个批次的损失
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = sum(criterion(outputs[j].squeeze(), batch_y[:, j]) for j in range(len(outputs)))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 如果是最后一轮，记录每个批次的损失
            if epoch == epochs - 1:
                batch_losses.append(loss.item())

        # 如果是最后一轮，保存每个批次的损失
        if epoch == epochs - 1:
            train_loss_per_batch = batch_losses
        
        # 测试集评估
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                test_loss += sum(criterion(outputs[j].squeeze(), batch_y[:, j]) for j, output in enumerate(outputs))
        test_loss_per_epoch.append(test_loss)
    
    # 保存真实值、预测值、误差
    y_test_np = y_test.detach().numpy().flatten()  # 确保 y_test_np 是一维数组
    outputs_numpy = [output.detach().numpy().flatten() for output in outputs]   
    y_true.append(y_test_np)
    y_pred.append([output[0] for output in outputs_numpy])
    errors.append([y_test_np[j] - outputs_numpy[j][0] for j in range(len(outputs))])

    print(f"Sample {i+1} completed. Epochs: {epochs}, Test Loss: {test_loss}")

# 绘制最后一轮每个批次的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_loss_per_batch)), train_loss_per_batch, marker="o", label="Batch Loss (Last Epoch)", color="green")
plt.title("Loss Curve for Each Batch in Last Epoch")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("loss_curve_batches_last_epoch.png")
plt.show()
