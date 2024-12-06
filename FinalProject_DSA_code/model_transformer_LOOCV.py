import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_definition import DenseNet1DModel  # 导入你定义的 DenseNet1DModel
import pandas as pd
import numpy as np

# 定义多任务回归模型
class MultiTaskDenseNetModel(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(MultiTaskDenseNetModel, self).__init__()
        # 假设 DenseNet1DModel 是你已经定义的模型
        self.densenet_model = DenseNet1DModel(input_dim=input_dim, output_dims=output_dims)
        
    def forward(self, x):
        # 获取 DenseNet1DModel 的输出
        outputs = self.densenet_model(x)
        return outputs

# 加载数据
data = torch.load('combined_protein_features.pt')  # 假设数据是一个tensor，形状为 [802, 39]
X = data[:, :33]  # 前33列是蛋白质序列特征
y = data[:, 33:]  # 后6列是蛋白质性质特征

# 检查并选择设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# 初始化模型并将模型放到设备上
model = MultiTaskDenseNetModel(input_dim=33, output_dims=[1, 1, 1, 1, 1, 1])  # 6个输出任务
model = model.to(device)

# 定义损失函数和优化器
# 每个任务使用MSE损失
criterion = nn.MSELoss()  # 使用 MSE 损失函数用于回归任务
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# LOOCV 初始化
num_samples = X.shape[0]

# 用于存储结果
y_true = []  # 真实标签
y_pred = []  # 预测值
errors = []  # 误差

# 用于记录每个样本的损失和误差
train_losses = []
test_losses = []

# 留一交叉验证 (LOOCV)
for i in range(num_samples):
    # 每次划分训练集和测试集
    X_train = torch.cat((X[:i], X[i+1:]), dim=0)  # 去掉第i个样本，作为训练集
    y_train = torch.cat((y[:i], y[i+1:]), dim=0)  # 去掉第i个样本，作为训练集
    X_test = X[i:i+1]  # 取出第i个样本作为测试集
    y_test = y[i:i+1]  # 取出第i个样本的对应标签
    
    # 创建数据加载器
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)  # 每次只测试一个样本

    # 训练模型
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # 模型预测
        outputs = model(batch_X)
        
        # 计算每个任务的损失
        loss = 0.0
        for j in range(len(outputs)):
            loss += criterion(outputs[j].squeeze(), batch_y[:, j])  # 对每个输出特征计算MSE损失
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # 记录训练损失
    train_losses.append(train_loss / len(train_loader))

    # 在测试集上评估模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
           # 将每个输出转换为 numpy 数组
            outputs_numpy = [output.detach().cpu().numpy() for output in outputs]

# 保存真实值、预测值、误差
            y_true.append(y_test[0].detach().cpu().numpy())  # 确保真实值在 CPU 上
            y_pred.append([output[0] for output in outputs_numpy])  # 直接从 numpy 数组获取预测值
            errors.append([(y_test[0].detach().cpu().numpy()[j] - outputs_numpy[j][0]) for j in range(len(outputs))])  # 计算误差

# 计算损失并累加到测试损失
            test_loss += sum(criterion(output.squeeze(), batch_y[:, j]) for j, output in enumerate(outputs))

    # 记录测试损失
    test_losses.append(test_loss / len(test_loader))

    print(f"Sample {i+1} completed. Error: {errors[-1]}")

# 假设 y_true, y_pred 和 errors 是嵌套的列表
# 我们将其转换为合适的形状以便保存为CSV文件
y_true = np.array(y_true)
y_pred = np.array(y_pred)
errors = np.array(errors)

# 确保它们的形状是 [num_samples, 6]
y_true = y_true.reshape(-1, 6)
y_pred = y_pred.reshape(-1, 6)
errors = errors.reshape(-1, 6)
# 保存结果到CSV文件
df_results = pd.DataFrame({
    'Sample Index': np.arange(1, num_samples + 1),
    'True Ex_max': np.array(y_true)[:, 0],
    'True Em_max': np.array(y_true)[:, 1],
    'True Stokes_Shift': np.array(y_true)[:, 2],
    'True Extinction_Coefficient': np.array(y_true)[:, 3],
    'True Quantum_Yield': np.array(y_true)[:, 4],
    'True Brightness': np.array(y_true)[:, 5],
    'Predicted Ex_max': np.array(y_pred)[:, 0],
    'Predicted Em_max': np.array(y_pred)[:, 1],
    'Predicted Stokes_Shift': np.array(y_pred)[:, 2],
    'Predicted Extinction_Coefficient': np.array(y_pred)[:, 3],
    'Predicted Quantum_Yield': np.array(y_pred)[:, 4],
    'Predicted Brightness': np.array(y_pred)[:, 5],
    'Error Ex_max': np.array(errors)[:, 0],
    'Error Em_max': np.array(errors)[:, 1],
    'Error Stokes_Shift': np.array(errors)[:, 2],
    'Error Extinction_Coefficient': np.array(errors)[:, 3],
    'Error Quantum_Yield': np.array(errors)[:, 4],
    'Error Brightness': np.array(errors)[:, 5],
    'Test Loss': test_losses,
})

# 保存为CSV
df_results.to_csv('model_results.csv', index=False)
