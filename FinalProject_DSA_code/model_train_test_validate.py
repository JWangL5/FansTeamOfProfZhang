import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from model_definition import DenseNet1DModel
from sklearn.metrics import r2_score

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

# 数据集划分
dataset = TensorDataset(X, y)
total_samples = len(dataset)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# 初始化模型和设置
epochs = 10
model = MultiTaskDenseNetModel(input_dim=33, output_dims=[1, 1, 1, 1, 1, 1])  # 6个输出任务
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# 动态权重初始化为 1.0
task_weights = torch.ones(6, device=device, requires_grad=True)

# 损失函数
criterion = nn.MSELoss()

# 存储损失
train_losses = []
val_losses = []

# 训练和验证
for epoch in range(epochs):
    # 训练模式
    model.train()
    epoch_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()

        # 模型预测
        outputs = model(batch_X)

        # 动态计算每个任务的损失
        task_losses = torch.stack([criterion(outputs[j].squeeze(), batch_y[:, j]) for j in range(len(outputs))])
        loss = torch.sum(task_weights * task_losses)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        # 动态调整权重：基于每个任务的损失的反比
        with torch.no_grad():
            task_weights = 1 / (task_losses + 1e-8).detach()  # 避免除以零
            task_weights = task_weights / task_weights.sum()
    train_losses.append(epoch_train_loss / len(train_loader))

    # 验证模式
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 模型预测
            outputs = model(batch_X)

            # 验证损失
            val_task_losses = torch.tensor([criterion(outputs[j].squeeze(), batch_y[:, j]) for j in range(len(outputs))])
            epoch_val_loss += torch.sum(val_task_losses).item()

    val_losses.append(epoch_val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# 保存模型
torch.save(model.state_dict(), "multi_task_densenet_model.pth")
print("Model saved to multi_task_densenet_model.pth")

# 验证集准确率评估
model.eval()
val_y_true = []
val_y_pred = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)

        # 保存真实值和预测值
        val_y_true.append(batch_y.cpu().numpy())
        val_y_pred.append(
            torch.cat([output.cpu() if output.dim() == 2 else output.unsqueeze(1).cpu() for output in outputs], dim=1).numpy()
        )

val_y_true = np.vstack(val_y_true)
val_y_pred = np.vstack(val_y_pred)

# 计算每个特征的单独准确率
feature_accuracies = []
for i in range(val_y_true.shape[1]):  # 遍历每个特征
    accuracy = np.mean(np.abs(val_y_true[:, i] - val_y_pred[:, i]) < 0.1)
    feature_accuracies.append(accuracy)

# 打印每个特征的准确率
feature_names = ['Ex_max', 'Em_max', 'Stokes_Shift', 'Extinction_Coefficient', 'Quantum_Yield', 'Brightness']
for feature_name, accuracy in zip(feature_names, feature_accuracies):
    print(f"Accuracy for {feature_name}: {accuracy:.4f}")

# 计算整体平均准确率
overall_accuracy = np.mean(feature_accuracies)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# 绘制验证集真实值和预测值的散点图
plt.figure(figsize=(18, 12))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.scatter(val_y_true[:, i], val_y_pred[:, i], alpha=0.6, label="Data Points")
    plt.plot([min(val_y_true[:, i]), max(val_y_true[:, i])], [min(val_y_true[:, i]), max(val_y_true[:, i])], 'r--', label="y = x")
    plt.xlabel(f"True {feature_names[i]}")
    plt.ylabel(f"Predicted {feature_names[i]}")
    plt.title(f"Scatter Plot for {feature_names[i]}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("validation_scatter_plots.png")
plt.show()
# Calculate R² values for each feature
r2_values = [r2_score(val_y_true[:, i], val_y_pred[:, i]) for i in range(val_y_true.shape[1])]

# Plot the R² values
plt.figure(figsize=(10, 6))
plt.bar(feature_names, r2_values, color="skyblue")
plt.xlabel("Protein Features")
plt.ylabel("$R^2$ Value")
plt.title("$R^2$ Values for Predicted Protein Features on Validation Set")
plt.ylim(0, 1)  # Optional: Limit the y-axis to [0, 1] for better visibility
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Annotate R² values on top of each bar
for i, r2 in enumerate(r2_values):
    plt.text(i, r2 + 0.02, f"{r2:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("r2_values_validation_set.png")
plt.show()