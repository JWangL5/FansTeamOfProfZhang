import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取当前文件夹中的 model_results.csv 文件
df = pd.read_csv('model_results.csv')

# 定义特征列的名称
features = ['Ex_max', 'Em_max', 'Stokes_Shift', 'Extinction_Coefficient', 'Quantum_Yield', 'Brightness']

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列的子图
axes = axes.flatten()  # 方便遍历子图

# 绘制每个特征的散点图
for i, feature in enumerate(features):
    # 真实值和预测值的列名
    true_col = f'True {feature}'
    pred_col = f'Predicted {feature}'
    
    # 获取真实值和预测值
    y_true = df[true_col]
    y_pred = df[pred_col]
    
    # 绘制散点图
    axes[i].scatter(y_true, y_pred, color='blue', label='Predicted vs True', alpha=0.5)
    axes[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--', label='Perfect Prediction')
    
    # 设置标题和标签
    axes[i].set_title(f'{feature} - True vs Predicted')
    axes[i].set_xlabel(f'True {feature}')
    axes[i].set_ylabel(f'Predicted {feature}')
    axes[i].legend()

# 调整布局
plt.tight_layout()

# 保存图形为图片文件
plt.savefig('protein_feature_scatter_plots.png', dpi=300)

# 显示图形
plt.show()

# 1. 绘制 Test Loss 的分布图
test_losses = df['Test Loss']  # 获取 'Test Loss' 列

# 绘制分布图
plt.figure(figsize=(10, 6))
plt.hist(test_losses, bins=30, color='green', edgecolor='black')
plt.title('Test Loss Distribution')
plt.xlabel('Test Loss')
plt.ylabel('Frequency')

# 保存 Test Loss 分布图
plt.savefig('test_loss_distribution.png')

# 显示图像
plt.show()

# 2. 绘制蛋白质特征误差的热力图
# 获取指定的误差数据列
error_columns = ['Error Ex_max', 'Error Em_max', 'Error Stokes_Shift', 'Error Extinction_Coefficient', 'Error Quantum_Yield', 'Error Brightness']
error_data = df[error_columns].values  # 获取所有误差数据

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(error_data, cmap='coolwarm', cbar=True, xticklabels=error_columns, yticklabels=range(1, 803), vmin=-1, vmax=1)

# 设置标题和标签
plt.title('Error Heatmap for Protein Features')
plt.xlabel('Protein Feature Error')
plt.ylabel('Sample Index')

# 保存热力图
plt.savefig('protein_feature_error_heatmap.png')

# 显示图像
plt.show()
