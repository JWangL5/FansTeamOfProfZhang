import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# **Step 1: 加载蛋白质特征数据并进行归一化**
# 加载蛋白质特征 CSV 文件
protein_features = pd.read_csv('protein_features_filled.csv')

# 查看数值列
print("Columns in protein_features:", protein_features.columns)

# 选择需要归一化的数值列
numeric_columns = protein_features.columns

# 初始化 MinMaxScaler
scaler = MinMaxScaler()

# 对数值特征进行归一化
normalized_features = scaler.fit_transform(protein_features[numeric_columns])

# 转换为 DataFrame，并保留列名
normalized_features_df = pd.DataFrame(normalized_features, columns=numeric_columns)

# 查看归一化后的特征
print("Normalized features preview:")
print(normalized_features_df.head())

# **Step 2: 加载蛋白质序列的嵌入数据**
# 加载 protein_embeddings.pt 文件
sequence_embeddings = torch.load('protein_embeddings.pt') 
print(f"Sequence embeddings shape: {sequence_embeddings.shape}")

# **Step 3: 合并归一化后的特征和嵌入**
# 将归一化特征转换为 PyTorch 张量
normalized_features_tensor = torch.tensor(normalized_features, dtype=torch.float32) 
print(f"Normalized features shape: {normalized_features_tensor.shape}")

# 检查是否可以直接合并
if sequence_embeddings.shape[0] != normalized_features_tensor.shape[0]:
    raise ValueError("Number of samples in embeddings and features do not match!")

# 合并特征
combined_features = torch.cat([sequence_embeddings, normalized_features_tensor], dim=1) 
print(f"Combined features shape: {combined_features.shape}")

# **Step 4: 保存合并后的特征**
torch.save(combined_features, 'combined_protein_features.pt')
print("Combined features saved to 'combined_protein_features.pt'")
