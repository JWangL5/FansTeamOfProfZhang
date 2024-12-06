import json
import pandas as pd

# 读取 JSON 文件
with open('combined_basic.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 创建两个空的列表，一个用于存储蛋白质的特征数据，一个用于存储蛋白质序列
protein_data = []
sequence_data = []

# 遍历 JSON 数据，提取每个蛋白质的序列和相关性质
for i, entry in enumerate(data):
    # 提取蛋白质名字，假设 JSON 中有一个字段 "Name"，没有则使用索引生成名字
    protein_name = entry.get("Name", f"protein_{i}")  # 如果没有 "Name" 字段，就用 "protein_{i}" 作为名字
    
    # 提取特征数据
    protein_info = {
        "Ex max (nm)": entry.get("Ex max (nm)", ""),
        "Em max (nm)": entry.get("Em max (nm)", ""),
        "Stokes Shift (nm)": entry.get("Stokes Shift (nm)", ""),
        "Extinction Coefficient": entry.get("Extinction Coefficient", ""),
        "Quantum Yield": entry.get("Quantum Yield", ""),
        "Brightness": entry.get("Brightness", ""),
    }
    
    # 提取蛋白质序列
    protein_sequence = entry.get("seq", "")
    
    # 将每个蛋白质的特征数据和序列添加到各自的列表中
    protein_data.append(protein_info)
    sequence_data.append({"Name": protein_name, "seq": protein_sequence})  # 添加蛋白质名字

# 将蛋白质特征数据列表转换为 DataFrame
df_features = pd.DataFrame(protein_data)

# 将蛋白质序列数据列表转换为 DataFrame
df_sequences = pd.DataFrame(sequence_data)

# 查看提取的特征数据
print("Protein Feature DataFrame (first few rows):")
print(df_features.head())

# 查看提取的蛋白质序列数据
print("\nProtein Sequence DataFrame (first few rows):")
print(df_sequences.head())

# 将数值列强制转换为数值类型（NaN 处理无效数据）
df_features = df_features.apply(pd.to_numeric, errors='coerce')

# 仅对数值型列填充中位数
numeric_columns = df_features.select_dtypes(include=['float64', 'int64']).columns  # 选择数值型列
df_features[numeric_columns] = df_features[numeric_columns].fillna(df_features[numeric_columns].median())  # 填充中位数

# 查看填充后的特征数据
print("\nProtein Feature DataFrame after filling missing values with median:")
print(df_features.head())

# 保存为 CSV 文件（如果需要）
df_features.to_csv('protein_features_filled.csv', index=False)
df_sequences.to_csv('protein_sequences.csv', index=False)
