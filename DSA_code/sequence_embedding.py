import torch
import pandas as pd
from esm import pretrained

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 ESM 模型
model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

# 加载蛋白质序列
df_sequences = pd.read_csv('protein_sequences.csv')
sequences = df_sequences['seq'].tolist()
names = df_sequences['Name'].tolist()

# 截断序列长度
max_length = 1024
sequences = [seq[:max_length] for seq in sequences]

# 分批处理
batch_size = 4
all_embeddings = []

for i in range(0, len(sequences), batch_size):
    batch = list(zip(names[i:i + batch_size], sequences[i:i + batch_size]))
    _, _, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens)
        logits = results["logits"]

        # 平均池化，将序列长度压缩为固定表示
        pooled_logits = torch.mean(logits, dim=1)  # [batch_size, hidden_dim]
        all_embeddings.append(pooled_logits.cpu())

# 合并所有批次的嵌入
all_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Total embeddings shape: {all_embeddings.shape}")

# 保存到文件
torch.save(all_embeddings, 'protein_embeddings.pt')
print("Protein embeddings saved to 'protein_embeddings.pt'")
