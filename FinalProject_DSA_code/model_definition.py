# model_definition.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# transformer 模型
class MultiTaskDPTModel(nn.Module):
    def __init__(self, input_dim, output_dims, hidden_dim=256, num_heads=11, num_layers=6):
        super(MultiTaskDPTModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.fc1 = nn.Linear(input_dim, 128)  # 输入通道转换

        # 输出层为每个任务设置一个单独的全连接层
        self.output_layers = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(output_dims)  # 每个任务独立预测
        ])

    def forward(self, x):
        # 调整输入形状为 (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1)  # 变成 [batch_size, 1, input_dim]
        
        # Transformer 编码
        x = self.transformer_encoder(x)
        
        # 平均池化
        x = x.mean(dim=1)  # 取 Transformer 输出的平均值作为表示
        
        # FC 层处理
        x = torch.relu(self.fc1(x))  # [batch_size, 128]

        # 为每个任务输出单独的结果
        outputs = [layer(x) for layer in self.output_layers]
        
        return outputs  # 返回每个任务的输出

# 1D densenet
# 定义1D DenseNet模型
class DenseNet1DModel(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(DenseNet1DModel, self).__init__()

        # 使用DenseNet结构，替换为1D卷积操作
        self.densenet_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 输出层为每个任务设置一个单独的输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(512, 1) for _ in range(len(output_dims))
        ])
    
    def forward(self, x):
        # 输入的x形状是[batch_size, input_dim=33]，将其转换为[batch_size, input_dim, seq_len]
        # 对于蛋白质序列数据，我们认为每个序列的特征是33维
        x = x.unsqueeze(2)  # 变成[batch_size, 1, 33]
        
        # 提取1D DenseNet特征
        features = self.densenet_block(x)
        
        # 通过池化层对1D特征进行降维
        features = F.adaptive_avg_pool1d(features, 1)  # 输出[batch_size, 512, 1]
        features = features.view(features.size(0), -1)  # 展平为[batch_size, 512]

        # 为每个任务输出单独的结果
        outputs = [layer(features) for layer in self.output_layers]
        
        return outputs