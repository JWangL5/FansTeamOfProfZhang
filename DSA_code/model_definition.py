# model_definition.py
import torch
import torch.nn as nn

class DPTModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_heads=4, num_layers=4):
        super(DPTModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.fc1 = nn.Linear(input_dim, 128)  # 输入通道转换
        self.fc2 = nn.Linear(128, output_dim)  # 输出预测

    def forward(self, x):
        # 调整输入形状为 (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1)  # 变成 [batch_size, 1, input_dim]
        
        # Transformer 编码
        x = self.transformer_encoder(x)
        
        # 平均池化
        x = x.mean(dim=1)  # 取 Transformer 输出的平均值作为表示
        
        # FC 层处理
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
