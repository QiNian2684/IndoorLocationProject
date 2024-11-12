# model_definition.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=520):
        """
        正弦位置编码。

        参数：
        - d_model: 模型的维度
        - max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为缓冲区，不作为模型参数

    def forward(self, x):
        """
        将位置编码添加到输入张量。

        参数：
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]

        返回：
        - x: 添加位置编码后的张量，形状为 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class WiFiTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=520, model_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        """
        定义用于特征提取的 Transformer 自编码器模型。

        参数：
        - input_dim: 输入特征维度（序列长度），默认为 520
        - model_dim: 模型内部特征维度，默认为 128
        - num_heads: 多头注意力头数，默认为 8
        - num_layers: Transformer 编码器和解码器的层数，默认为 2
        - dropout: Dropout 概率，默认为 0.1
        """
        super(WiFiTransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim

        # 编码器：将每个输入特征映射到 model_dim 维度
        self.encoder_embedding = nn.Linear(1, model_dim)  # 输入每个位置1个特征
        self.positional_encoding = PositionalEncoding(model_dim, max_len=input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 解码器：重构输入序列
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.decoder_output = nn.Linear(model_dim, 1)  # 重构原始1个特征

        self.activation = nn.ReLU()  # 保持 ReLU 激活函数

        # 回归头：用于预测经度和纬度
        self.regression_head = nn.Sequential(
            nn.Linear(model_dim * input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 输出经度和纬度
        )

        # 分类头：用于预测楼层
        self.classification_head = nn.Sequential(
            nn.Linear(model_dim * input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 假设楼层数为 5，可根据实际情况调整
        )

    def encode(self, x):
        """
        编码器部分，提取特征。

        参数：
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回：
        - memory: Transformer 编码器的输出，形状为 [batch_size, input_dim, model_dim]
        """
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        x = self.encoder_embedding(x)  # [batch_size, input_dim, model_dim]
        x = self.activation(x)
        x = self.positional_encoding(x)  # 添加位置编码
        memory = self.transformer_encoder(x)  # [batch_size, input_dim, model_dim]
        return memory

    def decode(self, memory):
        """
        解码器部分，重构输入。

        参数：
        - memory: Transformer 编码器的输出，形状为 [batch_size, input_dim, model_dim]

        返回：
        - x: 重构后的输入，形状为 [batch_size, input_dim]
        """
        # 使用编码器的输出作为解码器的输入
        decoder_input = memory  # [batch_size, input_dim, model_dim]
        decoder_input = self.positional_encoding(decoder_input)  # 添加位置编码
        output = self.transformer_decoder(decoder_input, memory)  # [batch_size, input_dim, model_dim]
        output = self.decoder_output(output)  # [batch_size, input_dim, 1]
        output = output.squeeze(-1)  # [batch_size, input_dim]
        return output

    def forward(self, x):
        """
        前向传播函数，完成自编码器的编码和解码。

        参数：
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回：
        - reconstructed: 重构后的输入，形状为 [batch_size, input_dim]
        - regression_output: 回归预测，形状为 [batch_size, 2]
        - classification_output: 分类预测，形状为 [batch_size, num_classes]
        """
        memory = self.encode(x)
        reconstructed = self.decode(memory)

        # 将编码器的输出展平成 [batch_size, input_dim * model_dim]
        memory_flat = memory.view(memory.size(0), -1)

        # 回归预测
        regression_output = self.regression_head(memory_flat)

        # 分类预测
        classification_output = self.classification_head(memory_flat)

        return reconstructed, regression_output, classification_output

    def extract_features(self, x):
        """
        提取编码器的特征。

        参数：
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回：
        - features: 提取的特征，形状为 [batch_size, input_dim * model_dim]
        """
        memory = self.encode(x)
        features = memory.view(memory.size(0), -1)
        return features
