import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key=None, value=None, mask=None):
        # 如果key和value为None，则是自注意力
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size = query.size(0)
        q_seq_len = query.size(1)
        kv_seq_len = key.size(1)
        
        # 生成Q, K, V
        Q = self.w_q(query).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, kv_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, kv_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        return self.w_o(output), attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(d_model, num_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(d_model, num_heads),
                'norm1': nn.LayerNorm(d_model),
                'cross_attn': MultiHeadSelfAttention(d_model, num_heads),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm3': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def create_padding_mask(self, x, pad_token=0):
        return (x != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
        
    def encode(self, src, src_mask):
        # 源语言编码
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            # 自注意力 - 修改这里
            attn_output, _ = layer['self_attn'](x, mask=src_mask)
            x = layer['norm1'](x + attn_output)
            
            # 前馈网络
            ffn_output = layer['ffn'](x)
            x = layer['norm2'](x + ffn_output)
            
        return x
    
    def decode(self, tgt, memory, tgt_mask, src_mask):
        # 目标语言解码
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            # 自注意力（带look-ahead mask）
            attn_output, _ = layer['self_attn'](x, mask=tgt_mask)
            x = layer['norm1'](x + attn_output)
            
            # 交叉注意力 - 修改这里
            cross_attn_output, attention_weights = layer['cross_attn'](x, memory, memory, src_mask)
            x = layer['norm2'](x + cross_attn_output)
            
            # 前馈网络
            ffn_output = layer['ffn'](x)
            x = layer['norm3'](x + ffn_output)
            
        return self.output_projection(x), attention_weights
    
    def forward(self, src, tgt):
        # 创建mask
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt) & self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        
        # 编码
        memory = self.encode(src, src_mask)
        
        # 解码
        output, attention_weights = self.decode(tgt, memory, tgt_mask, src_mask)
        
        return output, attention_weights

# 创建简单的词汇表
def create_vocab():
    # 英文词汇
    en_words = ['<pad>', '<sos>', '<eos>', 'i', 'love', 'you', 'hello', 'world', 'good', 'morning', 'how', 'are']
    # 中文词汇
    zh_words = ['<pad>', '<sos>', '<eos>', '我', '爱', '你', '你好', '世界', '早上', '好', '怎么', '样']
    
    en_vocab = {word: i for i, word in enumerate(en_words)}
    zh_vocab = {word: i for i, word in enumerate(zh_words)}
    
    en_idx2word = {i: word for word, i in en_vocab.items()}
    zh_idx2word = {i: word for word, i in zh_vocab.items()}
    
    return en_vocab, zh_vocab, en_idx2word, zh_idx2word

# 创建训练数据
def create_training_data(en_vocab, zh_vocab):
    # 简单的英中对照句子
    pairs = [
        (['i', 'love', 'you'], ['我', '爱', '你']),
        (['hello', 'world'], ['你好', '世界']),
        (['good', 'morning'], ['早上', '好']),
        (['how', 'are', 'you'], ['你', '怎么', '样'])
    ]
    
    src_data = []
    tgt_data = []
    
    for en_sent, zh_sent in pairs:
        # 编码英文句子
        src_ids = [en_vocab['<sos>']] + [en_vocab[w] for w in en_sent] + [en_vocab['<eos>']]
        # 编码中文句子
        tgt_ids = [zh_vocab['<sos>']] + [zh_vocab[w] for w in zh_sent] + [zh_vocab['<eos>']]
        
        # 填充到相同长度
        max_len = 8
        src_ids += [en_vocab['<pad>']] * (max_len - len(src_ids))
        tgt_ids += [zh_vocab['<pad>']] * (max_len - len(tgt_ids))
        
        src_data.append(src_ids[:max_len])
        tgt_data.append(tgt_ids[:max_len])
    
    return torch.tensor(src_data), torch.tensor(tgt_data)

# 主函数
def main():
    # 创建词汇表
    en_vocab, zh_vocab, en_idx2word, zh_idx2word = create_vocab()
    
    # 创建训练数据
    src_data, tgt_data = create_training_data(en_vocab, zh_vocab)
    
    print("训练数据示例:")
    print(f"英文: {src_data[0]}")
    print(f"中文: {tgt_data[0]}")
    
    # 创建模型
    model = SimpleTransformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(zh_vocab),
        d_model=128,
        num_heads=8,
        num_layers=2
    ).to(device)
    
    # 将数据移到GPU
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)
    
    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    
    print(f"\n开始训练... (设备: {device})")
    model.train()
    
    # 简单训练循环
    for epoch in range(100):
        optimizer.zero_grad()
        
        # 前向传播
        tgt_input = tgt_data[:, :-1]  # 去掉最后一个token作为输入
        tgt_output = tgt_data[:, 1:]  # 去掉第一个token作为目标
        
        output, attention_weights = model(src_data, tgt_input)
        
        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("\n训练完成！")
    
    # 测试翻译
    model.eval()
    with torch.no_grad():
        test_src = src_data[0:1]  # 取第一个句子测试
        print(f"\n测试翻译:")
        print(f"输入英文: {[en_idx2word[idx.item()] for idx in test_src[0] if idx.item() != 0]}")
        
        # 简单的贪心解码
        max_len = 8
        tgt_input = torch.tensor([[zh_vocab['<sos>']]]).to(device)
        
        for _ in range(max_len-1):
            output, attention_weights = model(test_src, tgt_input)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            
            if next_token.item() == zh_vocab['<eos>']:
                break
        
        translated = [zh_idx2word[idx.item()] for idx in tgt_input[0] if idx.item() not in [0, zh_vocab['<sos>']]]
        print(f"翻译结果: {translated}")
        
        # 显示注意力权重
        print(f"\n最后一层的注意力权重形状: {attention_weights.shape}")
        print("注意力权重可视化 (第一个头的权重):")
        attn_matrix = attention_weights[0, 0].cpu().numpy()
        print(f"注意力矩阵形状: {attn_matrix.shape}")
        
        # 简单的注意力权重显示
        src_tokens = [en_idx2word[idx.item()] for idx in test_src[0] if idx.item() != 0]
        tgt_tokens = [zh_idx2word[idx.item()] for idx in tgt_input[0] if idx.item() not in [0]]
        
        print(f"\n源语言tokens: {src_tokens}")
        print(f"目标语言tokens: {tgt_tokens}")
        print("注意力权重矩阵 (目标->源):")
        for i, tgt_token in enumerate(tgt_tokens[:attn_matrix.shape[0]]):
            print(f"{tgt_token}: ", end="")
            for j, src_token in enumerate(src_tokens[:attn_matrix.shape[1]]):
                if j < attn_matrix.shape[1]:
                    print(f"{src_token}:{attn_matrix[i,j]:.3f} ", end="")
            print()

if __name__ == "__main__":
    main()