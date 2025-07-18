import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size=5000):
        super().__init__()
        self.d_model = d_model  
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# 问题详解：
# super()的作用是什么？
# - super().__init__() 调用父类 nn.Module 的初始化方法，确保继承的模块正确初始化。
# - 必要性：若不调用，将无法正确注册模型参数（如权重矩阵），导致训练失败。
# - 扩展性：允许在自定义层中添加额外功能时保留父类的核心逻辑。

# nn.Embedding(vocab_size, d_model) 参数为何“反着写”？
# - PyTorch 规范：nn.Embedding 的官方定义是 (num_embeddings, embedding_dim)。
#   - num_embeddings：词汇表大小（vocab_size），表示唯一单词的数量。
#   - embedding_dim：每个词向量的维度（d_model）。
# - 设计逻辑：
#   - 输入形状：(batch_size, sequence_length) 的整数张量（单词索引）
#   - 输出形状：(batch_size, sequence_length, d_model)
#   - 参数顺序与输入/输出的维度变化逻辑一致。

# Embedding 的核心概念
# - 作用：将离散的单词符号（如 "apple"）映射为连续向量（深度学习可处理的数值形式）。
# - 数学本质：
#   - 一个可学习的查找表：EmbeddingMatrix ∈ ℝ^{vocab_size × d_model}
#   - 前向传播：对输入索引 i，输出第 i 行向量：output = EmbeddingMatrix[i]
# - Transformer 中的重要性：
#   - 提供单词的语义表示（相似的词有相似向量）
#   - 是模型处理文本输入的唯一接口

# nn.Module 的角色
# - 基类作用：PyTorch 中所有神经网络模块的基类（层、模型等均继承它）。
# - 关键功能：
#   - 参数管理：自动跟踪 nn.Parameter（如 nn.Embedding 的权重矩阵）
#   - 设备迁移：.to(device) 可将所有参数移至 GPU/CPU
#   - 模型序列化：支持 torch.save()/torch.load()
#   - 计算图构建：通过 forward 方法定义运算，实现自动微分

# forward 方法的作用
# - 核心功能：定义输入 x 如何通过模块计算得到输出。
# - PyTorch 调用机制：
#   - 直接调用模块时（如 embed = Embedding(); output = embed(x)）
#   - 内部自动调用 forward() 并执行计算图构建