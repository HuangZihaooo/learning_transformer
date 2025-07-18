import torch
import torch.nn.functional as F
import numpy as np

# 基本Softmax示例
def softmax_example():
    # 输入logits
    logits = torch.tensor([2.0, 1.0, 0.1])
    
    # 使用PyTorch内置函数
    probs = F.softmax(logits, dim=0)
    print(f"Softmax输出: {probs}")
    print(f"概率和: {probs.sum()}")
    
    # 手动实现
    exp_logits = torch.exp(logits)
    manual_probs = exp_logits / exp_logits.sum()
    print(f"手动计算: {manual_probs}")

# Transformer中的注意力Softmax
def attention_softmax():
    # 模拟注意力分数
    # batch_size = 2：批次大小，表示同时处理2个句子
    # seq_len = 4：序列长度，表示每个句子有4个词/token
    # d_k = 64：键向量的维度，表示每个词用64维向量表示
    batch_size, seq_len, d_k = 2, 4, 64
    
    # 随机生成Q, K
    Q = torch.randn(batch_size, seq_len, d_k) # 查询 可以理解为：2个句子，每个句子4个词，每个词的查询向量是64维
    K = torch.randn(batch_size, seq_len, d_k) # 键 可以理解为：2个句子，每个句子4个词，每个词的键向量是64维
    
    # 计算注意力分数
    # .transpose(-2, -1) 交换最后两个维度，将 K 的维度从 (batch_size, seq_len, d_k) 变为 (batch_size, d_k, seq_len)
    # 这样 Q 和 K 的维度就匹配了，可以进行矩阵乘法
    # Q形状：[2, 4, 64]
    # K转置后形状：[2, 64, 4]
    # 结果形状：[2, 4, 4]
    # scores[i, j, k] = Q的第j个词向量 与 K的第k个词向量 的点积
    # 例如：scores[0, 1, 3] = 第1个句子中第2个词对第4个词的"关注分数"
    # 结果是一个4x4的矩阵，表示每个词对其他词的"关注分数"
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # 应用Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"每行权重和: {attention_weights.sum(dim=-1)}")

import torch
import torch.nn.functional as F
import numpy as np

# 简化示例：1个句子，3个词，2维向量
def simple_attention_example():
    print("=== 简化的注意力计算示例 ===")
    
    # 步骤1：定义参数
    batch_size, seq_len, d_k = 1, 3, 2
    print(f"句子数: {batch_size}, 词数: {seq_len}, 向量维度: {d_k}")
    
    # 步骤2：创建简单的Q和K
    Q = torch.tensor([[[1.0, 2.0],    # 第1个词的查询向量
                       [3.0, 1.0],    # 第2个词的查询向量  
                       [2.0, 3.0]]])  # 第3个词的查询向量
    
    K = torch.tensor([[[2.0, 1.0],    # 第1个词的键向量
                       [1.0, 3.0],    # 第2个词的键向量
                       [3.0, 2.0]]])  # 第3个词的键向量
    
    print(f"\nQ矩阵形状: {Q.shape}")
    print(f"Q矩阵内容:\n{Q}")
    print(f"\nK矩阵形状: {K.shape}")
    print(f"K矩阵内容:\n{K}")
    
    # 步骤3：K转置
    K_transposed = K.transpose(-2, -1)
    print(f"\nK转置后形状: {K_transposed.shape}")
    print(f"K转置后内容:\n{K_transposed}")
    
    # 步骤4：计算原始分数（矩阵乘法）
    raw_scores = torch.matmul(Q, K_transposed)
    print(f"\n原始分数形状: {raw_scores.shape}")
    print(f"原始分数内容:\n{raw_scores}")
    
    # 手动验证第一行的计算
    print(f"\n=== 手动验证第1个词的注意力分数 ===")
    q1 = Q[0, 0, :]  # 第1个词的查询向量 [1.0, 2.0]
    k1 = K[0, 0, :]  # 第1个词的键向量 [2.0, 1.0]
    k2 = K[0, 1, :]  # 第2个词的键向量 [1.0, 3.0]
    k3 = K[0, 2, :]  # 第3个词的键向量 [3.0, 2.0]
    
    score_1_1 = torch.dot(q1, k1)  # 1*2 + 2*1 = 4
    score_1_2 = torch.dot(q1, k2)  # 1*1 + 2*3 = 7
    score_1_3 = torch.dot(q1, k3)  # 1*3 + 2*2 = 7
    
    print(f"第1个词对第1个词的分数: {q1} · {k1} = {score_1_1}")
    print(f"第1个词对第2个词的分数: {q1} · {k2} = {score_1_2}")
    print(f"第1个词对第3个词的分数: {q1} · {k3} = {score_1_3}")
    
    # 步骤5：缩放
    scaled_scores = raw_scores / np.sqrt(d_k)
    print(f"\n缩放后分数 (除以√{d_k}={np.sqrt(d_k):.2f}):\n{scaled_scores}")
    
    # 步骤6：应用Softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"\n注意力权重 (Softmax后):\n{attention_weights}")
    
    # 验证每行和为1
    row_sums = attention_weights.sum(dim=-1)
    print(f"\n每行权重和: {row_sums}")
    
    # 手动验证第一行的softmax
    print(f"\n=== 手动验证第1行的Softmax ===")
    first_row = scaled_scores[0, 0, :]  # [2.83, 4.95, 4.95]
    exp_values = torch.exp(first_row)
    manual_softmax = exp_values / exp_values.sum()
    print(f"第1行缩放分数: {first_row}")
    print(f"指数值: {exp_values}")
    print(f"手动计算的softmax: {manual_softmax}")
    print(f"PyTorch计算的softmax: {attention_weights[0, 0, :]}")

# 运行示例
import torch
import torch.nn.functional as F
import numpy as np

# 简化示例：1个句子，3个词，2维向量
def simple_attention_example():
    print("=== 简化的注意力计算示例 ===")
    
    # 步骤1：定义参数
    batch_size, seq_len, d_k = 1, 3, 2
    print(f"句子数: {batch_size}, 词数: {seq_len}, 向量维度: {d_k}")
    
    # 步骤2：创建简单的Q和K
    Q = torch.tensor([[[1.0, 2.0],    # 第1个词的查询向量
                       [3.0, 1.0],    # 第2个词的查询向量  
                       [2.0, 3.0]]])  # 第3个词的查询向量
    
    K = torch.tensor([[[2.0, 1.0],    # 第1个词的键向量
                       [1.0, 3.0],    # 第2个词的键向量
                       [3.0, 2.0]]])  # 第3个词的键向量
    
    print(f"\nQ矩阵形状: {Q.shape}")
    print(f"Q矩阵内容:\n{Q}")
    print(f"\nK矩阵形状: {K.shape}")
    print(f"K矩阵内容:\n{K}")
    
    # 步骤3：K转置
    K_transposed = K.transpose(-2, -1)
    print(f"\nK转置后形状: {K_transposed.shape}")
    print(f"K转置后内容:\n{K_transposed}")
    
    # 步骤4：计算原始分数（矩阵乘法）
    raw_scores = torch.matmul(Q, K_transposed)
    print(f"\n原始分数形状: {raw_scores.shape}")
    print(f"原始分数内容:\n{raw_scores}")
    
    # 手动验证第一行的计算
    print(f"\n=== 手动验证第1个词的注意力分数 ===")
    q1 = Q[0, 0, :]  # 第1个词的查询向量 [1.0, 2.0]
    k1 = K[0, 0, :]  # 第1个词的键向量 [2.0, 1.0]
    k2 = K[0, 1, :]  # 第2个词的键向量 [1.0, 3.0]
    k3 = K[0, 2, :]  # 第3个词的键向量 [3.0, 2.0]
    
    score_1_1 = torch.dot(q1, k1)  # 1*2 + 2*1 = 4
    score_1_2 = torch.dot(q1, k2)  # 1*1 + 2*3 = 7
    score_1_3 = torch.dot(q1, k3)  # 1*3 + 2*2 = 7
    
    print(f"第1个词对第1个词的分数: {q1} · {k1} = {score_1_1}")
    print(f"第1个词对第2个词的分数: {q1} · {k2} = {score_1_2}")
    print(f"第1个词对第3个词的分数: {q1} · {k3} = {score_1_3}")
    
    # 步骤5：缩放
    scaled_scores = raw_scores / np.sqrt(d_k)
    print(f"\n缩放后分数 (除以√{d_k}={np.sqrt(d_k):.2f}):\n{scaled_scores}")
    
    # 步骤6：应用Softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"\n注意力权重 (Softmax后):\n{attention_weights}")
    
    # 验证每行和为1
    row_sums = attention_weights.sum(dim=-1)
    print(f"\n每行权重和: {row_sums}")
    
    # 手动验证第一行的softmax
    print(f"\n=== 手动验证第1行的Softmax ===")
    first_row = scaled_scores[0, 0, :]  # [2.83, 4.95, 4.95]
    exp_values = torch.exp(first_row)
    manual_softmax = exp_values / exp_values.sum()
    print(f"第1行缩放分数: {first_row}")
    print(f"指数值: {exp_values}")
    print(f"手动计算的softmax: {manual_softmax}")
    print(f"PyTorch计算的softmax: {attention_weights[0, 0, :]}")

# 运行示例
softmax_example()
attention_softmax()
simple_attention_example()
