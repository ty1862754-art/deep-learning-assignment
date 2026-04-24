# 改动记录

## 改动1: FeedForwardBlock 实现

### 改动前
```python
# 创建前馈网络层
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        ## TODO: 实现前馈网络
        """
        前馈网络由两层线性变换和中间的 ReLU 激活组成：
        1. 输入和输出维度均为 d_model
        2. 中间隐藏层维度为 d_ff
        3. 在 ReLU 后加入 dropout 以降低过拟合风险
        """

    def forward(self, x):
        pass
```

### 改动后
```python
# 创建前馈网络层
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # 第一层线性映射：d_model -> d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        # ReLU 激活后使用 dropout 做正则化
        self.dropout = nn.Dropout(dropout)
        # 第二层线性映射：d_ff -> d_model
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # 逐位置前馈计算：Linear -> ReLU -> Dropout -> Linear
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

## 改动2: MultiHeadAttentionBlock.attention 实现

### 改动前
```python
@staticmethod
def attention(query, key, value, mask, dropout: nn.Dropout):
    ## TODO: 实现注意力机制
    """
    1. 先计算 query 与 key 的点积，并除以 sqrt(d_k) 做缩放
    2. 对分数做 softmax，得到注意力权重
    3. 用注意力权重对 value 加权求和
    4. 通过 attention_scores.masked_fill_(mask == 0, -1e9) 进行掩码
    """
    pass
```

### 改动后
```python
@staticmethod
def attention(query, key, value, mask, dropout: nn.Dropout):
    # query/key/value 形状: (batch, heads, seq_len, d_k)
    d_k = query.shape[-1]

    # 缩放点积注意力分数
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用掩码，屏蔽无效位置或未来位置
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    # 对分数归一化为概率
    attention_scores = torch.softmax(attention_scores, dim=-1)

    # 对注意力概率做 dropout（可选）
    if dropout is not None:
        attention_scores = dropout(attention_scores)

    # 与 value 相乘得到加权结果，并返回注意力权重
    return (attention_scores @ value), attention_scores
```


  



