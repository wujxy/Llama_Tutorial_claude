# LLaMA 从零实现 + LoRA 微调 - Ver 1.0 版本日志

## 项目概述

本项目是一个用于教学目的的 LLM 实现教程，实现了从零开始训练 LLaMA 架构模型以及使用 LoRA 进行参数高效微调的完整流程。项目代码简洁清晰，适合初学者理解 Transformer 架构和 LLM 训练的基本原理。

---

## 一、核心功能模块

### 1.1 基础模型训练 (Base Training)

**功能说明：** 从随机初始化开始训练一个完整的 LLaMA 模型

**核心文件：**
- 训练脚本：[src/train_base.py](src/train_base.py)
- 模型定义：[src/llama/model.py](src/llama/model.py)
- 配置文件：[configs/train.yaml](configs/train.yaml)

**输出目录：** `outputs/base_train/`
- `model.pt` - 训练好的模型权重
- `tokenizer.json` - 词表文件
- `config.json` - 模型配置
- `train_log.txt` - 训练日志

---

### 1.2 LoRA 微调 (LoRA Fine-tuning)

**功能说明：** 在冻结基础模型的前提下，仅训练 LoRA 适配器参数

**核心文件：**
- 训练脚本：[src/train_lora.py](src/train_lora.py)
- LoRA 实现：[src/lora/lora.py](src/lora/lora.py)
- 配置文件：[configs/lora_pretrain.yaml](configs/lora_pretrain.yaml)

**输出目录：** `outputs/lora_pretrain/`
- `lora.pt` - LoRA 权重（仅包含 LoRA 参数）
- `lora_config.json` - LoRA 配置
- `train_log.txt` - 训练日志

---

### 1.3 模型评估 (Model Evaluation)

**功能说明：** 评估基础模型或基础模型+LoRA 的性能，包括损失计算和文本生成

**核心文件：**
- 评估脚本：[src/eval.py](src/eval.py)
- 配置文件：[configs/eval.yaml](configs/eval.yaml)

**输出目录：** `outputs/eval/`
- `metrics.json` - 损失和困惑度指标
- `generations.txt` - 生成的文本样本
- `eval_log.txt` - 评估日志

---

## 二、关键组件实现详解

### 2.1 简易字符级 Tokenizer

**文件位置：** [src/tokenizer/simple_tokenizer.py](src/tokenizer/simple_tokenizer.py)

#### 实现原理

这是一个**字符级别**的分词器，适用于教学演示。词表由训练数据中的所有唯一字符构建而成。

#### 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `<pad>` | 0 | 填充符（用于变长序列补齐） |
| `<bos>` | 1 | 序列开始标记 |
| `<eos>` | 2 | 序列结束标记 |
| `<unk>` | 3 | 未知字符标记 |

#### 词表构建流程

```python
# 1. 初始化特殊 token
vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

# 2. 收集训练文本中的所有唯一字符
char_set = set()
for text in texts:
    char_set.update(text)

# 3. 按字符排序后分配 ID（确保词表确定性）
for char in sorted(char_set):
    vocab[char] = len(vocab)
```

#### 编码与解码

```python
# 编码：文本 → token ID 序列
def encode(text, add_bos=True, add_eos=True):
    ids = []
    if add_bos: ids.append(BOS_ID)
    for char in text:
        ids.append(vocab.get(char, UNK_ID))  # 未知字符用 UNK_ID
    if add_eos: ids.append(EOS_ID)
    return ids

# 解码：token ID 序列 → 文本
def decode(ids, skip_special=True):
    chars = []
    for idx in ids:
        if skip_special and idx in special_ids:
            continue
        chars.append(inverse_vocab[idx])
    return ''.join(chars)
```

#### 数据集集成

在训练时，数据按以下格式化：

```
### Instruction:
{prompt}

### Response:
{response}
```

然后进行 Tokenize、截断/填充、创建 causal LM 标签。

---

### 2.2 LLaMA 模型架构

**文件位置：** [src/llama/model.py](src/llama/model.py)

#### 整体架构

```
输入 (input_ids) → Token Embedding → [TransformerBlock × n_layers] → RMSNorm → Output Projection → 输出 (logits)
```

#### 核心组件

##### 2.2.1 RMSNorm（均方根层归一化）

```python
# 公式：output = (x / sqrt(mean(x^2) + eps)) * weight
def _norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

特点：相比 LayerNorm 去掉了均值中心化，计算更简单高效。

##### 2.2.2 RoPE（旋转位置编码）

```python
# 预计算频率
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

# 应用旋转（对 Q 和 K）
xq_ = torch.view_as_complex(xq.reshape(..., -1, 2))
xk_ = torch.view_as_complex(xk.reshape(..., -1, 2))
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
```

作用：通过旋转操作将位置信息编码到 Query 和 Key 中，无需额外的位置嵌入参数。

##### 2.2.3 多头注意力机制

```python
class Attention:
    # Q, K, V 投影
    self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
```

支持 **GQA（分组查询注意力）**：当 `n_kv_heads < n_heads` 时，KV 头会被重复以匹配 Q 头数量。

**因果掩码**：确保每个位置只能注意到之前的token。

##### 2.2.4 SwiGLU 前馈网络

```python
# SwiGLU 激活函数
hidden = F.silu(self.w1(x)) * self.w3(x)  # gate(x) * up(x)
output = self.w2(hidden)  # down projection
```

相比标准 ReLU，SwiGLU 使用门控机制，性能更好。

##### 2.2.5 Transformer Block（预归一化）

```python
# Pre-LN 架构（先归一化再计算）
h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
out = h + self.feed_forward(self.ffn_norm(h))
```

Pre-LN 相比 Post-LN 训练更稳定。

---

### 2.3 LoRA 实现

**文件位置：** [src/lora/lora.py](src/lora/lora.py)

#### 核心思想

LoRA 冻结预训练权重，并在每个层注入可训练的低秩分解矩阵：

```
output = W @ x + scale * B @ A @ x
```

其中：
- `W`：冻结的原始权重矩阵
- `A`：下投影矩阵 (d_in → r)，Kaiming 初始化
- `B`：上投影矩阵 (r → d_out)，零初始化（确保初始扰动为零）
- `scale = alpha / r`：缩放因子

#### LoRA Linear 层

```python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        self.original_layer = original_layer  # 冻结的原始层
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))  # 零初始化
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # 零初始化
        self.scaling = alpha / r

    def forward(self, x):
        # 原始路径
        original_output = self.original_layer(x)
        # LoRA 路径
        lora_output = x @ lora_A.T @ lora_B.T * self.scaling
        return original_output + lora_output
```

#### 参数效率

| 模块 | 原始参数 | LoRA 参数 (r=8) | 压缩比 |
|------|----------|-----------------|--------|
| Linear(128, 256) | 32,768 | 2,048 | ~6% |
| 全模型 (~200K) | ~200K | ~12K | ~6% |

---

### 2.4 损失函数设计

**文件位置：** [src/train_base.py:108-121](src/train_base.py#L108-L121)、[src/eval.py:121-131](src/eval.py#L121-L131)

#### 交叉熵损失（Cross-Entropy Loss）

本项目使用标准的**交叉熵损失**训练因果语言模型（Causal Language Model）。

```python
# Shift logits and labels for next-token prediction
shift_logits = logits[..., :-1, :].contiguous()   # 去掉最后一个位置的预测
shift_labels = labels[..., 1:].contiguous()       # 去掉第一个位置的标签

# Flatten for cross-entropy
shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch*(seq-1), vocab_size)
shift_labels = shift_labels.view(-1)                              # (batch*(seq-1),)

# Compute cross-entropy loss (ignore padding tokens with -100)
loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

#### 因果语言模型原理

这是一个**自回归**语言模型，核心思想是：**给定前面的 token，预测下一个 token**。

**对齐逻辑示例：**

假设输入序列（包含 BOS 和 EOS）为：

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|---|
| input_ids | `<bos>` | H | e | l | l | o |
| labels | H | e | l | l | o | `<eos>` |

- 位置 0 的 `<bos>` → 预测位置 1 的 `H`
- 位置 1 的 `H` → 预测位置 2 的 `e`
- 位置 2 的 `e` → 预测位置 3 的 `l`
- ...
- 位置 5 的 `o` → 预测位置 6 的 `<eos>`

通过 **shift（移位）** 实现：

```python
# logits 的形状：(batch, seq_len, vocab_size)
# 我们用位置 i 的 logits 预测位置 i+1 的 token

# 去掉最后一个位置的 logits（没有下一个 token 可预测）
shift_logits = logits[..., :-1, :]  # (batch, seq_len-1, vocab_size)

# 去掉第一个位置的 label（没有被预测的 token）
shift_labels = labels[..., 1:]      # (batch, seq_len-1)
```

#### 填充处理（Padding Masking）

**代码位置：** [src/data/dataset.py:127-133](src/data/dataset.py#L127-L133)

```python
# Pad to max_seq_len
pad_len = self.max_seq_len - len(input_ids)
if pad_len > 0:
    input_ids = input_ids + [pad_id] * pad_len
    labels = labels + [-100] * pad_len      # 关键：用 -100 标记填充位置
    attention_mask = attention_mask + [0] * pad_len
```

**为什么用 -100？**

PyTorch 的 `CrossEntropyLoss` 提供 `ignore_index` 参数，当 `label == -100` 时，该位置的损失**不会被计算**，也不会影响梯度更新。

示例：

```
原始序列: [BOS, H, e, l, l, o, EOS]           (长度 7)
填充后:   [BOS, H, e, l, l, o, EOS, PAD, PAD, PAD]  (长度 10)
labels:   [  H,  e,  l,  l,  o, EOS, -100, -100, -100]
                                            ^^^^ 这些位置不计算损失
```

#### 数学公式

交叉熵损失的数学定义：

```
L = -Σ log P(y_i | x_<i)
```

对于词汇表 V 上的分类：

```
L = -Σ log(softmax(logits_i)[y_i])
```

其中 `softmax(logits_i)[y_i]` 是真实 token `y_i` 的预测概率。

---

### 2.5 模型评估指标

**文件位置：** [src/eval.py:73-337](src/eval.py#L73-L337)

#### 2.5.1 平均损失（Average Loss）

```python
def evaluate_loss(...) -> tuple[float, int]:
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for item in data:
            # ... forward pass ...
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            total_loss += loss.item()
            num_samples += 1

    avg_loss = total_loss / num_samples
    return avg_loss, num_samples
```

**含义：** 所有样本的平均交叉熵损失，越低越好。

#### 2.5.2 困惑度（Perplexity, PPX）

```python
perplexity = torch.exp(torch.tensor(avg_loss))
```

**什么是困惑度？**

困惑度是衡量语言模型预测能力的重要指标：

```
PPX = exp(L)
```

其中 L 是平均交叉熵损失。

**直观理解：**

| PPX 值 | 含义 |
|--------|------|
| 1 | 模型完全确定下一个 token（理想情况） |
| 10 | 模型在 10 个等可能的候选中犹豫 |
| 100 | 模型在 100 个候选中犹豫 |
| ∞ | 模型完全随机猜测 |

**示例：** 如果 PPX = 10，说明模型平均需要在约 10 个候选 token 中做选择。

**与词汇表的关系：**

- 最小 PPX = 1（完美预测）
- 最大 PPX = 词汇表大小（完全随机）

对于本项目的字符级 tokenizer（词汇表约 100-200），PPX 应该远小于这个值。

#### 2.5.3 文本生成质量（Generation Quality）

```python
def generate_text(
    model, prompt, tokenizer, device,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = None,
) -> str:
    # 自回归生成
    for _ in range(max_new_tokens):
        logits = model(generated_ids)
        next_token_logits = logits[0, -1, :] / temperature

        # Top-k 采样（可选）
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < v[-1]] = float('-inf')

        # 采样下一个 token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 遇到 EOS 停止
        if next_token.item() == tokenizer.EOS_ID:
            break

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
```

**评估方式：**

生成文本后，与期望输出进行**人工对比**：

```python
# 输出示例
Prompt: What is AI?
Expected: AI stands for Artificial Intelligence.
Generated: AI is...
```

本项目使用人工对比，但实际生产中可以使用自动化指标如 BLEU、ROUGE。

#### 评估指标总结

| 指标 | 类型 | 用途 | 计算位置 |
|------|------|------|----------|
| **交叉熵损失** | 训练/评估 | 优化目标，衡量预测准确性 | [train_base.py:121](src/train_base.py#L121) |
| **平均损失** | 评估 | 整体性能概览 | [eval.py:136](src/eval.py#L136) |
| **困惑度** | 评估 | 直观的模型不确定性度量 | [eval.py:332](src/eval.py#L332) |
| **生成文本** | 评估 | 定性质量检查 | [eval.py:365](src/eval.py#L365) |

---

## 三、配置选项说明

### 3.1 基础训练配置 ([configs/train.yaml](configs/train.yaml))

```yaml
seed: 42                      # 随机种子
data_path: "data/train.jsonl" # 训练数据
output_dir: "outputs/base_train"
max_seq_len: 256              # 最大序列长度

model:
  dim: 128                    # 嵌入维度
  n_layers: 2                 # Transformer 层数
  n_heads: 4                  # 注意力头数
  n_kv_heads: 4               # KV 头数
  hidden_dim: 256             # FFN 隐藏维度
  dropout: 0.0                # Dropout 率

train:
  batch_size: 2
  lr: 0.001                   # 学习率
  max_steps: 100              # 最大训练步数
  weight_decay: 0.01
  log_every: 10
```

### 3.2 LoRA 训练配置 ([configs/lora_pretrain.yaml](configs/lora_pretrain.yaml))

```yaml
seed: 42
data_path: "data/lora.jsonl"
base_ckpt_dir: "outputs/base_train"  # 基础模型路径
output_dir: "outputs/lora_pretrain"

lora:
  r: 8                        # LoRA 秩
  alpha: 16.0                 # 缩放参数
  dropout: 0.0
  target_modules:             # 应用 LoRA 的模块
    - "wq"    # Query 投影
    - "wk"    # Key 投影
    - "wv"    # Value 投影
    - "wo"    # 输出投影
    - "w1"    # FFN gate
    - "w2"    # FFN down
    - "w3"    # FFN up

train:
  batch_size: 2
  lr: 0.001
  max_steps: 50
```

### 3.3 评估配置 ([configs/eval.yaml](configs/eval.yaml))

```yaml
seed: 42
data_path: "data/eval.jsonl"
base_ckpt_dir: "outputs/base_train"
lora_ckpt_dir: ""             # 留空=仅基础模型，"outputs/lora_pretrain"=基础+LoRA
output_dir: "outputs/eval"

eval:
  batch_size: 1
  max_eval_samples: 10        # 评估样本数
  gen_max_new_tokens: 64      # 生成最大 token 数
```

---

## 四、快速开始

### 4.1 环境准备

```bash
# 创建虚拟环境
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
# 或 ml_env\Scripts\activate  # Windows

# 安装依赖
pip install torch pyyaml tqdm numpy
```

### 4.2 准备数据

数据格式为 JSONL，每行一个样本：

```jsonl
{"prompt": "What is AI?", "response": "AI stands for Artificial Intelligence."}
{"prompt": "Hello!", "response": "Hi there! How can I help you?"}
```

需要准备三个数据文件：
- `data/train.jsonl` - 基础训练数据
- `data/lora.jsonl` - LoRA 微调数据
- `data/eval.jsonl` - 评估数据

### 4.3 完整训练流程

```bash
# 方法一：使用一键脚本
bash run_pipeline.sh

# 方法二：分步执行

# 1. 基础模型训练
python -m src.train_base --config configs/train.yaml

# 2. LoRA 微调
python -m src.train_lora --config configs/lora_pretrain.yaml

# 3. 评估基础模型
python -m src.eval --config configs/eval.yaml

# 4. 评估基础模型 + LoRA（需要先修改 eval.yaml 中 lora_ckpt_dir）
python -m src.eval --config configs/eval.yaml
```

### 4.4 VSCode 调试配置

确保 [`.vscode/launch.json`](.vscode/launch.json) 配置如下：

```json
{
    "name": "Python 调试程序: 当前文件",
    "type": "debugpy",
    "request": "launch",
    "python": "/path/to/your/python",
    "module": "src.train_base",  // 注意使用 "module" 而不是 "program"
    "args": ["--config", "configs/train.yaml"],
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}"
}
```

### 4.5 查看结果

```bash
# 查看训练日志
cat outputs/base_train/train_log.txt
cat outputs/lora_pretrain/train_log.txt

# 查看评估结果
cat outputs/eval/metrics.json
cat outputs/eval/generations.txt
```

---

## 五、项目结构

```
Llama_Tutorial_claude/
├── configs/              # 配置文件
│   ├── train.yaml       # 基础训练配置
│   ├── lora_pretrain.yaml
│   └── eval.yaml
├── data/                # 训练数据
│   ├── train.jsonl
│   ├── lora.jsonl
│   └── eval.jsonl
├── outputs/             # 输出目录
│   ├── base_train/
│   ├── lora_pretrain/
│   └── eval/
├── src/
│   ├── data/
│   │   └── dataset.py   # 数据集处理
│   ├── llama/
│   │   └── model.py     # LLaMA 模型
│   ├── lora/
│   │   └── lora.py      # LoRA 实现
│   ├── tokenizer/
│   │   └── simple_tokenizer.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── logging.py
│   │   └── seed.py
│   ├── train_base.py    # 基础训练脚本
│   ├── train_lora.py    # LoRA 训练脚本
│   └── eval.py          # 评估脚本
├── run_pipeline.sh      # 一键运行脚本
├── .vscode/
│   └── launch.json      # VSCode 调试配置
└── update_log.md        # 本日志文件
```

---

## 六、技术特点总结

| 特性 | 说明 |
|------|------|
| **字符级 Tokenizer** | 简单易懂，适合教学演示 |
| **LLaMA 架构** | 完整实现 RoPE、RMSNorm、SwiGLU、GQA |
| **LoRA 微调** | 参数高效，可训练参数仅约 6% |
| **纯 PyTorch** | 无外部依赖，代码清晰 |
| **模块化设计** | 易于扩展和修改 |

---

## 七、后续版本优化

1. **复杂或直接使用已有的Tokenizer**（如 BPE、SentencePiece）
2. **添加 KV Cache** 加速推理
3. **支持梯度累积** 和 **混合精度训练**
4. **添加更多评估指标**（BLEU、ROUGE 等）
5. **支持分布式训练**

---

*最后更新：2026年1月12日*
