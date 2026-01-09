# Claude.md — From-scratch LLaMA + LoRA: Hello World Pipeline (Python 3.10)

/project:ultrathink-task <LLaMA + LoRA: Hello World Pipeline>

## 0. 核心要求（必须严格遵守）

你需要为本仓库生成一套**最小可运行**代码，实现以下闭环：

1. **完整 LLaMA 模型从零开始训练**（不使用任何模型权重，不从 HuggingFace 下载权重）
2. **LoRA 预训练**（在冻结 base 模型参数的前提下，仅训练 LoRA 参数；目标是流程跑通）
3. **性能测试**（能跑通评测：loss / perplexity + 简单生成）

额外约束：

- Python 版本：**3.10**（环境已创建: /home/NagaiYoru/LLM_Tutorial/llm_py_venv/bin/activate）
- 若缺少依赖：可自行安装（但依赖要尽量少）
- dataset：**自行创建**，随机生成**10 条样本**（训练/LoRA/评测都要能跑通）
- LLaMA 模型实现：**参照** `/home/NagaiYoru/LLM_Tutorial/llama3/llama/model.py` 的结构/风格  
  （本仓库中需要落地一份可运行的 `src/llama/model.py`，而不是直接依赖外部路径）
- 第一版不追求模型效果，只追求：**能在 10 条样本上完成训练-保存-加载-评测-生成**

> 目标是 Hello World：代码要清晰可读、注释充分、运行步骤明确、端到端跑通。

---

## 1. 你需要创建的仓库结构

在仓库根目录创建如下结构（必须一致）：

.
├── Claude.md
├── README.md
├── requirements.txt
├── configs/
│ ├── train.yaml
│ ├── lora_pretrain.yaml
│ └── eval.yaml
├── data/
│ ├── train.jsonl
│ ├── lora.jsonl
│ └── eval.jsonl
├── outputs/
│ ├── base_train/
│ ├── lora_pretrain/
│ └── eval/
└── src/
├── init.py
├── utils/
│ ├── init.py
│ ├── seed.py
│ ├── io.py
│ └── logging.py
├── tokenizer/
│ ├── init.py
│ └── simple_tokenizer.py
├── data/
│ ├── init.py
│ └── dataset.py
├── llama/
│ ├── init.py
│ └── model.py
├── lora/
│ ├── init.py
│ └── lora.py
├── train_base.py
├── train_lora.py
└── eval.py

## 2. 数据：随机生成 10 条样本（必须写入 JSONL 文件）

### 2.1 JSONL 格式

每一行一个样本（必须是合法 JSON），字段如下：

```json
{"prompt":"...","response":"..."}

### 2.2 数据内容要求
必须随机生成 10 条训练样本（可以用固定 seed 以可复现）
prompt/response 都用简单 ASCII 文本（避免编码/分词复杂性），例如：
prompt: "Add two numbers: 3 + 5"
response: "8"
或 "Reverse string: abcde" -> "edcba"
生成 3 份文件：
data/train.jsonl：10 条
data/lora.jsonl：10 条（可与 train 相同，也可轻微扰动）
data/eval.jsonl：至少 5 条（建议 10 条）
你需要在仓库内直接生成并提交这些 jsonl 文件，不要要求用户运行额外脚本生成。

## 3. Tokenizer（必须自带，不依赖外部模型）
为了确保“从零训练”与最小依赖，使用简单字符级 tokenizer（推荐）：
从 train+lora+eval 全部文本中抽取字符集合，建立 vocab
特殊 token：<pad> <bos> <eos> <unk>
编码：将字符串映射到 id 序列
解码：将 id 序列映射回字符串
支持：
encode(text) -> List[int]
decode(ids) -> str
save(path) / load(path)（保存 vocab JSON）
训练样本拼接格式建议（统一指令模板）：
shell
复制代码
### Instruction:
{prompt}
### Response:
{response}
训练时以 causal LM 方式训练：输入为整段 token，labels 为右移一位（pad 部分 mask 为 -100）。

4. LLaMA 模型：在本仓库实现 src/llama/model.py
4.1 实现来源与要求
你的实现需要参照 /home/NagaiYoru/LLM_Tutorial/llama3/llama/model.py
但本仓库必须自包含：不可依赖该绝对路径 import（用户机器可能无该路径）
目标是“最小可跑通的 LLaMA-like 模型”：
RMSNorm
RoPE（旋转位置编码）
经典 LLaMA attention / MLP（SwiGLU 或类似）
KV 计算 + causal mask
支持 forward(input_ids) 输出 logits

4.2 配置（小模型，保证 CPU 也能跑）
在 configs/train.yaml 给一个非常小的默认配置，例如：
vocab_size：由 tokenizer 决定（运行时注入）
dim：128
n_layers：2
n_heads：4
n_kv_heads：4（可以与 n_heads 相同，先不做 GQA 也行）
hidden_dim：256
max_seq_len：256
dropout：0.0（可选）
关键：结构清晰、注释充分、能 forward/backward。

5. 训练 1：从零训练 base 模型（train_base.py）
5.1 训练目标
从随机初始化参数开始训练（torch.nn.init 默认即可）
在 data/train.jsonl 的 10 条样本上训练少量 steps（例如 50~200 steps）
输出保存到 outputs/base_train/：
model.pt（state_dict）
tokenizer.json（vocab）
config.json（模型超参）
train_log.txt（loss 打印）

5.2 训练最小规范
optimizer：AdamW
lr：1e-3 或 5e-4
batch_size：1~2
gradient_accumulation：可选（默认为 1）
device：自动选择 cuda 若可用，否则 cpu
打印：
step / loss
保存一次最终 checkpoint（必要）

6. 训练 2：LoRA 预训练（train_lora.py）
6.1 LoRA 定义（Hello World 版）
在 src/lora/lora.py 实现一个最小 LoRA：
针对 nn.Linear：实现 LoRALinear 或用 hook 替换
参数：
rank r
alpha
dropout
forward：W x + scale * (B(A(x)))

6.2 需要 LoRA 的模块
为了简单且接近 LLaMA：
attention: wq, wk, wv, wo
mlp: w1, w2, w3（或你实现里的线性层名字）
如果你的 LLaMA 实现里线性层命名不同，请在代码中集中维护一个列表 target_module_names，并写清楚注释。

6.3 LoRA 训练流程要求
加载 base 模型 checkpoint：outputs/base_train/model.pt
冻结 base 所有参数：requires_grad=False
仅 LoRA 参数可训练
在 data/lora.jsonl 上训练少量 steps（例如 30~100 steps）
保存到 outputs/lora_pretrain/：
lora.pt（仅 LoRA 参数 state_dict）
lora_config.json

7. 性能测试（eval.py）
必须支持两种评测模式：
base：加载 outputs/base_train/model.pt
base + lora：加载 base，再加载 outputs/lora_pretrain/lora.pt 合并生效
评测内容（最小化）：
在 data/eval.jsonl 上计算：
avg loss
perplexity = exp(loss)
生成测试：
从 eval 里取 2 条 prompt
使用 greedy 或 top-k 生成最多 64 tokens
打印/保存生成结果
输出写入：
outputs/eval/metrics.json
outputs/eval/generations.txt

8. 配置文件（YAML）字段规范（最少字段，清晰）
8.1 configs/train.yaml
必须包含：
seed
data_path
output_dir
max_seq_len
model: {dim, n_layers, n_heads, hidden_dim, dropout}
train: {batch_size, lr, max_steps, weight_decay, log_every}

8.2 configs/lora_pretrain.yaml
必须包含：
seed
data_path
base_ckpt_dir
output_dir
max_seq_len
lora: {r, alpha, dropout, target_modules}
train: {batch_size, lr, max_steps, weight_decay, log_every}

8.3 configs/eval.yaml
必须包含：
seed
data_path
base_ckpt_dir
lora_ckpt_dir: 可为空（base-only）
max_seq_len
eval: {batch_size, max_eval_samples, gen_max_new_tokens}

9. requirements.txt（尽量少依赖）
第一版建议：
torch
pyyaml
tqdm
不要引入 transformers / datasets / accelerate 等（你是从零实现模型，不需要）。

10. 运行命令（README 必须同样写出）
在仓库根目录执行：

10.1 安装依赖（环境已建好，仅补依赖）
bash
复制代码
pip install -r requirements.txt
10.2 从零训练 base
bash
复制代码
python -m src.train_base --config configs/train.yaml
10.3 LoRA 预训练（冻结 base，仅训练 LoRA）
bash
复制代码
python -m src.train_lora --config configs/lora_pretrain.yaml
10.4 性能测试（base 或 base+lora）
bash
复制代码
python -m src.eval --config configs/eval.yaml
11. 代码可读性与注释要求（必须做到）
每个脚本头部说明：用途 / 输入 / 输出

关键实现（RoPE、RMSNorm、mask、LoRA 注入）要有注释

日志要清晰：正在加载什么、保存到哪里、loss 数值

报错友好：缺文件时提示应该先跑哪个命令

12. 验收标准（第一版 Done）
满足以下即可视为完成：

10 条样本下：

base 训练能跑完并保存 checkpoint

LoRA 训练能跑完并保存 adapter

eval 能输出 loss/ppl，并生成文本

代码结构清晰、README 可复制粘贴运行

完全不依赖任何预训练权重或外部模型下载

13. Claude Code 执行顺序（你必须按此顺序落地）
创建目录结构与文件骨架

写 requirements.txt

写 data/.jsonl：随机 10 条样本（固定 seed，保证可复现）

写 tokenizer（字符级）+ dataset loader

写 src/llama/model.py（参照目标路径实现 LLaMA-like）

写 base 训练脚本 train_base.py

写 LoRA 实现与注入逻辑 + train_lora.py

写 eval.py

写 configs/.yaml

写 README.md（包含命令与输出说明）

自检：按 README 命令顺序确保端到端跑通（哪怕 CPU 慢，但 steps 很少）

完成后提交所有代码与数据文件，使用户 clone 后可直接运行三条命令跑通。
```
