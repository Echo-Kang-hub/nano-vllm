<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History


---

## 项目深度解析（中文）

### 一、这个项目是干什么的

Nano-vLLM 是一个**从零手写的轻量级大语言模型离线推理引擎**，目标是用约 1200 行纯 Python/PyTorch 代码，复现 [vLLM](https://github.com/vllm-project/vllm) 的核心功能，并达到相近甚至更高的推理吞吐量。

**它解决了什么问题？**

主流推理框架（如 vLLM）代码量庞大、架构复杂，对初学者极不友好。Nano-vLLM 将同一套生产级优化技术用最精简的方式重新实现，让开发者可以通过阅读代码彻底理解推理引擎的工作原理。

**核心能力一览：**

| 能力 | 说明 |
|------|------|
| **连续批处理（Continuous Batching）** | 动态将不同长度的请求拼成一个批次，GPU 利用率极高 |
| **PagedAttention / KV Cache 块管理** | 用分页内存管理 KV Cache，杜绝显存碎片 |
| **Prefix Caching（前缀缓存）** | 利用 xxhash 对 Token 块做哈希，相同前缀只计算一次 |
| **Tensor Parallelism（张量并行）** | 基于 NCCL + `torch.distributed`，多 GPU 分片执行 |
| **CUDA Graph 加速** | Decode 阶段用 CUDAGraph 消除 CPU 调度开销 |
| **Torch Compile** | RMSNorm、RoPE、Sampler 等算子用 `@torch.compile` 即时编译 |
| **FlashAttention 2** | Prefill 用 `flash_attn_varlen_func`，Decode 用 `flash_attn_with_kvcache` |
| **Triton 自定义算子** | KV Cache 写入用手写 Triton kernel，避免不必要的显存拷贝 |

目前支持的模型：**Qwen3 系列**（架构可扩展）。

---

### 二、代码结构

```
nano-vllm/
├── example.py              # 用法示例（快速上手入口）
├── bench.py                # 吞吐量基准测试脚本
├── pyproject.toml          # 项目依赖与打包配置
│
└── nanovllm/
    ├── __init__.py         # 导出 LLM、SamplingParams（对外 API）
    ├── llm.py              # LLM 类（继承 LLMEngine，薄封装）
    ├── config.py           # Config 数据类，存放所有运行时配置
    ├── sampling_params.py  # SamplingParams：temperature / max_tokens / ignore_eos
    │
    ├── engine/             # 推理引擎核心
    │   ├── sequence.py     # Sequence：单条请求的状态机（WAITING/RUNNING/FINISHED）
    │   ├── block_manager.py# BlockManager：KV Cache 分页管理 + Prefix Cache
    │   ├── scheduler.py    # Scheduler：调度 Prefill / Decode，处理抢占
    │   ├── llm_engine.py   # LLMEngine：推理主循环，管理多进程、tokenizer
    │   └── model_runner.py # ModelRunner：GPU 上的模型执行，含 CUDAGraph 捕获
    │
    ├── layers/             # 可复用的神经网络层
    │   ├── attention.py    # Attention：FlashAttention + Triton KVCache 写入
    │   ├── linear.py       # 张量并行线性层（Column/Row/QKV/Merged）
    │   ├── embed_head.py   # 词嵌入与 LM Head 的张量并行实现
    │   ├── rotary_embedding.py # RoPE 旋转位置编码（支持 torch.compile）
    │   ├── layernorm.py    # RMSNorm，含 fused add+norm（残差合并）
    │   ├── activation.py   # SiluAndMul（门控 FFN 激活函数）
    │   └── sampler.py      # Sampler：温度采样（Gumbel-max trick）
    │
    ├── models/
    │   └── qwen3.py        # Qwen3 完整模型定义（Attention/MLP/Layer/Model/ForCausalLM）
    │
    └── utils/
        ├── context.py      # 全局推理上下文（is_prefill、slot_mapping 等）
        └── loader.py       # safetensors 模型权重加载器（支持 packed modules）
```

**分层关系：**

```
example.py / bench.py
       ↓
   LLM (llm.py)
       ↓
 LLMEngine (engine/llm_engine.py)      ← tokenizer、请求队列、主循环
       ↓
  Scheduler (engine/scheduler.py)      ← 调度策略（prefill 优先、抢占）
  ModelRunner (engine/model_runner.py) ← GPU 执行、CUDAGraph、多进程通信
       ↓
 Qwen3ForCausalLM (models/qwen3.py)   ← 模型前向传播
       ↓
  layers/ (attention, linear, ...)    ← 基础算子层
  utils/context.py                    ← 注入推理上下文（无需传参）
```

---

### 三、阅读代码的收获

#### 3.1 连续批处理（Continuous Batching）的本质

传统静态批处理要等一批请求都完成才发起下一批，GPU 大量空转。Nano-vLLM 的 `Scheduler` 将所有请求维护在 `waiting` 和 `running` 两个队列中，每个 `step()` 都动态凑批：**Prefill 优先**，能装多少装多少；Prefill 队列空了再进 Decode 循环。这样请求一到达就能被调度，大幅提升 GPU 利用率。

#### 3.2 PagedAttention：KV Cache 的分页内存管理

`BlockManager` 把 KV Cache 切成固定大小的"块"（默认 256 tokens/块），`Sequence` 维护一个 `block_table`（物理块 ID 列表）。分配、释放均以块为单位，彻底消除了因序列长度不同带来的显存碎片。抢占（preemption）时只需将块归还空闲池，无需移动数据。

#### 3.3 Prefix Caching（前缀缓存）的精妙设计

`BlockManager.allocate` 在为序列分配块时，对每个**已满**的块用 `xxhash` 计算哈希（链式哈希，依赖前一块哈希值作为种子），并存入 `hash_to_block_id` 字典。下次相同前缀的请求进来时，直接命中缓存块，`num_cached_tokens` 字段告诉后续流程哪些 Token 不必重新计算，`prepare_prefill` 据此跳过已缓存的 Token，Attention 层用 `block_tables` 直接从缓存块读取 KV。

#### 3.4 全局 Context 模式（无侵入式参数传递）

`utils/context.py` 用一个模块级全局变量 `_CONTEXT` 存放当次推理的元信息（`is_prefill`、`cu_seqlens_q`、`slot_mapping`、`block_tables` 等）。模型层（Attention、LM Head）通过 `get_context()` 直接读取，无需在每一层的 `forward` 签名里传递这些参数。这是一种**隐式上下文注入**的设计模式，代价是全局状态，收益是模型代码极度简洁。

#### 3.5 张量并行（Tensor Parallelism）的实现

- **列并行（ColumnParallelLinear）**：每个 GPU 持有输出维度的一个分片，前向直接 `F.linear`，无需通信。
- **行并行（RowParallelLinear）**：每个 GPU 持有输入维度的一个分片，前向后做 `dist.all_reduce` 求和。
- **QKVParallelLinear / MergedColumnParallelLinear**：对 Q/K/V 或 gate/up 这类拼接权重，`weight_loader` 在加载时按分片 ID 切割并写入正确偏移。
- **VocabParallelEmbedding**：词表按 GPU 数量切分，每个 GPU 只处理自己范围内的 token，最后 `all_reduce`。
- 多进程通过**共享内存（SharedMemory）+ Event** 同步，主进程将调用指令序列化后写入 shm，子进程轮询读取并执行，避免了 pickle 序列化大型 tensor。

#### 3.6 CUDAGraph 加速 Decode

Decode 阶段每步只处理 1 个 token/序列，计算量极小，CPU 调度开销占比显著。`capture_cudagraph` 预先对 batch size 1、2、4、8、16、32… 各捕获一张 CUDA Graph；推理时找到最小的满足条件的 bs，replay graph，消除 PyTorch dispatcher 开销，吞吐可提升 10%~30%。

#### 3.7 Triton Kernel：KV Cache 写入

`store_kvcache_kernel` 是一个 Triton kernel，逐 token 根据 `slot_mapping` 把当前步的 K/V 写到对应物理块的正确位置。相比 `scatter` 或 `index_put`，Triton kernel 更灵活且无需额外中间张量，访存模式也更高效。

#### 3.8 Fused Add+Norm（残差融合归一化）

`RMSNorm.add_rms_forward` 将"加残差"和"RMSNorm"合并成一个算子（`torch.compile` 会进一步融合成单个 kernel）。Transformer 的每一层都有两处这样的操作，融合后显著减少显存读写轮次。

#### 3.9 Gumbel-max Trick 采样

`Sampler` 用 `probs / Exponential(1)` 的 argmax 代替传统的 `multinomial` 采样（即 Gumbel-max trick），两者数学等价，但 argmax 在 GPU 上更高效，且对 `torch.compile` 更友好。

#### 3.10 权重加载的 packed_modules_mapping 技巧

HuggingFace Qwen3 的权重文件中 Q/K/V 是分开存储的（`q_proj`、`k_proj`、`v_proj`），而本项目为性能将它们合并为一个 `qkv_proj`。`packed_modules_mapping` 字典定义了映射关系，`loader.py` 在加载时检测到被 packed 的权重名称后，调用对应的 `weight_loader(param, weight, shard_id)` 将各分块写入合并参数的正确偏移位置，实现了无缝兼容。

---

### 四、代码阅读顺序（从入口到底层逐层深入）

建议按以下顺序阅读，每一步都建立在前一步的理解之上：

#### 第 1 步：理解对外接口（5 分钟）
- `nanovllm/__init__.py` — 只导出 `LLM` 和 `SamplingParams`，接口极简。
- `nanovllm/sampling_params.py` — 采样参数：`temperature`、`max_tokens`、`ignore_eos`。
- `example.py` — 实际使用方式，建立整体感知。

#### 第 2 步：理解配置层（5 分钟）
- `nanovllm/config.py` — `Config` dataclass，所有超参集中管理，`__post_init__` 做合法性校验并自动读取 HuggingFace 配置。
- `nanovllm/llm.py` — `LLM` 只是 `LLMEngine` 的别名，薄封装以对齐 vLLM API。

#### 第 3 步：理解推理主循环（20 分钟）
- `nanovllm/engine/llm_engine.py` — 重点阅读 `__init__`（初始化多进程）、`add_request`、`step`、`generate`。理解"请求→序列→调度→模型执行→postprocess"的完整闭环。

#### 第 4 步：理解序列状态机（10 分钟）
- `nanovllm/engine/sequence.py` — `Sequence` 是单条请求的核心数据结构：状态枚举、`block_table`、`append_token`、`__getstate__`（跨进程序列化优化，只传必要字段）。

#### 第 5 步：理解 KV Cache 管理（30 分钟）★ 重点
- `nanovllm/engine/block_manager.py` — 先理解 `Block` 结构（`block_id`、`ref_count`、`hash`），再读 `allocate`（含前缀缓存命中逻辑）、`deallocate`、`can_append`、`may_append`（Decode 时扩块逻辑）。这是最精妙也最需耐心的部分。

#### 第 6 步：理解调度器（15 分钟）
- `nanovllm/engine/scheduler.py` — `schedule` 方法分两段：先尝试调度 Prefill（从 `waiting` 队列），有 Prefill 就直接返回；没有则调度 Decode（从 `running` 队列），资源不足时调用 `preempt` 抢占。

#### 第 7 步：理解全局上下文（5 分钟）
- `nanovllm/utils/context.py` — 读懂 `Context` dataclass 的各字段含义，理解为什么用全局变量而不是参数传递。

#### 第 8 步：理解 GPU 执行层（40 分钟）★ 重点
- `nanovllm/engine/model_runner.py` — 按顺序读：
  1. `__init__`：分布式初始化、模型加载、warmup、KV Cache 分配、CUDAGraph 捕获。
  2. `prepare_prefill` / `prepare_decode`：理解如何把 `Sequence` 列表转换成模型输入 tensor，以及 `slot_mapping`、`cu_seqlens` 等字段的含义。
  3. `run_model`：Prefill 直接运行，Decode 走 CUDAGraph replay。
  4. `capture_cudagraph`：理解对不同 batch size 预捕获多张图的策略。
  5. 多进程通信：`write_shm` / `read_shm` / `loop`。

#### 第 9 步：理解基础算子层（30 分钟）
- `nanovllm/utils/loader.py` — safetensors 加载 + packed modules 映射。
- `nanovllm/layers/attention.py` — Triton KVCache 写入 kernel + FlashAttention Prefill/Decode 两种调用方式。
- `nanovllm/layers/linear.py` — 列并行、行并行、QKV 并行线性层及其 `weight_loader`。
- `nanovllm/layers/embed_head.py` — 词表并行 Embedding 和 LM Head，以及 Prefill 时只取最后一个 token 的 logit。
- `nanovllm/layers/layernorm.py` — Fused Add+RMSNorm。
- `nanovllm/layers/rotary_embedding.py` — RoPE 预计算缓存 + `torch.compile`。
- `nanovllm/layers/sampler.py` — Gumbel-max trick 温度采样。
- `nanovllm/layers/activation.py` — SiluAndMul 门控激活。

#### 第 10 步：理解模型定义（15 分钟）
- `nanovllm/models/qwen3.py` — 自底向上：`Qwen3Attention` → `Qwen3MLP` → `Qwen3DecoderLayer` → `Qwen3Model` → `Qwen3ForCausalLM`。理解 `packed_modules_mapping` 如何指导权重加载器将分离的 Q/K/V 权重合并。

#### 第 11 步：运行基准测试（实践验证）
- `bench.py` — 修改 `enforce_eager=True/False`、`tensor_parallel_size` 等参数，自行对比各优化选项对吞吐量的影响。
