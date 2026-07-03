# QuantizedKVTransfer

> **Quantized KV Cache Transfer Plugin for vLLM Disaggregated Serving**

基于 **Ray Object Store** 的量化 KV Cache 传输插件，用于 **vLLM Prefill-Decode Disaggregated Serving**。

---

## Overview

在 Prefill-Decode 分离部署中，Prefill 节点负责计算 Prompt 对应的 KV Cache，而 Decode 节点负责后续 Token 生成。

由于 KV Cache 体积巨大，跨节点传输成为系统瓶颈。

**QuantizedKVTransfer** 通过：

- INT8 Quantization（约 50% 数据压缩）
- Ray Object Store Shared Memory
- Vectorized Tensor Indexing
- Plugin-based KV Connector

实现无需修改 vLLM 源码即可完成高效 KV Cache 传输。

---

## Features

- 🚀 INT8 Quantization（约 50% 数据压缩）
- ⚡ Ray Object Store 共享内存
- 🔥 PyTorch 向量化批量索引
- 📦 多请求并发
- 🔌 基于 KVConnectorBase_V1 插件化实现

---

## Architecture

---
```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Ray Cluster                                      │
│  ┌─────────────────────────────┐          ┌─────────────────────────────┐  │
│  │   PrefillActor (Producer)   │          │   DecodeActor (Consumer)    │  │
│  │                             │          │                             │  │
│  │  ┌───────────────────────┐  │          │  ┌───────────────────────┐  │  │
│  │  │  LLMEngine            │  │          │  │  LLMEngine            │  │  │
│  │  │  ┌─────────────────┐  │  │          │  │  ┌─────────────────┐  │  │  │
│  │  │  │ Attention Layer │  │  │          │  │  │ Attention Layer │  │  │  │
│  │  │  │ (逐层)          │  │  │          │  │  │ (逐层)          │  │  │  │
│  │  │  └────────┬────────┘  │  │          │  │  └────────┬────────┘  │  │  │
│  │  │           │            │  │          │  │           │            │  │  │
│  │  │  ┌────────▼────────┐  │  │          │  │  ┌────────▼────────┐  │  │  │
│  │  │  │ 1. 提取 KV      │  │  │          │  │  │ 5. 反量化 KV    │  │  │  │
│  │  │  │ 2. int8 量化    │  │  │          │  │  │ 6. 填充 Cache   │  │  │  │
│  │  │  └────────┬────────┘  │  │          │  │  └────────┬────────┘  │  │  │
│  │  └───────────┼────────────┘  │          │  └───────────┼────────────┘  │
│  │               │               │          │               │               │
│  │      ┌────────▼────────┐      │          │      ┌────────▼────────┐      │
│  │      │  ray.put(k_quant)│      │          │      │  ray.get(k_ref) │      │
│  │      └────────┬────────┘      │          │      └────────┬────────┘      │
│  └───────────────┼────────────────┘          └───────────────┼────────────────┘
│                  │                                            │
│                  │  ┌────────────────────────────────────┐    │
│                  └──►│  Ray Object Store (共享内存)     │◄───┘
│                     │  ├─ k_ref → [int8 KV data]       │
│                     │  ├─ v_ref → [int8 KV data]       │
│                     │  └─ scale_ref → [scale]          │
│                     └────────────────────────────────────┘
│                                     ▲
│                                     │
│                        ┌────────────┴────────────┐
│                        │  ProducerActor          │
│                        │  ┌──────────────────┐   │
│                        │  │ req_id → refs    │   │
│                        │  │  └─ k_ref, v_ref │   │
│                        │  │  └─ scale_ref    │   │
│                        │  └──────────────────┘   │
│                        └─────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────┘
工作流程（详细调用链）
```text
[启动] ray.init(namespace="kv_namespace")
   │
   ├─ [Producer] PrefillActor.__init__()
   │    └─ Connector.__init__() → 创建 ProducerActor (detached, namespace=kv_namespace)
   │
   ├─ [Consumer] DecodeActor.__init__()
   │    └─ Connector.__init__() → 尝试获取 ProducerActor (可能失败，稍后重试)
   │
   ├─ [Producer] run_prefill()
   │    ├─ add_request()
   │    ├─ engine.step() [循环]
   │    │    ├─ build_connector_meta() → 返回 {"request_ids": [...]}
   │    │    ├─ start_save_kv() → 捕获 request_ids
   │    │    ├─ 逐层 Attention:
   │    │    │    └─ save_kv_layer()
   │    │    │         ├─ 根据 seq_lens 切分 slot_mapping
   │    │    │         ├─ flat_k[valid_slots] 批量提取（向量化）
   │    │    │         ├─ int8 量化 (GPU)
   │    │    │         └─ ray.put(k_quant) → 存共享内存（零拷贝）
   │    │    └─ wait_for_save() → ProducerActor.add_refs(req_id, refs)
   │    └─ abort_request()
   │
   └─ [Consumer] run_decode()
        ├─ add_request()
        ├─ engine.step() [循环，逐 token 生成]
        │    ├─ build_connector_meta() → 返回 {"request_ids": [...]}
        │    ├─ start_load_kv() → 获取 ProducerActor 句柄 (重试 20 次)
        │    ├─ 逐层 Attention:
        │    │    └─ save_kv_layer() [Consumer 侧]
        │    │         ├─ 判断 _is_consumer → 执行加载
        │    │         ├─ ProducerActor.get_kv_refs(req_id)
        │    │         ├─ ray.get(refs) → 取回量化数据（零拷贝）
        │    │         ├─ 反量化 → dequantized
        │    │         └─ flat_k[valid_slots] = k_dequant 批量填充（向量化）
        │    └─ 生成下一个 token (使用加载的 KV)
        └─ 返回完整生成文本
```
---

## Project Structure

```text
QuantizedKVTransfer/
├── __init__.py
├── quantized_ray_connector.py
├── test_single_request.py
├── test_multi_request.py
├── eval_quantization.py
└── README.md
```

---

## Core Components

### ProducerActor

负责维护：

- request_id → ObjectRefs 映射
- add_refs()
- get_kv_refs()
- clear_refs()

ProducerActor 不保存真实 KV 数据，仅维护 Ray ObjectRef。

### QuantizedRaySharedMemoryConnector

Producer：

1. 提取 KV
2. INT8 量化
3. ray.put()
4. 注册 ObjectRef

Consumer：

1. 查询 ObjectRef
2. ray.get()
3. 反量化
4. 写入 KV Cache

---

## Why Quantization?

KV Cache 随上下文长度线性增长。

INT8 量化可以：

| Format | Transfer Size |
|---------|---------------|
| FP16 | 100% |
| INT8 | ~50% |

在保证生成质量基本一致的情况下，大幅降低网络/共享内存传输压力。

---

## Why Ray Object Store?

| Solution | Advantages | Limitations |
|----------|------------|-------------|
| Shared Memory | 简单 | CPU 拷贝 |
| CUDA IPC | 高性能 | 生命周期复杂 |
| NCCL | GPU 通信 | 更适合集体通信 |
| **Ray Object Store** | 对象管理简单、共享内存、易集成 | 依赖 Ray |

---

## Installation

```bash
pip install torch ray vllm
```

---

## Quick Start

### Single Request

```bash
python test_single_request.py
```

### Multiple Requests

```bash
python test_multi_request.py
```

### Quantization Evaluation

```bash
python eval_quantization.py
```

---

## Configuration

Producer

```python
KVTransferConfig(
    kv_connector="QuantizedRaySharedMemoryConnector",
    kv_role="kv_producer",
)
```

Consumer

```python
KVTransferConfig(
    kv_connector="QuantizedRaySharedMemoryConnector",
    kv_role="kv_consumer",
)
```

---

## Evaluation

| Scenario | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|------|----------|----------|----------|
| FP16 vs INT8 | 1.00 | 1.00 | 1.00 | 1.00 |
| Corrupted Quantization | 0.45 | 0.65 | 0.54 | 0.66 |

结果表明正常量化几乎不会影响生成质量。

---

## Optimizations

| Optimization | Description |
|--------------|-------------|
| INT8 Quantization | 减少约 50% 传输数据 |
| Ray Object Store | 共享内存对象管理 |
| Vectorized Indexing | 减少 Python 循环 |
| GPU Tensor Pipeline | 降低额外数据搬运 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ProducerActor Not Found | 先启动 Producer |
| CUDA OOM | 降低 GPU Memory Utilization |
| Ray Init Failed | ray stop --force |

---

## Roadmap

- [x] INT8 Quantization
- [x] Ray Shared Memory
- [x] Multi-request Support
- [x] Vectorized KV Extraction
- [ ] FP8 Quantization
- [ ] Async KV Transfer
- [ ] Pipeline Overlap
- [ ] Multi-node Benchmark

---

## Tech Stack

- vLLM
- PyTorch
- Ray
- CUDA
- INT8 Quantization

---

## License

This project follows the same license as the vLLM project.
