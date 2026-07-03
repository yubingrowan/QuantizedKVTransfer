# QuantizedKVTransfer

基于 Ray 共享内存的量化 KV cache 传输模块，用于 vLLM 的 Prefill-Decode 分离架构。

## 特性

- **多 Request 支持**：支持同时处理多个并发请求
- **量化传输**：使用 INT8 量化减少 KV cache 传输带宽
- **Triton 加速**：可选使用 Triton GPU 量化算子加速
- **Ray 共享内存**：利用 Ray 的分布式共享内存实现高效传输
- **模块化设计**：清晰的代码结构，易于维护和扩展

## 文件结构

```
QuantizedKVTransfer/
├── __init__.py              # 模块导出
├── actor.py                 # Ray Actor 管理 KV 引用
├── quantization.py          # 量化器实现（CPU/Triton）
├── connector.py             # 主连接器逻辑
├── quantized_ray_connector.py  # 单文件实现（兼容版本）
├── test_single_request.py   # 单请求测试脚本
├── test_multi_request.py    # 多请求测试脚本
└── README.md               # 本文档
```

## 架构说明

### 组件

1. **ProducerActor** (`actor.py`)
   - Ray Actor，用于存储和管理 KV cache 的引用
   - 支持多 request 的引用管理
   - 提供引用的添加、获取、清理接口

2. **Quantizer** (`quantization.py`)
   - `CPUQuantizer`：CPU 线性量化（回退方案）
   - `TritonQuantizer`：Triton GPU 量化（高性能）
   - 自动选择可用的量化器

3. **QuantizedRaySharedMemoryConnector** (`connector.py`)
   - 实现 vLLM KVConnectorBase_V1 接口
   - Producer 端：执行 KV cache 量化并存储到 Ray
   - Consumer 端：从 Ray 加载量化数据并反量化
   - 支持多 request 并发处理

## 使用方法

### 1. 单请求测试

```bash
cd /home/developer/vllm/QuantizedKVTransfer
python test_single_request.py
```

### 2. 多请求测试

```bash
cd /home/developer/vllm/QuantizedKVTransfer
python test_multi_request.py
```

### 3. 配置参数

在 `KVTransferConfig` 中配置连接器：

```python
kv_config = KVTransferConfig(
    kv_connector="QuantizedRaySharedMemoryConnector",
    kv_role="kv_producer",  # 或 "kv_consumer"
    kv_rank=rank,
    kv_connector_module_path="QuantizedKVTransfer.connector",
    kv_connector_extra_config={
        "shm_prefix": "my_quantized_kv",  # 共享内存前缀
        "is_producer": True,               # Producer 端设为 True
        "is_consumer": False,              # Producer 端设为 False
        "use_triton": True,                # 是否使用 Triton 量化
    }
)
```

### 4. 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `shm_prefix` | str | Ray 共享内存前缀，用于区分不同的 KV cache |
| `is_producer` | bool | 是否为 Producer 端（Prefill 节点） |
| `is_consumer` | bool | 是否为 Consumer 端（Decode 节点） |
| `use_triton` | bool | 是否使用 Triton GPU 量化（默认 True） |

## 工作流程

### Prefill 阶段（Producer）

1. 接收 prefill 请求
2. 对每个层的 KV cache 进行量化
3. 将量化后的数据存储到 Ray 共享内存
4. 通过 ProducerActor 管理引用
5. 完成后清理本地引用

### Decode 阶段（Consumer）

1. 接收 decode 请求
2. 从 ProducerActor 获取 KV 引用
3. 从 Ray 共享内存加载量化数据
4. 反量化并写入本地 KV cache
5. 继续正常的 decode 推理

## 多 Request 支持

连接器自动支持多 request 并发处理：

- 每个 request 有独立的 KV cache 存储
- 使用 request_id 区分不同的请求
- 支持同时处理多个 prefill 和 decode 请求
- 自动清理已完成的请求资源

## 性能优化

### Triton 量化

当 `use_triton=True` 时，优先使用 Triton GPU 量化算子：

- 更快的量化速度
- 更低的延迟
- 如果 Triton 不可用，自动回退到 CPU 量化

### 并发处理

- Ray Actor 支持并发访问
- 多 request 可以并行处理
- 减少序列化等待时间

## 日志调试

所有日志输出到 stderr 并立即刷新，便于调试：

```bash
python ray.py 2>&1 | tee nixl.log
```

关键日志标识：
- `[Yubing Init]` - 连接器初始化
- `[Yubing Save]` - KV cache 保存
- `[Yubing Load]` - KV cache 加载
- `[Yubing Quant]` - 量化操作
- `[Yubing Actor]` - Ray Actor 操作

## 故障排查

### 问题：Producer Actor 未找到

**原因**：Consumer 端启动时 Producer 端还未初始化

**解决**：Consumer 端会自动重试 5 次，间隔 0.5 秒

### 问题：量化失败

**原因**：Triton 不可用或 GPU 内存不足

**解决**：设置 `use_triton=False` 使用 CPU 量化

### 问题：KV cache 加载失败

**原因**：request_id 不匹配或引用未正确推送

**解决**：检查 Prefill 和 Decode 端使用相同的 request_id

## 扩展开发

### 添加新的量化器

在 `quantization.py` 中继承 `Quantizer` 类：

```python
class CustomQuantizer(Quantizer):
    def quantize(self, tensor: torch.Tensor):
        # 实现自定义量化逻辑
        pass
    
    def dequantize(self, quantized, scale, zero_point, dtype):
        # 实现自定义反量化逻辑
        pass
```

### 修改 Ray Actor

在 `actor.py` 中扩展 `ProducerActor` 的功能。

## 依赖项

- vLLM
- Ray
- PyTorch
- Triton (可选，用于 GPU 加速)

## 许可证

遵循 vLLM 项目的许可证。
