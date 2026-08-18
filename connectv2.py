import ray
import torch
import re
import time
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple, Set

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheConfig

logger = init_logger(__name__)
KV_NAMESPACE = "kv_namespace"

@dataclass
class ReqMeta:
    request_id: str
    slot_mapping: torch.Tensor
    is_store: bool

@dataclass
class QuantizedConnectorMetadata(KVConnectorMetadata):
    requests: List[ReqMeta] = field(default_factory=list)
    def add_request(self, request_id, slot_mapping, is_store):
        self.requests.append(ReqMeta(request_id, slot_mapping, is_store))

@ray.remote
class ProducerActor:
    def __init__(self):
        self._kv_refs = {}
    def add_refs(self, req_id, refs):
        self._kv_refs[req_id] = refs
    def get_kv_refs(self, req_id):
        return self._kv_refs.get(req_id, {})
    def clear_refs(self, req_id):
        self._kv_refs.pop(req_id, None)

class QuantizedRaySharedMemoryConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._is_producer = extra.get("is_producer", False)
        self._is_consumer = extra.get("is_consumer", False)
        self._shm_prefix = extra.get("shm_prefix", "async_kv")

        self._kv_refs = {}
        self._saved_layers = set()
        self._pending_load = {}
        self._loading_threads = {}
        self._finished_recving = set()
        self._current_load_plan = []   # (req_id, clean_id, slot_mapping)
        self._full_matched = {}
        self._lock = threading.Lock()
        self._call_counts = {}

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        actor_name = f"{self._shm_prefix}_producer"
        if self._is_producer:
            try:
                self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
            except:
                self._producer_actor = ProducerActor.options(
                    name=actor_name, namespace=KV_NAMESPACE, lifetime="detached"
                ).remote()
        else:
            try:
                self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
            except:
                self._producer_actor = None

    def _log(self, method, msg):
        self._call_counts[method] = self._call_counts.get(method, 0) + 1
        count = self._call_counts[method]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        if count <= 20 or count % 100 == 0:
            print(f"[{timestamp}] [{method}] call #{count}: {msg}", flush=True)
        if count == 100:
            print(f"[{timestamp}] [{method}] subsequent calls will be summarized every 100.", flush=True)

    def _get_clean_req_id(self, req_id):
        return req_id.split('-')[0] if '-' in req_id else req_id

    def _split_kv(self, kv):
        if kv.dim() == 5 and kv.shape[1] == 2:
            return kv[:, 0, ...], kv[:, 1, ...]
        elif kv.dim() == 4 and kv.shape[1] == 2:
            return kv[:, 0, ...], kv[:, 1, ...]
        elif kv.dim() == 4 and kv.shape[-1] % 2 == 0:
            h = kv.shape[-1] // 2
            return kv[..., :h], kv[..., h:]
        else:
            raise ValueError(f"Unsupported shape: {kv.shape}")

    def _quantize(self, t):
        t = t.float()
        mn, mx = t.min(), t.max()
        scale = max(abs(mn), abs(mx)) / 127.0
        if scale == 0:
            scale = torch.tensor(1.0, device=t.device)
        q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
        return q, scale.cpu(), torch.tensor(0, device='cpu', dtype=torch.int8)

    def _dequantize(self, q, s, dtype):
        return (q.float() * s.to(q.device)).to(dtype)

    def _try_get_producer_actor(self):
        if self._producer_actor is not None:
            return self._producer_actor
        actor_name = f"{self._shm_prefix}_producer"
        try:
            self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
            return self._producer_actor
        except ValueError:
            return None

    # ---------- Scheduler ----------
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._log("get_num_new_matched_tokens", 
                  f"req={request.request_id}, clean={self._get_clean_req_id(request.request_id)}, "
                  f"num_computed={num_computed_tokens}, is_producer={self._is_producer}")
        if self._is_producer:
            return (0, False)

        clean = self._get_clean_req_id(request.request_id)
        actor = self._try_get_producer_actor()
        if actor is None:
            self._log("get_num_new_matched_tokens", "actor is None, returning (0,False)")
            return (0, False)

        refs = ray.get(actor.get_kv_refs.remote(clean))
        if refs:
            full_matched = len(request.prompt_token_ids) - num_computed_tokens
            self._log("get_num_new_matched_tokens", f"full_matched={full_matched}, returning ({max(0, full_matched-1)}, True)")
            self._full_matched[request.request_id] = full_matched
            if full_matched <= 1:
                return (0, False)
            matched = full_matched - 1
            return (matched, True)
        self._log("get_num_new_matched_tokens", "cache miss, returning (0,False)")
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._log("update_state_after_alloc", 
                  f"req={request.request_id}, num_external_tokens={num_external_tokens}, blocks={blocks}")
        if self._is_producer or num_external_tokens <= 0:
            return

        req_id = request.request_id
        block_ids = []
        if blocks and hasattr(blocks, 'blocks'):
            if isinstance(blocks.blocks, tuple):
                for bl in blocks.blocks:
                    for b in bl:
                        block_ids.append(b.block_id)
            elif isinstance(blocks.blocks, list):
                for b in blocks.blocks:
                    if hasattr(b, 'block_id'):
                        block_ids.append(b.block_id)
        if block_ids:
            self._pending_load[req_id] = (request, num_external_tokens, block_ids)
            self._log("update_state_after_alloc", f"stored pending_load for {req_id} with {len(block_ids)} blocks, num_external_tokens={num_external_tokens}")

    def build_connector_meta(self, scheduler_output):
        self._log("build_connector_meta", f"pending_load count: {len(self._pending_load)}")
        meta = QuantizedConnectorMetadata()
        if self._is_producer:
            return meta

        for req_id, (request, num_ext, block_ids) in list(self._pending_load.items()):
            if not block_ids:
                continue
            slot_mapping = []
            tokens_needed = num_ext
            for b in block_ids:
                for off in range(self._block_size):
                    if tokens_needed <= 0:
                        break
                    slot_mapping.append(b * self._block_size + off)
                    tokens_needed -= 1
                if tokens_needed <= 0:
                    break
            if slot_mapping:
                meta.add_request(req_id, torch.tensor(slot_mapping, dtype=torch.long), is_store=False)
                self._log("build_connector_meta", f"added load request for {req_id}, slots={len(slot_mapping)}")
            del self._pending_load[req_id]
            self._full_matched.pop(req_id, None)
        return meta

    # ---------- Worker ----------
    def start_load_kv(self, forward_context, **kwargs):
        self._log("start_load_kv", f"is_consumer={self._is_consumer}")
        if not self._is_consumer:
            return

        actor = self._try_get_producer_actor()
        if actor is None:
            self._log("start_load_kv", "producer actor not available yet, will retry later")
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, QuantizedConnectorMetadata) or not metadata.requests:
            self._log("start_load_kv", "no metadata or empty requests, returning")
            return

        new_plan = []
        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue
            req_id = req_meta.request_id
            clean = self._get_clean_req_id(req_id)
            slot_mapping = req_meta.slot_mapping
            if slot_mapping is None or slot_mapping.numel() == 0:
                continue
            new_plan.append((req_id, clean, slot_mapping))

        if new_plan:
            self._current_load_plan.extend(new_plan)
            self._log("start_load_kv", f"plan extended: added {len(new_plan)} requests, total {len(self._current_load_plan)}")

        for req_id, clean, slot_mapping in new_plan:
            with self._lock:
                if clean in self._loading_threads:
                    self._log("start_load_kv", f"{clean} already loading, skip")
                    continue
                # 不再需要 load_events，直接启动线程

            self._log("start_load_kv", f"starting async load for {clean} (req_id={req_id})")
            thread = threading.Thread(
                target=self._async_load,
                args=(clean, req_id, slot_mapping, forward_context.no_compile_layers, actor),
                daemon=True
            )
            with self._lock:
                self._loading_threads[clean] = thread
            thread.start()

    def _async_load(self, clean_req_id, original_req_id, slot_mapping, no_compile_layers, actor):
        self._log("_async_load", f"START for {clean_req_id} (orig={original_req_id})")
        try:
            refs = ray.get(actor.get_kv_refs.remote(clean_req_id))
            self._log("_async_load", f"got {len(refs)} refs for {clean_req_id}")
            if not refs:
                raise RuntimeError(f"No refs for {clean_req_id}")

            valid_slots = slot_mapping[slot_mapping >= 0]
            num_tokens = valid_slots.numel()
            block_idxs = valid_slots // self._block_size
            offsets = valid_slots % self._block_size
            self._log("_async_load", f"{clean_req_id} num_tokens={num_tokens}")

            if no_compile_layers is None:
                raise RuntimeError("no_compile_layers is None")

            # 遍历所有层，反量化并直接写入 KV cache
            for layer_name, layer in no_compile_layers.items():
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue
                match = re.search(r'layers\.(\d+)', layer_name)
                if not match:
                    continue
                layer_idx = int(match.group(1))
                k_ref = refs.get(f"k_layer{layer_idx}")
                v_ref = refs.get(f"v_layer{layer_idx}")
                k_scale_ref = refs.get(f"k_scale_layer{layer_idx}")
                v_scale_ref = refs.get(f"v_scale_layer{layer_idx}")
                if any(x is None for x in [k_ref, v_ref, k_scale_ref, v_scale_ref]):
                    continue

                k_quant = ray.get(k_ref)
                v_quant = ray.get(v_ref)
                k_scale = ray.get(k_scale_ref)
                v_scale = ray.get(v_scale_ref)
                k_dequant = self._dequantize(k_quant, k_scale, kv_cache.dtype)
                v_dequant = self._dequantize(v_quant, v_scale, kv_cache.dtype)

                # 截取到实际需要的 token 数
                if k_dequant.shape[0] != num_tokens:
                    self._log("_async_load", f"WARNING: layer {layer_idx} has {k_dequant.shape[0]} tokens, num_tokens={num_tokens}, trimming")
                    k_dequant = k_dequant[:num_tokens]
                    v_dequant = v_dequant[:num_tokens]

                # 写入 KV cache（与原来 wait_for_layer_load 中的逻辑相同）
                try:
                    if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
                        kv_cache[block_idxs, 0, :, offsets] = k_dequant
                        kv_cache[block_idxs, 1, :, offsets] = v_dequant
                    elif kv_cache.dim() == 4 and kv_cache.shape[1] == 2:
                        kv_cache[block_idxs, 0, offsets] = k_dequant
                        kv_cache[block_idxs, 1, offsets] = v_dequant
                    elif kv_cache.dim() == 4 and kv_cache.shape[-1] % 2 == 0:
                        h = kv_cache.shape[-1] // 2
                        kv_cache[block_idxs, :, offsets, :h] = k_dequant
                        kv_cache[block_idxs, :, offsets, h:] = v_dequant
                    else:
                        kv_cache[block_idxs, :, offsets] = k_dequant
                        v_cache = getattr(layer, "v_cache", None)
                        if v_cache is not None:
                            v_cache[block_idxs, :, offsets] = v_dequant
                    self._log("_async_load", f"wrote layer {layer_idx} for {clean_req_id}")
                except Exception as e:
                    self._log("_async_load", f"write failed for layer {layer_idx}: {e}")
                    raise

            # 所有层写入完成后，标记请求为已就绪
            with self._lock:
                self._finished_recving.add(original_req_id)
                self._loading_threads.pop(clean_req_id, None)
                self._log("_async_load", f"COMPLETE for {clean_req_id}, added to _finished_recving: {original_req_id}")

        except Exception as e:
            self._log("_async_load", f"ERROR for {clean_req_id}: {e}")
            with self._lock:
                self._loading_threads.pop(clean_req_id, None)
                # 即使失败也加入 finished，防止死等；但调度器会尝试计算，可能导致错误
                self._finished_recving.add(original_req_id)

    # ---------- 关键修改：wait_for_layer_load 变为空 ----------
    def wait_for_layer_load(self, layer_name):
        # 不再做任何等待或写入，因为数据已经在 _async_load 中写好了
        # 调度器只会将已经 finished 的请求送入 forward，所以这里直接返回
        if not self._is_consumer:
            return
        # 可选：加一个安全性检查，如果还有未完成的加载线程，可以等待（但通常不会发生）
        # 为了演示，这里直接返回
        pass

    def get_finished(self, finished_req_ids):
        self._log("get_finished", f"finished_req_ids={finished_req_ids}")
        with self._lock:
            recving = self._finished_recving.copy()
            self._finished_recving.clear()
            self._log("get_finished", f"returning recving={recving}")
            for req_id in finished_req_ids:
                clean = self._get_clean_req_id(req_id)
                # 清理相关资源（无需再清理 _loaded_cache 等）
                self._loading_threads.pop(clean, None)
                self._current_load_plan = [p for p in self._current_load_plan if p[1] != clean]
        return set(), recving

    # ---------- 保存 ----------
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self._is_producer:
            return

        request_ids = getattr(attn_metadata, "request_ids", None)
        if not request_ids:
            seq_lens = getattr(attn_metadata, "seq_lens", None)
            if seq_lens is not None:
                request_ids = [f"req_{i:03d}" for i in range(len(seq_lens))]
            else:
                request_ids = ["req_001"]

        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            seq_lens = [len(getattr(attn_metadata, "slot_mapping", []))]
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            return

        cum_len = [0]
        for l in seq_lens:
            cum_len.append(cum_len[-1] + l)
        match = re.search(r'layers\.(\d+)', layer_name)
        layer_idx = int(match.group(1)) if match else 0

        k, v = self._split_kv(kv_layer)
        flat_k = k.permute(0, 2, 1, 3).reshape(-1, k.shape[1], k.shape[3])
        flat_v = v.permute(0, 2, 1, 3).reshape(-1, v.shape[1], v.shape[3])

        for idx, req_id_full in enumerate(request_ids):
            clean_id = self._get_clean_req_id(req_id_full)
            start, end = cum_len[idx], cum_len[idx+1]
            req_slots = slot_mapping[start:end]
            valid_slots = req_slots[req_slots >= 0]
            if valid_slots.numel() == 0:
                continue
            if (clean_id, layer_idx) in self._saved_layers:
                continue

            k_stack = flat_k[valid_slots]
            v_stack = flat_v[valid_slots]
            k_quant, k_scale, _ = self._quantize(k_stack)
            v_quant, v_scale, _ = self._quantize(v_stack)
            k_ref = ray.put(k_quant)
            v_ref = ray.put(v_quant)
            k_scale_ref = ray.put(k_scale)
            v_scale_ref = ray.put(v_scale)

            if clean_id not in self._kv_refs:
                self._kv_refs[clean_id] = {}
            self._kv_refs[clean_id][f"k_layer{layer_idx}"] = k_ref
            self._kv_refs[clean_id][f"v_layer{layer_idx}"] = v_ref
            self._kv_refs[clean_id][f"k_scale_layer{layer_idx}"] = k_scale_ref
            self._kv_refs[clean_id][f"v_scale_layer{layer_idx}"] = v_scale_ref
            self._saved_layers.add((clean_id, layer_idx))
            self._log("save_kv_layer", f"saved layer {layer_idx} for {clean_id}, tokens={valid_slots.numel()}")

    def wait_for_save(self):
        if not self._is_producer or self._producer_actor is None:
            return
        for clean_id, refs in self._kv_refs.items():
            if refs:
                ray.get(self._producer_actor.add_refs.remote(clean_id, refs))
        self._kv_refs.clear()

    def get_block_ids_with_load_errors(self):
        return set()

    def start_save_kv(self, forward_context, **kwargs):
        pass
    def load_kv_layer(self, layer_name, kv_cache, attn_metadata):
        pass
    def register_request(self, request_id, meta):
        pass
    def unregister_request(self, request_id):
        clean = self._get_clean_req_id(request_id)
        with self._lock:
            self._loading_threads.pop(clean, None)
            self._pending_load.pop(request_id, None)
            self._finished_recving.discard(request_id)
            self._full_matched.pop(request_id, None)
            self._current_load_plan = [p for p in self._current_load_plan if p[1] != clean]
        if self._is_producer and self._producer_actor is not None:
            ray.get(self._producer_actor.clear_refs.remote(clean))
    def get_connector_metadata(self, request_ids):
        return QuantizedConnectorMetadata()
    def get_kv_connector_metrics(self):
        return None
    def get_cross_layer_attn_metadata(self, *args, **kwargs):
        return None
    def close(self):
        pass