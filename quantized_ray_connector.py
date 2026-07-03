import ray
import torch
import numpy as np
import re
import sys
import time
from typing import Optional, Any, Dict, List, Tuple

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheConfig

logger = init_logger(__name__)

KV_NAMESPACE = "kv_namespace"

@ray.remote
class ProducerActor:
    def __init__(self):
        self._kv_refs: Dict[str, Dict[str, ray.ObjectRef]] = {}
        print("[QuantizedKVTransfer Actor] ProducerActor initialized", file=sys.stderr, flush=True)

    def add_refs(self, request_id: str, refs: Dict[str, ray.ObjectRef]):
        print(f"[QuantizedKVTransfer Actor] add_refs for {request_id}: {list(refs.keys())}", file=sys.stderr, flush=True)
        if request_id not in self._kv_refs:
            self._kv_refs[request_id] = {}
        self._kv_refs[request_id].update(refs)

    def get_kv_refs(self, request_id: str) -> Dict[str, ray.ObjectRef]:
        print(f"[QuantizedKVTransfer Actor] get_kv_refs for {request_id}, found {len(self._kv_refs.get(request_id, {}))} refs", file=sys.stderr, flush=True)
        return self._kv_refs.get(request_id, {})

    def clear_refs(self, request_id: str):
        if request_id in self._kv_refs:
            del self._kv_refs[request_id]
            print(f"[QuantizedKVTransfer Actor] cleared refs for {request_id}", file=sys.stderr, flush=True)


class QuantizedRaySharedMemoryConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: str,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._dtype = vllm_config.cache_config.cache_dtype
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._shm_prefix = extra.get("shm_prefix", "quantized_kv")
        self._role = role
        self._is_producer = extra.get("is_producer", False)
        self._is_consumer = extra.get("is_consumer", False)

        self._kv_refs: Dict[str, Dict[str, ray.ObjectRef]] = {}
        self._saved_layers = set()    # (req_id, layer_idx)
        self._loaded_layers = set()   # (req_id, layer_idx)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        actor_name = f"{self._shm_prefix}_producer"

        if self._is_producer:
            try:
                self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
                print(f"[QuantizedKVTransfer Init] Producer got existing actor {actor_name} in namespace {KV_NAMESPACE}", file=sys.stderr, flush=True)
            except ValueError:
                self._producer_actor = ProducerActor.options(
                    name=actor_name,
                    namespace=KV_NAMESPACE,
                    lifetime="detached"
                ).remote()
                print(f"[QuantizedKVTransfer Init] Producer created new actor {actor_name} in namespace {KV_NAMESPACE}", file=sys.stderr, flush=True)
        else:
            try:
                self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
                print(f"[QuantizedKVTransfer Init] Consumer found actor {actor_name} in namespace {KV_NAMESPACE}", file=sys.stderr, flush=True)
            except ValueError:
                print(f"[QuantizedKVTransfer Init] Consumer: Producer actor not found in namespace {KV_NAMESPACE}, will retry later", file=sys.stderr, flush=True)
                self._producer_actor = None

        print(f"[QuantizedKVTransfer Init] role={role}, prefix={self._shm_prefix}, is_producer={self._is_producer}, is_consumer={self._is_consumer}", file=sys.stderr, flush=True)

    # ---------- Helper methods ----------
    def _quantize_tensor(self, tensor: torch.Tensor):
        """Symmetric quantization, returns quantized GPU tensor, scale (CPU), and zero_point (CPU)."""
        tensor_float = tensor.float()
        min_val = tensor_float.min()
        max_val = tensor_float.max()
        scale = max(abs(min_val), abs(max_val)) / 127.0
        if scale == 0:
            scale = torch.tensor(1.0, device=tensor.device)
        quantized = torch.clamp(torch.round(tensor_float / scale), -128, 127).to(torch.int8)
      
        return quantized, scale.cpu(), torch.tensor(0, device='cpu', dtype=torch.int8)

    def _dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor,
                           zp: torch.Tensor, dtype: torch.dtype):
        return (quantized.float() * scale.to(quantized.device) + zp.float()).to(dtype)

    def _get_clean_req_id(self, req_id: str) -> str:
        """Remove the suffix automatically added by vLLM, e.g., req_001-94792588 -> req_001."""
        if '-' in req_id:
            return req_id.split('-')[0]
        return req_id

    # ---------- Required interface ----------
    def build_connector_meta(self, scheduler_output) -> Dict:
        request_ids = []
        for req_data in getattr(scheduler_output, "scheduled_new_reqs", []):
            request_ids.append(req_data.req_id)
        print(f"[QuantizedKVTransfer Meta] Build metadata with request_ids={request_ids}", file=sys.stderr, flush=True)
        return {"shm_prefix": self._shm_prefix, "request_ids": request_ids}

    def get_num_new_matched_tokens(self, request_ids: List[str], kv_cache_groups: Any) -> Tuple[int, bool]:
        return 0, False

    def start_save_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        print(f"[QuantizedKVTransfer start_save_kv] called with kwargs keys: {list(kwargs.keys())}", file=sys.stderr, flush=True)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Any,
    ) -> None:
        print(f"[QuantizedKVTransfer Save] ENTER for layer {layer_name}", file=sys.stderr, flush=True)

        # ---------- Consumer side: directly perform loading ----------
        if self._is_consumer:
            self._consumer_load_kv_layer(layer_name, kv_cache, attn_metadata)
            return

        # ---------- Producer side: save ----------
        if not self._is_producer:
            print("[QuantizedKVTransfer Save] Not a producer, skipping", file=sys.stderr, flush=True)
            return

        # Get all request_ids and their token counts
        request_ids = getattr(attn_metadata, "request_ids", None)
        if not request_ids:
            request_ids = ["req_001"]
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            print("[QuantizedKVTransfer Save] No seq_lens, assuming single request", file=sys.stderr, flush=True)
            seq_lens = [len(getattr(attn_metadata, "slot_mapping", []))]

        # Cumulative starting positions for each request
        cum_len = [0]
        for length in seq_lens:
            cum_len.append(cum_len[-1] + length)

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            print("[QuantizedKVTransfer Save] No slot_mapping, skipping", file=sys.stderr, flush=True)
            return

        match = re.search(r'layers\.(\d+)', layer_name)
        layer_idx = int(match.group(1)) if match else 0

        key_cache, value_cache = kv_cache
        page_size = key_cache.shape[1]
        # ---------- Optimization: flatten KV cache for bulk indexing ----------
        # shape: [num_pages * page_size, num_heads, head_dim]
        flat_k = key_cache.view(-1, key_cache.shape[2], key_cache.shape[3])
        flat_v = value_cache.view(-1, value_cache.shape[2], value_cache.shape[3])

        for req_idx, req_id_full in enumerate(request_ids):
            req_id = self._get_clean_req_id(req_id_full)
            start = cum_len[req_idx]
            end = cum_len[req_idx + 1]
            req_slots = slot_mapping[start:end]
            valid_slots = req_slots[req_slots >= 0]   # GPU tensor, used directly as indices
            num_tokens = valid_slots.numel()
            if num_tokens == 0:
                continue

            if (req_id, layer_idx) in self._saved_layers:
                print(f"[QuantizedKVTransfer Save] Layer {layer_idx} for {req_id} already saved, skipping", file=sys.stderr, flush=True)
                continue

            # ---------- Optimization: extract in bulk using advanced indexing ----------
            k_stack = flat_k[valid_slots]  # [num_tokens, num_heads, head_dim], GPU
            v_stack = flat_v[valid_slots]

            # Quantize (all on GPU)
            k_quant, k_scale, _ = self._quantize_tensor(k_stack)
            v_quant, v_scale, _ = self._quantize_tensor(v_stack)

            # ---------- Optimization: directly put GPU tensor, avoid .cpu().numpy() ----------
            k_ref = ray.put(k_quant)
            v_ref = ray.put(v_quant)
            # scale is a CPU scalar, can stay on CPU
            k_scale_ref = ray.put(k_scale)
            v_scale_ref = ray.put(v_scale)
            # zero_point is always 0, no need to transfer

            if req_id not in self._kv_refs:
                self._kv_refs[req_id] = {}
            self._kv_refs[req_id][f"k_layer{layer_idx}"] = k_ref
            self._kv_refs[req_id][f"v_layer{layer_idx}"] = v_ref
            self._kv_refs[req_id][f"k_scale_layer{layer_idx}"] = k_scale_ref
            self._kv_refs[req_id][f"v_scale_layer{layer_idx}"] = v_scale_ref

            self._saved_layers.add((req_id, layer_idx))
            print(f"[QuantizedKVTransfer Save] Saved layer {layer_idx} for {req_id}, {num_tokens} tokens (GPU direct)", file=sys.stderr, flush=True)

    def _consumer_load_kv_layer(
        self,
        layer_name: str,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Any,
    ) -> None:
        """Consumer-side loading logic, supports multiple requests."""
        request_ids = getattr(attn_metadata, "request_ids", None)
        if not request_ids:
            request_ids = ["req_001"]
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            seq_lens = [len(getattr(attn_metadata, "slot_mapping", []))]
        cum_len = [0]
        for length in seq_lens:
            cum_len.append(cum_len[-1] + length)

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            print("[QuantizedKVTransfer Load] No slot_mapping, skipping", file=sys.stderr, flush=True)
            return

        match = re.search(r'layers\.(\d+)', layer_name)
        layer_idx = int(match.group(1)) if match else 0

        # Ensure producer actor is available (with retries)
        if self._producer_actor is None:
            actor_name = f"{self._shm_prefix}_producer"
            for attempt in range(30):
                try:
                    self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
                    print(f"[QuantizedKVTransfer Load] Got producer actor on attempt {attempt+1}", file=sys.stderr, flush=True)
                    break
                except ValueError:
                    if attempt < 29:
                        time.sleep(0.5)
                    else:
                        print("[QuantizedKVTransfer Load] Producer actor not available after 30 attempts, abort", file=sys.stderr, flush=True)
                        return

        key_cache, value_cache = kv_cache
        page_size = key_cache.shape[1]
        # Flatten KV cache
        flat_k = key_cache.view(-1, key_cache.shape[2], key_cache.shape[3])
        flat_v = value_cache.view(-1, value_cache.shape[2], value_cache.shape[3])

        for req_idx, req_id_full in enumerate(request_ids):
            req_id = self._get_clean_req_id(req_id_full)
            start = cum_len[req_idx]
            end = cum_len[req_idx + 1]
            req_slots = slot_mapping[start:end]
            valid_slots = req_slots[req_slots >= 0]
            num_tokens = valid_slots.numel()
            if num_tokens == 0:
                continue

            if (req_id, layer_idx) in self._loaded_layers:
                print(f"[QuantizedKVTransfer Load] Layer {layer_idx} for {req_id} already loaded, skipping", file=sys.stderr, flush=True)
                continue

            # Attempt to fetch refs
            refs = None
            for rid in [req_id, req_id_full, "req_001"]:
                try:
                    refs = ray.get(self._producer_actor.get_kv_refs.remote(rid))
                    if refs:
                        print(f"[QuantizedKVTransfer Load] Got refs for {rid}", file=sys.stderr, flush=True)
                        break
                except Exception as e:
                    print(f"[QuantizedKVTransfer Load] Failed to get refs for {rid}: {e}", file=sys.stderr, flush=True)
            if not refs:
                print(f"[QuantizedKVTransfer Load] No refs found for {req_id}, skipping", file=sys.stderr, flush=True)
                continue

            k_ref = refs.get(f"k_layer{layer_idx}")
            v_ref = refs.get(f"v_layer{layer_idx}")
            k_scale_ref = refs.get(f"k_scale_layer{layer_idx}")
            v_scale_ref = refs.get(f"v_scale_layer{layer_idx}")
            if any(ref is None for ref in [k_ref, v_ref, k_scale_ref, v_scale_ref]):
                print(f"[QuantizedKVTransfer Load] Missing refs for layer {layer_idx}, request {req_id}", file=sys.stderr, flush=True)
                continue

            try:
                # ---------- Optimization: directly get GPU tensor, no numpy needed ----------
                k_quant = ray.get(k_ref)      # GPU tensor
                v_quant = ray.get(v_ref)
                k_scale = ray.get(k_scale_ref)  # CPU scalar
                v_scale = ray.get(v_scale_ref)
            except Exception as e:
                print(f"[QuantizedKVTransfer Load] Failed to get data: {e}", file=sys.stderr, flush=True)
                continue

            # Dequantize (k_scale is on CPU, move to GPU first)
            k_dequant = (k_quant.float() * k_scale.to(k_quant.device)).to(key_cache.dtype)
            v_dequant = (v_quant.float() * v_scale.to(v_quant.device)).to(value_cache.dtype)

            # ---------- Optimization: bulk fill, avoid Python loops ----------
            flat_k[valid_slots] = k_dequant
            flat_v[valid_slots] = v_dequant

            self._loaded_layers.add((req_id, layer_idx))
            print(f"[QuantizedKVTransfer Load] Loaded layer {layer_idx} for {req_id}, {num_tokens} tokens (GPU direct)", file=sys.stderr, flush=True)

    def update_state_after_alloc(self, request_id: str, new_kv_cache: Any, num_tokens: int) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self) -> None:
        print("[QuantizedKVTransfer WaitSave] ENTER", file=sys.stderr, flush=True)
        if not self._is_producer:
            return
        if self._producer_actor is None:
            print("[QuantizedKVTransfer WaitSave] Producer actor not available, cannot push refs", file=sys.stderr, flush=True)
            return
        for req_id, refs in self._kv_refs.items():
            if refs:
                try:
                    ray.get(self._producer_actor.add_refs.remote(req_id, refs))
                    print(f"[QuantizedKVTransfer WaitSave] Pushed refs for {req_id}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[QuantizedKVTransfer WaitSave] Failed to push refs: {e}", file=sys.stderr, flush=True)
        self._kv_refs.clear()
        print("[QuantizedKVTransfer WaitSave] Save completed", file=sys.stderr, flush=True)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        print("[QuantizedKVTransfer start_load_kv] ENTER", file=sys.stderr, flush=True)
        if not self._is_consumer:
            print("[QuantizedKVTransfer start_load_kv] Not a consumer, skipping", file=sys.stderr, flush=True)
            return

        if self._producer_actor is None:
            actor_name = f"{self._shm_prefix}_producer"
            for attempt in range(20):
                try:
                    self._producer_actor = ray.get_actor(actor_name, namespace=KV_NAMESPACE)
                    print(f"[QuantizedKVTransfer Load] Successfully got producer actor (attempt {attempt+1})", file=sys.stderr, flush=True)
                    break
                except ValueError:
                    if attempt < 19:
                        time.sleep(0.5)
            else:
                print("[QuantizedKVTransfer Load] Producer actor not available after retries, cannot load", file=sys.stderr, flush=True)
                return

    def load_kv_layer(
        self,
        layer_name: str,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Any,
    ) -> None:
        # This method may not be called; kept for future use
        print(f"[QuantizedKVTransfer load_kv_layer] ENTER for layer {layer_name} (not used)", file=sys.stderr, flush=True)

    def register_request(self, request_id: str, meta: Any) -> None:
        print(f"[QuantizedKVTransfer Register] request_id={request_id}", file=sys.stderr, flush=True)

    def unregister_request(self, request_id: str) -> None:
        self._saved_layers = {item for item in self._saved_layers if item[0] != request_id}
        self._loaded_layers = {item for item in self._loaded_layers if item[0] != request_id}
        if self._is_producer and self._producer_actor is not None:
            try:
                ray.get(self._producer_actor.clear_refs.remote(request_id))
            except Exception:
                pass

    def get_kv_connector_metrics(self):
        return None

    def get_connector_metadata(self, request_ids: list[str]) -> dict:
        return {"shm_prefix": self._shm_prefix}

    def get_cross_layer_attn_metadata(self, request_ids, kv_cache_groups, layer_name, num_tokens):
        return None

    def close(self):
        pass