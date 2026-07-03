#!/usr/bin/env python3
import sys
import os
# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ray
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm import SamplingParams
from vllm.config import KVTransferConfig

# Using the original single‑file implementation
KV_NAMESPACE = "kv_namespace"

@ray.remote(num_gpus=0.4)
class PrefillActor:
    def __init__(self, model_path: str, rank: int = 0):
        kv_config = KVTransferConfig(
            kv_connector="QuantizedRaySharedMemoryConnector",
            kv_role="kv_producer",
            kv_rank=rank,
            kv_connector_module_path="quantized_ray_connector",
            kv_connector_extra_config={
                "shm_prefix": "my_quantized_kv",
                "is_producer": True,
                "is_consumer": False,
                "use_triton": True,  # Use Triton quantization
            }
        )
        engine_args = EngineArgs(
            model=model_path,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            max_num_seqs=2,
            max_model_len=256,
            kv_transfer_config=kv_config,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def run_prefill(self, prompt: str, request_id: str):
        print(f"[QuantizedKVTransfer Prefill] Starting for {request_id}", flush=True)
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        self.engine.add_request(request_id, prompt, sampling_params)
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
        print(f"[QuantizedKVTransfer Prefill] Request {request_id} completed", flush=True)
        self.engine.abort_request(request_id)
        return outputs

@ray.remote(num_gpus=0.5)
class DecodeActor:
    def __init__(self, model_path: str, rank: int = 1):
        kv_config = KVTransferConfig(
            kv_connector="QuantizedRaySharedMemoryConnector",
            kv_role="kv_consumer",
            kv_rank=rank,
            kv_connector_module_path="quantized_ray_connector",
            kv_connector_extra_config={
                "shm_prefix": "my_quantized_kv",
                "is_producer": False,
                "is_consumer": True,
                "use_triton": True,  # Use Triton quantization
            }
        )
        engine_args = EngineArgs(
            model=model_path,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            max_num_seqs=2,
            max_model_len=256,
            kv_transfer_config=kv_config,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def run_decode(self, prompt: str, request_id: str, max_tokens: int = 10):
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        self.engine.add_request(request_id, prompt, sampling_params)
        final_text = ""
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for out in outputs:
                if out.request_id == request_id:
                    final_text = out.outputs[0].text
        return final_text

def main():
    ray.init(ignore_reinit_error=True, namespace=KV_NAMESPACE)
    model_path = "/mnt/workspace/.cache/modelscope/models/facebook/opt-125m"
    prompt = "Python programming language"
    request_id = "req_001"

    prefill = PrefillActor.remote(model_path, rank=0)
    decode = DecodeActor.remote(model_path, rank=1)

    ray.get(prefill.run_prefill.remote(prompt, request_id))
    generated = ray.get(decode.run_decode.remote(prompt, request_id, max_tokens=10))

    print(f"\nFull generation: {prompt}{generated}")

if __name__ == "__main__":
    main()