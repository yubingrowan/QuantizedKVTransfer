#!/usr/bin/env python3
"""
Multi‑request test script
Tests the QuantizedRaySharedMemoryConnector under multiple concurrent requests.
"""
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
                "use_triton": True,
            }
        )
        engine_args = EngineArgs(
            model=model_path,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            max_num_seqs=4,  # Support more concurrent requests
            max_model_len=256,
            kv_transfer_config=kv_config,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def run_prefill(self, prompts: list[str], request_ids: list[str]):
        """Process multiple prefill requests in batch."""
        print(f"[QuantizedKVTransfer Prefill] Starting {len(prompts)} requests", flush=True)
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        
        # Add all requests
        for prompt, req_id in zip(prompts, request_ids):
            self.engine.add_request(req_id, prompt, sampling_params)
        
        # Process all requests
        outputs = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            outputs.extend(step_outputs)
        
        print(f"[QuantizedKVTransfer Prefill] Completed {len(outputs)} requests", flush=True)
        
        # Clean up requests
        for req_id in request_ids:
            self.engine.abort_request(req_id)
        
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
                "use_triton": True,
            }
        )
        engine_args = EngineArgs(
            model=model_path,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            max_num_seqs=4,
            max_model_len=256,
            kv_transfer_config=kv_config,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def run_decode(self, prompts: list[str], request_ids: list[str], max_tokens: int = 10):
        """Process multiple decode requests in batch."""
        print(f"[QuantizedKVTransfer Decode] Starting {len(prompts)} requests", flush=True)
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.7,          # Increase randomness, reduce repetition
            top_p=0.9,
            repetition_penalty=1.1,   # Penalise repeated tokens to avoid "I'm I'm I'm"
            frequency_penalty=0.3,    # Lower high‑frequency word repetition
            presence_penalty=0.3,     # Encourage new words to appear
            stop=["\n"]               # Early termination to avoid meaningless continuation
        )
        
        # Add all requests
        for prompt, req_id in zip(prompts, request_ids):
            self.engine.add_request(req_id, prompt, sampling_params)
        
        # Process all requests
        results = {}
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for out in outputs:
                if out.request_id in request_ids:
                    results[out.request_id] = out.outputs[0].text
        
        print(f"[QuantizedKVTransfer Decode] Completed {len(results)} requests", flush=True)
        return results

def main():
    ray.init(ignore_reinit_error=True, namespace=KV_NAMESPACE)
    model_path = "/mnt/workspace/.cache/modelscope/models/facebook/opt-125m"
    
    # Multiple requests
    prompts = [
        "What should I do when",
        "The capital of France is",
        "Machine learning is a",
        "Python programming language can",
    ]
    request_ids = [f"req_{i:03d}" for i in range(len(prompts))]
    
    print(f"Testing {len(prompts)} concurrent requests", flush=True)
    
    prefill = PrefillActor.remote(model_path, rank=0)
    decode = DecodeActor.remote(model_path, rank=1)
    
    # Prefill phase
    ray.get(prefill.run_prefill.remote(prompts, request_ids))
    
    # Decode phase
    results = ray.get(decode.run_decode.remote(prompts, request_ids, max_tokens=10))
    
    # Output results
    print("\n" + "="*50)
    for req_id, prompt in zip(request_ids, prompts):
        generated = results.get(req_id, "")
        print(f"{prompt}{generated}")
    print("="*50)

if __name__ == "__main__":
    main()