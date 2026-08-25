import ray
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm import SamplingParams
from vllm.config import KVTransferConfig

KV_NAMESPACE = "kv_namespace"

@ray.remote(num_gpus=0.45)
class PrefillActor:
    def __init__(self, model_path, rank=0):
        kv_config = KVTransferConfig(
            kv_connector="QuantizedRaySharedMemoryConnector",
            kv_connector_module_path="connector",
            kv_role="kv_producer",
            kv_rank=rank,
            kv_connector_extra_config={
                "shm_prefix": "async_kv",
                "is_producer": True,
                "is_consumer": False,
            }
        )
        engine_args = EngineArgs(
            model=model_path,
            gpu_memory_utilization=0.42,
            enforce_eager=True,
            max_num_seqs=4,
            max_model_len=256,
            kv_transfer_config=kv_config,
            enable_chunked_prefill=False,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def run_prefill(self, prompts, request_ids):
        print(f"[Prefill] Starting {len(prompts)} requests")
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        for prompt, req_id in zip(prompts, request_ids):
            self.engine.add_request(req_id, prompt, sampling_params)
        while self.engine.has_unfinished_requests():
            self.engine.step()
        print("[Prefill] Done")

@ray.remote(num_gpus=0.5)
class DecodeActor:
    def __init__(self, model_path, rank=1):
        kv_config = KVTransferConfig(
            kv_connector="QuantizedRaySharedMemoryConnector",
            kv_connector_module_path="connector",
            kv_role="kv_consumer",
            kv_rank=rank,
            kv_connector_extra_config={
                "shm_prefix": "async_kv",
                "is_producer": False,
                "is_consumer": True,
            }
        )
        engine_args = EngineArgs(
            model=model_path,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
            max_num_seqs=4,
            max_model_len=256,
            kv_transfer_config=kv_config,
            enable_chunked_prefill=False,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def run_decode(self, prompts, request_ids, max_tokens=10):
        print(f"[Decode] Starting {len(prompts)} requests")
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["\n"]
        )
        for prompt, req_id in zip(prompts, request_ids):
            self.engine.add_request(req_id, prompt, sampling_params)
        results = {}
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for out in outputs:
                if out.request_id in request_ids:
                    results[out.request_id] = out.outputs[0].text
        print("[Decode] Done")
        return results

def main():
    ray.init(ignore_reinit_error=True, namespace=KV_NAMESPACE)
    model_path = "/mnt/workspace/.cache/modelscope/models/facebook/opt-125m"

    prompts = [
        "What should I do when",
        "The capital of France is",
        "Machine learning is a",
        "Python programming language can",
        "My name is",
        "I want to find someone",
    ]
    request_ids = [f"req_{i:03d}" for i in range(len(prompts))]

    prefill = PrefillActor.remote(model_path, rank=0)
    decode = DecodeActor.remote(model_path, rank=1)

    print("\n=== PREFILL ===")
    ray.get(prefill.run_prefill.remote(prompts, request_ids))

    print("\n=== DECODE ===")
    results = ray.get(decode.run_decode.remote(prompts, request_ids, max_tokens=10))

    print("\n" + "="*50)
    for req_id, prompt in zip(request_ids, prompts):
        gen = results.get(req_id, "")
        print(f"{prompt}{gen}")
    print("="*50)

if __name__ == "__main__":
    main()