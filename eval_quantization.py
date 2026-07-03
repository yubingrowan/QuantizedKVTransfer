#!/usr/bin/env python3
"""
Quantized Transfer Quality Evaluation Script (Full Version)
- Baseline mode (no quantized transfer): directly uses vLLM single‑process inference
- Quantized mode (int8 + Ray transfer): uses PrefillActor + DecodeActor separated inference
- Computes BLEU and ROUGE scores (fully local computation, no network dependencies)
"""

import sys
import os
import time
import ray
import torch
import pandas as pd
from tqdm import tqdm

# vLLM related
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.config import KVTransferConfig

# ==================== Configuration ====================
MODEL_PATH = "/mnt/workspace/.cache/modelscope/models/facebook/opt-125m"
TEST_PROMPTS = [
    "What should I do when",
    "The capital of France is",
    "Machine learning is a",
    "Python programming language can",
]

SAMPLING_PARAMS = SamplingParams(
    max_tokens=20,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    frequency_penalty=0.3,
    presence_penalty=0.3,
    stop=["\n"],
)

# ==================== Ray Actor Definitions ====================
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

    def run_prefill(self, prompts: list[str], request_ids: list[str]):
        print(f"[Prefill] Starting {len(prompts)} requests", flush=True)
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        for prompt, req_id in zip(prompts, request_ids):
            self.engine.add_request(req_id, prompt, sampling_params)
        while self.engine.has_unfinished_requests():
            self.engine.step()
        for req_id in request_ids:
            self.engine.abort_request(req_id)
        print(f"[Prefill] Completed {len(prompts)} requests", flush=True)

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

    def run_decode(self, prompts: list[str], request_ids: list[str], max_tokens: int = 20):
        print(f"[Decode] Starting {len(prompts)} requests", flush=True)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            frequency_penalty=0.3,
            presence_penalty=0.3,
            stop=["\n"],
        )
        for prompt, req_id in zip(prompts, request_ids):
            self.engine.add_request(req_id, prompt, sampling_params)
        results = {}
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for out in outputs:
                if out.request_id in request_ids:
                    results[out.request_id] = out.outputs[0].text
        print(f"[Decode] Completed {len(results)} requests", flush=True)
        return results


# ==================== 1. Baseline Inference ====================
def run_baseline(prompts):
    print("[Baseline] Running without quantization...")
    engine_args = EngineArgs(
        model=MODEL_PATH,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        max_model_len=256,
    )
    llm = LLM(**engine_args.__dict__)
    outputs = llm.generate(prompts, SAMPLING_PARAMS)
    result_texts = [out.outputs[0].text for out in outputs]
    del llm
    torch.cuda.empty_cache()
    time.sleep(1)
    return result_texts


# ==================== 2. Quantized Inference ====================
def run_quantized(prompts):
    print("[Quantized] Running with int8 quantization + Ray transfer...")
    
    if ray.is_initialized():
        ray.shutdown()
    time.sleep(2)
    torch.cuda.empty_cache()
    time.sleep(1)
    
    ray.init(ignore_reinit_error=True, namespace=KV_NAMESPACE)
    print("[Quantized] Ray initialized")

    request_ids = [f"req_{i:03d}" for i in range(len(prompts))]
    prefill = PrefillActor.remote(MODEL_PATH, rank=0)
    decode = DecodeActor.remote(MODEL_PATH, rank=1)

    ray.get(prefill.run_prefill.remote(prompts, request_ids))
    results = ray.get(decode.run_decode.remote(prompts, request_ids, max_tokens=20))

    ray.shutdown()
    torch.cuda.empty_cache()
    return [results.get(req_id, "") for req_id in request_ids]


# ==================== 3. Local Metric Computation (fully local, no network) ====================
def tokenize(text):
    """Simple tokenizer: splits by spaces and punctuation, no nltk dependency."""
    # Replace punctuation with spaces, then split by whitespace
    for char in ['.', ',', '!', '?', ';', ':', '(', ')', '"', "'"]:
        text = text.replace(char, ' ')
    return text.lower().split()

def compute_metrics(references, hypotheses):
    """Fully local BLEU and ROUGE computation, no network dependencies."""
    print("[compute_metrics] Start")
    
    # If lists are empty, return 0
    if not references or not hypotheses:
        return {"BLEU": 0.0, "ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
    
    # ========== BLEU computation (simplified, using exact n-gram match) ==========
    def get_ngrams(tokens, n):
        return set(zip(*[tokens[i:] for i in range(n)]))
    
    def bleu_single(ref_tokens, hyp_tokens, n=4):
        if not hyp_tokens:
            return 0.0
        # Compute precisions: how many hyp n-grams appear in ref
        precisions = []
        for i in range(1, n+1):
            ref_ngrams = get_ngrams(ref_tokens, i)
            hyp_ngrams = get_ngrams(hyp_tokens, i)
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            overlap = len(ref_ngrams & hyp_ngrams)
            precision = overlap / len(hyp_ngrams) if len(hyp_ngrams) > 0 else 0.0
            precisions.append(precision)
        # If any precision is zero, BLEU is zero
        if any(p == 0 for p in precisions):
            return 0.0
        # Geometric mean
        import math
        bleu = math.exp(sum(math.log(p) for p in precisions) / n)
        # Brevity penalty (if hyp is much shorter than ref)
        if len(hyp_tokens) < len(ref_tokens):
            brevity_penalty = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
            bleu *= brevity_penalty
        return min(bleu, 1.0)
    
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)
        if not hyp_tokens:
            bleu_scores.append(0.0)
        else:
            bleu_scores.append(bleu_single(ref_tokens, hyp_tokens))
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    # ========== ROUGE computation (simplified) ==========
    def rouge_single(ref_tokens, hyp_tokens, n):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        if not ref_ngrams or not hyp_ngrams:
            return 0.0
        overlap = len(ref_ngrams & hyp_ngrams)
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def rouge_l_single(ref_tokens, hyp_tokens):
        # Longest Common Subsequence (LCS) simplified
        if not ref_tokens or not hyp_tokens:
            return 0.0
        # Dynamic programming for LCS
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        recall = lcs / m if m > 0 else 0.0
        precision = lcs / n if n > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)
        if not hyp_tokens:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue
        rouge1_scores.append(rouge_single(ref_tokens, hyp_tokens, 1))
        rouge2_scores.append(rouge_single(ref_tokens, hyp_tokens, 2))
        rougeL_scores.append(rouge_l_single(ref_tokens, hyp_tokens))
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    
    print("[compute_metrics] Complete")
    
    return {
        "BLEU": avg_bleu,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL,
    }


# ==================== 4. Main Workflow ====================
def main():
    print("=" * 60)
    print("📊 Quantized Transfer Quality Evaluation")
    print("=" * 60)

    # Run quantized first, then baseline (to avoid resource residue)
    quantized_outputs = run_quantized(TEST_PROMPTS)
    print("\n[Quantized Results]")
    for p, o in zip(TEST_PROMPTS, quantized_outputs):
        print(f"  {p} -> {o}")

    baseline_outputs = run_baseline(TEST_PROMPTS)
    print("\n[Baseline Results]")
    for p, o in zip(TEST_PROMPTS, baseline_outputs):
        print(f"  {p} -> {o}")

    metrics = compute_metrics(baseline_outputs, quantized_outputs)
    
    print("\n" + "=" * 60)
    print("📊 Quantized Transfer Quality Evaluation Results")
    print("=" * 60)
    print(f"  BLEU-4:     {metrics['BLEU']:.4f}")
    print(f"  ROUGE-1:    {metrics['ROUGE-1']:.4f}")
    print(f"  ROUGE-2:    {metrics['ROUGE-2']:.4f}")
    print(f"  ROUGE-L:    {metrics['ROUGE-L']:.4f}")
    print("=" * 60)

    df = pd.DataFrame({
        "prompt": TEST_PROMPTS,
        "baseline": baseline_outputs,
        "quantized": quantized_outputs,
    })
    df.to_csv("quantization_evaluation.csv", index=False)
    print("\n✅ Detailed results saved to quantization_evaluation.csv")
    return metrics

if __name__ == "__main__":
    main()