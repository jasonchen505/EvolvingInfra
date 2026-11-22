#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Lower bound on the GPU time of one decode step: weights, KV cache and recurrent state.

    python byte_floor.py --model-dir DIR --requests B --rows-per-request R --context-tokens L \
        --peak-tflops X --peak-tbps Y [--kv-bytes-per-token K] [--state-bytes-per-request S] \
        [--draft-model-dir DIR --draft-forwards N] [--gpus G] [options]

A class of work cannot take less than the larger of its FLOPs at 60 % of the GPU's peak and the
bytes it must move at 90 % of peak bandwidth; the floor of a step is the sum over classes. Take the
peaks of one GPU from its datasheet, for the precision the kernels run in. With tensor parallelism
over G GPUs pass --gpus G: each GPU holds 1/G of the work, so the floor uses G times the peaks.

Classes modeled here:
- weights (GEMMs): every weight tensor is read once per step. Bytes come from the safetensors
  headers of --model-dir (and --draft-model-dir, read --draft-forwards times). Tensors matching
  --exclude are not read in decode (vision tower; the embedding table, of which a step reads only a
  few rows, unless the model ties it to the output layer). FLOPs are 2 x parameters x rows, with
  the parameter count from the tensor shapes; for packed low-precision weights pass --params.
  Mixture of experts: tensors matching --expert-pattern are read only for the experts the step's
  rows reach, E * (1 - (1 - k/E) ** rows) of E per layer with k experts per token.
- attention: the KV cache of every request is read once per step. --kv-bytes-per-token is the
  whole model's KV per token: every attention layer, every KV head (2 x layers x KV heads x head
  dim x bytes per element). The engine's start log reports one GPU's pool; with G GPUs and at
  least G KV heads, multiply its KV cache size per token by G. Each GPU reads 1/G of the KV, or
  1/H with H KV heads when G > H: the heads are then copied to several GPUs (--kv-heads, default
  num_key_value_heads of config.json). FLOPs are 4 x heads x head dim x rows x context per
  attention layer (--attention-layers overrides the layer count).
- recurrent state (linear attention, state-space layers): read and written once per step per
  request. --state-bytes-per-request is the whole model's state per request; the engine's start
  log reports one GPU's pool (state pool size / requests), so multiply by G.
Work these classes do not cover (communication between GPUs, sampling, drafting kernels) needs its
own floor by the same rule.
"""
import argparse
import glob
import json
import math
import os
import re
import struct


def safetensor_tensors(model_dir):
    """(name, bytes, elements) of every tensor in the model directory's safetensors files."""
    tensors = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, stop = info["data_offsets"]
            tensors.append((name, stop - start, math.prod(info["shape"])))
    if not tensors:
        raise SystemExit(f"no *.safetensors in {model_dir}")
    return tensors


def text_config(model_dir):
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        config = json.load(f)
    return {**config, **config.get("text_config", {})}


def experts_reached(experts, per_token, rows):
    return experts * (1 - (1 - per_token / experts) ** rows)


def attention_layers(config):
    types = config.get("layer_types")
    if types:
        return sum(1 for t in types if "full" in t or t == "attention")
    interval = config.get("full_attention_interval")
    layers = config.get("num_hidden_layers", 0)
    return layers // interval if interval else layers


def floor(args):
    rows = args.requests * args.rows_per_request
    config = text_config(args.model_dir)
    tied = bool(config.get("tie_word_embeddings"))
    exclude = re.compile(args.exclude)
    expert = re.compile(args.expert_pattern)
    dense_bytes = expert_bytes = dense_params = expert_params = 0
    names = []
    for name, nbytes, elements in safetensor_tensors(args.model_dir):
        if exclude.search(name) and not (tied and "embed_tokens" in name):
            continue
        names.append(name)
        if expert.search(name):
            expert_bytes += nbytes
            expert_params += elements
        else:
            dense_bytes += nbytes
            dense_params += elements
    draft_bytes = draft_params = 0
    if args.draft_model_dir:
        for name, nbytes, elements in safetensor_tensors(args.draft_model_dir):
            if exclude.search(name):
                continue
            draft_bytes += nbytes * args.draft_forwards
            draft_params += elements * args.draft_forwards
    share = 1.0
    if expert_bytes:
        experts = args.experts or config.get("num_experts") or config.get("n_routed_experts") \
            or config.get("num_local_experts")
        per_token = args.experts_per_token or config.get("num_experts_per_tok")
        if not experts or not per_token:
            raise SystemExit("expert tensors found: pass --experts and --experts-per-token")
        share = experts_reached(experts, per_token, rows) / experts
        expert_params_active = expert_params * per_token / experts
    else:
        expert_params_active = 0
    params = args.params if args.params else dense_params + expert_params_active + draft_params
    weight_bytes = dense_bytes + expert_bytes * share + draft_bytes
    flops_peak = args.peak_tflops * 1e12 * 0.6 * args.gpus
    bandwidth = args.peak_tbps * 1e12 * 0.9 * args.gpus
    classes = {}
    gemm_flops = 2 * params * rows
    classes["weights"] = {"bytes": weight_bytes, "flops": gemm_flops,
                          "floor_ms": 1e3 * max(gemm_flops / flops_peak, weight_bytes / bandwidth)}
    if args.kv_bytes_per_token:
        heads = config.get("num_attention_heads", 0)
        head_dim = config.get("head_dim") or (config.get("hidden_size", 0) // heads if heads else 0)
        layers = args.attention_layers if args.attention_layers is not None else attention_layers(config)
        kv_heads = args.kv_heads or config.get("num_key_value_heads") or heads
        # bytes read by all GPUs together: each GPU reads 1/min(G, KV heads) of the model's KV
        kv_bytes = (args.kv_bytes_per_token * args.context_tokens * args.requests
                    * args.gpus / min(args.gpus, kv_heads))
        attn_flops = 4 * heads * head_dim * layers * rows * args.context_tokens
        classes["attention"] = {"bytes": kv_bytes, "flops": attn_flops,
                                "floor_ms": 1e3 * max(attn_flops / flops_peak, kv_bytes / bandwidth)}
    if args.state_bytes_per_request:
        state_bytes = 2 * args.state_bytes_per_request * args.requests
        classes["recurrent_state"] = {"bytes": state_bytes, "flops": 0, "floor_ms": 1e3 * state_bytes / bandwidth}
    return {"rows": rows, "expert_share_read": share if expert_bytes else None, "params": params,
            "tensors_counted": len(names), "classes": classes,
            "floor_ms": sum(c["floor_ms"] for c in classes.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--requests", type=int, required=True, help="requests in the batch")
    parser.add_argument("--rows-per-request", type=int, required=True,
                        help="tokens the target model processes per request per step (1 without speculation)")
    parser.add_argument("--context-tokens", type=float, default=0, help="mean context length per request")
    parser.add_argument("--peak-tflops", type=float, required=True)
    parser.add_argument("--peak-tbps", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--kv-bytes-per-token", type=float, default=0, help="whole model, all layers and KV heads")
    parser.add_argument("--kv-heads", type=int, default=None, help="default: num_key_value_heads of config.json")
    parser.add_argument("--state-bytes-per-request", type=float, default=0)
    parser.add_argument("--draft-model-dir", default=None)
    parser.add_argument("--draft-forwards", type=int, default=1)
    parser.add_argument("--params", type=float, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--experts-per-token", type=int, default=None)
    parser.add_argument("--attention-layers", type=int, default=None)
    parser.add_argument("--expert-pattern", default=r"\.experts\.")
    parser.add_argument("--exclude", default=r"(^|\.)(visual|vision_tower|vision_model|mm_projector)\.|embed_tokens")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    result = floor(args)
    print(f"{result['rows']} rows per step, {result['tensors_counted']} weight tensors counted, "
          f"{result['params'] / 1e9:.2f} B parameters in GEMMs"
          + (f", {100 * result['expert_share_read']:.1f} % of expert bytes read" if result["expert_share_read"] else ""))
    for name, c in result["classes"].items():
        print(f"  {name:16s} {c['bytes'] / 1e9:9.3f} GB  {c['flops'] / 1e12:9.3f} TFLOP  floor {c['floor_ms']:.3f} ms")
    print(f"floor of the modeled classes: {result['floor_ms']:.3f} ms per step")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
