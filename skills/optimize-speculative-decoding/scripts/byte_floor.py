#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Lower bound on the GPU time of one decode step: weights, KV cache and recurrent state.

    python byte_floor.py --model-dir DIR (--requests N | --batch-mix N:STEPS[,N:STEPS...]) \
        --rows-per-request R --context-tokens L --peak-tflops X --peak-tbps Y [--kv-bytes-per-token K] \
        [--state-bytes-per-request S] [--draft-model-dir DIR --draft-forwards N] [--gpus G] [options]

--requests N is the engine's batch of the step: the requests it decodes together. A run's steps have
different batches; --batch-mix N:STEPS,... (decode_steps.py prints it for a window) gives the number
of steps at each batch, and the floor is then the mean over those steps of each step's floor at its
own batch, class by class.

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
  rows reach, E * (1 - (1 - k/E) ** rows) of E per layer with k experts per token; a prefill forward
  reaches the experts of its chunk's tokens.
- attention: the KV cache of every request is read once per step. --kv-bytes-per-token is the
  whole model's KV per token: every attention layer, every KV head (2 x layers x KV heads x head
  dim x bytes per element). The engine's start log reports one GPU's pool; with G GPUs and at
  least G KV heads, multiply its KV cache size per token by G. Each GPU reads 1/G of the KV, or
  1/H with H KV heads when G > H: the heads are then copied to several GPUs (--kv-heads, default
  num_key_value_heads of config.json). FLOPs are 4 x heads x head dim x rows x context per
  attention layer (--attention-layers overrides the count of full-attention layers). A sliding-window
  layer (layer_types "sliding_attention", window "sliding_window" in config.json) reads and attends
  to at most the window: each attention layer holds an equal share of --kv-bytes-per-token.
- recurrent state (linear attention, state-space layers): read and written once per step per
  request. --state-bytes-per-request is the whole model's state per request; the engine's start
  log reports one GPU's pool (state pool size / requests), so multiply by G.
- drafter: --draft-model-dir, its weights read --draft-forwards times per step, and 2 x its
  parameters x its rows FLOPs; a class of its own, with its own shapes and its own config.json: its
  experts reached by its rows, and its embedding table read when it ties its output layer to it.
- recurrent state: also --recurrent-flops-per-row, the recurrent layers' FLOPs per processed token,
  computed from the model's layer shapes (for a gated delta rule layer, a few times heads x key dim x
  value dim), since a wider tree updates the state for every row.
- prefill: --prefill-tokens P prompt tokens processed per step on average (new requests' prompts,
  in their own forward or admitted into the verify step), in chunks of --prefill-chunk: 2 x
  parameters x P FLOPs, attention to the prefix and within the chunk, and their KV written. A prefill
  forward of its own reads the weights again: --prefill-passes-per-step F, the average number of such
  forwards per step, adds F weight reads.
Shared tensors are counted once per GPU group (divided by G): a tensor copied to every GPU, such as a
small table, is read G times; pass such corrections through --params or by hand.
Work these classes do not cover (communication between GPUs, sampling, drafting kernels) needs its
own floor by the same rule.
"""
import argparse
import copy
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
    """(full-attention layers, sliding-window layers, window)."""
    types = config.get("layer_types")
    window = config.get("sliding_window") or 0
    if types:
        return (sum(1 for t in types if "full" in t or t == "attention"), sum(1 for t in types if "sliding" in t),
                window)
    interval = config.get("full_attention_interval")
    layers = config.get("num_hidden_layers", 0)
    return (layers // interval if interval else layers), 0, window


def weights(model_dir, exclude, expert, tied):
    """(dense bytes, expert bytes, dense parameters, expert parameters, names) of the tensors a step reads."""
    dense_bytes = expert_bytes = dense_params = expert_params = 0
    names = []
    for name, nbytes, elements in safetensor_tensors(model_dir):
        if exclude.search(name) and not (tied and "embed_tokens" in name):
            continue
        names.append(name)
        if expert.search(name):
            expert_bytes += nbytes
            expert_params += elements
        else:
            dense_bytes += nbytes
            dense_params += elements
    return dense_bytes, expert_bytes, dense_params, expert_params, names


def expert_counts(config, experts=None, per_token=None):
    experts = experts or config.get("num_experts") or config.get("n_routed_experts") \
        or config.get("num_local_experts")
    per_token = per_token or config.get("num_experts_per_tok")
    if not experts or not per_token:
        raise SystemExit("expert tensors found: pass --experts and --experts-per-token")
    return experts, per_token


def floor(args):
    rows = args.requests * args.rows_per_request
    config = text_config(args.model_dir)
    exclude = re.compile(args.exclude)
    expert = re.compile(args.expert_pattern)
    dense_bytes, expert_bytes, dense_params, expert_params, names = weights(
        args.model_dir, exclude, expert, bool(config.get("tie_word_embeddings")))
    share = 1.0
    if expert_bytes:
        experts, per_token = expert_counts(config, args.experts, args.experts_per_token)
        share = experts_reached(experts, per_token, rows) / experts
        expert_params_active = expert_params * per_token / experts
    else:
        expert_params_active = 0
    params = args.params if args.params else dense_params + expert_params_active
    weight_bytes = dense_bytes + expert_bytes * share
    flops_peak = args.peak_tflops * 1e12 * 0.6 * args.gpus
    bandwidth = args.peak_tbps * 1e12 * 0.9 * args.gpus

    def entry(nbytes, flops):
        return {"bytes": nbytes, "flops": flops, "floor_ms": 1e3 * max(flops / flops_peak, nbytes / bandwidth)}

    classes = {"weights": entry(weight_bytes, 2 * params * rows)}
    if args.draft_model_dir:
        draft_config = text_config(args.draft_model_dir)
        d_dense, d_expert, d_dense_params, d_expert_params, _ = weights(
            args.draft_model_dir, exclude, expert, bool(draft_config.get("tie_word_embeddings")))
        draft_rows = args.requests * (args.draft_rows_per_request or args.rows_per_request)
        d_share, d_active = 1.0, 0
        if d_expert:
            d_experts, d_per_token = expert_counts(draft_config)
            d_share = experts_reached(d_experts, d_per_token, draft_rows) / d_experts
            d_active = d_expert_params * d_per_token / d_experts
        classes["drafter"] = entry((d_dense + d_expert * d_share) * args.draft_forwards,
                                   2 * (d_dense_params + d_active) * draft_rows * args.draft_forwards)
    heads = config.get("num_attention_heads", 0)
    head_dim = config.get("head_dim") or (config.get("hidden_size", 0) // heads if heads else 0)
    full, sliding, window = attention_layers(config)
    if args.attention_layers is not None:
        full = args.attention_layers
    layers = full + sliding

    def seen(tokens):
        """Tokens a sliding-window layer attends to."""
        return min(tokens, window) if window else tokens

    def per_layer_tokens(tokens):
        """KV tokens read, averaged over the attention layers."""
        return (full * tokens + sliding * seen(tokens)) / layers if layers else tokens

    kv_heads = args.kv_heads or config.get("num_key_value_heads") or heads or 1
    per_gpu_kv = args.gpus / min(args.gpus, kv_heads)
    if args.kv_bytes_per_token:
        # bytes read by all GPUs together: each GPU reads 1/min(G, KV heads) of the model's KV; the step's own
        # rows also attend to each other (a triangle of rows_per_request per request)
        kv_bytes = args.kv_bytes_per_token * per_layer_tokens(args.context_tokens) * args.requests * per_gpu_kv
        within = (args.within_step_pairs if args.within_step_pairs is not None
                  else args.rows_per_request * (args.rows_per_request + 1) / 2)  # a chain; a tree: its ancestor pairs
        pairs = (full * (rows * args.context_tokens + args.requests * within)
                 + sliding * (rows * seen(args.context_tokens) + args.requests * within))
        classes["attention"] = entry(kv_bytes, 4 * heads * head_dim * pairs)
    if args.state_bytes_per_request or args.recurrent_flops_per_row:
        classes["recurrent_state"] = entry(2 * args.state_bytes_per_request * args.requests,
                                           args.recurrent_flops_per_row * rows)
    if args.prefill_tokens:
        chunk = args.prefill_chunk or args.prefill_tokens
        # each chunk attends to the prefix and to itself: P x context + P x (chunk + 1) / 2 pairs per step
        pairs = (full * (args.prefill_tokens * args.prefill_prefix_tokens + args.prefill_tokens * (chunk + 1) / 2)
                 + sliding * (args.prefill_tokens * seen(args.prefill_prefix_tokens)
                              + args.prefill_tokens * (seen(chunk) + 1) / 2))
        flops = (2 * params * args.prefill_tokens + 4 * heads * head_dim * pairs
                 + args.recurrent_flops_per_row * args.prefill_tokens)
        chunks = args.prefill_tokens / chunk
        # a prefill forward of its own reads the dense weights and the experts its chunk's tokens reach
        prefill_weights = dense_bytes + (expert_bytes * experts_reached(experts, per_token, chunk) / experts
                                         if expert_bytes else 0)
        prefill_bytes = (args.kv_bytes_per_token * (args.prefill_tokens + chunks * per_layer_tokens(args.prefill_prefix_tokens))
                         * per_gpu_kv + prefill_weights * args.prefill_passes_per_step)
        classes["prefill"] = entry(prefill_bytes, flops)
    return {"rows": rows, "expert_share_read": share if expert_bytes else None, "params": params,
            "tensors_counted": len(names), "classes": classes,
            "floor_ms": sum(c["floor_ms"] for c in classes.values())}


def mixed_floor(args, mix):
    """The mean over steps of each class's floor, each step at its own batch: mix is {batch: steps}."""
    total = sum(mix.values())
    out = {"rows": 0.0, "classes": {}, "floor_ms": 0.0, "expert_share_read": None, "batch_mix": mix}
    for batch, count in sorted(mix.items()):
        one = copy.copy(args)
        one.requests = batch
        result = floor(one)
        weight = count / total
        out["rows"] += weight * result["rows"]
        out["floor_ms"] += weight * result["floor_ms"]
        out["params"], out["tensors_counted"] = result["params"], result["tensors_counted"]
        if result["expert_share_read"] is not None:
            out["expert_share_read"] = (out["expert_share_read"] or 0.0) + weight * result["expert_share_read"]
        for name, c in result["classes"].items():
            acc = out["classes"].setdefault(name, {"bytes": 0.0, "flops": 0.0, "floor_ms": 0.0})
            for key in acc:
                acc[key] += weight * c[key]
    return out


def parse_mix(text):
    mix = {}
    for item in text.split(","):
        batch, steps = item.split(":")
        if int(batch) <= 0 or int(steps) <= 0:
            raise SystemExit(f"--batch-mix: {item} is not a batch and a positive step count")
        mix[int(batch)] = mix.get(int(batch), 0) + int(steps)
    return mix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--requests", type=int, default=None, help="the engine's batch: requests in the step")
    parser.add_argument("--batch-mix", default=None, help="N:STEPS per engine batch N, from decode_steps.py")
    parser.add_argument("--rows-per-request", type=int, required=True,
                        help="tokens the target model processes per request per step (1 without speculation)")
    parser.add_argument("--context-tokens", type=float, default=0, help="mean context length per request")
    parser.add_argument("--peak-tflops", type=float, required=True)
    parser.add_argument("--peak-tbps", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--kv-bytes-per-token", type=float, default=0, help="whole model, all layers and KV heads")
    parser.add_argument("--kv-heads", type=int, default=None, help="default: num_key_value_heads of config.json")
    parser.add_argument("--state-bytes-per-request", type=float, default=0)
    parser.add_argument("--prefill-tokens", type=float, default=0, help="prompt tokens processed per step, on average")
    parser.add_argument("--prefill-chunk", type=float, default=0, help="tokens per prefill chunk")
    parser.add_argument("--prefill-prefix-tokens", type=float, default=0,
                        help="cached prefix tokens a prefill chunk reads, on average")
    parser.add_argument("--within-step-pairs", type=float, default=None,
                        help="attention pairs among a request's own rows in a step: its tree's ancestor pairs")
    parser.add_argument("--prefill-passes-per-step", type=float, default=0,
                        help="prefill forwards of their own per step, on average (each reads the weights again)")
    parser.add_argument("--recurrent-flops-per-row", type=float, default=0,
                        help="FLOPs of the recurrent layers per processed token, from the model's layer shapes")
    parser.add_argument("--draft-rows-per-request", type=int, default=0,
                        help="rows the drafter processes per request per forward (default: --rows-per-request)")
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
    if (args.requests is None) == (args.batch_mix is None):
        parser.error("give the engine's batch with --requests or the steps per batch with --batch-mix")
    result = mixed_floor(args, parse_mix(args.batch_mix)) if args.batch_mix else floor(args)
    print(f"{result['rows']:g} rows per step, {result['tensors_counted']} weight tensors counted, "
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
