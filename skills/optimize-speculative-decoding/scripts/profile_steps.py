#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""GPU time per decode step, by kernel class, from torch profiler traces of one GPU.

    python profile_steps.py TRACE.json[.gz] [...] [--classes CLASSES.json] [--drop-factor 2.5] [--top 15]
    python profile_steps.py TRACE.json[.gz] [...] --steps N  # when the traces have no CUDA graph launches

The engine writes a trace with POST /start_profile {"output_dir": ..., "num_steps": N,
"activities": ["CPU", "GPU"]}, one file per GPU. During a run the engine's batch varies, so profile
windows spread through the run (for example a quarter, half and three quarters of the way) and pass
one GPU's traces of all of them: their steps are pooled. Steps are cut at the start of the target model's
exact pass: with --anchor-kernel REGEX at every kernel whose name matches (name the first kernel of
the exact pass when a step holds several forwards, such as a low-precision pre-check); otherwise at
the launches of the CUDA graph with the most kernels. A step runs until the next cut, so it holds the
drafting and everything else between two exact passes; a kernel that crosses a cut is split between
the two steps. Steps longer than --drop-factor times the trace's median are left out: they hold a
prefill, or the profiler's own start and stop; a decode step of the run's largest batch stays below
it. --steps N instead divides each trace into N steps' worth of time.

Per step: wall time, GPU busy time (the union of kernel, copy and memset intervals) and the busy
time of each kernel class, averaged over the pooled steps (T_prof is the mean busy time). A kernel belongs to the first class whose regular expression matches
its name; --classes takes a JSON list of [class, regex] pairs to replace the defaults below. Check
the top kernels list against the classes: kernel names differ between models and GPUs.
"""
import argparse
import bisect
import collections
import gzip
import json
import re
import statistics
import sys

CLASSES = [
    ["communication", r"nccl|all_?reduce|AllReduce|all_?gather|AllGather|reduce_?scatter|Lamport|cross_device|"
                      r"one_?shot|two_?shot|custom_ar"],
    ["attention", r"attn|Attn|attention|Attention|fmha|Fmha|flash_fwd|flashmla|FlashMLA|[^a-z]mla|MLA|PagedKV|"
                  r"paged_|decode_kernel|_fwd_kernel|_fwd_grouped|sparse|delta_rule|gated_delta|gdn|GDN|mamba|"
                  r"Mamba|ssm_|selective_scan|conv1d|causal_conv|recurrent|chunk_"],
    ["moe", r"moe|Moe|MoE|expert|Expert|topk_softmax|routing|permute"],
    ["gemm", r"gemm|Gemm|GEMM|nvjet|cublas|matmul|Matmul|bmm|xmma|splitK"],
    ["sampling_and_drafting", r"sampl|softmax|Softmax|argmax|top_?k|top_?p|verify|accept|tree|draft|eagle|dflash|"
                              r"RadixSort|radix_sort|DeviceScan|murmur|[Gg]umbel|multinomial"],
    ["norm_activation_quant", r"norm|Norm|silu|gelu|act_and_mul|rope|Rope|rotary|quant|Quant|cast|elementwise|"
                              r"vectorized|reduce"],
    ["copy_and_index", r"memcpy|Memcpy|memset|Memset|copy|Copy|cat|index|Index|gather|scatter|fill"],
    ["other", r"."],
]
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def union(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def boundaries(events):
    """Start times of the launches of the graph with the most kernels (the target forward)."""
    launches = {e["args"]["correlation"] for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in e.get("name", "")
                and "correlation" in e.get("args", {})}
    kernels = collections.defaultdict(list)
    for e in events:
        if e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in launches:
            kernels[e["args"]["correlation"]].append(e["ts"])
    if not kernels:
        return []
    most = max(len(v) for v in kernels.values())
    return sorted(min(v) for v in kernels.values() if len(v) >= 0.9 * most)


def classify(name, classes):
    for label, pattern in classes:
        if pattern.search(name):
            return label
    return "other"


def anchor_cuts(events, pattern):
    return sorted(e["ts"] for e in events if e.get("cat") == "kernel" and pattern.search(e.get("name", "")))


def steps(events, classes, cuts, drop_factor):
    gpu = sorted((e for e in events if e.get("cat") in GPU and "dur" in e), key=lambda e: e["ts"])
    starts = [e["ts"] for e in gpu]
    longest = max((e["dur"] for e in gpu), default=0)
    rows = []
    for start, stop in zip(cuts, cuts[1:]):
        inside = []
        for e in gpu[bisect.bisect_left(starts, start - longest):bisect.bisect_left(starts, stop)]:
            lo, hi = max(e["ts"], start), min(e["ts"] + e["dur"], stop)
            if hi > lo:
                inside.append((e, lo, hi))
        by_class = collections.defaultdict(list)
        names = collections.Counter()
        for e, lo, hi in inside:
            by_class[classify(e["name"], classes)].append((lo, hi))
            names[e["name"]] += hi - lo
        busy = union([i for v in by_class.values() for i in v])
        per_class = {c: union(v) for c, v in by_class.items()}
        rows.append({"wall": stop - start, "busy": busy, "classes": per_class,
                     "overlap": sum(per_class.values()) - busy, "names": names})
    if not rows:
        return [], 0
    median = statistics.median(r["wall"] for r in rows)
    kept = [r for r in rows if r["wall"] <= drop_factor * median]
    return kept, len(rows) - len(kept)


def summarize(events, classes, drop_factor=2.5, n_steps=None, anchor=None):
    cuts = anchor_cuts(events, anchor) if anchor is not None else ([] if n_steps else boundaries(events))
    if len(cuts) >= 3:
        rows, dropped = steps(events, classes, cuts, drop_factor)
        count = len(rows)
    else:
        if not n_steps:
            raise SystemExit("no CUDA graph launches found: pass --steps N (the decode steps the trace covers)")
        gpu = [e for e in events if e.get("cat") in GPU and "dur" in e]
        whole, _ = steps(events, classes, [min(e["ts"] for e in gpu), max(e["ts"] + e["dur"] for e in gpu)],
                         float("inf"))
        row = whole[0]
        rows = [{"wall": row["wall"] / n_steps, "busy": row["busy"] / n_steps,
                 "classes": {c: v / n_steps for c, v in row["classes"].items()},
                 "overlap": row["overlap"] / n_steps,
                 "names": collections.Counter({k: v / n_steps for k, v in row["names"].items()})}]
        dropped, count = 0, n_steps
    names = collections.Counter()
    for r in rows:
        names.update(r["names"])
    labels = sorted({c for r in rows for c in r["classes"]})
    return {
        "steps": count, "dropped": dropped,
        "wall_us_median": statistics.median(r["wall"] for r in rows),
        "wall_us_mean": statistics.mean(r["wall"] for r in rows),
        "busy_us_median": statistics.median(r["busy"] for r in rows),
        "busy_us_mean": statistics.mean(r["busy"] for r in rows),
        "classes_us_mean": {c: statistics.mean(r["classes"].get(c, 0.0) for r in rows) for c in labels},
        "overlap_us_mean": statistics.mean(r.get("overlap", 0.0) for r in rows),
        "top": [(n, t / len(rows)) for n, t in names.most_common()],
    }


def pool(summaries):
    """The means over the steps of several traces (windows of one run), each trace weighted by its steps."""
    steps = sum(s["steps"] for s in summaries)

    def mean(key):
        return sum(s[key] * s["steps"] for s in summaries) / steps

    labels = sorted({c for s in summaries for c in s["classes_us_mean"]})
    top = collections.Counter()
    for s in summaries:
        for name, us in s["top"]:
            top[name] += us * s["steps"]
    return {"steps": steps, "dropped": sum(s["dropped"] for s in summaries), "traces": len(summaries),
            "wall_us_mean": mean("wall_us_mean"), "busy_us_mean": mean("busy_us_mean"),
            "classes_us_mean": {c: sum(s["classes_us_mean"].get(c, 0.0) * s["steps"] for s in summaries) / steps
                                for c in labels},
            "overlap_us_mean": mean("overlap_us_mean"),
            "top": [(n, t / steps) for n, t in top.most_common()]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--drop-factor", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--anchor-kernel", default=None, help="regex of the first kernel of the exact pass")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    table = CLASSES
    if args.classes:
        with open(args.classes) as f:
            table = json.load(f)
    classes = [(label, re.compile(pattern)) for label, pattern in table]
    anchor = re.compile(args.anchor_kernel) if args.anchor_kernel else None
    s = pool([summarize(load(trace), classes, args.drop_factor, args.steps, anchor) for trace in args.traces])
    busy = s["busy_us_mean"]
    print(f"{s['steps']} steps from {s['traces']} traces ({s['dropped']} left out), wall {s['wall_us_mean'] / 1e3:.3f} ms "
          f"mean, GPU busy {busy / 1e3:.3f} ms mean, idle {100 * (1 - busy / s['wall_us_mean']):.1f} %")
    for label, us in sorted(s["classes_us_mean"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:24s} {us / 1e3:8.3f} ms  {100 * us / busy:5.1f} % of busy")
    print(f"  classes overlap by {s['overlap_us_mean'] / 1e3:.3f} ms per step (kernels on concurrent streams)")
    print("top kernels (ms per step, class):")
    for name, us in s["top"][: args.top]:
        print(f"  {us / 1e3:8.3f}  {classify(name, classes):22s} {name[:110]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
