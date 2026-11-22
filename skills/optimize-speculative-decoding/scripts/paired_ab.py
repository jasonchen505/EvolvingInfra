#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Paired comparison of decode speed between a base arm and candidate arms, and the promotion decision.

    python paired_ab.py --base ARM --cand ARM [--cand ARM ...] RUN_DIR [RUN_DIR ...] [--json OUT]

Each RUN_DIR is a client run (fixed_batch_client.py): run.json gives its arm, node, cycle,
concurrency, wall-clock window and the engine's trace directory. Speed is committed tokens per
second over the run's plain decode steps (decode_steps.py). A pair is a base run and a candidate
run with the same node, cycle and concurrency; runs of both arms alternate within a job, so a pair
shares the GPU and the time.

Per candidate and concurrency it prints every pair's speed ratio (candidate / base), the mean
ratio with its 95 % interval (t distribution over pairs), the mean ratio per node, and the split of
the ratio into the tau ratio and the step-time ratio (base step time / candidate step time). Noise
is the root mean
square of (ratio - 1) over base-against-base pairs: consecutive cycles of the base arm on one
node. A candidate is promoted when the lower end of its interval exceeds 1 + (0.01 + 2 * noise).

Exits 1 without a decision when a run has no plain steps, a run has no partner, there are fewer
than two pairs, or no base-against-base pair exists.
"""
import argparse
import collections
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_steps  # noqa: E402

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t95(dof):
    keys = [k for k in sorted(T95) if k <= dof]
    return T95[keys[-1]] if keys else T95[1]


def load_run(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        meta = json.load(f)
    trace = meta.get("trace_dir") or os.path.join(run_dir, "trace")
    if not os.path.isabs(trace):
        trace = os.path.join(run_dir, trace)
    stats = decode_steps.measure(trace, meta["concurrency"], meta.get("t_start"), meta.get("t_end"))
    return {"dir": run_dir, "arm": meta["arm"], "node": meta["node"], "cycle": meta["cycle"],
            "concurrency": meta["concurrency"], **{k: stats.get(k) for k in ("steps", "tau", "step_ms_mean",
                                                                            "tokens_per_s")}}


def interval(ratios):
    mean = statistics.mean(ratios)
    if len(ratios) < 2:
        return mean, float("nan"), float("nan")
    half = t95(len(ratios) - 1) * statistics.stdev(ratios) / math.sqrt(len(ratios))
    return mean, mean - half, mean + half


def compare(runs, base, cands):
    """Decision per (candidate, concurrency); raises ValueError when the data cannot support one."""
    empty = [r["dir"] for r in runs if not r["steps"]]
    if empty:
        raise ValueError(f"runs without plain decode steps: {empty}")
    by_key = {}
    for r in runs:
        key = (r["arm"], r["node"], r["cycle"], r["concurrency"])
        if key in by_key:
            raise ValueError(f"two runs share arm, node, cycle and concurrency: {by_key[key]['dir']}, {r['dir']}")
        by_key[key] = r
    unpaired = [r["dir"] for r in runs if r["arm"] != base and (base, r["node"], r["cycle"], r["concurrency"])
                not in by_key]
    unpaired += [r["dir"] for r in runs if r["arm"] == base and not any(
        (c, r["node"], r["cycle"], r["concurrency"]) in by_key for c in cands)]
    if unpaired:
        raise ValueError(f"runs without a partner of the other arm on the same node and cycle: {unpaired}")
    decisions = []
    for concurrency in sorted({r["concurrency"] for r in runs}):
        bases = collections.defaultdict(list)
        for r in runs:
            if r["arm"] == base and r["concurrency"] == concurrency:
                bases[r["node"]].append(r)
        aa = []
        for node_runs in bases.values():
            node_runs.sort(key=lambda r: r["cycle"])
            aa += [b["tokens_per_s"] / a["tokens_per_s"] for a, b in zip(node_runs, node_runs[1:])]
        if not aa:
            raise ValueError(f"batch {concurrency}: no base-against-base pair (two base cycles on one node)")
        noise = math.sqrt(statistics.mean([(x - 1) ** 2 for x in aa]))
        for cand in cands:
            pairs = [(by_key[(base, n, c, b)], r) for (a, n, c, b), r in by_key.items()
                     if a == cand and b == concurrency]
            if len(pairs) < 2:
                raise ValueError(f"{cand} at batch {concurrency}: {len(pairs)} pairs, need at least 2")
            ratios = [c["tokens_per_s"] / b["tokens_per_s"] for b, c in pairs]
            mean, low, high = interval(ratios)
            threshold = 0.01 + 2 * noise
            decisions.append({
                "cand": cand, "concurrency": concurrency, "pairs": len(pairs), "ratio": mean, "low": low,
                "high": high, "tau_ratio": statistics.mean(c["tau"] / b["tau"] for b, c in pairs),
                "step_ratio": statistics.mean(b["step_ms_mean"] / c["step_ms_mean"] for b, c in pairs),
                "noise": noise, "aa_pairs": len(aa), "aa_interval": interval(aa) if len(aa) > 1 else None,
                "threshold": threshold, "promote": low - 1 > threshold,
                "per_pair": [{"node": b["node"], "cycle": b["cycle"], "ratio": c["tokens_per_s"] / b["tokens_per_s"],
                              "base_tau": b["tau"], "cand_tau": c["tau"], "base_step_ms": b["step_ms_mean"],
                              "cand_step_ms": c["step_ms_mean"]} for b, c in pairs],
                "per_node": {node: statistics.mean(c["tokens_per_s"] / b["tokens_per_s"] for b, c in pairs
                                                   if b["node"] == node)
                             for node in sorted({b["node"] for b, _ in pairs})},
            })
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cand", action="append", required=True)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    runs = [load_run(d) for d in args.runs]
    try:
        decisions = compare(runs, args.base, args.cand)
    except ValueError as error:
        print(f"no decision: {error}", file=sys.stderr)
        return 1
    for d in decisions:
        print(f"{d['cand']} vs {args.base}, batch {d['concurrency']}, {d['pairs']} pairs")
        for p in d["per_pair"]:
            print(f"  {p['node']} cycle {p['cycle']}: ratio {p['ratio']:.4f}  tau {p['base_tau']:.3f} -> "
                  f"{p['cand_tau']:.3f}  step {p['base_step_ms']:.3f} -> {p['cand_step_ms']:.3f} ms")
        aa = d["aa_interval"]
        aa_text = f", base-against-base mean {aa[0]:.4f} [{aa[1]:.4f}, {aa[2]:.4f}]" if aa else ""
        print(f"  ratio {d['ratio']:.4f} [{d['low']:.4f}, {d['high']:.4f}] = tau x{d['tau_ratio']:.4f} "
              f"* step x{d['step_ratio']:.4f}")
        print("  per node: " + ", ".join(f"{n} {r:.4f}" for n, r in d["per_node"].items()))
        print(f"  noise {d['noise']:.4f} over {d['aa_pairs']} base-against-base pairs{aa_text}")
        print(f"  promote: {'yes' if d['promote'] else 'no'} (lower end - 1 = {d['low'] - 1:+.4f}, "
              f"threshold {d['threshold']:.4f})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(decisions, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
