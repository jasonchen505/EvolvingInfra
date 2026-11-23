#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Paired comparison of a candidate with the current best system, and the promotion decision.

    python paired_ab.py --base ARM --cand ARM RUN_DIR [...] [--noise B:X[,B:X...]] --expect-pairs N
        --expect-batches B[,B...] [--groups REQUESTS.jsonl --group-field FIELD] [--json OUT]
    python paired_ab.py --noise-only --base ARM RUN_DIR [...] --expect-batches B[,B...]
        [--groups REQUESTS.jsonl --group-field FIELD]
    python paired_ab.py --sealed --base stock --cand ARM RUN_DIR [...] --expect-pairs 2
        --expect-batches B[,B...] [--groups REQUESTS.jsonl --group-field FIELD] [--json OUT]

SpecPilot (arXiv 2611.02501, sections 3.2 and 5.1). A pair is one run of each arm at the same time
on two slots (GPUs or GPU groups of one machine), serving the same requests at the same batch size
to the end; from one pair to the next the arms swap slots. Base-against-base pairs run the base arm
on both slots at once. Each RUN_DIR is a client run (fixed_batch_client.py): run.json gives the arm,
the slot (--node), the cycle (the pair's index), the batch size, the window, the request file's
sha256, the sampling and the cap; outputs.jsonl the requests.

The script refuses (exit 1, no decision) when the runs cannot support a decision: a cycle without
exactly one run on each of the two slots of the study, two runs of a pair that did not start
together (starts more than 5 % of the shorter run apart, or less than 95 % of the shorter run shared;
the faster arm may finish first), runs of a batch size that differ in request file, sampling, cap,
mode, task field, task copies or --logprobs, a run with fewer requests (or, with a task field, fewer
tasks) than its batch size, which cannot keep that many in flight, a run that did not finish every
request it was given, a request without its output tokens, two
runs of a pair with different requests, an arm that did not run equally often on each slot, a
missing group label, an arm with no runs, a batch size missing from --expect-batches or a number of
pairs other than --expect-pairs.

--noise-only computes the frozen noise of step 2 from base-against-base pairs alone (at least four
per batch size), with the groups the comparisons use: the root mean square of (gain - 1) and its
interval, per batch size, the gain of a pair being the run on the slot that sorts second over the
run on the other.

--sealed judges a sealed final (step 7): the final passes at a batch size when its gain over stock,
the geometric mean over the pairs, is above 1. It needs no noise.

Gain of a pair: the per-request output rate is the number of output tokens the engine returned
(the length of output_ids, not the count it reports) over the request's latency, from sending it to
its last token. The gain is the geometric mean of the per-request rate ratios
(candidate / base) within each group of requests (the paper's benchmarks), then the equal-weight
geometric mean over the groups.

Over the pairs at one batch size it prints every pair, the mean gain with its 95 % interval (t
distribution over the log gains) and per slot. Promote when the lower end of the interval exceeds
1 + threshold: 0.03 when the two arms' outputs differ on any request, else 0.01 + 2 x noise, with
noise the stock-against-stock noise of that batch size frozen in step 2 (--noise). The noise is
needed only at a batch size where the outputs are identical; there, without it, no decision.
A request of a task copy ("<id>#c<k>") takes the group of its original request. It also prints the combined gain, the equal-weight geometric mean
over batch sizes.

Exits 0 when the candidate is promoted (with --sealed: passes) at every batch size, 3 when it is not,
1 without a decision.
"""
import argparse
import collections
import json
import math
import os
import re
import statistics
import sys

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
SAME = ("sampling", "max_new_tokens_cap", "mode", "session_field", "logprobs")
PER_BATCH = ("requests_sha256", "session_copies", "n_requests")


def t95(dof):
    keys = [k for k in sorted(T95) if k <= dof]
    return T95[keys[-1]] if keys else T95[1]


def load_run(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        meta = json.load(f)
    requests = {}
    with open(os.path.join(run_dir, "outputs.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids, latency = r.get("output_ids"), r.get("latency_s")
            if not ids or not latency:
                raise ValueError(f"{run_dir}: request {r.get('id')} has no output tokens or latency")
            requests[str(r["id"])] = {"rate": len(ids) / latency, "output_ids": ids}
    run = {"dir": run_dir, "arm": meta["arm"], "slot": meta["node"], "cycle": meta["cycle"],
           "concurrency": meta["concurrency"], "t_start": meta.get("t_start"), "t_end": meta.get("t_end"),
           "failed": meta.get("n_failed", 0),
           "n_requests": meta.get("n_requests"),
           "tasks": meta.get("n_sessions") or meta.get("n_requests"),
           "same": {k: meta.get(k) for k in SAME}, "per_batch": {k: meta.get(k) for k in PER_BATCH},
           "requests": requests}
    return run


def load_groups(path, field):
    groups = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[str(r["id"])] = r.get(field)
    return groups


def group_of(groups, rid):
    """The group of a request; a task copy "<id>#c<k>" takes its original request's."""
    if rid in groups:
        return groups[rid]
    return groups.get(re.sub(r"#c\d+$", "", rid))


def pair_gain(base, cand, groups):
    """(gain, requests compared, outputs differ); raises ValueError when the runs finished different requests."""
    if set(base["requests"]) != set(cand["requests"]):
        missing = sorted(set(base["requests"]) ^ set(cand["requests"]))
        raise ValueError(f"{base['dir']} and {cand['dir']} finished different requests, e.g. {missing[:3]}")
    if not base["requests"]:
        raise ValueError(f"{base['dir']} and {cand['dir']} finished no request")
    by_group = collections.defaultdict(list)
    diverged = False
    for rid, b in base["requests"].items():
        c = cand["requests"][rid]
        group = "all"
        if groups is not None:
            group = group_of(groups, rid)
            if group is None:
                raise ValueError(f"request {rid} has no group")
        by_group[group].append(math.log(c["rate"] / b["rate"]))
        if b["output_ids"] != c["output_ids"]:
            diverged = True
    log_gain = statistics.mean(statistics.mean(v) for v in by_group.values())
    return math.exp(log_gain), len(base["requests"]), diverged


def simultaneous(a, b):
    """Started together, and the shorter run lies inside the other: the faster arm may finish first."""
    if None in (a["t_start"], a["t_end"], b["t_start"], b["t_end"]):
        return False
    shorter = min(a["t_end"] - a["t_start"], b["t_end"] - b["t_start"])
    shared = min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])
    return shorter > 0 and abs(a["t_start"] - b["t_start"]) <= 0.05 * shorter and shared >= 0.95 * shorter


def interval(log_gains):
    mean = statistics.mean(log_gains)
    if len(log_gains) < 2:
        return math.exp(mean), float("nan"), float("nan")
    half = t95(len(log_gains) - 1) * statistics.stdev(log_gains) / math.sqrt(len(log_gains))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def validate(runs, arms, expect_batches):
    """Checks shared by both modes; returns {(batch, cycle): [run, run]}."""
    runs = [r for r in runs if r["arm"] in arms]
    for arm in arms:
        if not any(r["arm"] == arm for r in runs):
            raise ValueError(f"no run of arm {arm!r}")
    slots = sorted({r["slot"] for r in runs})
    if len(slots) != 2:
        raise ValueError(f"the runs use slots {slots}; a study compares on exactly two slots")
    for r in runs:
        if r["same"] != runs[0]["same"]:
            raise ValueError(f"{r['dir']} differs from {runs[0]['dir']} in "
                             f"{[k for k in SAME if r['same'].get(k) != runs[0]['same'].get(k)]}")
        if r.get("tasks") is not None and r["tasks"] < r["concurrency"]:
            raise ValueError(f"{r['dir']}: {r['tasks']} requests or tasks at batch size {r['concurrency']}; "
                             "the run could not keep that many in flight")
        if r["failed"] or r["n_requests"] != len(r["requests"]):
            raise ValueError(f"{r['dir']}: {len(r['requests'])} of {r['n_requests']} requests finished "
                             f"({r['failed']} failed); a paired run serves all to the end")
    batches = sorted({r["concurrency"] for r in runs})
    if expect_batches is not None and batches != sorted(expect_batches):
        raise ValueError(f"batch sizes {batches}, expected {sorted(expect_batches)}")
    cycles = collections.defaultdict(list)
    for r in runs:
        same_batch = [x for x in runs if x["concurrency"] == r["concurrency"]]
        if r["per_batch"] != same_batch[0]["per_batch"]:
            raise ValueError(f"{r['dir']} differs from {same_batch[0]['dir']} in "
                             f"{[k for k in PER_BATCH if r['per_batch'][k] != same_batch[0]['per_batch'][k]]}")
        cycles[(r["concurrency"], r["cycle"])].append(r)
    for (b, cycle), members in cycles.items():
        if len(members) != 2 or members[0]["slot"] == members[1]["slot"]:
            raise ValueError(f"batch {b} cycle {cycle}: need one run on each of the two slots, got "
                             f"{[(r['arm'], r['slot']) for r in members]}")
        if not simultaneous(*members):
            raise ValueError(f"batch {b} cycle {cycle}: the two runs did not run at the same time")
    return cycles


def noise_only(runs, base, expect_batches=None, groups=None):
    by_cycle = collections.defaultdict(list)
    for r in runs:
        by_cycle[(r["concurrency"], r["cycle"])].append(r)
    base_pairs = [r for members in by_cycle.values() if all(m["arm"] == base for m in members) for r in members]
    if not base_pairs:
        raise ValueError(f"no cycle runs {base!r} on both slots")
    cycles = validate(base_pairs, (base,), expect_batches)
    result = {}
    for concurrency in sorted({key[0] for key in cycles}):
        gains = []
        for (b, _), members in sorted(cycles.items()):
            if b == concurrency:
                first, second = sorted(members, key=lambda r: r["slot"])
                gains.append(pair_gain(first, second, groups)[0])
        if len(gains) < 4:
            raise ValueError(f"batch {concurrency}: {len(gains)} base-against-base pairs, need at least 4")
        result[concurrency] = {"noise": math.sqrt(statistics.mean([(g - 1) ** 2 for g in gains])),
                               "pairs": len(gains), "interval": interval([math.log(g) for g in gains])}
    return result


def compare(runs, base, cand, groups=None, noise=None, expect_pairs=None, expect_batches=None, sealed=False):
    """Decision per batch size; raises ValueError when the data cannot support one."""
    cycles = validate(runs, (base, cand), expect_batches)
    decisions = []
    for concurrency in sorted({key[0] for key in cycles}):
        pairs = []
        for (b, cycle), members in sorted(cycles.items()):
            if b != concurrency:
                continue
            arms = sorted(r["arm"] for r in members)
            if set(arms) != {base, cand}:
                continue
            b_run = next(r for r in members if r["arm"] == base)
            c_run = next(r for r in members if r["arm"] == cand)
            gain, n, diverged = pair_gain(b_run, c_run, groups)
            pairs.append({"cycle": cycle, "base_slot": b_run["slot"], "cand_slot": c_run["slot"], "gain": gain,
                          "requests": n, "diverged": diverged})
        if len(pairs) < 2 or (expect_pairs is not None and len(pairs) != expect_pairs):
            raise ValueError(f"{cand} at batch {concurrency}: {len(pairs)} pairs, expected "
                             f"{expect_pairs if expect_pairs is not None else 'at least 2'}")
        on_slot = collections.Counter(p["cand_slot"] for p in pairs)
        if len(on_slot) < 2 or max(on_slot.values()) - min(on_slot.values()) > 1:
            raise ValueError(f"{cand} at batch {concurrency}: the arms did not swap slots evenly ({dict(on_slot)})")
        gain, low, high = interval([math.log(p["gain"]) for p in pairs])
        diverged = any(p["diverged"] for p in pairs)
        if not sealed and not diverged and concurrency not in (noise or {}):
            raise ValueError(f"batch {concurrency}: the outputs are identical, so the threshold needs the "
                             "stock-against-stock noise of this batch size (step 2): pass --noise")
        batch_noise = None if sealed or diverged else noise[concurrency]
        threshold = None if sealed else 0.03 if diverged else 0.01 + 2 * batch_noise
        per_slot = {slot: math.exp(statistics.mean(math.log(p["gain"]) for p in pairs if p["cand_slot"] == slot))
                    for slot in sorted(on_slot)}
        decisions.append({"cand": cand, "base": base, "concurrency": concurrency, "pairs": len(pairs), "gain": gain,
                          "low": low, "high": high, "per_slot": per_slot, "noise": batch_noise,
                          "diverged": diverged, "threshold": threshold, "sealed": sealed,
                          "promote": gain > 1 if sealed else low > 1 + threshold,
                          "per_pair": pairs})
    return decisions


def parse_noise(text):
    noise = {}
    for item in text.split(","):
        batch, value = item.split(":")
        noise[int(batch)] = float(value)
    return noise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--base", required=True, help="the current best system's arm (stock with --noise-only)")
    parser.add_argument("--cand", default=None)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--sealed", action="store_true", help="judge a sealed final: faster than stock at every batch size")
    parser.add_argument("--noise", default=None,
                        help="B:X per batch size, the noise frozen in step 2; needed where outputs are identical")
    parser.add_argument("--expect-pairs", type=int, default=None)
    parser.add_argument("--expect-batches", default=None, help="the batch sizes of the study, comma-separated")
    parser.add_argument("--groups", default=None, help="request file giving each request's group")
    parser.add_argument("--group-field", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    groups = load_groups(args.groups, args.group_field) if args.groups and args.group_field else None
    batches = [int(b) for b in args.expect_batches.split(",")] if args.expect_batches else None
    try:
        runs = [load_run(d) for d in args.runs]
        if args.noise_only:
            result = noise_only(runs, args.base, batches, groups)
            for b, n in result.items():
                print(f"batch {b}: noise {n['noise']:.4f} over {n['pairs']} pairs, base against base "
                      f"{n['interval'][0]:.4f} [{n['interval'][1]:.4f}, {n['interval'][2]:.4f}]")
            print("--noise " + ",".join(f"{b}:{n['noise']:.5f}" for b, n in result.items()))
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(result, f, indent=1)
            return 0
        if not args.cand:
            raise ValueError("--cand is required")
        decisions = compare(runs, args.base, args.cand, groups, parse_noise(args.noise) if args.noise else None,
                            args.expect_pairs, batches, args.sealed)
    except ValueError as error:
        print(f"no decision: {error}", file=sys.stderr)
        return 1
    for d in decisions:
        print(f"{d['cand']} vs {d['base']}, batch {d['concurrency']}, {d['pairs']} pairs")
        for p in d["per_pair"]:
            print(f"  cycle {p['cycle']}: {d['cand']} on {p['cand_slot']}, {d['base']} on {p['base_slot']}: "
                  f"gain {p['gain']:.4f} over {p['requests']} requests" + (", outputs differ" if p["diverged"] else ""))
        print(f"  gain {d['gain']:.4f} [{d['low']:.4f}, {d['high']:.4f}]; per slot of the candidate: "
              + ", ".join(f"{s} {g:.4f}" for s, g in d["per_slot"].items()))
        if d["sealed"]:
            print(f"  faster than {d['base']}: {'yes' if d['promote'] else 'no'} (gain {d['gain']:.4f})")
            continue
        print(f"  threshold {d['threshold']:.4f}"
              + (" (outputs differ)" if d["diverged"] else f" (frozen noise {d['noise']:.4f})"))
        print(f"  promote: {'yes' if d['promote'] else 'no'} (lower end {d['low']:.4f} against {1 + d['threshold']:.4f})")
    if len(decisions) > 1:
        combined = math.exp(statistics.mean(math.log(d["gain"]) for d in decisions))
        print(f"combined gain over {len(decisions)} batch sizes: {combined:.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(decisions, f, indent=1)
    return 0 if all(d["promote"] for d in decisions) else 3


if __name__ == "__main__":
    raise SystemExit(main())
