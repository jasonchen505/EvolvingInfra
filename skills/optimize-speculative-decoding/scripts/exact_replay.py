#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Exact replay of draft trees on recorded drafter lattices: tokens per step against the node budget.

    python exact_replay.py LATTICES.jsonl [...] --outputs OUTPUTS.jsonl [...]
        [--budgets 8,16,24,32,48,64] [--costs 8:14.3,16:15.1,...] [--walk] [--json OUT]

Exact replay reruns tree selection and verification offline (SpecPilot, arXiv 2611.02501, sections
3.1 and 6.4): it measures a tree on recorded lattices without timing noise.

LATTICES.jsonl holds one record per request per drafting step, written by an engine hook (off by
default) that the study adds when the engine has none:
  {"id": the request id the client sent (rid), "pos": committed output tokens before the step,
   "cands": [[token ids], ...],      one list per drafted position: the drafter's top candidates
   "logp": [[scores], ...],          the drafter's score of each candidate
   "pair": [root, [[...]], ...],     optional: pair[s][i][j] scores candidate j at position s given
                                     candidate i at position s - 1 (a DFlash2 selector, a DSpark
                                     Markov head); pair[0] is one row (a list of scores, or a list
                                     holding that one list), the scores given the committed anchor
                                     token, or null
   "accept_prob": [...]}             optional: the probability that the engine's own verify accepts
                                     the drafted chain's token at each position, given the ones before
OUTPUTS.jsonl is the outputs.jsonl of the run that recorded the lattices: its committed tokens are
the target's own (greedy: its argmax up to near-ties; under sampling: samples of its distribution).
The target's tokens after a step are output_ids[pos:].

For each step it builds the best-first tree: nodes in order of path score, the sum of edge scores
from the root, where an edge scores log_softmax over its parent's children (of the pair row of the
parent when the lattice has pair scores, else of the position's own scores). Edge scores are at most
0, so a parent precedes its children and every prefix of the order is a tree: the tree of budget N
is its first N nodes (N proposed tokens; the engine's verify adds a row for the committed anchor).
A tree accepts the longest root path whose tokens equal the target's, plus one token when the
output continues. Under sampling this is a verify that samples the target's token at each node and
continues only through the child holding it, which keeps the target's distribution; it is not the
rejection sampling of a single chain, whose expected tokens per step the record's accept_prob gives
(chain_verify). Also reported:
- chain: the greedy path through the lattice (the best child at every position);
- oracle: keep accepting while the target's token is among the position's candidates, the most any
  tree on this lattice can accept.
A step whose lattice drafts nothing still commits the target's token: one token for every policy.
By default every budget is replayed at the recorded steps (the recording run's step starts). With
--walk, when lattices exist at the positions a policy reaches (a recorder that drafts at every
position of the target's tokens), each policy walks its own trajectory through each request; the
coverage line says how many requests it walked to the end. Validate a chosen tree live: its live
tokens per step must agree with the replay within the run's noise before it is trusted.
With --costs N:t, where t is the full step time in milliseconds (drafting, tree construction, verify
and commit) with a tree of N nodes per request at the batch size, it prints tau(N) / t(N), the tokens
each request commits per millisecond, and the best budget; tau_best is tau at that budget.
"""
import argparse
import collections
import heapq
import json
import math
import statistics
import sys


def log_softmax(scores):
    top = max(scores)
    total = math.log(sum(math.exp(s - top) for s in scores)) + top
    return [s - total for s in scores]


def edge_table(step):
    """edges[s][i] = log_softmax of the children's scores at position s given candidate i at s - 1
    (i = 0 for the root's children, and for every parent when the lattice has no pair scores)."""
    cands, logp, pair = step["cands"], step["logp"], step.get("pair")
    table = []
    for s in range(len(cands)):
        if pair and s < len(pair) and pair[s] is not None:
            rows = pair[s] if pair[s] and isinstance(pair[s][0], list) else [pair[s]]
            table.append([log_softmax(row) for row in rows])
        else:
            table.append([log_softmax(logp[s])])
    return table


def best_first(step, max_nodes):
    """(children, ranks): children[(parent, token)] = node, and node -> insertion rank (0-based)."""
    cands = step["cands"]
    edges = edge_table(step)
    depth = len(cands)
    nodes = []  # (position, candidate index, parent node or -1)
    children = {}
    heap = [(-score, 0, j, -1) for j, score in enumerate(edges[0][0])] if depth else []
    heapq.heapify(heap)
    while heap and len(nodes) < max_nodes:
        neg, s, j, parent = heapq.heappop(heap)
        node = len(nodes)
        nodes.append((s, j, parent))
        children[(parent, cands[s][j])] = node
        if s + 1 < depth:
            row = edges[s + 1][j] if len(edges[s + 1]) > 1 else edges[s + 1][0]
            for k, score in enumerate(row):
                heapq.heappush(heap, (neg - score, s + 1, k, node))
    return children


def greedy_chain(step):
    edges = edge_table(step)
    chain, prev = [], 0
    for s in range(len(step["cands"])):
        row = edges[s][prev] if len(edges[s]) > 1 else edges[s][0]
        j = max(range(len(row)), key=row.__getitem__)
        chain.append(step["cands"][s][j])
        prev = j
    return chain


def committed(accepted, remaining):
    """Accepted draft tokens plus the target's token after them, when the output continues."""
    return min(accepted + 1, remaining)


def replay_step(step, target, budgets):
    """Tokens this step commits: {"chain", "oracle", N: ...}; target is the target's tokens from pos on."""
    if not target:
        return None
    depth = min(len(step["cands"]), len(target))
    result = {}
    chain = greedy_chain(step)
    accepted = 0
    while accepted < depth and chain[accepted] == target[accepted]:
        accepted += 1
    result["chain"] = committed(accepted, len(target))
    accepted = 0
    while accepted < depth and target[accepted] in step["cands"][accepted]:
        accepted += 1
    result["oracle"] = committed(accepted, len(target))
    children = best_first(step, max(budgets))
    path_ranks, parent = [], -1
    for s in range(depth):
        node = children.get((parent, target[s]))
        if node is None:
            break
        path_ranks.append(node)  # insertion order: a node's rank is its index
        parent = node
    for n in budgets:
        result[n] = committed(sum(1 for rank in path_ranks if rank < n), len(target))
    probs = step.get("accept_prob")
    if probs:
        expected, alive = 1.0, 1.0
        for k in range(min(len(probs), len(target) - 1)):
            alive *= probs[k]
            expected += alive
        result["chain_verify"] = expected
    return result


def load_outputs(paths):
    outputs = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("output_ids") and not r.get("aborted"):
                        outputs[str(r["id"])] = r["output_ids"]
    return outputs


def load_lattices(paths):
    steps, seen = [], set()
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                step = json.loads(line)
                key = (str(step["id"]), int(step["pos"]))
                if key in seen:
                    raise ValueError(f"two lattices for request {key[0]} at position {key[1]}")
                seen.add(key)
                steps.append(step)
    return steps


def replay(steps, outputs, budgets):
    """Mean tokens per step at the recorded steps."""
    sums = collections.defaultdict(list)
    skipped = 0
    for step in steps:
        target = outputs.get(str(step["id"]))
        result = replay_step(step, target[int(step["pos"]):], budgets) if target is not None else None
        if result is None:
            skipped += 1
            continue
        for key, value in result.items():
            sums[key].append(value)
    return {"steps": len(sums["chain"]), "skipped": skipped,
            "tau": {str(k): statistics.mean(v) for k, v in sums.items() if v}}


def walk(steps, outputs, budgets):
    """Each policy's own trajectory from the request's first drafting step; the means cover the requests
    that every policy walked to the end."""
    by_pos = {(str(s["id"]), int(s["pos"])): s for s in steps}
    first = {}
    for rid, pos in by_pos:
        first[rid] = min(pos, first.get(rid, pos))
    policies = ["chain", "oracle", *budgets]
    commits = {str(p): {} for p in policies}
    for rid, target in outputs.items():
        if rid not in first:
            continue
        for policy in policies:
            pos, seen = first[rid], []
            while pos < len(target) and (rid, pos) in by_pos:
                n = replay_step(by_pos[(rid, pos)], target[pos:], budgets)[policy]
                seen.append(n)
                pos += n
            if seen and pos >= len(target):
                commits[str(policy)][rid] = seen
    common = set.intersection(*(set(c) for c in commits.values())) if commits else set()
    tau = {p: statistics.mean(n for rid in common for n in c[rid]) if common else None for p, c in commits.items()}
    return {"tau": tau, "requests_walked": len(common), "requests": len(outputs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("lattices", nargs="+")
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--budgets", default="8,16,24,32,48,64")
    parser.add_argument("--costs", default=None, help="N:t pairs, t = the full step time in ms with N nodes")
    parser.add_argument("--walk", action="store_true", help="walk each policy's own trajectory")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    budgets = sorted({int(x) for x in args.budgets.split(",")})
    try:
        steps = load_lattices(args.lattices)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    outputs = load_outputs(args.outputs)
    result = replay(steps, outputs, budgets)
    if not result["steps"]:
        print(f"no step had the target's tokens ({result['skipped']} skipped)", file=sys.stderr)
        return 1
    if args.walk:
        walked = walk(steps, outputs, budgets)
        result["walk"] = walked
        if not walked["requests_walked"]:
            print("--walk: no request has lattices along every policy's whole trajectory: inconclusive",
                  file=sys.stderr)
            return 1
        tau = walked["tau"]
        print(f"walked trajectories: {walked['requests_walked']} of {walked['requests']} requests, by every policy")
    else:
        tau = result["tau"]
        print(f"{result['steps']} recorded steps ({result['skipped']} skipped)")
    print(f"chain {tau['chain']:.3f}, oracle {tau['oracle']:.3f} tokens per step"
          + (f", the engine's chain verify {result['tau']['chain_verify']:.3f}" if "chain_verify" in result["tau"] else ""))
    costs = {}
    if args.costs:
        for item in args.costs.split(","):
            n, c = item.split(":")
            if float(c) <= 0:
                print(f"--costs: {item} is not a positive cost", file=sys.stderr)
                return 1
            costs[int(n)] = float(c)
        missing = [n for n in budgets if n not in costs]
        if missing:
            print(f"--costs has no cost for budgets {missing}; every ranked budget needs its measured cost",
                  file=sys.stderr)
            return 1
    best = None
    for n in budgets:
        if str(n) not in tau:
            continue
        line = f"  best-first tree, {n:3d} nodes: tau {tau[str(n)]:.3f}"
        if n in costs:
            s = tau[str(n)] / costs[n]
            line += f", step {costs[n]:.3f} ms, {s:.4f} tokens per ms"
            if best is None or s > best[1]:
                best = (n, s)
        print(line)
    if best:
        result["best_budget"], result["best_tokens_per_ms"] = best
        result["tau_best"] = tau[str(best[0])]
        print(f"tau_best {result['tau_best']:.3f} at {best[0]} nodes ({best[1]:.4f} tokens per ms)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
