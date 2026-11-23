#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Token forcing: is a system's output as faithful to the target model as stock's?

    python token_forcing.py calibrate --ar AR_RUN [...] --stock STOCK_RUN [...] --scored STOCK_SCORED [...]
    python token_forcing.py check --base STOCK_SCORED [...] --cand SCORED [...] --delta D
        --rescored SCORED [...] --rescored-base SCORED [...] [--beyond N] [--audit AUDIT.jsonl]
    python token_forcing.py probability --base STOCK_SCORED [...] --cand SCORED [...] [--beyond N]
    python token_forcing.py rules --tests TESTS.jsonl

SpecPilot (arXiv 2611.02501, sections 2.1, 3.3 and 6.2). The scorer feeds a system's outputs back
through the target, teacher-forced, on the stock engine without speculation and with an FP32 output
layer (fixed_batch_client.py --score-from). Each SCORED directory holds scores.jsonl (per request the
output tokens, the scorer's first choice and the log-probabilities; with --score-top the K most likely
tokens) and run.json (the scorer's /server_info and the scored run). RUN directories hold
outputs.jsonl. --base and --cand list scored runs in matching order: the first base run pairs with the
first candidate run, and so on. calibrate, check and probability refuse (exit 1) a scorer with
speculative decoding or without the FP32 output layer, scorers, sampling or K that differ between the
scored runs of both sides, a scored run whose source run did not finish every request, a request of
the source run without a score, a request scored only in part or whose scored tokens are not its
run's output, and a base and candidate with different requests.

calibrate (greedy decoding): at each request's first token where stock's output departs from AR's
(the target without speculation), the margin is the scorer's log-probability gap between the two
tokens, read from the top list there (score stock with --score-top). delta is the largest margin
rounded up to a multiple of 1/8. Exits 1 when a departure lacks either token or none exists.

check (greedy decoding). A token is a mismatch when it is not the scorer's first choice; its margin
is lp_top1 - lp_tok. Paired by request, with an interval that resamples requests:
- rate rule: the candidate fails when its mismatch rate exceeds stock's by more than 0.5 points at
  95 % confidence (lower end of the interval above 0.005); it passes when the upper end is at most
  0.005; in between it is inconclusive and does not pass: score more requests.
- event rule: large-margin events are mismatches with margin above 2 x delta that the audit does not
  explain; it fails when a one-sided binomial test of the event counts given the positions finds the
  candidate's more frequent (p < 0.05). An event is explained when stock has the same one (same
  request, position, prefix and token), or when the same outputs scored with another prefill chunk
  size (--rescored for the candidate, --rescored-base for stock) have another first choice there.
  The rescored runs list one run per scored run, in the same order: the same source run, scored on a
  scorer that differs only in --chunked-prefill-size.
--beyond N keeps only positions >= N: the long-output check of the sealed final.

probability (lossless.md: the probability check, and the check of stock against AR). Each
scored run's source run was generated at temperature 1 with fixed_batch_client.py --logprobs K and
scored with --score-top K, so at every output position both the system's own distribution (from the
forward pass that produced the token) and the scorer's are untempered. d is a lower bound of their
total variation distance from the two top-K lists. A token missing from a list has at most that
list's smallest listed probability, and the missing tokens together at most its unlisted mass
(1 minus the listed probability). With C the tokens in both lists and A the tokens only the system
lists:
  up = sum over C of max(0, p_sys - p_scorer) + max(0, p_sys(A) - min(scorer's unlisted mass,
       |A| x the scorer's smallest listed probability))
down is the same with the two sides swapped, and d = max(up, down). With the same tokens in both lists
d is the distance over them with every other token merged into one category. At every position the
scorer's K tokens must hold every token the
decoding rule can output: K >= top_k when top_k > 0, else probability at least top_p (0.999 when top_p
is 1). Stock's d measure rounding. At each batch size, with at least 10 000 positions on each side,
the candidate fails under either rule:
- level rule: at the 50th, 90th, 99th or 99.9th percentile its d exceeds max(2 x stock's d, 1e-4);
- event rule: it has positions with d above max(2 x the largest d of stock, 1e-4), and a one-sided
  Fisher exact test against stock's count there (zero) gives p < 0.05.
--beyond N keeps only positions >= N.

rules (a patch that changes the sampling code; lossless.md, the rule check). TESTS.jsonl holds
one line per test case and position of the chain (node of a tree), written by the patch's GPU test:
{"case", "kind", "position", "expected": {token: probability}, "counts": {token: count},
"ranks_agree": true or false}, and "control" on the lines of a broken copy of the code. kind is one
of flat, peaked, near_tie, draft_excess, target_outside_draft, top_p; control one of no_q, wrong_q,
residual_p, bias. Each line gets a chi-square goodness-of-fit test of its counts against its expected
probabilities at level 0.01 / m, m the number of test lines: tokens with expected counts below 5 are
pooled, in file order, into cells of expected count at least 5; when they cannot fill one cell beside
a single other cell, an exact binomial test of their total count replaces the chi-square test; a
count on a token of expected probability 0 fails the line. The code passes when no test line
fails and every control fails at least one line. No verdict when a line has fewer than 100 000 draws,
its expected probabilities do not sum to 1, a test line's GPUs disagree, or a kind or a control is
missing.

Exits 0 when every rule passes, 1 otherwise.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys

IDENTITY = ("model_path", "revision", "tokenizer_path", "dtype", "enable_fp32_lm_head", "speculative_algorithm")
AUDIT_SAME = IDENTITY + ("quantization", "kv_cache_dtype", "attention_backend", "sampling_backend", "tp_size")
LEVELS = (0.5, 0.9, 0.99, 0.999)
FLOOR = 1e-4
MIN_POSITIONS = 10000
KINDS = ("flat", "peaked", "near_tie", "draft_excess", "target_outside_draft", "top_p")
CONTROLS = ("no_q", "wrong_q", "residual_p", "bias")
MIN_DRAWS = 100000


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_outputs(dirs):
    out = {}
    for d in dirs:
        for r in read_jsonl(os.path.join(d, "outputs.jsonl")):
            if r.get("output_ids"):
                out[str(r["id"])] = r["output_ids"]
    return out


def scorer_info(directory):
    with open(os.path.join(directory, "run.json")) as f:
        meta = json.load(f)
    info = meta.get("server_info") or {}
    info = info.get("server_args", info) if isinstance(info, dict) else {}
    if not info:
        raise ValueError(f"{directory}: the scorer's /server_info was not recorded")
    if info.get("speculative_algorithm") not in (None, "", "NONE"):
        raise ValueError(f"{directory}: the scorer runs speculative decoding ({info.get('speculative_algorithm')})")
    if info.get("enable_fp32_lm_head") is not True:
        raise ValueError(f"{directory}: the scorer's output layer is not FP32 (--enable-fp32-lm-head)")
    return meta, dict({k: info.get(k) for k in IDENTITY}, sampling=meta.get("sampling"))


def source_run(directory, meta):
    """(source directory, its run.json, its outputs by request id) of a scored run, when that run finished
    every request it was given."""
    source_dir = meta.get("source_run")
    if not source_dir:
        raise ValueError(f"{directory}: the scored run is not recorded")
    with open(os.path.join(source_dir, "run.json")) as f:
        source = json.load(f)
    outputs = {str(r["id"]): r for r in read_jsonl(os.path.join(source_dir, "outputs.jsonl"))}
    if source.get("n_failed") or source.get("n_requests", len(outputs)) != len(outputs):
        raise ValueError(f"{source_dir}: {len(outputs)} of {source.get('n_requests')} requests finished "
                         f"({source.get('n_failed')} failed); a checked run serves every request")
    return source_dir, source, outputs


def all_scored(directory, outputs, scored_ids):
    missing = sorted(set(outputs) - set(scored_ids))
    if missing:
        raise ValueError(f"{directory}: {len(missing)} requests of the scored run have no score, e.g. {missing[:3]}")


def load_scored(dirs, top=False):
    """[(run index, record)]: every request of every source run scored, complete and equal to its output."""
    records, identity = [], None
    arrays = ("tok", "top1", "lp_tok", "lp_top1") + (("top_ids", "top_lps") if top else ())
    for index, d in enumerate(dirs):
        meta, ident = scorer_info(d)
        if identity is None:
            identity = ident
        elif ident != identity:
            raise ValueError(f"{d}: scored with another scorer or sampling than {dirs[0]}")
        _, _, source = source_run(d, meta)
        scored = []
        for r in read_jsonl(os.path.join(d, "scores.jsonl")):
            n = r.get("n_out")
            if not n or r.get("n_scored") != n or any(len(r.get(k) or []) != n for k in arrays):
                raise ValueError(f"{d}: request {r.get('id')} scored {r.get('n_scored')} of {n} tokens")
            if (source.get(str(r["id"])) or {}).get("output_ids") != r["tok"]:
                raise ValueError(f"{d}: request {r.get('id')}'s scored tokens are not its source run's output")
            records.append((index, r))
            scored.append(str(r["id"]))
        all_scored(d, source, scored)
    if not records:
        raise ValueError(f"{dirs}: no scored request")
    return records


def same_scorer(dirs):
    """Every scored run from one scorer with one K: the two sides of a check are comparable."""
    first = None
    for d in dirs:
        meta, ident = scorer_info(d)
        key = (ident, meta.get("score_top"))
        if first is None:
            first = (d, key)
        elif key != first[1]:
            raise ValueError(f"{d} was scored with another scorer, sampling or K than {first[0]}")


def server_args(directory):
    with open(os.path.join(directory, "run.json")) as f:
        meta = json.load(f)
    info = meta.get("server_info") or {}
    return meta, (info.get("server_args", info) if isinstance(info, dict) else {})


def check_audit(scored_dirs, rescored_dirs, flag):
    """The audit rescores the same runs on the same scorer with only another prefill chunk size."""
    if len(rescored_dirs) != len(scored_dirs):
        raise ValueError(f"{flag} must list one rescored run per scored run, in matching order")
    for scored, rescored in zip(scored_dirs, rescored_dirs):
        s_meta, s_args = server_args(scored)
        r_meta, r_args = server_args(rescored)
        if os.path.realpath(s_meta.get("source_run") or "") != os.path.realpath(r_meta.get("source_run") or ""):
            raise ValueError(f"{rescored} rescored another run than {scored} scored")
        differ = [k for k in AUDIT_SAME if s_args.get(k) != r_args.get(k)]
        if differ:
            raise ValueError(f"{rescored}: the audit's scorer differs from {scored}'s in {differ}; "
                             "only the prefill chunk size may differ")
        if s_args.get("chunked_prefill_size") == r_args.get("chunked_prefill_size"):
            raise ValueError(f"{rescored}: the audit used the same prefill chunk size as {scored}")


def unit_key(index, r):
    return f"{index}|{r['id']}"


def same_requests(base, cand, base_dirs, cand_dirs):
    if len(base_dirs) != len(cand_dirs):
        raise ValueError("--base and --cand must list the same number of scored runs, in matching order")
    if any(os.path.realpath(b) == os.path.realpath(c) for b, c in zip(base_dirs, cand_dirs)):
        raise ValueError("a scored run is passed as both stock and candidate")
    b = {unit_key(i, r) for i, r in base}
    c = {unit_key(i, r) for i, r in cand}
    if b != c:
        raise ValueError(f"stock and candidate runs hold different requests, e.g. {sorted(b ^ c)[:3]}")


def calibrate(ar, stock, scored):
    """(delta, margins at the first departures, departures without both tokens in the top list)."""
    scores = {str(r["id"]): r for _, r in scored}
    margins, missing = [], 0
    for rid, s in stock.items():
        a, sc = ar.get(rid), scores.get(rid)
        if a is None or sc is None:
            continue
        k = next((i for i, (x, y) in enumerate(zip(a, s)) if x != y), None)
        if k is None:
            continue
        top = dict(zip(sc["top_ids"][k], sc["top_lps"][k])) if k < len(sc.get("top_ids") or []) else {}
        if a[k] not in top or s[k] not in top:
            missing += 1
            continue
        margins.append(abs(top[a[k]] - top[s[k]]))
    if not margins or missing:
        return None, margins, missing
    return math.ceil(max(margins) * 8 - 1e-9) / 8, margins, missing


def prefix_key(rid, tokens, pos):
    return (rid, pos, hashlib.sha256(json.dumps(tokens[:pos]).encode()).hexdigest(), tokens[pos])


def per_request(scored, delta, beyond, explained):
    """unit -> (positions, mismatches, events), and the event list."""
    units, events = {}, []
    for index, r in scored:
        positions = mismatches = count = 0
        for pos, (tok, top1, l1, lt) in enumerate(zip(r["tok"], r["top1"], r["lp_top1"], r["lp_tok"])):
            if pos < beyond:
                continue
            positions += 1
            if tok == top1:
                continue
            mismatches += 1
            margin = l1 - lt
            if margin > 2 * delta:
                key = prefix_key(str(r["id"]), r["tok"], pos)
                if explained(key):
                    continue
                count += 1
                events.append({"run": index, "id": str(r["id"]), "pos": pos, "tok": tok, "top1": top1,
                               "margin": margin})
        units[unit_key(index, r)] = (positions, mismatches, count)
    return units, events


def large_events(scored, delta):
    keys = set()
    for _, r in scored:
        for pos, (tok, top1, l1, lt) in enumerate(zip(r["tok"], r["top1"], r["lp_top1"], r["lp_tok"])):
            if tok != top1 and l1 - lt > 2 * delta:
                keys.add(prefix_key(str(r["id"]), r["tok"], pos))
    return keys


def chunk_flips(scored, rescored):
    flips = set()
    original = {(i, str(r["id"])): r for i, r in scored}
    for i, r in rescored:
        o = original.get((i, str(r["id"])))
        if o is None or o["tok"] != r["tok"]:
            continue
        for pos, (t1, t2) in enumerate(zip(o["top1"], r["top1"])):
            if t1 != t2:
                flips.add(prefix_key(str(r["id"]), r["tok"], pos))
    return flips


def paired_interval(base, cand, rate, resamples=2000, seed=0):
    """95 % interval of rate(cand) - rate(base), resampling requests jointly."""
    keys = sorted(base)
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append(rate([cand[k] for k in sample]) - rate([base[k] for k in sample]))
    diffs.sort()
    return diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1]


def verdict(low, high, limit):
    return "fail" if low > limit else "pass" if high <= limit else "inconclusive"


def binomial_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        term = math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log1p(-p))
        total += term
        if i > n * p and term < total * 1e-17:  # past the mode the terms only shrink
            break
    return min(1.0, total)


def check(base_scored, cand_scored, delta, rescored=(), rescored_base=(), beyond=0, limit=0.005, resamples=2000):
    shared = large_events(base_scored, delta) & large_events(cand_scored, delta)
    base_flips, cand_flips = chunk_flips(base_scored, rescored_base), chunk_flips(cand_scored, rescored)
    base, base_events = per_request(base_scored, delta, beyond, lambda key: key in shared or key in base_flips)
    cand, cand_events = per_request(cand_scored, delta, beyond, lambda key: key in shared or key in cand_flips)
    if set(base) != set(cand):
        raise ValueError("stock and candidate runs hold different requests")

    def rate(units):
        return sum(u[1] for u in units) / max(1, sum(u[0] for u in units))

    n_b, n_c = sum(u[0] for u in base.values()), sum(u[0] for u in cand.values())
    if not n_b or not n_c:
        raise ValueError(f"no position to compare (stock {n_b}, candidate {n_c}; --beyond {beyond})")
    low, high = paired_interval(base, cand, rate, resamples)
    x_b, x_c = len(base_events), len(cand_events)
    p_value = binomial_tail(x_c, x_b + x_c, n_c / (n_b + n_c))
    return {"delta": delta, "beyond": beyond, "base_positions": n_b, "cand_positions": n_c,
            "base_rate": rate(base.values()), "cand_rate": rate(cand.values()), "rate_interval": [low, high],
            "rate_verdict": verdict(low, high, limit), "base_events": x_b, "cand_events": x_c,
            "shared_events": len(shared), "event_p_value": p_value,
            "event_verdict": "pass" if p_value >= 0.05 else "fail",
            "events": {"base": base_events, "cand": cand_events}}


def chi2_sf(x, dof):
    """P(X >= x) for a chi-square variable with dof degrees of freedom (regularized upper gamma)."""
    a, z = dof / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(100000):
            n += 1
            term *= z / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b = z + 1.0 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - math.lgamma(a)) * h)


def load_probability(dirs):
    """Per scored run: (batch size, sampling, {request id: [(own ids, own lps, scorer ids, scorer lps)]})."""
    runs, identity = [], None
    for d in dirs:
        meta, ident = scorer_info(d)
        if identity is None:
            identity = ident
        elif ident != identity:
            raise ValueError(f"{d}: scored with another scorer than {dirs[0]}")
        source_dir, source, own = source_run(d, meta)
        sampling = source.get("sampling") or {}
        if float(sampling.get("temperature", 0.0)) != 1.0:
            raise ValueError(f"{source_dir}: generated at temperature {sampling.get('temperature', 0.0)}; "
                             "the probability check runs at temperature 1")
        k = source.get("logprobs")
        if not k or meta.get("score_top") != k:
            raise ValueError(f"{d}: the run kept --logprobs {k} and was scored with --score-top "
                             f"{meta.get('score_top')}; both must be one K")
        positions = {}
        for r in read_jsonl(os.path.join(d, "scores.jsonl")):
            n, rid = r.get("n_out"), str(r.get("id"))
            if not n or r.get("n_scored") != n or len(r.get("top_ids") or []) != n:
                raise ValueError(f"{d}: request {rid} scored {r.get('n_scored')} of {n} tokens")
            o = own.get(rid)
            if o is None or o.get("output_ids") != r["tok"]:
                raise ValueError(f"{d}: request {rid}'s scored tokens are not its source run's output")
            if len(o.get("top_ids") or []) != n:
                raise ValueError(f"{source_dir}: request {rid} kept log-probabilities for "
                                 f"{len(o.get('top_ids') or [])} of {n} tokens")
            positions[rid] = list(zip(o["top_ids"], o["top_lps"], r["top_ids"], r["top_lps"]))
        if not positions:
            raise ValueError(f"{d}: no scored request")
        all_scored(d, own, positions)
        runs.append((source.get("concurrency"), sampling, positions))
    return runs


def total_variation(own_ids, own_lps, ref_ids, ref_lps):
    """d at one position: a lower bound of the total variation distance between the run's distribution
    and the scorer's, from their two top-K lists. A token missing from a list has at most that list's
    smallest listed probability, and the missing tokens together at most its unlisted mass. One way,
    d sums over the tokens in both lists the excess of the run's probability over the scorer's, plus
    the run's probability on the tokens only it lists minus the most the scorer can give them; the
    other way swaps the two; d is the larger. With the same tokens in both lists it is the distance
    over those tokens with every other token merged into one category."""
    own = {t: math.exp(x) for t, x in zip(own_ids, own_lps)}
    ref = {t: math.exp(x) for t, x in zip(ref_ids, ref_lps)}
    common = own.keys() & ref.keys()

    def one_way(p, q):
        only = [t for t in p if t not in q]
        most = min(max(0.0, 1.0 - sum(q.values())), len(only) * min(q.values(), default=0.0))
        return sum(max(0.0, p[t] - q[t]) for t in common) + max(0.0, sum(p[t] for t in only) - most)

    return max(one_way(own, ref), one_way(ref, own))


def covered(ref_lps, top_k, top_p):
    """Whether the scorer's list holds every token the decoding rule can output at this position."""
    if top_k > 0 and len(ref_lps) >= top_k:
        return True
    return sum(math.exp(x) for x in ref_lps) >= min(top_p, 0.999) - 1e-6


def percentile(sorted_values, q):
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def all_in_candidate(k, n_cand, n_base):
    """One-sided Fisher exact p-value of k events in the candidate's n_cand positions and none in stock's
    n_base: the chance that all k fall in the candidate's run if every position were alike."""
    p = 1.0
    for i in range(k):
        p *= (n_cand - i) / (n_cand + n_base - i)
    return p


def probability(base_runs, cand_runs, beyond=0):
    if len(base_runs) != len(cand_runs):
        raise ValueError("--base and --cand must list the same number of scored runs, in matching order")
    sides = collections.defaultdict(lambda: ([], []))
    for (b_batch, b_sampling, b_pos), (c_batch, c_sampling, c_pos) in zip(base_runs, cand_runs):
        if b_batch != c_batch:
            raise ValueError(f"a stock run at batch size {b_batch} is paired with a candidate run at {c_batch}")
        if b_sampling != c_sampling:
            raise ValueError("stock and candidate runs used different sampling")
        if set(b_pos) != set(c_pos):
            raise ValueError(f"stock and candidate runs hold different requests, e.g. {sorted(set(b_pos) ^ set(c_pos))[:3]}")
        top_k, top_p = int(b_sampling.get("top_k") or -1), float(b_sampling.get("top_p", 1.0))
        for side, positions in ((0, b_pos), (1, c_pos)):
            for rid, rows in positions.items():
                for pos, (own_ids, own_lps, ref_ids, ref_lps) in enumerate(rows):
                    if pos < beyond:
                        continue
                    if not covered(ref_lps, top_k, top_p):
                        raise ValueError(f"request {rid} position {pos}: the scorer's {len(ref_ids)} tokens do not "
                                         "hold every token the decoding rule can output; raise --logprobs and --score-top")
                    sides[b_batch][side].append(total_variation(own_ids, own_lps, ref_ids, ref_lps))
    if not sides:
        raise ValueError(f"no position from {beyond} on")
    results = {}
    for batch, (base, cand) in sorted(sides.items()):
        for name, values in (("stock", base), ("candidate", cand)):
            if len(values) < MIN_POSITIONS:
                raise ValueError(f"batch {batch}: {name} has {len(values)} positions, fewer than {MIN_POSITIONS}; "
                                 "run more requests")
        base.sort()
        cand.sort()
        levels = []
        for q in LEVELS:
            s, c = percentile(base, q), percentile(cand, q)
            limit = max(2 * s, FLOOR)
            levels.append({"percentile": q, "stock": s, "candidate": c, "limit": limit, "pass": c <= limit})
        threshold = max(2 * base[-1], FLOOR)
        events = sum(1 for x in cand if x > threshold)
        p_value = all_in_candidate(events, len(cand), len(base))
        results[batch] = {"stock_positions": len(base), "cand_positions": len(cand), "levels": levels,
                          "level_verdict": "pass" if all(x["pass"] for x in levels) else "fail",
                          "stock_largest": base[-1], "event_threshold": threshold, "cand_events": events,
                          "event_p_value": p_value, "event_verdict": "pass" if p_value >= 0.05 else "fail"}
    return {"beyond": beyond, "batches": results}


def binomial_two_sided(x, n, p):
    """Exact two-sided binomial p-value of x successes in n draws: twice the smaller tail, at most 1."""
    if p <= 0:
        return 1.0 if x == 0 else 0.0
    lower = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                         + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(x + 1))
    return min(1.0, 2 * min(lower, binomial_tail(x, n, p)))


def goodness_of_fit(expected, counts):
    """(p-value, statistic) of a chi-square test of counts against probabilities. A token of expected
    count 5 or more is a cell of its own; the others are pooled in file order into cells of expected
    count at least 5, a last short pool joining the cell before it. A count on a token of probability
    0 gives p = 0. When the pooled tokens together expect fewer than 5 draws, they join the smallest
    cell if two cells remain, and otherwise an exact binomial test of their total count replaces the
    chi-square test (statistic None). A single token of probability 1 leaves nothing to test: p = 1."""
    n = sum(counts.values())
    if any(c > 0 and expected.get(t, 0.0) <= 0.0 for t, c in counts.items()):
        return 0.0, math.inf
    cells, pools, pool = [], [], [0, 0.0]
    for token, p in expected.items():
        if p <= 0.0:
            continue
        observed, mean = counts.get(token, 0), n * p
        if mean >= 5:
            cells.append([observed, mean])
            continue
        pool[0] += observed
        pool[1] += mean
        if pool[1] >= 5:
            pools.append(pool)
            pool = [0, 0.0]
    if pool[1] > 0:
        if pools:
            pools[-1][0] += pool[0]
            pools[-1][1] += pool[1]
        elif len(cells) >= 2:
            lowest = min(cells, key=lambda cell: cell[1])
            lowest[0] += pool[0]
            lowest[1] += pool[1]
        elif cells:
            return binomial_two_sided(pool[0], n, pool[1] / n), None
        else:
            pools.append(pool)
    cells += pools
    if len(cells) < 2:
        return 1.0, 0.0
    stat = sum((o - e) ** 2 / e for o, e in cells)
    return chi2_sf(stat, len(cells) - 1), stat


def rules(lines):
    tests = [line for line in lines if not line.get("control")]
    if not tests:
        raise ValueError("no test line")
    missing = [k for k in KINDS if k not in {line.get("kind") for line in tests}]
    if missing:
        raise ValueError(f"no test case of kind {', '.join(missing)}")
    missing = [c for c in CONTROLS if c not in {line.get("control") for line in lines}]
    if missing:
        raise ValueError(f"no control {', '.join(missing)}")
    alpha = 0.01 / len(tests)
    out = []
    for line in lines:
        where = f"case {line.get('case')} position {line.get('position')}"
        if line.get("control"):
            where += f" control {line['control']}"
        expected = {str(k): float(v) for k, v in line["expected"].items()}
        counts = {str(k): int(v) for k, v in line["counts"].items()}
        draws = sum(counts.values())
        if draws < MIN_DRAWS:
            raise ValueError(f"{where}: {draws} draws, fewer than {MIN_DRAWS}")
        if abs(sum(expected.values()) - 1.0) > 1e-6:
            raise ValueError(f"{where}: the expected probabilities sum to {sum(expected.values())}")
        if not line.get("control") and line.get("ranks_agree") is not True:
            raise ValueError(f"{where}: the engine's GPUs did not output the same token in every draw")
        p_value, stat = goodness_of_fit(expected, counts)
        out.append({"case": line.get("case"), "kind": line.get("kind"), "position": line.get("position"),
                    "control": line.get("control"), "draws": draws, "chi2": stat, "p_value": p_value,
                    "rejected": p_value < alpha})
    failed = [r for r in out if not r["control"] and r["rejected"]]
    not_rejected = [c for c in CONTROLS if not any(r["rejected"] for r in out if r["control"] == c)]
    return {"alpha": alpha, "test_lines": len(tests), "failed": failed, "controls_not_rejected": not_rejected,
            "lines": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--ar", nargs="+", required=True)
    cal.add_argument("--stock", nargs="+", required=True)
    cal.add_argument("--scored", nargs="+", required=True, help="stock's outputs scored with --score-top")
    chk = sub.add_parser("check")
    chk.add_argument("--delta", type=float, required=True)
    chk.add_argument("--rescored", nargs="+", required=True,
                     help="the candidate's outputs scored with another prefill chunk size (the audit)")
    chk.add_argument("--rescored-base", nargs="+", required=True,
                     help="stock's outputs scored with another prefill chunk size (the audit)")
    chk.add_argument("--audit", default=None)
    prob = sub.add_parser("probability")
    rul = sub.add_parser("rules")
    rul.add_argument("--tests", required=True, help="the patch's GPU test lines (JSONL)")
    for p in (chk, prob):
        p.add_argument("--base", nargs="+", required=True, help="stock's scored outputs")
        p.add_argument("--cand", nargs="+", required=True)
        p.add_argument("--beyond", type=int, default=0)
    for p in (cal, chk, prob, rul):
        p.add_argument("--json", default=None)
    args = parser.parse_args()
    try:
        if args.mode == "calibrate":
            delta, margins, missing = calibrate(load_outputs(args.ar), load_outputs(args.stock),
                                                load_scored(args.scored, top=True))
            result = {"delta": delta, "departures": len(margins), "without_both_tokens": missing,
                      "largest_margin": max(margins, default=None)}
            print(f"{len(margins)} first departures of stock from AR with both tokens scored ({missing} without); "
                  f"largest margin {result['largest_margin']}; delta {delta}")
            ok = delta is not None
        elif args.mode == "check":
            same_scorer(args.base + args.cand)
            check_audit(args.cand, args.rescored, "--rescored")
            check_audit(args.base, args.rescored_base, "--rescored-base")
            base, cand = load_scored(args.base), load_scored(args.cand)
            same_requests(base, cand, args.base, args.cand)
            result = check(base, cand, args.delta, load_scored(args.rescored), load_scored(args.rescored_base),
                           args.beyond)
            if args.audit:
                with open(args.audit, "w") as f:
                    for side in ("base", "cand"):
                        for e in result["events"][side]:
                            f.write(json.dumps(dict(e, system=side)) + "\n")
            low, high = result["rate_interval"]
            print(f"positions from {args.beyond}: stock {result['base_positions']}, candidate {result['cand_positions']}")
            print(f"rate rule: stock {100 * result['base_rate']:.3f} %, candidate {100 * result['cand_rate']:.3f} %, "
                  f"difference 95 % [{100 * low:+.3f}, {100 * high:+.3f}] pt: {result['rate_verdict']}")
            print(f"event rule (margin > {2 * args.delta:g}, unexplained): stock {result['base_events']}, candidate "
                  f"{result['cand_events']}, p = {result['event_p_value']:.4f}: {result['event_verdict']}")
            ok = result["rate_verdict"] == "pass" and result["event_verdict"] == "pass"
            result.pop("events")
        elif args.mode == "probability":
            if any(os.path.realpath(b) == os.path.realpath(c) for b, c in zip(args.base, args.cand)):
                raise ValueError("a scored run is passed as both stock and candidate")
            same_scorer(args.base + args.cand)
            result = probability(load_probability(args.base), load_probability(args.cand), args.beyond)
            ok = True
            for batch, r in result["batches"].items():
                print(f"batch {batch}, positions from {args.beyond}: stock {r['stock_positions']}, "
                      f"candidate {r['cand_positions']}")
                for x in r["levels"]:
                    print(f"  level rule, percentile {100 * x['percentile']:g}: stock d {x['stock']:.3g}, candidate "
                          f"{x['candidate']:.3g}, limit {x['limit']:.3g}: {'pass' if x['pass'] else 'fail'}")
                print(f"  event rule: d above {r['event_threshold']:.3g} (stock's largest {r['stock_largest']:.3g}): "
                      f"candidate {r['cand_events']}, p = {r['event_p_value']:.4f}: {r['event_verdict']}")
                ok = ok and r["level_verdict"] == "pass" and r["event_verdict"] == "pass"
        else:
            result = rules(read_jsonl(args.tests))
            print(f"{result['test_lines']} test lines at level {result['alpha']:.3g}: {len(result['failed'])} failed")
            for r in result["failed"][:10]:
                print(f"  failed: case {r['case']} ({r['kind']}) position {r['position']}, p = {r['p_value']:.3g}")
            for c in CONTROLS:
                print(f"control {c}: {'kept' if c in result['controls_not_rejected'] else 'rejected'}")
            ok = not result["failed"] and not result["controls_not_rejected"]
    except ValueError as error:
        print(f"no verdict: {error}", file=sys.stderr)
        return 1
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
