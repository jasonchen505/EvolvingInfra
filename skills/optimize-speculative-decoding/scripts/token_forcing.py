#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Lossless check: are a candidate's outputs as faithful to the target model as the base system's?

    python token_forcing.py --base SCORED_DIR [SCORED_DIR ...] --cand SCORED_DIR [SCORED_DIR ...]
        [--gap-edges 0,0.25,0.5,1,2,4] [--event-gap 2] [--max-rate-increase 0.005] [--resamples 2000]

Each SCORED_DIR holds scores.jsonl: the outputs of one system scored by the target model without
speculation, one request per line with the committed tokens (tok) and, at every position, the
reference token (top1) and how far the reference was from changing (its gap). A position is a
mismatch when the committed token is not the reference.
- Greedy decoding (fixed_batch_client.py --score-from): the reference is the target's first choice,
  and the gap is its log-probability minus that of its second choice (lp_top1, lp_top2).
- Sampling with a keyed rule: the reference is the token the target draws itself with the same key,
  recorded on an engine without speculation while it replays the outputs; the record gives the gap
  directly (gap), for example the margin of the draw over the runner-up.

How often a mismatch happens depends on the gap: near-ties flip with any numerical difference, and
the share of near-ties depends on what the outputs say, which differs between runs once their
outputs diverge. So the rates are compared within gap bins and weighted by the base's share of
positions in each bin.

Rules (both one-sided, 95 % from a bootstrap over requests):
- rate rule: the candidate's weighted mismatch rate exceeds the base's by at most
  --max-rate-increase (0.005 = 0.5 percentage points);
- event rule: mismatches at positions with a gap above --event-gap, per position, are not more
  frequent in the candidate than in the base.
Exits 0 when both pass, 1 otherwise.
"""
import argparse
import bisect
import json
import os
import random
import sys


def load(dirs, edges, event_gap):
    """Per request: (positions per bin, mismatches per bin, positions, mismatches above the event gap)."""
    units = []
    for d in dirs:
        with open(os.path.join(d, "scores.jsonl")) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_bins = [0] * len(edges)
                m_bins = [0] * len(edges)
                events = 0
                gaps = r.get("gap")
                if gaps is None:
                    gaps = [None if lp2 is None else lp1 - lp2 for lp1, lp2 in zip(r["lp_top1"], r["lp_top2"])]
                for tok, top1, gap in zip(r["tok"], r["top1"], gaps):
                    gap = float("inf") if gap is None else float(gap)
                    b = bisect.bisect_right(edges, gap) - 1
                    n_bins[b] += 1
                    if tok != top1:
                        m_bins[b] += 1
                        if gap > event_gap:
                            events += 1
                if sum(n_bins):
                    units.append((n_bins, m_bins, sum(n_bins), events))
    return units


def totals(units, n_edges):
    n = [0] * n_edges
    m = [0] * n_edges
    positions = events = 0
    for nb, mb, p, e in units:
        for i in range(n_edges):
            n[i] += nb[i]
            m[i] += mb[i]
        positions += p
        events += e
    return n, m, positions, events


def statistics_of(base, cand, n_edges):
    """(weighted base rate, weighted cand rate, base event rate, cand event rate)."""
    bn, bm, bp, be = totals(base, n_edges)
    cn, cm, cp, ce = totals(cand, n_edges)
    bins = [i for i in range(n_edges) if bn[i] and cn[i]]
    weight = sum(bn[i] for i in bins)
    base_rate = sum(bm[i] for i in bins) / weight
    cand_rate = sum(bn[i] / weight * cm[i] / cn[i] for i in bins)
    return base_rate, cand_rate, be / bp, ce / cp


def check(base, cand, edges, max_rate_increase=0.005, resamples=2000, seed=0):
    n_edges = len(edges)
    point = statistics_of(base, cand, n_edges)
    rng = random.Random(seed)
    rate_diffs, event_diffs = [], []
    for _ in range(resamples):
        b = [base[rng.randrange(len(base))] for _ in base]
        c = [cand[rng.randrange(len(cand))] for _ in cand]
        s = statistics_of(b, c, n_edges)
        rate_diffs.append(s[1] - s[0])
        event_diffs.append(s[3] - s[2])
    rate_diffs.sort()
    event_diffs.sort()
    rate_upper = rate_diffs[int(0.95 * resamples) - 1]
    event_lower = event_diffs[int(0.05 * resamples)]
    return {
        "base_positions": totals(base, n_edges)[2], "cand_positions": totals(cand, n_edges)[2],
        "base_requests": len(base), "cand_requests": len(cand),
        "base_rate": point[0], "cand_rate_weighted": point[1], "rate_increase": point[1] - point[0],
        "rate_increase_upper95": rate_upper, "rate_rule_pass": rate_upper <= max_rate_increase,
        "base_event_rate": point[2], "cand_event_rate": point[3], "event_increase": point[3] - point[2],
        "event_increase_lower95": event_lower, "event_rule_pass": event_lower <= 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--cand", nargs="+", required=True)
    parser.add_argument("--gap-edges", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--event-gap", type=float, default=2.0)
    parser.add_argument("--max-rate-increase", type=float, default=0.005)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    edges = [float(x) for x in args.gap_edges.split(",")]
    if edges[0] != 0 or edges != sorted(edges):
        print("--gap-edges must start at 0 and increase", file=sys.stderr)
        return 1
    base = load(args.base, edges, args.event_gap)
    cand = load(args.cand, edges, args.event_gap)
    if not base or not cand:
        print("no scored positions on one side", file=sys.stderr)
        return 1
    r = check(base, cand, edges, args.max_rate_increase, args.resamples)
    print(f"base {r['base_requests']} requests / {r['base_positions']} positions, "
          f"candidate {r['cand_requests']} / {r['cand_positions']}")
    print(f"rate rule: base {100 * r['base_rate']:.3f} %, candidate weighted {100 * r['cand_rate_weighted']:.3f} %, "
          f"increase {100 * r['rate_increase']:+.3f} pt (95 % upper {100 * r['rate_increase_upper95']:+.3f} pt, "
          f"limit {100 * args.max_rate_increase:.2f} pt): {'pass' if r['rate_rule_pass'] else 'FAIL'}")
    print(f"event rule (gap > {args.event_gap:g}): base {100 * r['base_event_rate']:.4f} %, candidate "
          f"{100 * r['cand_event_rate']:.4f} % of positions, increase {100 * r['event_increase']:+.4f} pt "
          f"(95 % lower {100 * r['event_increase_lower95']:+.4f} pt): {'pass' if r['event_rule_pass'] else 'FAIL'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0 if r["rate_rule_pass"] and r["event_rule_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
