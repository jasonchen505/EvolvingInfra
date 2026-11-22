#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Tokens per step and time per step of an engine's plain decode steps at one batch size.

    python decode_steps.py TRACE_DIR --concurrency B [--start T0 --end T1] [--json]

TRACE_DIR is the engine's --sched-trace-dir: one iter_trace.*.jsonl per scheduler, one record per
forward. A step counts when it is a decode forward of exactly B requests whose previous and next
forwards are decode forwards (consecutive forward numbers). Its time is the time from its launch to
the next launch. Two kinds of steps are left out:
- the first decode step after a prefill forward (extend, mixed, split prefill) or an idle forward:
  with the overlap scheduler its interval also holds the end of the prefill's GPU work;
- steps with fewer than B requests: the start and end of a run, and moments when a request
  finished and the next had not started yet.
--start/--end (wall-clock seconds, as in a client's run.json) keep the steps of one run when the
engine served several.

Tokens per step (tau) is the committed tokens per request per step: decode_tokens / batch size,
where decode_tokens counts each request's accepted draft tokens plus its one bonus token (1 per
request without speculation). Tokens per second is the sum of committed tokens over the sum of
step times.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

DECODE = {"decode", "target_verify"}


def load(trace_dir):
    """One list of records per iter_trace file, sorted by forward number."""
    files = sorted(glob.glob(os.path.join(trace_dir, "iter_trace*.jsonl")))
    streams = []
    for path in files:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        streams.append(sorted(records, key=lambda r: r["forward_iter"]))
    return streams


def plain_steps(records, concurrency, start=None, end=None):
    """(steps, left out after a prefill or idle): steps are (seconds, committed tokens, batch size)."""
    steps, after_prefill = [], 0
    for i in range(1, len(records) - 1):
        prev, cur, nxt = records[i - 1], records[i], records[i + 1]
        if cur["mode"] not in DECODE:
            continue
        batch = cur.get("launch_bs") or cur["bs"]
        if batch != concurrency:
            continue
        if start is not None and cur["ts"] < start or end is not None and cur["ts"] > end:
            continue
        if nxt["forward_iter"] != cur["forward_iter"] + 1:
            continue
        if (prev["forward_iter"] != cur["forward_iter"] - 1 or prev["mode"] not in DECODE
                or cur.get("after_idle")):
            after_prefill += 1
            continue
        seconds = nxt["launch_mono"] - cur["launch_mono"]
        if seconds <= 0:
            continue
        steps.append((seconds, cur.get("decode_tokens") or batch, batch))
    return steps, after_prefill


def measure(trace_dir, concurrency, start=None, end=None):
    """Summary of the plain steps at this batch size, pooled over every scheduler's trace."""
    streams = load(trace_dir)
    if not streams:
        raise SystemExit(f"no iter_trace*.jsonl in {trace_dir} (start the engine with --sched-trace-dir)")
    steps, after_prefill = [], 0
    for records in streams:
        s, a = plain_steps(records, concurrency, start, end)
        steps += s
        after_prefill += a
    result = {"trace_dir": trace_dir, "concurrency": concurrency, "steps": len(steps),
              "left_out_after_prefill": after_prefill}
    if not steps:
        return result
    ms = [s[0] * 1e3 for s in steps]
    tokens = sum(s[1] for s in steps)
    seconds = sum(s[0] for s in steps)
    result.update(
        tau=tokens / sum(s[2] for s in steps),
        step_ms_mean=statistics.mean(ms),
        step_ms_median=statistics.median(ms),
        step_ms_se=statistics.stdev(ms) / math.sqrt(len(ms)) if len(ms) > 1 else float("nan"),
        tokens_per_s=tokens / seconds,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("trace_dir")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = measure(args.trace_dir, args.concurrency, args.start, args.end)
    if args.json:
        print(json.dumps(result))
    elif not result["steps"]:
        print(f"no plain decode steps at batch {args.concurrency} "
              f"({result['left_out_after_prefill']} left out after a prefill)", file=sys.stderr)
    else:
        print(f"batch {args.concurrency}: {result['steps']} plain steps "
              f"({result['left_out_after_prefill']} left out after a prefill), "
              f"tau {result['tau']:.3f}, step {result['step_ms_mean']:.3f} ms "
              f"(median {result['step_ms_median']:.3f}, se {result['step_ms_se']:.3f}), "
              f"{result['tokens_per_s']:.1f} tokens/s")
    return 0 if result["steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
