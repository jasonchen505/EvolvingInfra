#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Tokens per step and time per step over every decode step of a run, each at its own batch.

    python decode_steps.py TRACE_DIR [--start T0 --end T1 | --windows T0:T1[,T0:T1...]] [--json]

TRACE_DIR is the engine's --sched-trace-dir: one iter_trace.*.jsonl per scheduler, one record per
forward. During a run at batch size B (B tasks or requests in flight, replay.md rule 4) the engine's
batch, the requests one decode step processes, varies between 1 and B: a task between two turns has
no request in the engine. Every decode forward counts, at the batch it ran, when its previous and next
forwards are decode forwards (consecutive forward numbers). Its time is the time from its launch to
the next launch. Left out: the first decode step after a prefill forward (extend, mixed, split
prefill) or an idle forward, whose interval with the overlap scheduler also holds the end of the
prefill's GPU work, and a step whose next launch is a prefill or follows an idle wait.
--start/--end (wall-clock seconds, as in a client's run.json) keep the steps of one run when the
engine served several; --windows pools the steps of several windows, such as the profile windows of
one run.

Over the counted steps: tau is the committed tokens per request per step, the sum of decode_tokens
(each request's accepted draft tokens plus its one bonus token, 1 per request without speculation)
over the sum of the steps' batches; the step time is the mean time of a step; tokens per second is
the sum of committed tokens over the sum of step times. Context is the mean number of tokens per
request (the scheduler's summed sequence lengths over the sum of the batches). The steps per batch
(the batch mix) and the context are byte_floor.py's --batch-mix and --context-tokens for the same
window: give it the windows of the profiled steps when the floor is compared with a profile.
"""
import argparse
import collections
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


def plain_steps(records, start=None, end=None):
    """(steps, left out after a prefill or idle): steps are (seconds, committed tokens, batch,
    summed context or None)."""
    steps, after_prefill = [], 0
    for i in range(1, len(records) - 1):
        prev, cur, nxt = records[i - 1], records[i], records[i + 1]
        if cur["mode"] not in DECODE:
            continue
        batch = cur.get("launch_bs") or cur["bs"]
        if not batch:
            continue
        # the step runs from its launch to the next one: both records must lie inside the window
        if (start is not None and cur["ts"] < start) or (end is not None and max(cur["ts"], nxt["ts"]) > end):
            continue
        # the step's time runs to the next launch: that launch must be the next decode step,
        # not a prefill or a forward after an idle wait (between requests at B=1)
        if (nxt["forward_iter"] != cur["forward_iter"] + 1 or nxt["mode"] not in DECODE
                or nxt.get("after_idle")):
            continue
        if (prev["forward_iter"] != cur["forward_iter"] - 1 or prev["mode"] not in DECODE
                or cur.get("after_idle")):
            after_prefill += 1
            continue
        seconds = nxt["launch_mono"] - cur["launch_mono"]
        if seconds <= 0:
            continue
        steps.append((seconds, cur.get("decode_tokens") or batch, batch, cur.get("launch_seq_lens_sum")))
    return steps, after_prefill


def measure(trace_dir, start=None, end=None, windows=None):
    """Summary of the counted decode steps, pooled over every scheduler's trace and every window."""
    streams = load(trace_dir)
    if not streams:
        raise SystemExit(f"no iter_trace*.jsonl in {trace_dir} (start the engine with --sched-trace-dir)")
    steps, after_prefill = [], 0
    for window_start, window_end in windows or [(start, end)]:
        for records in streams:
            s, a = plain_steps(records, window_start, window_end)
            steps += s
            after_prefill += a
    result = {"trace_dir": trace_dir, "steps": len(steps), "left_out_after_prefill": after_prefill}
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
        batch_mean=statistics.mean(s[2] for s in steps),
        batch_mix=dict(sorted(collections.Counter(s[2] for s in steps).items())),
    )
    with_context = [s for s in steps if s[3]]
    result["context_mean"] = (sum(s[3] for s in with_context) / sum(s[2] for s in with_context)
                              if with_context else None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("trace_dir")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--windows", default=None, help="T0:T1 pairs, comma-separated: pool these windows")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    windows = [tuple(float(x) for x in w.split(":")) for w in args.windows.split(",")] if args.windows else None
    result = measure(args.trace_dir, args.start, args.end, windows)
    if args.json:
        print(json.dumps(result))
    elif not result["steps"]:
        print(f"no decode steps ({result['left_out_after_prefill']} left out after a prefill)", file=sys.stderr)
    else:
        print(f"{result['steps']} decode steps ({result['left_out_after_prefill']} left out after a prefill), "
              f"mean batch {result['batch_mean']:.2f}, tau {result['tau']:.3f}, step {result['step_ms_mean']:.3f} ms "
              f"(median {result['step_ms_median']:.3f}, se {result['step_ms_se']:.3f}), "
              f"{result['tokens_per_s']:.1f} tokens/s"
              + (f", context {result['context_mean']:.0f} tokens per request" if result.get("context_mean") else ""))
        print("--batch-mix " + ",".join(f"{b}:{n}" for b, n in result["batch_mix"].items()))
    return 0 if result["steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
