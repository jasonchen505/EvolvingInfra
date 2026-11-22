#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Send a request set to one engine's /generate at a fixed concurrency and keep what it returns.

    python fixed_batch_client.py --endpoint http://HOST:PORT --requests REQUESTS.jsonl --concurrency B \
        --out RUN_DIR --arm NAME --node NAME --cycle K [--trace-dir DIR] [options]

Requests: one JSON object per line with "id", the prompt as "input_ids" (a list of token ids) or
"text" (with "image_data" for images), and "max_new_tokens". The client keeps B requests in
flight: when one finishes, the next starts, so the engine decodes a batch of B except at the start
and the end of the run.

Modes:
  default            generate. Greedy unless --sampling says otherwise; stops at the end-of-sequence
                     token unless --ignore-eos. Writes RUN_DIR/outputs.jsonl. With --keyed-seed and
                     sampling, the first request runs twice with its seed before the run: different
                     tokens mean the engine ignores the seed, and the client exits 1.
  --forced-from RUN  exact replay: every request commits the output tokens of RUN (normally the run
                     without speculation), so each draft setting is measured on the same tokens. The
                     engine rejects forced outputs for draft methods that do not support them.
  --score-from RUN   scoring: sends prompt + RUN's output tokens and returns, at every output position,
                     the log-probability of that token and of the target's two most likely tokens.
                     Send it to an engine without speculation. Writes RUN_DIR/scores.jsonl. The engine
                     returns input log-probabilities from logprob_start_len on, the first of them empty,
                     so the request starts one token before the output. A request given as "text" is
                     rebuilt from RUN's output text; its positions are scored up to the first token where
                     the rebuilt prompt differs from RUN's tokens.

RUN_DIR/run.json records the arm, node, cycle, concurrency, wall-clock start and end, the engine's
/server_info and the trace directory, which decode_steps.py and paired_ab.py read. The client exits
1 if any request failed.
"""
import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def http_json(url, body=None, timeout=3600.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def server_info(endpoint):
    for path in ("/server_info", "/get_server_info"):
        try:
            return http_json(endpoint + path, timeout=60)
        except (urllib.error.URLError, ValueError, TimeoutError):
            continue
    return None


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def keyed_seed(request_id):
    return int.from_bytes(hashlib.sha256(str(request_id).encode()).digest()[:4], "big") & 0x7FFFFFFF


def build_body(request, args, previous=None):
    """The /generate body of one request; previous is this request's record from --forced-from/--score-from."""
    # the text keeps special tokens and the stop token, so --score-from can rebuild the tokens from it
    sampling = {"max_new_tokens": int(request["max_new_tokens"]), "temperature": 0.0,
                "skip_special_tokens": False, "spaces_between_special_tokens": False, "no_stop_trim": True}
    sampling.update(args.sampling)
    if args.ignore_eos:
        sampling["ignore_eos"] = True
    if args.keyed_seed:
        sampling["sampling_seed"] = keyed_seed(request["id"])
    body = {"sampling_params": sampling}
    if args.score_from:
        output_ids = previous["output_ids"]
        if "input_ids" in request:
            body["input_ids"] = list(request["input_ids"]) + list(output_ids)
            start = len(request["input_ids"]) - 1
        else:
            body["text"] = request["text"] + previous["text"]
            start = previous["prompt_tokens"] - 1
        body.update(return_logprob=True, logprob_start_len=start, top_logprobs_num=2)
        sampling.update(max_new_tokens=1, temperature=0.0)
        sampling.pop("ignore_eos", None)
        sampling.pop("sampling_seed", None)
    elif "input_ids" in request:
        body["input_ids"] = request["input_ids"]
    else:
        body["text"] = request["text"]
    if "image_data" in request:
        body["image_data"] = request["image_data"]
    if args.forced_from:
        sampling["forced_output_token_ids"] = list(previous["output_ids"])
        sampling["max_new_tokens"] = len(previous["output_ids"])
    return body


def seed_check(endpoint, request, args, tokens=32):
    """Keyed sampling must be reproducible: the same request with the same seed gives the same tokens
    twice. Returns the problem, or None."""
    probe = dict(request, max_new_tokens=min(tokens, int(request["max_new_tokens"])))
    outputs = [http_json(endpoint + "/generate", build_body(probe, args), timeout=args.timeout).get("output_ids") or []
               for _ in range(2)]
    if outputs[0] == outputs[1]:
        return None
    k = next((i for i, (a, b) in enumerate(zip(*outputs)) if a != b), min(len(o) for o in outputs))
    return (f"--keyed-seed: two runs of {request['id']} with the same seed differ at output token {k}, so the "
            "engine does not draw with the request's seed. decoupled-engine passes seeds to its sampler only "
            "with --enable-deterministic-inference, and a speculative verify must draw with the same key.")


def entry(item):
    """(logprob, token_id) of one logprob entry: [logprob, token_id, text] or a dict."""
    if isinstance(item, dict):
        return item.get("logprob"), item.get("token_id")
    return item[0], item[1]


def score_record(request_id, output_ids, meta):
    """Per output position: token, target's top-1 token, and the three log-probabilities. Entry 0 of the
    engine's lists is the last prompt token (no log-probability); output position p is entry p + 1."""
    tokens = meta.get("input_token_logprobs") or []
    tops = meta.get("input_top_logprobs") or []
    record = {"id": request_id, "n_out": len(output_ids), "tok": [], "top1": [], "lp_tok": [], "lp_top1": [],
              "lp_top2": []}
    for position, expected in enumerate(output_ids):
        j = position + 1
        if j >= len(tokens) or j >= len(tops) or not tops[j]:
            break
        lp_tok, token = entry(tokens[j])
        if token != expected or lp_tok is None:
            break  # the rebuilt prompt diverged from the recorded tokens
        ranked = [entry(t) for t in tops[j]]
        record["tok"].append(token)
        record["lp_tok"].append(lp_tok)
        record["top1"].append(ranked[0][1])
        record["lp_top1"].append(ranked[0][0])
        record["lp_top2"].append(ranked[1][0] if len(ranked) > 1 else None)
    record["n_scored"] = len(record["tok"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", default="run")
    parser.add_argument("--node", default=None, help="the GPU node of the engine; pairs are formed per node")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--trace-dir", default=None, help="the engine's --sched-trace-dir, as seen from here")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--sampling", type=json.loads, default={}, help='JSON merged into sampling_params')
    parser.add_argument("--keyed-seed", action="store_true",
                        help="sampling_seed from each request id; with sampling, first checks that the engine uses it")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--forced-from", default=None)
    mode.add_argument("--score-from", default=None)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    requests = read_jsonl(args.requests)
    if args.max_requests is not None:
        requests = requests[: args.max_requests]
    previous = {}
    source = args.forced_from or args.score_from
    if source:
        previous = {str(r["id"]): r for r in read_jsonl(os.path.join(source, "outputs.jsonl"))}
        missing = [r["id"] for r in requests if str(r["id"]) not in previous]
        if missing:
            print(f"{len(missing)} requests have no output in {source}, e.g. {missing[:3]}", file=sys.stderr)
            return 1
    os.makedirs(args.out, exist_ok=True)
    out_name = "scores.jsonl" if args.score_from else "outputs.jsonl"
    out_path = os.path.join(args.out, out_name)
    if os.path.exists(out_path):
        print(f"{out_path} exists; use a new run directory", file=sys.stderr)
        return 1

    with open(args.requests, "rb") as f:
        requests_sha = hashlib.sha256(f.read()).hexdigest()
    info = server_info(endpoint)
    if args.keyed_seed and float(args.sampling.get("temperature", 0.0)) > 0 and not source:
        problem = seed_check(endpoint, requests[0], args)
        if problem:
            print(problem, file=sys.stderr)
            return 1
    lock = threading.Lock()
    pending = iter(requests)
    results, failures = [], []
    out = open(out_path, "w")

    def worker():
        while True:
            with lock:
                request = next(pending, None)
            if request is None:
                return
            rid = str(request["id"])
            began = time.time()
            try:
                body = build_body(request, args, previous.get(rid))
                reply = http_json(endpoint + "/generate", body, timeout=args.timeout)
                meta = reply.get("meta_info", {})
                if args.score_from:
                    record = score_record(rid, previous[rid]["output_ids"], meta)
                else:
                    record = {"id": rid, "prompt_tokens": meta.get("prompt_tokens"),
                              "completion_tokens": meta.get("completion_tokens"),
                              "output_ids": reply.get("output_ids"), "text": reply.get("text"),
                              "spec_verify_ct": meta.get("spec_verify_ct"),
                              "spec_num_correct_drafts": meta.get("spec_num_correct_drafts"),
                              "spec_accept_length": meta.get("spec_accept_length"),
                              "finish_reason": meta.get("finish_reason")}
                record["latency_s"] = time.time() - began
                with lock:
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    results.append(record)
            except Exception as error:  # noqa: BLE001 - every failure is recorded and fails the run
                detail = error.read().decode(errors="replace")[:500] if isinstance(error, urllib.error.HTTPError) else ""
                with lock:
                    failures.append({"id": rid, "error": f"{type(error).__name__}: {error} {detail}".strip()})

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for _ in range(args.concurrency):
            pool.submit(worker)
    t_end = time.time()
    out.close()

    run = {"arm": args.arm, "node": args.node or endpoint.split("//")[-1].split(":")[0], "cycle": args.cycle,
           "concurrency": args.concurrency, "mode": "score" if args.score_from else "forced" if args.forced_from
           else "generate", "source_run": source, "endpoint": endpoint, "requests": os.path.abspath(args.requests),
           "requests_sha256": requests_sha, "n_requests": len(requests), "n_done": len(results),
           "n_failed": len(failures), "failures": failures[:20], "t_start": t_start, "t_end": t_end,
           "trace_dir": args.trace_dir, "client_host": socket.gethostname(), "sampling": args.sampling,
           "ignore_eos": args.ignore_eos, "keyed_seed": args.keyed_seed, "server_info": info}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1)

    print(f"{args.arm} cycle {args.cycle} on {run['node']}: {len(results)} done, {len(failures)} failed, "
          f"{t_end - t_start:.1f} s")
    if args.score_from:
        scored = sum(r["n_scored"] for r in results)
        total = sum(r["n_out"] for r in results)
        print(f"scored {scored} of {total} output positions")
    elif results:
        tokens = sum(r["completion_tokens"] or 0 for r in results)
        print(f"output tokens {tokens}, {tokens / (t_end - t_start):.1f} tokens/s over the whole run")
        verify = [r for r in results if r.get("spec_verify_ct")]
        if verify:
            per_request = statistics.mean(r["spec_accept_length"] for r in verify)
            pooled = (sum(r["spec_num_correct_drafts"] + r["spec_verify_ct"] for r in verify)
                      / sum(r["spec_verify_ct"] for r in verify))
            print(f"accepted tokens per verify step: {pooled:.3f} pooled, {per_request:.3f} mean over requests")
    for failure in failures[:5]:
        print(f"failed {failure['id']}: {failure['error']}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
