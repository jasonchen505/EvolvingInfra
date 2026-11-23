#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Send a request set to one engine's /generate at a fixed concurrency and keep what it returns.

    python fixed_batch_client.py --endpoint http://HOST:PORT --requests REQUESTS.jsonl --concurrency B \
        --out RUN_DIR --arm NAME --node SLOT --cycle K [--trace-dir DIR] [--max-new-tokens-cap N] \
        [--sampling JSON] [--session-field FIELD [--session-copies C]] [--logprobs K]
    python fixed_batch_client.py --endpoint http://HOST:PORT --requests REQUESTS.jsonl --concurrency B \
        --out SCORED_DIR --score-from RUN_DIR [--score-top K] [--sampling JSON] [--session-field FIELD
        [--session-copies C]]

Requests: one JSON object per line with "id", the prompt as "input_ids" (a list of token ids) or
"text" (with "image_data" for images), and "max_new_tokens". Without --session-field every request
stands alone and the client keeps B requests in flight: when one finishes, the next starts.

Tasks of several turns (a robot's episode, an agent's session): with --session-field FIELD the
requests that share FIELD's value are the turns of one task, in file order, and B counts tasks: the
client keeps B tasks in flight, sends a task's turns one after another, each only after the previous
one returned, and starts the next task when one ends. Each turn's prompt is built as the task's own
client builds it: the prompt the previous turn sent, the answer the engine just gave it, and the new
part the recorded turn adds (its recorded prompt minus the previous turn's recorded prompt and
recorded answer, "output_text" for a text prompt or "output_ids" for a token prompt, with its
images). The client refuses a file where a turn's recorded prompt does not extend the previous
turn's prompt and answer. The answer keeps special tokens and drops the stop token, as the recorded
answers do: the engine's text leaves out the stop token it matched; its output_ids end with it, so a
token answer loses that last token, unless the turn's recorded output_ids end with the same token.
Every task, and every copy of it, sends its own cache key, so tasks never share
cached KV or recurrent state; a turn reuses only its own task's earlier context. --session-copies C
runs the task list C times, for batches larger than the set has tasks; copy k >= 1 of a request gets
the id "<id>#c<k>".

Decoding is greedy unless --sampling says otherwise (JSON merged into sampling_params; it must not
set max_new_tokens); --max-new-tokens-cap caps every request's max_new_tokens. Writes
RUN_DIR/outputs.jsonl (per request its output token ids and text, finish reason, latency from
sending to the last token, and the engine's speculative counters) and RUN_DIR/run.json (the arm, the
slot, the cycle, the batch size, the session field and copies, the wall-clock start and end, the
request file's sha256, the sampling, the cap, the engine's /server_info and the trace directory,
which decode_steps.py and paired_ab.py read). A request the engine aborted counts as failed, and a
failed turn ends its task. Exits 1 if any request failed or was not sent, and refuses a request set
with fewer requests (or, with --session-field, tasks times copies) than B: it cannot keep B in flight.

--logprobs K also keeps, at every output position, the engine's own log-probability of the output
token and its K most likely tokens (lp_out, top_ids, top_lps), from the forward pass that produced the
token, normalized over the vocabulary before top-k and top-p. Under sampling the engine applies the
temperature to them, so the probability check of lossless.md runs at temperature 1.

--score-from RUN scores RUN's outputs, teacher-forced. Each request's prompt is the one RUN sent,
rebuilt the same way from RUN's answers: the task field and copies are RUN's (read from its run.json;
a different --session-field or --session-copies is refused), and REQUESTS.jsonl must be RUN's file. For each request it opens an engine session, sends that
prompt for one token, then appends RUN's output token ids to that context (the engine joins token
ids without re-tokenizing them and keeps the prompt's images) and returns, at every output position,
the log-probability of the output token and of the target's two most likely tokens. With
--score-top K the record adds the K most likely tokens and their log-probabilities (top_ids,
top_lps), untempered. Send it to the stock engine without speculation, with an FP32 output layer.
Writes SCORED_DIR/scores.jsonl and SCORED_DIR/run.json, which names RUN as source_run.
"""
import argparse
import collections
import hashlib
import json
import os
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


def prompt_fields(request):
    fields = {"input_ids": list(request["input_ids"])} if "input_ids" in request else {"text": request["text"]}
    if "image_data" in request:
        fields["image_data"] = request["image_data"]
    return fields


def new_part(previous, request):
    """What a turn's recorded prompt adds after the previous turn's recorded prompt and answer."""
    if "input_ids" in request:
        head = list(previous["input_ids"]) + list(previous["output_ids"])
        if list(request["input_ids"][: len(head)]) != head:
            raise ValueError(f"turn {request['id']} does not extend the previous turn's prompt and output_ids")
        return list(request["input_ids"][len(head):])
    head = previous["text"] + previous["output_text"]
    if not request["text"].startswith(head):
        raise ValueError(f"turn {request['id']} does not extend the previous turn's prompt and output_text")
    images = previous.get("image_data") or []
    if (request.get("image_data") or [])[: len(images)] != images:
        raise ValueError(f"turn {request['id']} does not keep the previous turn's images")
    return request["text"][len(head):]


def tasks(requests, field=None, copies=1, namespace=""):
    """[[request, ...]]: one list per task in the order it is sent. Without a field every request is a task of
    its own. With one, the requests sharing its value are one task's turns in file order; each turn after the
    first carries the new part of its recorded prompt; every copy of every task gets its own cache key."""
    if not field:
        return [[dict(r, id=str(r["id"]))] for r in requests]
    turns = collections.OrderedDict()
    for r in requests:
        if field not in r:
            raise ValueError(f"request {r.get('id')} has no field {field!r}")
        turns.setdefault(str(r[field]), []).append(r)
    parts = {name: [None] + [new_part(rows[t - 1], rows[t]) for t in range(1, len(rows))]
             for name, rows in turns.items()}
    out = []
    for copy in range(copies):
        for name, rows in turns.items():
            out.append([dict(r, id=f"{r['id']}#c{copy}" if copy else str(r["id"]), _turn=t, new_part=parts[name][t],
                             cache_key=f"{namespace}/c{copy}/{len(name)}:{name}") for t, r in enumerate(rows)])
    return out


def extend(prompt, answer, request):
    """The prompt a turn sends: the previous turn's prompt, its answer and the turn's new part, with its images."""
    if "input_ids" in request:
        fields = {"input_ids": prompt["input_ids"] + list(answer) + request["new_part"]}
    else:
        fields = {"text": prompt["text"] + answer + request["new_part"]}
    if "image_data" in request:
        fields["image_data"] = request["image_data"]
    return fields


def answer_of(record, request):
    """The previous turn's answer as the next turn's prompt holds it. The engine's text leaves out the stop
    token it matched; its output_ids end with it, and the token answer drops it unless the turn's recorded
    output_ids end with that token too."""
    if "input_ids" not in request:
        return record["text"]
    ids = list(record["output_ids"] or [])
    reason = record.get("finish_reason")
    matched = reason.get("matched") if isinstance(reason, dict) else None
    recorded = list(request.get("output_ids") or [])
    if isinstance(matched, int) and ids and ids[-1] == matched and recorded[-1:] != [matched]:
        ids = ids[:-1]
    return ids


def build_body(request, args, fields):
    """The /generate body of one request: its prompt, its cap and the run's sampling."""
    max_new = int(request["max_new_tokens"])
    if args.max_new_tokens_cap:
        max_new = min(max_new, args.max_new_tokens_cap)
    sampling = {"temperature": 0.0, "skip_special_tokens": False, "spaces_between_special_tokens": False}
    sampling.update(args.sampling)
    sampling["max_new_tokens"] = max_new
    body = dict(fields, sampling_params=sampling, rid=str(request["id"]))
    if request.get("cache_key"):
        body["extra_key"] = request["cache_key"]
    if getattr(args, "logprobs", None):
        body.update(return_logprob=True, top_logprobs_num=args.logprobs, logprob_start_len=-1)
    return body


def score_in_session(endpoint, request, fields, output_ids, args):
    """The meta_info of the scoring request: in one engine session, the prompt for one token, then the output
    ids appended to the prompt (without the generated token), scored from the prompt's last position on."""
    sid = http_json(endpoint + "/open_session", {"capacity_of_str_len": 1 << 30}, timeout=60)
    if not isinstance(sid, str) or not sid:
        raise RuntimeError(f"/open_session returned {sid!r}")
    try:
        first = dict(fields, rid=f"{request['id']}~prompt", session_params={"id": sid},
                     sampling_params={"max_new_tokens": 1, "temperature": 0.0})
        prompt_tokens = http_json(endpoint + "/generate", first, timeout=args.timeout)["meta_info"]["prompt_tokens"]
        second = {"input_ids": list(output_ids), "rid": str(request["id"]),
                  "session_params": {"id": sid, "rid": first["rid"], "drop_previous_output": True},
                  "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}, "return_logprob": True,
                  "logprob_start_len": prompt_tokens - 1, "top_logprobs_num": max(2, args.score_top or 2)}
        return finished(http_json(endpoint + "/generate", second, timeout=args.timeout).get("meta_info", {}))
    finally:
        try:
            http_json(endpoint + "/close_session", {"session_id": sid}, timeout=60)
        except Exception:  # noqa: BLE001 - an unclosed session expires; the score is already taken
            pass


def finished(meta):
    """A reply's meta_info, unless the engine aborted the request (it can still answer HTTP 200)."""
    reason = meta.get("finish_reason")
    if isinstance(reason, dict) and reason.get("type") == "abort":
        raise RuntimeError(f"the engine aborted the request: {reason}")
    return meta


def entry(item):
    """(logprob, token_id) of one logprob entry: [logprob, token_id, text] or a dict."""
    if isinstance(item, dict):
        return item.get("logprob"), item.get("token_id")
    return item[0], item[1]


def own_logprobs(output_ids, meta):
    """The engine's log-probabilities of a generated output (--logprobs): per output position the output
    token's and the K most likely tokens', up to the first position where the engine's token differs."""
    tokens = meta.get("output_token_logprobs") or []
    tops = meta.get("output_top_logprobs") or []
    record = {"lp_out": [], "top_ids": [], "top_lps": []}
    for position, expected in enumerate(output_ids or []):
        if position >= len(tokens) or position >= len(tops) or not tops[position]:
            break
        lp, token = entry(tokens[position])
        if token != expected or lp is None:
            break
        ranked = [entry(t) for t in tops[position]]
        record["lp_out"].append(lp)
        record["top_ids"].append([t for _, t in ranked])
        record["top_lps"].append([x for x, _ in ranked])
    return record


def score_record(request_id, output_ids, meta, top=False):
    """Per output position: token, target's top-1 token, and the three log-probabilities. Entry 0 of the
    engine's lists is the last prompt token (no log-probability); output position p is entry p + 1.
    With top, also the K most likely tokens and their log-probabilities (see --score-top)."""
    tokens = meta.get("input_token_logprobs") or []
    tops = meta.get("input_top_logprobs") or []
    record = {"id": request_id, "n_out": len(output_ids), "tok": [], "top1": [], "lp_tok": [], "lp_top1": [],
              "lp_top2": []}
    if top:
        record.update(top_ids=[], top_lps=[])
    for position, expected in enumerate(output_ids):
        j = position + 1
        if j >= len(tokens) or j >= len(tops) or not tops[j]:
            break
        lp_tok, token = entry(tokens[j])
        if token != expected or lp_tok is None:
            break
        ranked = [entry(t) for t in tops[j]]
        record["tok"].append(token)
        record["lp_tok"].append(lp_tok)
        record["top1"].append(ranked[0][1])
        record["lp_top1"].append(ranked[0][0])
        record["lp_top2"].append(ranked[1][0] if len(ranked) > 1 else None)
        if top:
            record["top_ids"].append([t for _, t in ranked])
            record["top_lps"].append([lp for lp, _ in ranked])
    record["n_scored"] = len(record["tok"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--concurrency", type=int, required=True, help="requests, or with --session-field tasks, in flight")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", default="run")
    parser.add_argument("--node", default=None, help="the engine's slot; paired_ab.py pairs runs of a cycle on two slots")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--trace-dir", default=None, help="the engine's --sched-trace-dir, as seen from here")
    parser.add_argument("--session-field", default=None, help="field naming each request's task; B then counts tasks")
    parser.add_argument("--session-copies", type=int, default=None, help="run the task list this many times")
    parser.add_argument("--max-new-tokens-cap", type=int, default=None, help="cap every request's max_new_tokens")
    parser.add_argument("--sampling", type=json.loads, default={}, help="JSON merged into sampling_params")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--score-from", default=None, help="score this run's outputs instead of generating")
    parser.add_argument("--score-top", type=int, default=None, help="with --score-from: the K most likely tokens")
    parser.add_argument("--logprobs", type=int, default=None,
                        help="keep the engine's own log-probabilities of the K most likely tokens at every output position")
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    if "max_new_tokens" in args.sampling:
        print("--sampling must not set max_new_tokens: the request file and --max-new-tokens-cap set it",
              file=sys.stderr)
        return 1
    if args.logprobs and args.score_from:
        print("--logprobs keeps a generated run's own log-probabilities; a scoring run uses --score-top",
              file=sys.stderr)
        return 1
    with open(args.requests, "rb") as f:
        requests_sha = hashlib.sha256(f.read()).hexdigest()
    if args.score_from:
        with open(os.path.join(args.score_from, "run.json")) as f:
            source = json.load(f)
        for name in ("session_field", "session_copies"):
            given, recorded = getattr(args, name), source.get(name)
            if given is not None and given != recorded:
                print(f"--{name.replace('_', '-')} {given}: {args.score_from} ran with {recorded}", file=sys.stderr)
                return 1
            setattr(args, name, recorded)
        if source.get("requests_sha256") not in (None, requests_sha):
            print(f"{args.requests} is not the request file {args.score_from} ran", file=sys.stderr)
            return 1
    args.session_copies = args.session_copies or 1
    namespace = hashlib.sha256(f"{os.path.abspath(args.out)}:{time.time()}".encode()).hexdigest()[:12]
    try:
        items = tasks(read_jsonl(args.requests), args.session_field, args.session_copies, namespace)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if not args.score_from and len(items) < args.concurrency:
        print(f"{len(items)} {'tasks' if args.session_field else 'requests'} at batch size {args.concurrency}: "
              "the run could not keep that many in flight; pass --session-copies", file=sys.stderr)
        return 1
    previous = {}
    if args.score_from:
        previous = {str(r["id"]): r for r in read_jsonl(os.path.join(args.score_from, "outputs.jsonl"))}
        missing = [r["id"] for item in items for r in item if r["id"] not in previous]
        if missing:
            print(f"{len(missing)} requests have no output in {args.score_from}, e.g. {missing[:3]}", file=sys.stderr)
            return 1
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "scores.jsonl" if args.score_from else "outputs.jsonl")
    if os.path.exists(out_path):
        print(f"{out_path} exists; use a new run directory", file=sys.stderr)
        return 1
    info = server_info(endpoint)
    lock = threading.Lock()
    pending = iter(items)
    results, failures = [], []
    out = open(out_path, "w")

    def send(request, fields):
        rid = request["id"]
        began = time.time()
        if args.score_from:
            output_ids = previous[rid]["output_ids"]
            meta = score_in_session(endpoint, request, fields, output_ids, args)
            return score_record(rid, output_ids, meta, bool(args.score_top))
        body = build_body(request, args, fields)
        reply = http_json(endpoint + "/generate", body, timeout=args.timeout)
        meta = finished(reply.get("meta_info", {}))
        record = {"id": rid, "output_ids": reply.get("output_ids"), "text": reply.get("text"),
                  "finish_reason": meta.get("finish_reason"), "prompt_tokens": meta.get("prompt_tokens"),
                  "completion_tokens": meta.get("completion_tokens"),
                  "max_new_tokens": body["sampling_params"]["max_new_tokens"], "turn": request.get("_turn"),
                  "spec_verify_ct": meta.get("spec_verify_ct"),
                  "spec_num_correct_drafts": meta.get("spec_num_correct_drafts"),
                  "spec_accept_length": meta.get("spec_accept_length"), "latency_s": time.time() - began}
        if args.logprobs:
            record.update(own_logprobs(record["output_ids"], meta))
        return record

    def worker():
        while True:
            with lock:
                item = next(pending, None)
            if item is None:
                return
            prompt = answer = None
            for request in item:
                try:
                    fields = prompt_fields(request) if not request.get("_turn") else extend(prompt, answer, request)
                    record = send(request, fields)
                except Exception as error:  # noqa: BLE001 - every failure is recorded and fails the run
                    detail = error.read().decode(errors="replace")[:500] if isinstance(error, urllib.error.HTTPError) else ""
                    with lock:
                        failures.append({"id": request["id"], "error": f"{type(error).__name__}: {error} {detail}".strip()})
                    break  # the task's later turns build on this one
                with lock:
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    results.append(record)
                source = previous[request["id"]] if args.score_from else record
                try:
                    prompt, answer = fields, answer_of(source, request)
                except Exception as error:  # noqa: BLE001 - recorded; the task cannot continue
                    with lock:
                        failures.append({"id": request["id"], "error": f"{type(error).__name__}: {error}"})
                    break

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(args.concurrency)]
    for future in futures:
        if future.exception() is not None:
            failures.append({"id": None, "error": f"a worker stopped: {future.exception()!r}"})
    t_end = time.time()
    out.close()

    n_requests = sum(len(item) for item in items)
    run = {"arm": args.arm, "node": args.node or endpoint.split("//")[-1].split(":")[0], "cycle": args.cycle,
           "concurrency": args.concurrency, "mode": "score" if args.score_from else "generate",
           "session_field": args.session_field, "session_copies": args.session_copies,
           "n_sessions": len(items) if args.session_field else None, "source_run": args.score_from,
           "requests": os.path.abspath(args.requests), "requests_sha256": requests_sha, "n_requests": n_requests,
           "n_done": len(results), "n_failed": len(failures), "failures": failures[:20], "t_start": t_start,
           "t_end": t_end, "max_new_tokens_cap": args.max_new_tokens_cap, "sampling": args.sampling,
           "score_top": args.score_top, "logprobs": args.logprobs, "trace_dir": args.trace_dir,
           "server_info": info}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1)

    print(f"{args.arm} cycle {args.cycle} on {run['node']}: {len(results)} of {n_requests} done, "
          f"{len(failures)} failed, {t_end - t_start:.1f} s")
    if args.score_from:
        print(f"scored {sum(r['n_scored'] for r in results)} of {sum(r['n_out'] for r in results)} output positions")
    elif results:
        tokens = sum(len(r["output_ids"] or []) for r in results)
        print(f"output tokens {tokens}, {tokens / (t_end - t_start):.1f} tokens/s over the whole run")
        verify = [r for r in results if r.get("spec_verify_ct")]
        if verify:
            pooled = (sum(r["spec_num_correct_drafts"] + r["spec_verify_ct"] for r in verify)
                      / sum(r["spec_verify_ct"] for r in verify))
            per_request = statistics.mean(r["spec_accept_length"] for r in verify)
            print(f"accepted tokens per verify step: {pooled:.3f} pooled, {per_request:.3f} mean over requests")
    for failure in failures[:5]:
        print(f"failed {failure['id']}: {failure['error']}", file=sys.stderr)
    unsent = n_requests - len(results) - len(failures)
    if unsent and not failures:
        print(f"{unsent} requests were never sent", file=sys.stderr)
    return 1 if failures or len(results) != n_requests else 0


if __name__ == "__main__":
    raise SystemExit(main())
