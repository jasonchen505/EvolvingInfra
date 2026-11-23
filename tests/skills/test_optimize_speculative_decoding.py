# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The measurement and decision scripts of skills/optimize-speculative-decoding, on synthetic inputs."""
import collections
import hashlib
import importlib.util
import json
import math
import random
import re
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "skills" / "optimize-speculative-decoding" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_is_disjoint_stable_and_rejects_duplicates():
    split = script("split_requests")
    lines = [json.dumps({"id": f"r{i}"}) for i in range(200)]
    explore, (sealed,) = split.split(lines, "study", 0.3)
    again = split.split(list(reversed(lines)), "study", 0.3)
    ids = lambda rows: {json.loads(r)["id"] for r in rows}  # noqa: E731
    assert ids(explore) | ids(sealed) == {f"r{i}" for i in range(200)}
    assert not ids(explore) & ids(sealed)
    assert ids(again[1][0]) == ids(sealed)
    assert 30 < len(sealed) < 90
    with pytest.raises(ValueError, match="duplicate"):
        split.split(lines + lines[:1], "study", 0.3)


def test_split_keeps_the_requests_of_a_group_together():
    split = script("split_requests")
    lines = [json.dumps({"id": f"g{g}-r{r}", "group": f"g{g}"}) for g in range(60) for r in range(4)]
    explore, (sealed,) = split.split(lines, "study", 0.3, group_field="group")
    groups = lambda rows: {json.loads(r)["group"] for r in rows}  # noqa: E731
    assert not groups(explore) & groups(sealed)
    assert len(sealed) % 4 == 0 and len(explore) % 4 == 0


def record(i, mode, bs, launch, tokens=0):
    return {"forward_iter": i, "mode": mode, "bs": bs, "launch_bs": bs, "launch_mono": launch,
            "ts": 1000.0 + launch, "decode_tokens": tokens, "after_idle": False}


def test_decode_steps_count_every_decode_step_at_its_own_batch_but_the_one_after_a_prefill(tmp_path):
    steps = script("decode_steps")
    records = [
        record(0, "extend", 2, 0.000),
        record(1, "decode", 2, 0.050, 6),   # first decode after a prefill: left out
        record(2, "decode", 2, 0.060, 8),   # 12 ms to the next launch
        record(3, "decode", 2, 0.072, 4),   # 8 ms
        record(4, "decode", 1, 0.080, 3),   # one request in the batch while a task is between turns: 10 ms
        record(5, "decode", 2, 0.090, 6),   # 10 ms
        record(6, "decode", 2, 0.100, 6),   # last record: no next launch
    ]
    found, after_prefill = steps.plain_steps(records)
    assert after_prefill == 1
    assert [round(s[0] * 1e3, 6) for s in found] == [12.0, 8.0, 10.0, 10.0]
    assert [(s[1], s[2]) for s in found] == [(8, 2), (4, 2), (3, 1), (6, 2)]
    contexts = [dict(r, launch_seq_lens_sum=100 * r["bs"]) for r in records]
    (tmp_path / "iter_trace.dp0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in contexts))
    result = steps.measure(str(tmp_path))
    # tokens per request per step: committed tokens over the requests the steps processed
    assert result["tau"] == pytest.approx(21 / 7) and result["step_ms_mean"] == pytest.approx(10.0)
    assert result["batch_mix"] == {1: 1, 2: 3} and result["batch_mean"] == pytest.approx(1.75)
    assert result["context_mean"] == pytest.approx(100.0)
    # two windows, as the profile windows of one run: the steps of both, pooled
    pooled = steps.measure(str(tmp_path), windows=[(1000.0, 1000.075), (1000.075, 1000.2)])
    assert pooled["steps"] == 3 and pooled["batch_mix"] == {1: 1, 2: 2} and pooled["tau"] == pytest.approx(17 / 5)


def test_plain_steps_leave_out_a_step_whose_next_launch_is_a_prefill_or_follows_an_idle_wait():
    steps = script("decode_steps")
    records = [
        record(0, "decode", 1, 0.000, 1),
        record(1, "decode", 1, 0.010, 1),   # plain: 10 ms
        record(2, "decode", 1, 0.020, 1),   # next launch is a prefill 5 s later: left out
        record(3, "extend", 1, 5.020),
        record(4, "decode", 1, 5.100, 1),   # first decode after the prefill: left out
        record(5, "decode", 1, 5.110, 1),   # next forward follows an idle wait: left out
        dict(record(6, "decode", 1, 9.000, 1), after_idle=True),
        record(7, "decode", 1, 9.010, 1),
    ]
    found, _ = steps.plain_steps(records)
    assert [round(s[0] * 1e3, 6) for s in found] == [10.0]


def test_scoring_skips_the_empty_first_entry_and_stops_where_tokens_differ():
    client = script("fixed_batch_client")
    # the engine's lists start at logprob_start_len (the last prompt token), whose entry has no log-probability
    meta = {"input_token_logprobs": [[None, 9, None], [-0.1, 5, None], [-1.2, 6, None], [-0.3, 8, None]],
            "input_top_logprobs": [None, [[-0.1, 5, None], [-2.0, 1, None]], [[-0.4, 3, None], [-1.2, 6, None]],
                                   [[-0.3, 8, None], [-0.9, 2, None]]]}
    record = client.score_record("r", [5, 6, 7], meta)
    assert record["n_scored"] == 2
    assert record["tok"] == [5, 6] and record["top1"] == [5, 3]
    assert record["lp_top2"] == [-2.0, -1.2]


def test_profile_steps_cut_at_the_largest_graph():
    profile = script("profile_steps")
    events, correlation = [], 0
    for step in range(4):
        start = step * 1000.0
        for graph, count in (("draft", 3), ("target", 10)):
            correlation += 1
            events.append({"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ph": "X", "ts": start,
                           "dur": 5, "args": {"correlation": correlation}})
            offset = 0 if graph == "draft" else 200
            for k in range(count):
                events.append({"cat": "kernel", "ph": "X", "name": f"{graph}_gemm_{k}", "ts": start + offset + 50 * k,
                               "dur": 40, "args": {"correlation": correlation}})
    classes = [(label, re.compile(pattern)) for label, pattern in profile.CLASSES]
    summary = profile.summarize(events, classes)
    assert summary["steps"] == 3
    assert summary["wall_us_median"] == pytest.approx(1000.0)
    assert summary["busy_us_median"] == pytest.approx(13 * 40.0)


def test_byte_floor_reads_weight_bytes_from_safetensors_headers(tmp_path):
    floor = script("byte_floor")
    header = {"model.layers.0.mlp.experts.0.w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
              "model.layers.0.mlp.experts.1.w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [32, 64]},
              "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [64, 128]},
              "model.embed_tokens.weight": {"dtype": "BF16", "shape": [16, 4], "data_offsets": [128, 256]}}
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(256))
    (tmp_path / "config.json").write_text(json.dumps({"num_experts": 2, "num_experts_per_tok": 1}))
    args = SimpleNamespace(model_dir=str(tmp_path), requests=1, rows_per_request=1, context_tokens=0, kv_heads=None,
                           peak_tflops=1.0, peak_tbps=1.0, gpus=1, kv_bytes_per_token=0, state_bytes_per_request=0,
                           draft_model_dir=None, draft_forwards=1, params=None, experts=None, experts_per_token=None,
                           attention_layers=None, prefill_tokens=0, prefill_chunk=0, prefill_passes_per_step=0,
                           recurrent_flops_per_row=0, draft_rows_per_request=0,
                           within_step_pairs=None, prefill_prefix_tokens=0,
                           expert_pattern=r"\.experts\.",
                           exclude=r"(^|\.)(visual|vision_tower|vision_model|mm_projector)\.|embed_tokens")
    result = floor.floor(args)
    assert result["tensors_counted"] == 3
    assert result["expert_share_read"] == pytest.approx(0.5)
    assert result["classes"]["weights"]["bytes"] == pytest.approx(64 + 64 * 0.5)
    assert floor.experts_reached(8, 2, 1) == pytest.approx(2.0)


def test_profile_steps_puts_the_samplers_sort_scan_and_hash_kernels_in_sampling():
    profile = script("profile_steps")
    classes = [(label, re.compile(pattern)) for label, pattern in profile.CLASSES]
    for name in ("void cub::DeviceRadixSortOnesweepKernel<float>", "void cub::DeviceScanKernel<int>",
                 "murmur_hash32_kernel", "gumbel_noise_kernel"):
        assert profile.classify(name, classes) == "sampling_and_drafting", name
    assert profile.classify("selective_scan_fwd_kernel", classes) == "attention"


def test_byte_floor_reads_kv_heads_copied_to_several_gpus_once_per_gpu(tmp_path):
    floor = script("byte_floor")
    header = {"model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(64))
    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 8, "num_key_value_heads": 4,
                                                      "head_dim": 4, "num_hidden_layers": 2}))

    def attention_ms(gpus):
        args = SimpleNamespace(model_dir=str(tmp_path), requests=2, rows_per_request=1, context_tokens=100,
                               peak_tflops=1e9, peak_tbps=1.0, gpus=gpus, kv_bytes_per_token=1000, kv_heads=None,
                               state_bytes_per_request=0, draft_model_dir=None, draft_forwards=1, params=None,
                               experts=None, experts_per_token=None, attention_layers=None, prefill_tokens=0,
                               prefill_chunk=0, prefill_passes_per_step=0, recurrent_flops_per_row=0,
                               draft_rows_per_request=0, within_step_pairs=None, prefill_prefix_tokens=0,
                               expert_pattern=r"\.experts\.", exclude=r"embed_tokens")
        return floor.floor(args)["classes"]["attention"]["floor_ms"]

    # 4 KV heads: past 4 GPUs each GPU still reads a quarter of the KV
    assert attention_ms(8) == pytest.approx(attention_ms(4))
    assert attention_ms(2) == pytest.approx(2 * attention_ms(4))
    assert attention_ms(4) == pytest.approx(1e3 * 1000 * 100 * 2 / 4 / 0.9e12)


def test_tasks_send_their_turns_in_order_with_the_live_answers_and_their_own_cache_key(monkeypatch, tmp_path):
    client = script("fixed_batch_client")
    rows = []
    for s in range(3):
        prompt = f"<s{s}>obs0"
        for t in range(3):
            rows.append({"id": f"s{s}-t{t}", "episode": f"s{s}", "text": prompt, "image_data": [f"s{s}f{u}" for u in range(t + 1)],
                         "output_text": f"<rec{s}{t}>", "max_new_tokens": 9000})
            prompt = prompt + f"<rec{s}{t}>" + f"<end>obs{t + 1}"
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
    threading = __import__("threading")
    lock, full, in_flight, sent = threading.Lock(), threading.Event(), {}, []

    def engine(url, body=None, timeout=0):
        key = body["extra_key"]
        with lock:
            assert key not in in_flight, "two turns of one task in flight"
            in_flight[key] = body["text"]
            if len(in_flight) == 4:
                full.set()
            sent.append((key, body["rid"], body["text"], body["image_data"]))
        full.wait(timeout=5)  # the first four requests return only once four tasks are in flight
        with lock:
            del in_flight[key]
        return {"output_ids": [1, 0], "text": f"<live:{body['rid']}>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": 0}}}

    monkeypatch.setattr(client, "http_json", engine)
    monkeypatch.setattr(client, "server_info", lambda endpoint: None)
    monkeypatch.setattr(sys, "argv", ["client", "--endpoint", "http://e", "--requests", str(requests), "--concurrency",
                                      "4", "--out", str(tmp_path / "run"), "--session-field", "episode",
                                      "--session-copies", "2", "--max-new-tokens-cap", "8192"])
    assert client.main() == 0
    assert len(sent) == 18 and len({key for key, *_ in sent}) == 6  # every copy of every task its own cache key
    by_key = collections.defaultdict(list)
    for key, rid, text, images in sent:
        by_key[key].append((rid, text, images))
    for turns in by_key.values():
        rid0 = turns[0][0]
        s, copy = rid0[1], rid0.split("#c")[1] if "#c" in rid0 else None
        suffix = f"#c{copy}" if copy else ""
        assert [rid for rid, *_ in turns] == [f"s{s}-t{t}{suffix}" for t in range(3)]
        # each turn: the previous prompt, the answer the engine just gave, then the recorded new observation
        assert turns[1][1] == f"<s{s}>obs0" + f"<live:s{s}-t0{suffix}>" + "<end>obs1"
        assert turns[2][1] == turns[1][1] + f"<live:s{s}-t1{suffix}>" + "<end>obs2"
        assert turns[2][2] == [f"s{s}f{u}" for u in range(3)]
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["n_sessions"] == 6 and run["n_done"] == 18 and run["session_copies"] == 2
    # a turn whose recorded prompt does not extend the previous prompt and answer is refused
    rows[1]["text"] = "something else"
    requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["client", "--endpoint", "http://e", "--requests", str(requests), "--concurrency",
                                      "4", "--out", str(tmp_path / "run2"), "--session-field", "episode"])
    assert client.main() == 1


def test_client_refuses_max_new_tokens_in_sampling(monkeypatch, tmp_path):
    client = script("fixed_batch_client")
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps({"id": "r", "text": "x", "max_new_tokens": 10}) + "\n")
    monkeypatch.setattr(sys, "argv", ["client", "--endpoint", "http://engine", "--requests", str(requests),
                                      "--concurrency", "1", "--out", str(tmp_path / "run"),
                                      "--sampling", '{"max_new_tokens": 4096}'])
    assert client.main() == 1


def test_decode_steps_reports_the_mean_context_of_the_counted_steps(tmp_path):
    steps = script("decode_steps")
    records = [dict(record(i, "decode", 2, 0.01 * i, 6), launch_seq_lens_sum=1000 * (i + 1)) for i in range(5)]
    (tmp_path / "iter_trace.dp0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    result = steps.measure(str(tmp_path))
    assert result["steps"] == 3
    assert result["context_mean"] == pytest.approx((2000 + 3000 + 4000) / 3 / 2)


def test_exact_replay_accepts_the_longest_path_of_the_best_first_tree():
    replay = script("exact_replay")
    step = {"id": "r", "pos": 0, "cands": [[10, 11], [20, 21], [30, 31]],
            "logp": [[-0.1, -2.3], [-0.2, -1.7], [-0.3, -1.4]]}
    # nodes by path score: 10, 10-20, 10-20-30, 10-20-31, 10-21, 10-21-30, 11, ...
    result = replay.replay_step(step, [10, 21, 30, 99], [2, 5, 6])
    assert result["chain"] == 2      # the chain 10-20-30 stops at 21
    assert result["oracle"] == 4     # 10, 21 and 30 are all candidates
    assert (result[2], result[5], result[6]) == (2, 3, 4)


def test_exact_replay_scores_children_given_their_parent_with_pair_scores():
    replay = script("exact_replay")
    step = {"id": "r", "pos": 0, "cands": [[1, 2], [3, 4]], "logp": [[0.0, -1.0], [0.0, 0.0]],
            "pair": [None, [[5.0, -5.0], [-5.0, 5.0]]]}
    # after 1 the drafter wants 3, after 2 it wants 4: nodes 1, 1-3, 2, 2-4
    result = replay.replay_step(step, [2, 4, 99], [3, 4])
    assert result["chain"] == 1 and result[3] == 2 and result[4] == 3
    # the output ends after 4: nothing follows the last accepted token
    assert replay.replay_step(step, [2, 4], [4])[4] == 2


def pair_run(arm, slot, cycle, rates, outputs=None, concurrency=8, window=None):
    requests = {f"r{i}": {"rate": rate, "output_ids": (outputs or {}).get(f"r{i}", [i])}
                for i, rate in enumerate(rates)}
    t0 = window if window is not None else 100.0 * cycle
    return {"dir": f"{arm}-{slot}-{cycle}", "arm": arm, "slot": slot, "cycle": cycle, "concurrency": concurrency,
            "t_start": t0, "t_end": t0 + 90.0, "failed": 0, "n_requests": len(requests),
            "same": {"sampling": {}, "max_new_tokens_cap": 2048, "mode": "generate", "session_field": None,
                     "logprobs": None},
            "per_batch": {"requests_sha256": "x", "session_copies": 1, "n_requests": len(requests)},
            "requests": requests}


def paired_runs(gain, cand_outputs=None):
    rates = [100.0 + 7 * i for i in range(20)]
    runs = [pair_run("base", "a", 0, rates), pair_run("base", "b", 0, [r * 1.002 for r in rates]),
            pair_run("base", "a", 1, [r * 0.999 for r in rates]), pair_run("base", "b", 1, rates)]
    for cycle, (base_slot, cand_slot) in enumerate([("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")], start=2):
        jitter = 1 + 0.002 * (cycle % 2)
        runs.append(pair_run("base", base_slot, cycle, rates))
        runs.append(pair_run("cand", cand_slot, cycle, [r * gain * jitter for r in rates], cand_outputs))
    return runs


def test_paired_comparison_promotes_on_per_request_rates_against_the_current_best():
    ab = script("paired_ab")
    (decision,) = ab.compare(paired_runs(1.10), "base", "cand", noise={8: 0.002}, expect_pairs=4)
    assert decision["pairs"] == 4
    assert decision["gain"] == pytest.approx(1.101, abs=0.002)
    assert set(decision["per_slot"]) == {"a", "b"}
    assert not decision["diverged"] and decision["threshold"] < 0.02
    assert decision["promote"]


def test_paired_comparison_asks_three_percent_when_outputs_diverge():
    ab = script("paired_ab")
    other = {f"r{i}": [i, 7] for i in range(20)}
    (small,) = ab.compare(paired_runs(1.02, other), "base", "cand", noise={8: 0.002})
    assert small["diverged"] and small["threshold"] == pytest.approx(0.03)
    assert not small["promote"]
    (clear,) = ab.compare(paired_runs(1.06, other), "base", "cand", noise={8: 0.002})
    assert clear["promote"]


def test_sealed_final_passes_when_faster_than_stock_at_every_batch_size_without_noise():
    ab = script("paired_ab")
    rates = [100.0 + 7 * i for i in range(20)]

    def sealed(gains):
        runs = []
        for cycle, ((base_slot, cand_slot), gain) in enumerate(zip([("a", "b"), ("b", "a")], gains)):
            runs.append(pair_run("base", base_slot, cycle, rates))
            runs.append(pair_run("cand", cand_slot, cycle, [r * gain for r in rates]))
        return runs

    # two pairs, outputs differ, too few for step 5's interval rule: the sealed final only asks for faster
    (faster,) = ab.compare(sealed([1.13, 1.29]), "base", "cand", expect_pairs=2, sealed=True)
    assert faster["promote"] and faster["threshold"] is None
    (slower,) = ab.compare(sealed([0.99, 1.005]), "base", "cand", expect_pairs=2, sealed=True)
    assert not slower["promote"]


def test_paired_comparison_needs_frozen_noise_and_swapped_slots():
    ab = script("paired_ab")
    with pytest.raises(ValueError, match="needs the stock-against-stock noise"):
        ab.compare(paired_runs(1.1), "base", "cand")
    runs = [dict(r, slot="b" if r["arm"] == "cand" else "a") if r["cycle"] >= 2 else r for r in paired_runs(1.1)]
    with pytest.raises(ValueError, match="swap slots evenly"):
        ab.compare(runs, "base", "cand", noise={8: 0.002})
    third = [dict(r, slot="c") if r["arm"] == "cand" and r["cycle"] == 5 else r for r in paired_runs(1.1)]
    with pytest.raises(ValueError, match="exactly two slots"):
        ab.compare(third, "base", "cand", noise={8: 0.002})
    with pytest.raises(ValueError, match="pairs, expected 6"):
        ab.compare(paired_runs(1.1), "base", "cand", noise={8: 0.002}, expect_pairs=6)


def test_noise_only_needs_four_base_pairs_per_batch():
    ab = script("paired_ab")
    with pytest.raises(ValueError, match="need at least 4"):
        ab.noise_only(paired_runs(1.1), "base")
    rates = [100.0 + 7 * i for i in range(20)]
    runs = []
    for cycle, jitter in enumerate((1.002, 0.998, 1.001, 0.999)):
        runs += [pair_run("base", "a", cycle, rates), pair_run("base", "b", cycle, [r * jitter for r in rates])]
    result = ab.noise_only(runs, "base", [8])
    assert result[8]["pairs"] == 4 and result[8]["noise"] == pytest.approx(0.0016, abs=2e-4)


def test_paired_comparison_refuses_runs_that_are_not_simultaneous_or_complete():
    ab = script("paired_ab")
    runs = paired_runs(1.1)
    late = [dict(r, t_start=r["t_start"] + 30, t_end=r["t_end"] + 30) if r["arm"] == "cand" else r for r in runs]
    with pytest.raises(ValueError, match="same time"):
        ab.compare(late, "base", "cand", noise={8: 0.002})
    partial = [dict(r, requests={k: v for k, v in r["requests"].items() if k != "r19"}) if r["arm"] == "cand" else r
               for r in runs]
    with pytest.raises(ValueError, match="to the end"):
        ab.compare(partial, "base", "cand", noise={8: 0.002})
    failed = [dict(r, failed=1) if r["arm"] == "cand" and r["cycle"] == 3 else r for r in runs]
    with pytest.raises(ValueError, match="to the end"):
        ab.compare(failed, "base", "cand", noise={8: 0.002})
    with pytest.raises(ValueError, match="no run of arm"):
        ab.compare(runs, "base", "other", noise={8: 0.002})


def test_paired_rates_count_the_returned_tokens(tmp_path):
    ab = script("paired_ab")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"arm": "a", "node": "s", "cycle": 0, "concurrency": 1,
                                              "n_requests": 1}))
    (run / "outputs.jsonl").write_text(json.dumps({"id": "r", "output_ids": [1] * 20, "completion_tokens": 22,
                                                   "latency_s": 1.0}) + "\n")
    assert ab.load_run(str(run))["requests"]["r"]["rate"] == pytest.approx(20.0)


def test_paired_gain_weights_each_group_equally():
    ab = script("paired_ab")
    base = pair_run("base", "a", 2, [100.0] * 4)
    cand = pair_run("cand", "b", 2, [120.0, 100.0, 100.0, 100.0])
    groups = {"r0": "math", "r1": "code", "r2": "code", "r3": "code"}
    gain, n, _ = ab.pair_gain(base, cand, groups)
    assert n == 4 and gain == pytest.approx((1.2 * 1.0) ** 0.5)


def write_jsonl(path, rows, name):
    path.mkdir(exist_ok=True)
    with open(path / name, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def test_token_forcing_sets_delta_from_the_gap_between_the_two_departing_tokens():
    forcing = script("token_forcing")
    ar = {"r0": [1, 2, 3, 4], "r1": [5, 6, 7]}
    stock = {"r0": [1, 2, 9, 4], "r1": [5, 6, 7]}
    scored = [{"id": "r0", "tok": [1, 2, 9, 4], "top1": [1, 2, 3, 4], "lp_tok": [-0.1, -0.1, -0.8, -0.1],
               "lp_top1": [-0.1, -0.1, -0.6, -0.1], "top_ids": [[1], [2], [3, 9], [4]],
               "top_lps": [[-0.1], [-0.1], [-0.6, -0.8], [-0.1]]}]
    delta, margins, missing = forcing.calibrate(ar, stock, [(0, r) for r in scored])
    assert margins == [pytest.approx(0.2)] and delta == 0.25 and missing == 0
    no_top = [dict(scored[0], top_ids=[[1], [2], [3], [4]], top_lps=[[-0.1], [-0.1], [-0.6], [-0.1]])]
    assert forcing.calibrate(ar, stock, [(0, r) for r in no_top])[0] is None


def forced(rows, run=0):
    return [(run, {"id": f"r{i}", "tok": [p[0] for p in positions], "top1": [p[1] for p in positions],
                   "lp_tok": [p[2] for p in positions], "lp_top1": [p[3] for p in positions],
                   "n_out": len(positions), "n_scored": len(positions)})
            for i, positions in enumerate(rows)]


def test_token_forcing_rejects_mismatches_with_margins_above_two_delta():
    forcing = script("token_forcing")
    near_tie = (1, 2, -0.7, -0.6)          # mismatch, margin 0.1
    agree = (5, 5, -0.01, -0.01)
    confident = (3, 4, -3.0, -0.01)        # mismatch, margin about 3
    base = forced([[near_tie] + [agree] * 49] * 40)
    same = forced([[near_tie] + [agree] * 49] * 40)
    worse = forced([[near_tie, confident] + [agree] * 48] * 40)
    equal = forcing.check(base, same, 0.25, resamples=300)
    assert equal["rate_verdict"] == "pass" and equal["event_verdict"] == "pass" and equal["cand_events"] == 0
    bad = forcing.check(base, worse, 0.25, resamples=300)
    assert bad["cand_events"] == 40 and bad["event_verdict"] == "fail" and bad["rate_verdict"] == "fail"
    # the scorer's first choice changes with the prefill chunk at those positions: the audit explains them
    rescored = [(i, dict(r, top1=[r["top1"][0], 9] + r["top1"][2:])) for i, r in worse]
    audited = forcing.check(base, worse, 0.25, rescored=rescored, resamples=300)
    assert audited["cand_events"] == 0 and audited["event_verdict"] == "pass"


def test_token_forcing_pairs_requests_and_refuses_missing_evidence(tmp_path):
    forcing = script("token_forcing")
    clean = (5, 5, -0.01, -0.01)
    miss = (1, 2, -0.7, -0.6)
    rows = [[clean] * 100, [miss] * 10 + [clean] * 90]
    paired = forcing.check(forced(rows), forced(rows), 0.25, resamples=300)
    assert paired["rate_interval"] == [0.0, 0.0] and paired["rate_verdict"] == "pass"
    with pytest.raises(ValueError, match="different requests"):
        forcing.check(forced(rows), forced(rows[:1]), 0.25)
    # events both systems have are explained on both sides; the candidate's own ones still count
    shared = (3, 4, -3.0, -0.01)
    both = [[shared] + [clean] * 99] * 6
    only = [[shared] + [clean] * 99] * 6 + [[clean] * 50 + [shared] + [clean] * 49] * 6
    result = forcing.check(forced(both + [[clean] * 100] * 6), forced(only), 0.25, resamples=300)
    assert result["base_events"] == 0 and result["cand_events"] == 6 and result["event_verdict"] == "fail"
    with pytest.raises(ValueError, match="both stock and candidate"):
        forcing.same_requests([], [], ["run1"], ["run1"])
    run = tmp_path / "run"
    write_jsonl(run, [{"id": "r0", "output_ids": [1] * 4096}], "outputs.jsonl")
    (run / "run.json").write_text(json.dumps({"n_requests": 1, "n_failed": 0}))
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "scores.jsonl").write_text(json.dumps({"id": "r0", "tok": [1], "top1": [1], "lp_tok": [0], "lp_top1": [0],
                                                     "n_out": 4096, "n_scored": 1}) + "\n")
    (scored / "run.json").write_text(json.dumps({"server_info": {"speculative_algorithm": None,
                                                                 "enable_fp32_lm_head": True},
                                                 "source_run": str(run)}))
    with pytest.raises(ValueError, match="scored 1 of 4096"):
        forcing.load_scored([str(scored)])
    (scored / "run.json").write_text(json.dumps({"server_info": {"speculative_algorithm": "DFLASH",
                                                                 "enable_fp32_lm_head": True},
                                                 "source_run": str(run)}))
    with pytest.raises(ValueError, match="speculative"):
        forcing.load_scored([str(scored)])


SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": -1}
PROBS = [0.4, 0.2, 0.15, 0.1, 0.07, 0.05, 0.02, 0.01]


def noisy(rng, probs, sd=1e-3):
    """A distribution as one engine computes it: each probability off by a relative rounding error."""
    raw = [p * (1 + rng.gauss(0, sd)) for p in probs]
    return [math.log(x / sum(raw)) for x in raw]


def positions(rng, n_requests, n_positions, own=lambda rng, r, k: PROBS):
    ids = list(range(len(PROBS)))
    return {f"r{r}": [(ids, noisy(rng, own(rng, r, k)), ids, noisy(rng, PROBS)) for k in range(n_positions)]
            for r in range(n_requests)}


def test_probability_check_passes_rounding_and_fails_a_shift_or_a_few_wrong_positions():
    forcing = script("token_forcing")
    rng = random.Random(5)
    base = [(8, SAMPLING, positions(rng, 120, 100))]
    same = forcing.probability(base, [(8, SAMPLING, positions(rng, 120, 100))])["batches"][8]
    assert same["level_verdict"] == "pass" and same["event_verdict"] == "pass" and same["cand_events"] == 0
    # every distribution moved 3 % toward uniform, as a lower-precision kernel moves them a little
    shifted = lambda rng, r, k: [0.97 * p + 0.03 / len(PROBS) for p in PROBS]  # noqa: E731
    shift = forcing.probability(base, [(8, SAMPLING, positions(rng, 120, 100, shifted))])["batches"][8]
    assert shift["level_verdict"] == "fail"
    # ten positions of one request wrong, as a state reused by a later turn would make them
    wrong = lambda rng, r, k: PROBS[::-1] if r == 7 and k < 10 else PROBS  # noqa: E731
    few = forcing.probability(base, [(8, SAMPLING, positions(rng, 120, 100, wrong))])["batches"][8]
    assert few["level_verdict"] == "pass" and few["cand_events"] == 10 and few["event_verdict"] == "fail"


def test_probability_check_refuses_an_uncovered_decoding_rule_too_few_positions_or_other_requests():
    forcing = script("token_forcing")
    rng = random.Random(6)
    base = [(8, SAMPLING, positions(rng, 120, 100))]
    short = {rid: [(o, ol, ids[:3], lps[:3]) for o, ol, ids, lps in rows]
             for rid, rows in positions(rng, 120, 100).items()}
    with pytest.raises(ValueError, match="every token the decoding rule can output"):
        forcing.probability(base, [(8, SAMPLING, short)])
    assert forcing.covered(short["r0"][0][3], top_k=3, top_p=0.95)
    with pytest.raises(ValueError, match="fewer than 10000"):
        forcing.probability([(8, SAMPLING, positions(rng, 50, 100))], [(8, SAMPLING, positions(rng, 50, 100))])
    with pytest.raises(ValueError, match="different requests"):
        forcing.probability(base, [(8, SAMPLING, positions(rng, 121, 100))])
    with pytest.raises(ValueError, match="batch size 8 is paired with a candidate run at 16"):
        forcing.probability(base, [(16, SAMPLING, positions(rng, 120, 100))])


def test_probability_check_reads_runs_kept_at_temperature_one_with_one_k(tmp_path):
    forcing = script("token_forcing")
    ids, lps = [3, 1], [math.log(0.9), math.log(0.1)]
    run = tmp_path / "run"
    write_jsonl(run, [{"id": "r0", "output_ids": [3, 1], "top_ids": [ids, ids], "top_lps": [lps, lps]}], "outputs.jsonl")
    (run / "run.json").write_text(json.dumps({"concurrency": 8, "sampling": SAMPLING, "logprobs": 2}))
    scored = tmp_path / "scored"
    write_jsonl(scored, [{"id": "r0", "tok": [3, 1], "n_out": 2, "n_scored": 2, "top_ids": [ids, ids],
                          "top_lps": [lps, lps]}], "scores.jsonl")
    meta = {"server_info": {"speculative_algorithm": None, "enable_fp32_lm_head": True}, "source_run": str(run),
            "score_top": 2}
    (scored / "run.json").write_text(json.dumps(meta))
    [(batch, sampling, by_request)] = forcing.load_probability([str(scored)])
    assert batch == 8 and by_request["r0"][0] == (ids, lps, ids, lps)
    assert forcing.total_variation(*by_request["r0"][0]) == 0.0
    (scored / "run.json").write_text(json.dumps(dict(meta, score_top=4)))
    with pytest.raises(ValueError, match="one K"):
        forcing.load_probability([str(scored)])
    (scored / "run.json").write_text(json.dumps(meta))
    (run / "run.json").write_text(json.dumps({"concurrency": 8, "sampling": dict(SAMPLING, temperature=0.7),
                                              "logprobs": 2}))
    with pytest.raises(ValueError, match="temperature 1"):
        forcing.load_probability([str(scored)])


def rule_lines(rng, draws=100000, controls=None):
    expected = {"0": 0.5, "1": 0.3, "2": 0.15, "3": 0.05}
    kinds = ["flat", "peaked", "near_tie", "draft_excess", "target_outside_draft", "top_p"]

    def counts(probs):
        drawn = collections.Counter(rng.choices(list(probs), weights=list(probs.values()), k=draws))
        return dict(drawn)

    lines = [{"case": kind, "kind": kind, "position": 1, "expected": expected, "counts": counts(expected),
              "ranks_agree": True} for kind in kinds]
    biased = {"0": 0.47, "1": 0.33, "2": 0.15, "3": 0.05}
    for name in ("no_q", "wrong_q", "residual_p", "bias"):
        source = (controls or {}).get(name, biased)
        lines.append({"case": "peaked", "kind": "peaked", "position": 1, "control": name, "expected": expected,
                      "counts": counts(source), "ranks_agree": True})
    return lines


def test_rule_check_passes_exact_sampling_and_needs_every_control_to_fail():
    forcing = script("token_forcing")
    rng = random.Random(7)
    exact = forcing.rules(rule_lines(rng))
    assert not exact["failed"] and not exact["controls_not_rejected"] and exact["alpha"] == pytest.approx(0.01 / 6)
    blind = forcing.rules(rule_lines(rng, controls={"bias": {"0": 0.5, "1": 0.3, "2": 0.15, "3": 0.05}}))
    assert blind["controls_not_rejected"] == ["bias"]
    lines = rule_lines(rng)
    lines[0]["counts"]["9"] = 1  # a token the target gives probability 0
    assert [r["case"] for r in forcing.rules(lines)["failed"]] == ["flat"]


def test_rule_check_refuses_missing_kinds_controls_draws_or_disagreeing_gpus():
    forcing = script("token_forcing")
    rng = random.Random(8)
    with pytest.raises(ValueError, match="kind top_p"):
        forcing.rules([line for line in rule_lines(rng) if line["kind"] != "top_p" or line.get("control")])
    with pytest.raises(ValueError, match="control bias"):
        forcing.rules([line for line in rule_lines(rng) if line.get("control") != "bias"])
    with pytest.raises(ValueError, match="fewer than 100000"):
        forcing.rules(rule_lines(rng, draws=50000))
    lines = rule_lines(rng)
    lines[1]["ranks_agree"] = False
    with pytest.raises(ValueError, match="same token"):
        forcing.rules(lines)


def test_generation_keeps_the_engines_own_log_probabilities_with_logprobs():
    client = script("fixed_batch_client")
    request = {"id": "r", "text": "go", "max_new_tokens": 4}
    args = SimpleNamespace(max_new_tokens_cap=None, sampling=SAMPLING, logprobs=2)
    body = client.build_body(request, args, client.prompt_fields(request))
    assert body["return_logprob"] is True and body["top_logprobs_num"] == 2 and body["logprob_start_len"] == -1
    meta = {"output_token_logprobs": [[-0.1, 7, None], [-0.5, 8, None]],
            "output_top_logprobs": [[[-0.1, 7, None], [-2.4, 3, None]], [[-0.4, 9, None], [-0.5, 8, None]]]}
    own = client.own_logprobs([7, 8], meta)
    assert own == {"lp_out": [-0.1, -0.5], "top_ids": [[7, 3], [9, 8]], "top_lps": [[-0.1, -2.4], [-0.4, -0.5]]}
    assert client.own_logprobs([7, 6], meta)["lp_out"] == [-0.1]  # stops where the engine's token differs


def test_scoring_appends_the_output_ids_to_the_prompt_in_an_engine_session(monkeypatch):
    client = script("fixed_batch_client")
    calls = []

    def engine(url, body=None, timeout=0):
        calls.append((url.rsplit("/", 1)[1], body))
        if url.endswith("/open_session"):
            return "s1"
        if url.endswith("/close_session"):
            return None
        if body["rid"] == "r~prompt":  # the prompt as the run sent it: five positions after the image
            return {"meta_info": {"prompt_tokens": 5}}
        return {"meta_info": {"input_token_logprobs": [[None, 13, ""], [-0.1, 7, ""], [-0.2, 8, ""]],
                              "input_top_logprobs": [None, [[-0.1, 7, ""]], [[-0.2, 8, ""]]]}}

    monkeypatch.setattr(client, "http_json", engine)
    request = {"id": "r", "text": "look <image>", "image_data": ["frame.png"], "max_new_tokens": 4}
    meta = client.score_in_session("http://e", request, client.prompt_fields(request), [7, 8],
                                   SimpleNamespace(score_top=8, timeout=10))
    assert [name for name, _ in calls] == ["open_session", "generate", "generate", "close_session"]
    first, second = calls[1][1], calls[2][1]
    assert first["text"] == "look <image>" and first["image_data"] == ["frame.png"]
    assert second["input_ids"] == [7, 8] and second["logprob_start_len"] == 4
    assert second["session_params"] == {"id": "s1", "rid": "r~prompt", "drop_previous_output": True}
    assert client.score_record("r", [7, 8], meta)["n_scored"] == 2


def test_scoring_records_the_most_likely_tokens_with_score_top():
    client = script("fixed_batch_client")
    meta = {"input_token_logprobs": [[None, 9, ""], [math.log(0.3), 2, ""]],
            "input_top_logprobs": [None, [[math.log(0.5), 1, ""], [math.log(0.3), 2, ""], [math.log(0.15), 3, ""]]]}
    record = client.score_record("r", [2], meta, top=True)
    assert record["top_ids"] == [[1, 2, 3]]
    assert record["top_lps"] == [[pytest.approx(math.log(0.5)), pytest.approx(math.log(0.3)), pytest.approx(math.log(0.15))]]
    assert "top_ids" not in client.score_record("r", [2], meta)


def test_byte_floor_adds_the_prompts_processed_in_a_step(tmp_path):
    floor = script("byte_floor")
    header = {"model.layers.0.mlp.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(64))
    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 2, "num_key_value_heads": 2,
                                                      "head_dim": 4, "num_hidden_layers": 1}))

    def classes(passes):
        args = SimpleNamespace(model_dir=str(tmp_path), requests=1, rows_per_request=1, context_tokens=10,
                               peak_tflops=1.0, peak_tbps=1.0, gpus=1, kv_bytes_per_token=16, kv_heads=None,
                               state_bytes_per_request=0, draft_model_dir=None, draft_forwards=1, params=None,
                               experts=None, experts_per_token=None, attention_layers=None, prefill_tokens=100,
                               prefill_chunk=0, prefill_passes_per_step=passes, recurrent_flops_per_row=0,
                               draft_rows_per_request=0, within_step_pairs=None, prefill_prefix_tokens=10,
                               expert_pattern=r"\.experts\.", exclude=r"embed_tokens")
        return floor.floor(args)["classes"]

    shared, own = classes(0), classes(0.01)
    # one chunk of 100 tokens: 100 x 10 pairs with the prefix plus 100 x 101 / 2 within the chunk
    assert shared["prefill"]["flops"] == pytest.approx(2 * 32 * 100 + 4 * 2 * 4 * 1 * (100 * 10 + 100 * 101 / 2))
    # its own KV written, and the cached prefix read once per chunk
    assert shared["prefill"]["bytes"] == pytest.approx(16 * (100 + 10))
    assert own["prefill"]["bytes"] == pytest.approx(16 * (100 + 10) + 0.01 * 64)


def test_exact_replay_walks_each_policy_along_its_own_trajectory():
    replay = script("exact_replay")
    target = [1, 2, 3, 4]
    # lattices at every position: the chain proposes nothing right, the oracle accepts two at a time
    steps = [{"id": "r", "pos": p, "cands": [[target[p] + 100, target[p]], [target[min(p + 1, 3)]]],
              "logp": [[0.0, -5.0], [0.0]]} for p in range(4)]
    result = replay.walk(steps, {"r": target}, [1])
    assert result["tau"]["chain"] == pytest.approx(1.0)
    assert result["tau"]["oracle"] == pytest.approx(2.0)
    assert result["requests_walked"] == 1


def test_exact_replay_reports_the_engines_own_chain_verify():
    replay = script("exact_replay")
    step = {"id": "r", "pos": 0, "cands": [[1], [2]], "logp": [[0.0], [0.0]], "accept_prob": [0.5, 0.5]}
    assert replay.replay_step(step, [1, 2, 3], [2])["chain_verify"] == pytest.approx(1.75)


def test_probability_d_counts_the_mass_a_run_puts_on_tokens_the_scorer_does_not_list():
    forcing = script("token_forcing")
    half = [math.log(0.4999), math.log(0.4999)]
    # the run lists tokens 3 and 4, the scorer 1 and 2: the scorer leaves 0.0002 for every other token
    assert forcing.total_variation([3, 4], half, [1, 2], half) == pytest.approx(0.9996)
    # the same tokens on both sides: the distance over them with the rest merged
    own, ref = [math.log(0.5), math.log(0.3)], [math.log(0.4), math.log(0.3)]
    assert forcing.total_variation([1, 2], own, [1, 2], ref) == pytest.approx(0.1)
    # ten positions of one request drawn from a distribution on other tokens fail the event rule
    rng = random.Random(10)
    base = [(8, SAMPLING, positions(rng, 120, 100))]
    cand = positions(rng, 120, 100)
    cand["r7"][:10] = [([t + 100 for t in o], ol, ids, lps) for o, ol, ids, lps in cand["r7"][:10]]
    result = forcing.probability(base, [(8, SAMPLING, cand)])["batches"][8]
    assert result["cand_events"] == 10 and result["event_verdict"] == "fail"


def test_rule_check_pools_rare_tokens_into_cells_and_tests_a_thin_tail_exactly():
    forcing = script("token_forcing")
    uniform = {str(t): 1 / 32768 for t in range(32768)}
    assert forcing.goodness_of_fit(uniform, {"0": 100000})[0] < 1e-12
    rng = random.Random(9)
    fair = collections.Counter(str(rng.randrange(32768)) for _ in range(100000))
    assert forcing.goodness_of_fit(uniform, dict(fair))[0] > 1e-3
    tail = {"a": 0.99999, "b": 0.00001}  # token b expects one draw in 100 000
    assert forcing.goodness_of_fit(tail, {"a": 99990, "b": 10})[0] < 1e-4
    assert forcing.goodness_of_fit(tail, {"a": 99999, "b": 1})[0] > 0.5


def scored_run(tmp_path, name, outputs, scores, server_args=None, score_top=None, source=None):
    run = tmp_path / f"{name}-run"
    write_jsonl(run, outputs, "outputs.jsonl")
    (run / "run.json").write_text(json.dumps(dict({"n_requests": len(outputs), "n_failed": 0}, **(source or {}))))
    scored = tmp_path / f"{name}-scored"
    write_jsonl(scored, scores, "scores.jsonl")
    args = dict({"speculative_algorithm": None, "enable_fp32_lm_head": True, "model_path": "target",
                 "chunked_prefill_size": 8192}, **(server_args or {}))
    (scored / "run.json").write_text(json.dumps({"server_info": args, "source_run": str(run), "score_top": score_top}))
    return str(scored)


def score(rid, tokens):
    n = len(tokens)
    return {"id": rid, "tok": tokens, "top1": tokens, "lp_tok": [0.0] * n, "lp_top1": [0.0] * n, "n_out": n,
            "n_scored": n}


def test_token_forcing_needs_every_request_of_a_finished_run_scored(tmp_path):
    forcing = script("token_forcing")
    outputs = [{"id": "r0", "output_ids": [1, 2]}, {"id": "r1", "output_ids": [3]}]
    one = scored_run(tmp_path, "one", outputs, [score("r0", [1, 2])])
    with pytest.raises(ValueError, match="1 requests of the scored run have no score"):
        forcing.load_scored([one])
    unfinished = scored_run(tmp_path, "unfinished", outputs, [score("r0", [1, 2]), score("r1", [3])],
                            source={"n_requests": 3, "n_failed": 1})
    with pytest.raises(ValueError, match="2 of 3 requests finished"):
        forcing.load_scored([unfinished])
    assert len(forcing.load_scored([scored_run(tmp_path, "all", outputs, [score("r0", [1, 2]), score("r1", [3])])])) == 2


def test_token_forcing_compares_runs_of_one_scorer_and_one_k(tmp_path):
    forcing = script("token_forcing")
    outputs = [{"id": "r0", "output_ids": [1]}]
    a = scored_run(tmp_path, "a", outputs, [score("r0", [1])], score_top=8)
    forcing.same_scorer([a, scored_run(tmp_path, "b", outputs, [score("r0", [1])], score_top=8)])
    with pytest.raises(ValueError, match="another scorer, sampling or K"):
        forcing.same_scorer([a, scored_run(tmp_path, "k", outputs, [score("r0", [1])], score_top=16)])
    with pytest.raises(ValueError, match="another scorer, sampling or K"):
        forcing.same_scorer([a, scored_run(tmp_path, "m", outputs, [score("r0", [1])], {"model_path": "other"}, 8)])


def test_token_forcing_audit_rescores_the_same_run_with_only_another_chunk_size(tmp_path):
    forcing = script("token_forcing")
    outputs = [{"id": "r0", "output_ids": [1]}]
    scored = scored_run(tmp_path, "scored", outputs, [score("r0", [1])])
    run = json.loads(Path(scored, "run.json").read_text())["source_run"]

    def rescored(name, **server_args):
        directory = tmp_path / name
        write_jsonl(directory, [score("r0", [1])], "scores.jsonl")
        args = dict({"speculative_algorithm": None, "enable_fp32_lm_head": True, "model_path": "target",
                     "chunked_prefill_size": 4096}, **server_args)
        (directory / "run.json").write_text(json.dumps({"server_info": args, "source_run": run}))
        return str(directory)

    forcing.check_audit([scored], [rescored("audit")], "--rescored")
    with pytest.raises(ValueError, match="differs from"):
        forcing.check_audit([scored], [rescored("model", model_path="other")], "--rescored")
    with pytest.raises(ValueError, match="same prefill chunk size"):
        forcing.check_audit([scored], [rescored("chunk", chunked_prefill_size=8192)], "--rescored")
    with pytest.raises(ValueError, match="rescored another run"):
        forcing.check_audit([scored], [scored_run(tmp_path, "other", outputs, [score("r0", [1])],
                                                  {"chunked_prefill_size": 4096})], "--rescored")
    with pytest.raises(ValueError, match="one rescored run per scored run"):
        forcing.check_audit([scored, scored], [rescored("single")], "--rescored")


def test_task_copies_take_their_original_requests_group():
    ab = script("paired_ab")
    base = pair_run("base", "a", 2, [100.0] * 4)
    cand = pair_run("cand", "b", 2, [120.0, 100.0, 100.0, 100.0])
    for run in (base, cand):
        run["requests"] = {**run["requests"], **{f"{rid}#c1": v for rid, v in run["requests"].items()}}
    groups = {"r0": "math", "r1": "code", "r2": "code", "r3": "code"}
    gain, n, _ = ab.pair_gain(base, cand, groups)
    assert n == 8 and gain == pytest.approx((1.2 * 1.0) ** 0.5)


def test_noise_only_uses_the_groups_and_one_numerator_slot():
    ab = script("paired_ab")
    groups = {"r0": "math", "r1": "code", "r2": "code", "r3": "code"}
    runs = []
    for cycle in range(4):
        runs += [pair_run("base", "a", cycle, [100.0] * 4), pair_run("base", "b", cycle, [104.0, 100.0, 100.0, 100.0])]
    noise = ab.noise_only(runs, "base", [8], groups)[8]["noise"]
    assert noise == pytest.approx(math.sqrt(1.04) - 1)  # every pair: the equal-weight mean of 1.04 and 1
    assert ab.noise_only(list(reversed(runs)), "base", [8], groups)[8]["noise"] == pytest.approx(noise)


def test_paired_comparison_needs_no_noise_where_outputs_differ():
    ab = script("paired_ab")
    other = {f"r{i}": [i, 7] for i in range(20)}
    (decision,) = ab.compare(paired_runs(1.06, other), "base", "cand")
    assert decision["diverged"] and decision["threshold"] == pytest.approx(0.03) and decision["promote"]


def test_paired_comparison_refuses_runs_that_differ_in_logprobs_or_cannot_keep_the_batch_in_flight():
    ab = script("paired_ab")
    runs = paired_runs(1.1)
    logged = [dict(r, same=dict(r["same"], logprobs=128)) if r["arm"] == "cand" else r for r in runs]
    with pytest.raises(ValueError, match="logprobs"):
        ab.compare(logged, "base", "cand", noise={8: 0.002})
    with pytest.raises(ValueError, match="could not keep that many in flight"):
        ab.compare([dict(r, tasks=4) for r in runs], "base", "cand", noise={8: 0.002})


def two_turns(tmp_path):
    rows = [{"id": "t0", "episode": "e", "text": "obs0", "output_text": "<a0>", "max_new_tokens": 4},
            {"id": "t1", "episode": "e", "text": "obs0<a0><end>obs1", "output_text": "<a1>", "max_new_tokens": 4}]
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return requests


def test_client_counts_an_aborted_request_as_failed_and_ends_its_task(monkeypatch, tmp_path):
    client = script("fixed_batch_client")
    requests, sent = two_turns(tmp_path), []

    def engine(url, body=None, timeout=0):
        sent.append(body["rid"])
        return {"output_ids": [3], "text": "<cut>", "meta_info": {"finish_reason": {"type": "abort", "status_code": None}}}

    monkeypatch.setattr(client, "http_json", engine)
    monkeypatch.setattr(client, "server_info", lambda endpoint: None)
    monkeypatch.setattr(sys, "argv", ["client", "--endpoint", "http://e", "--requests", str(requests), "--concurrency",
                                      "1", "--out", str(tmp_path / "run"), "--session-field", "episode"])
    assert client.main() == 1
    assert sent == ["t0"]
    assert json.loads((tmp_path / "run" / "run.json").read_text())["n_failed"] == 1


def test_client_sends_independent_requests_that_carry_a_turn_field(monkeypatch, tmp_path):
    client = script("fixed_batch_client")
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps({"id": "r", "input_ids": [1, 2], "max_new_tokens": 1, "turn": 1}) + "\n")
    monkeypatch.setattr(client, "http_json", lambda url, body=None, timeout=0: {
        "output_ids": [5], "text": "x", "meta_info": {"finish_reason": {"type": "length"}}})
    monkeypatch.setattr(client, "server_info", lambda endpoint: None)
    monkeypatch.setattr(sys, "argv", ["client", "--endpoint", "http://e", "--requests", str(requests), "--concurrency",
                                      "1", "--out", str(tmp_path / "run")])
    assert client.main() == 0
    assert json.loads((tmp_path / "run" / "run.json").read_text())["n_done"] == 1


def test_client_refuses_a_request_set_smaller_than_the_batch(monkeypatch, tmp_path):
    client = script("fixed_batch_client")
    requests, sent = tmp_path / "requests.jsonl", []
    requests.write_text("".join(json.dumps({"id": f"r{i}", "input_ids": [1], "max_new_tokens": 1}) + "\n"
                                for i in range(2)))
    monkeypatch.setattr(client, "http_json", lambda url, body=None, timeout=0: sent.append(body) or {
        "output_ids": [5], "text": "x", "meta_info": {}})
    monkeypatch.setattr(client, "server_info", lambda endpoint: None)
    monkeypatch.setattr(sys, "argv", ["client", "--endpoint", "http://e", "--requests", str(requests), "--concurrency",
                                      "8", "--out", str(tmp_path / "run")])
    assert client.main() == 1
    assert not sent


def test_scoring_rebuilds_the_runs_turns_with_its_task_field_and_copies(monkeypatch, tmp_path):
    client = script("fixed_batch_client")
    requests = two_turns(tmp_path)
    run, outputs = tmp_path / "run", []
    for copy in ("", "#c1"):
        outputs += [{"id": f"t0{copy}", "output_ids": [5, 6], "text": f"<live0{copy}>",
                     "finish_reason": {"type": "stop", "matched": 6}},
                    {"id": f"t1{copy}", "output_ids": [7], "text": f"<live1{copy}>",
                     "finish_reason": {"type": "stop", "matched": 7}}]
    write_jsonl(run, outputs, "outputs.jsonl")
    (run / "run.json").write_text(json.dumps({"session_field": "episode", "session_copies": 2,
                                              "requests_sha256": hashlib.sha256(requests.read_bytes()).hexdigest()}))
    prompts = {}

    def engine(url, body=None, timeout=0):
        if url.endswith("/open_session"):
            return "s"
        if url.endswith("/close_session"):
            return None
        if body["rid"].endswith("~prompt"):
            prompts[body["rid"][: -len("~prompt")]] = body["text"]
            return {"meta_info": {"prompt_tokens": 3}}
        ids = body["input_ids"]
        return {"meta_info": {"input_token_logprobs": [[None, 0, ""]] + [[-0.1, t, ""] for t in ids],
                              "input_top_logprobs": [None] + [[[-0.1, t, ""]] for t in ids]}}

    monkeypatch.setattr(client, "http_json", engine)
    monkeypatch.setattr(client, "server_info", lambda endpoint: None)
    argv = ["client", "--endpoint", "http://e", "--requests", str(requests), "--concurrency", "2",
            "--score-from", str(run), "--score-top", "1"]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(tmp_path / "scored")])
    assert client.main() == 0
    # the second turn's prompt holds the answer the run's engine gave, in each copy
    assert prompts["t1"] == "obs0<live0><end>obs1" and prompts["t1#c1"] == "obs0<live0#c1><end>obs1"
    scored = [json.loads(line)["id"] for line in (tmp_path / "scored" / "scores.jsonl").read_text().splitlines()]
    assert sorted(scored) == ["t0", "t0#c1", "t1", "t1#c1"]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(tmp_path / "copies"), "--session-copies", "1"])
    assert client.main() == 1
    other = tmp_path / "other.jsonl"
    other.write_text(requests.read_text().replace("obs1", "obs9"))
    monkeypatch.setattr(sys, "argv", [str(other) if a == str(requests) else a for a in argv]
                        + ["--out", str(tmp_path / "other")])
    assert client.main() == 1


def test_a_token_answer_drops_the_stop_token_the_engine_matched_unless_the_recording_keeps_it():
    client = script("fixed_batch_client")
    live = {"output_ids": [8, 0], "text": "<a>", "finish_reason": {"type": "stop", "matched": 0}}
    assert client.answer_of(live, {"input_ids": [90], "output_ids": [6]}) == [8]
    assert client.answer_of(live, {"input_ids": [90], "output_ids": [6, 0]}) == [8, 0]
    assert client.answer_of(dict(live, finish_reason={"type": "length"}), {"input_ids": [90], "output_ids": [6]}) == [8, 0]
    assert client.answer_of(live, {"text": "q", "output_text": "<r>"}) == "<a>"


def test_exact_replay_reads_a_flat_root_row_and_counts_an_empty_draft_as_one_token():
    replay = script("exact_replay")
    step = {"id": "r", "pos": 0, "cands": [[1, 2], [3, 4]], "logp": [[0.0, -1.0], [0.0, 0.0]],
            "pair": [[-5.0, 5.0], [[5.0, -5.0], [-5.0, 5.0]]]}
    # the root row prefers 2, and after 2 the drafter wants 4
    assert replay.replay_step(step, [2, 4, 7, 9], [2])[2] == 3
    empty = {"id": "r", "pos": 0, "cands": [], "logp": []}
    assert replay.replay_step(empty, [2, 4], [2]) == {"chain": 1, "oracle": 1, 2: 1}
    assert replay.replay_step(empty, [], [2]) is None


def model_dir(path, tensors, config):
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.mkdir(exist_ok=True)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def floor_args(model, **overrides):
    args = dict(model_dir=model, requests=1, rows_per_request=1, context_tokens=0, peak_tflops=1.0, peak_tbps=1.0,
                gpus=1, kv_bytes_per_token=0, kv_heads=None, state_bytes_per_request=0, draft_model_dir=None,
                draft_forwards=1, params=None, experts=None, experts_per_token=None, attention_layers=None,
                prefill_tokens=0, prefill_chunk=0, prefill_passes_per_step=0, recurrent_flops_per_row=0,
                draft_rows_per_request=0, within_step_pairs=None, prefill_prefix_tokens=0,
                expert_pattern=r"\.experts\.", exclude=r"embed_tokens")
    return SimpleNamespace(**dict(args, **overrides))


def test_byte_floor_reads_and_attends_a_sliding_window_layer_up_to_its_window(tmp_path):
    floor = script("byte_floor")
    model = model_dir(tmp_path / "m", {"model.layers.0.mlp.weight": [4, 8]},
                      {"num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
                       "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 2})
    attention = floor.floor(floor_args(model, context_tokens=10, kv_bytes_per_token=32))["classes"]["attention"]
    # each layer holds 16 of the 32 KV bytes per token: the full layer reads 10 tokens, the window 2
    assert attention["bytes"] == pytest.approx(16 * 10 + 16 * 2)
    # one row: 10 + 1 pairs in the full layer, 2 + 1 in the sliding one
    assert attention["flops"] == pytest.approx(4 * 2 * 4 * (11 + 3))


def test_byte_floor_counts_the_experts_a_prefill_chunk_reaches_and_the_drafters_own_weights(tmp_path):
    floor = script("byte_floor")
    experts = {f"model.layers.0.mlp.experts.{e}.weight": [5, 10] for e in range(4)}  # 100 bytes each
    config = {"num_experts": 4, "num_experts_per_tok": 1, "num_attention_heads": 2, "head_dim": 4,
              "num_hidden_layers": 1}
    model = model_dir(tmp_path / "m", experts, config)
    prefill = floor.floor(floor_args(model, prefill_tokens=4, prefill_passes_per_step=1))["classes"]["prefill"]
    assert prefill["bytes"] == pytest.approx(400 * (1 - 0.75 ** 4))  # four tokens reach more experts than one row
    drafter = model_dir(tmp_path / "d", experts, config)
    sparse = floor.floor(floor_args(model, draft_model_dir=drafter))["classes"]["drafter"]
    assert sparse["bytes"] == pytest.approx(100) and sparse["flops"] == pytest.approx(2 * 50)
    tied = model_dir(tmp_path / "t", {"model.embed_tokens.weight": [10, 10]}, {"tie_word_embeddings": True})
    output = floor.floor(floor_args(model, draft_model_dir=tied))["classes"]["drafter"]
    assert output["bytes"] == pytest.approx(200) and output["flops"] == pytest.approx(2 * 100)


def test_byte_floor_over_a_batch_mix_is_the_mean_of_each_steps_floor(tmp_path):
    floor = script("byte_floor")
    model = model_dir(tmp_path / "m", {"model.layers.0.mlp.weight": [4, 8]},
                      {"num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4, "num_hidden_layers": 1})
    one, two = (floor.floor(floor_args(model, context_tokens=10, kv_bytes_per_token=32, requests=n)) for n in (1, 2))
    mixed = floor.mixed_floor(floor_args(model, context_tokens=10, kv_bytes_per_token=32), {1: 3, 2: 1})
    assert mixed["floor_ms"] == pytest.approx(0.75 * one["floor_ms"] + 0.25 * two["floor_ms"])
    assert mixed["classes"]["attention"]["bytes"] == pytest.approx(0.75 * 320 + 0.25 * 640)
    assert mixed["rows"] == pytest.approx(1.25)


def test_profile_windows_of_one_run_pool_their_steps():
    profile = script("profile_steps")
    first = {"steps": 2, "dropped": 0, "wall_us_mean": 100.0, "busy_us_mean": 80.0, "overlap_us_mean": 0.0,
             "classes_us_mean": {"gemm": 50.0}, "top": [("k", 50.0)]}
    second = {"steps": 6, "dropped": 1, "wall_us_mean": 200.0, "busy_us_mean": 160.0, "overlap_us_mean": 0.0,
              "classes_us_mean": {"gemm": 90.0, "attention": 40.0}, "top": [("k", 90.0)]}
    pooled = profile.pool([first, second])
    assert pooled["steps"] == 8 and pooled["dropped"] == 1
    assert pooled["busy_us_mean"] == pytest.approx((2 * 80 + 6 * 160) / 8)
    assert pooled["classes_us_mean"] == {"attention": pytest.approx(6 * 40 / 8),
                                         "gemm": pytest.approx((2 * 50 + 6 * 90) / 8)}
