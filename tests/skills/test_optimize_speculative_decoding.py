# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The measurement and decision scripts of skills/optimize-speculative-decoding, on synthetic inputs."""
import importlib.util
import json
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
    explore, sealed = split.split(lines, "study", 0.3)
    again = split.split(list(reversed(lines)), "study", 0.3)
    ids = lambda rows: {json.loads(r)["id"] for r in rows}  # noqa: E731
    assert ids(explore) | ids(sealed) == {f"r{i}" for i in range(200)}
    assert not ids(explore) & ids(sealed)
    assert ids(again[1]) == ids(sealed)
    assert 30 < len(sealed) < 90
    with pytest.raises(ValueError, match="duplicate"):
        split.split(lines + lines[:1], "study", 0.3)


def test_split_keeps_the_turns_of_a_session_together():
    split = script("split_requests")
    lines = [json.dumps({"id": f"s{s}-t{t}", "session": f"s{s}"}) for s in range(60) for t in range(4)]
    explore, sealed = split.split(lines, "study", 0.3, group_field="session")
    sessions = lambda rows: {json.loads(r)["session"] for r in rows}  # noqa: E731
    assert not sessions(explore) & sessions(sealed)
    assert len(sealed) % 4 == 0 and len(explore) % 4 == 0


def record(i, mode, bs, launch, tokens=0):
    return {"forward_iter": i, "mode": mode, "bs": bs, "launch_bs": bs, "launch_mono": launch,
            "ts": 1000.0 + launch, "decode_tokens": tokens, "after_idle": False}


def test_plain_steps_leave_out_the_step_after_a_prefill_and_smaller_batches():
    steps = script("decode_steps")
    records = [
        record(0, "extend", 2, 0.000),
        record(1, "decode", 2, 0.050, 6),   # first decode after a prefill: left out
        record(2, "decode", 2, 0.060, 8),   # plain: 12 ms to the next launch
        record(3, "decode", 2, 0.072, 4),   # plain: 8 ms
        record(4, "decode", 1, 0.080, 3),   # smaller batch
        record(5, "decode", 2, 0.090, 6),   # plain: 10 ms
        record(6, "decode", 2, 0.100, 6),   # last record: no next launch
    ]
    found, after_prefill = steps.plain_steps(records, 2)
    assert after_prefill == 1
    assert [round(s[0] * 1e3, 6) for s in found] == [12.0, 8.0, 10.0]
    assert [s[1] for s in found] == [8, 4, 6]


def run(arm, node, cycle, speed, tau=3.0):
    return {"dir": f"{arm}-{node}-{cycle}", "arm": arm, "node": node, "cycle": cycle, "concurrency": 8,
            "steps": 500, "tau": tau, "step_ms_mean": 8 * tau / speed * 1e3, "tokens_per_s": speed}


def test_paired_comparison_promotes_a_clear_gain_and_splits_it():
    ab = script("paired_ab")
    runs = []
    for node in ("n1", "n2"):
        for cycle, jitter in enumerate((1.0, 1.004, 0.997)):
            runs.append(run("base", node, cycle, 1000 * jitter))
            runs.append(run("cand", node, cycle, 1100 * jitter, tau=3.3))
    (decision,) = ab.compare(runs, "base", ["cand"])
    assert decision["pairs"] == 6
    assert decision["ratio"] == pytest.approx(1.1)
    assert decision["tau_ratio"] == pytest.approx(1.1)
    assert decision["step_ratio"] == pytest.approx(1.0)
    assert decision["promote"]


def test_paired_comparison_refuses_a_run_without_partner():
    ab = script("paired_ab")
    runs = [run("base", "n1", 0, 1000), run("cand", "n1", 0, 1010), run("base", "n1", 1, 1000),
            run("cand", "n1", 1, 1010), run("cand", "n2", 0, 1010)]
    with pytest.raises(ValueError, match="partner"):
        ab.compare(runs, "base", ["cand"])


def scored(path, rows):
    path.mkdir()
    with open(path / "scores.jsonl", "w") as f:
        for i, positions in enumerate(rows):
            f.write(json.dumps({"id": str(i), "tok": [p[0] for p in positions], "top1": [p[1] for p in positions],
                                "lp_top1": [p[2] for p in positions], "lp_top2": [p[3] for p in positions]}) + "\n")
    return str(path)


def test_token_forcing_passes_equal_outputs_and_fails_extra_confident_mismatches(tmp_path):
    forcing = script("token_forcing")
    edges = [0, 0.25, 0.5, 1, 2, 4]
    near_tie = (1, 2, -0.6, -0.7)   # mismatch at gap 0.1
    agree = (5, 5, -0.01, -5.0)     # match at gap 5
    confident_miss = (3, 4, -0.01, -5.0)  # mismatch at gap 5
    base = forcing.load([scored(tmp_path / "base", [[near_tie] + [agree] * 49] * 40)], edges, 2.0)
    same = forcing.load([scored(tmp_path / "same", [[near_tie] + [agree] * 49] * 40)], edges, 2.0)
    worse = forcing.load([scored(tmp_path / "worse", [[near_tie, confident_miss] + [agree] * 48] * 40)], edges, 2.0)
    equal = forcing.check(base, same, edges, resamples=300)
    assert equal["rate_rule_pass"] and equal["event_rule_pass"]
    assert equal["rate_increase"] == pytest.approx(0.0)
    bad = forcing.check(base, worse, edges, resamples=300)
    assert not bad["event_rule_pass"]
    assert bad["cand_event_rate"] == pytest.approx(1 / 50)


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
    args = SimpleNamespace(model_dir=str(tmp_path), requests=1, rows_per_request=1, context_tokens=0,
                           peak_tflops=1.0, peak_tbps=1.0, gpus=1, kv_bytes_per_token=0, state_bytes_per_request=0,
                           draft_model_dir=None, draft_forwards=1, params=None, experts=None, experts_per_token=None,
                           attention_layers=None, expert_pattern=r"\.experts\.",
                           exclude=r"(^|\.)(visual|vision_tower|vision_model|mm_projector)\.|embed_tokens")
    result = floor.floor(args)
    assert result["tensors_counted"] == 3
    assert result["expert_share_read"] == pytest.approx(0.5)
    assert result["classes"]["weights"]["bytes"] == pytest.approx(64 + 64 * 0.5)
    assert floor.experts_reached(8, 2, 1) == pytest.approx(2.0)


def test_paired_comparison_reports_each_nodes_ratio():
    ab = script("paired_ab")
    runs = []
    for node, gain in (("n1", 1.1), ("n2", 1.3)):
        for cycle, jitter in enumerate((1.0, 1.004, 0.997)):
            runs.append(run("base", node, cycle, 1000 * jitter))
            runs.append(run("cand", node, cycle, 1000 * gain * jitter, tau=3.3))
    (decision,) = ab.compare(runs, "base", ["cand"])
    assert decision["per_node"] == pytest.approx({"n1": 1.1, "n2": 1.3})


def test_token_forcing_reads_a_given_reference_and_gap(tmp_path):
    forcing = script("token_forcing")
    edges = [0, 0.25, 0.5, 1, 2, 4]

    def keyed(path, rows):
        path.mkdir()
        with open(path / "scores.jsonl", "w") as f:
            for i, positions in enumerate(rows):
                f.write(json.dumps({"id": str(i), "tok": [p[0] for p in positions], "top1": [p[1] for p in positions],
                                    "gap": [p[2] for p in positions]}) + "\n")
        return str(path)

    near_tie, agree, confident_miss = (1, 2, 0.1), (5, 5, 5.0), (3, 4, 3.0)
    base = forcing.load([keyed(tmp_path / "base", [[near_tie] + [agree] * 49] * 40)], edges, 2.0)
    worse = forcing.load([keyed(tmp_path / "worse", [[near_tie, confident_miss] + [agree] * 48] * 40)], edges, 2.0)
    result = forcing.check(base, worse, edges, resamples=300)
    assert result["base_event_rate"] == 0
    assert result["cand_event_rate"] == pytest.approx(1 / 50)
    assert not result["event_rule_pass"]


def test_keyed_seed_check_flags_an_engine_that_ignores_the_seed(monkeypatch):
    client = script("fixed_batch_client")
    args = SimpleNamespace(sampling={"temperature": 1.0}, ignore_eos=False, keyed_seed=True, score_from=None,
                           forced_from=None, timeout=10)
    request = {"id": "r1", "text": "hello", "max_new_tokens": 100}
    replies = iter([{"output_ids": [1, 2, 3]}, {"output_ids": [1, 2, 4]}])
    monkeypatch.setattr(client, "http_json", lambda url, body, timeout: next(replies))
    assert "differ at output token 2" in client.seed_check("http://engine", request, args)
    monkeypatch.setattr(client, "http_json", lambda url, body, timeout: {"output_ids": [1, 2, 3]})
    assert client.seed_check("http://engine", request, args) is None


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
                               experts=None, experts_per_token=None, attention_layers=None,
                               expert_pattern=r"\.experts\.", exclude=r"embed_tokens")
        return floor.floor(args)["classes"]["attention"]["floor_ms"]

    # 4 KV heads: past 4 GPUs each GPU still reads a quarter of the KV
    assert attention_ms(8) == pytest.approx(attention_ms(4))
    assert attention_ms(2) == pytest.approx(2 * attention_ms(4))
    assert attention_ms(4) == pytest.approx(1e3 * 1000 * 100 * 2 / 4 / 0.9e12)
