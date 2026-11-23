# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The `/generate` contract gate admits this engine's own sampling parameters.

Needs the engine (``decoupled-engine``).
"""
import pytest

pytest.importorskip("torch", reason="needs the engine (decoupled-engine)")
pytest.importorskip("decoupled_engine", reason="needs the engine (decoupled-engine)")

from decoupled_engine.srt.entrypoints.api_contract import generate_contract_error  # noqa: E402
from decoupled_engine.srt.sampling.sampling_params import SamplingParams  # noqa: E402

SAMPLING = {"temperature": 0.0, "top_k": 1, "ignore_eos": True, "max_new_tokens": 2}
FORCED = {**SAMPLING, "forced_output_token_ids": [5, 6]}


def body(sampling, **fields):
    return {"input_ids": [1, 2, 3], "sampling_params": sampling, **fields}


def test_strict_replay_sampling_parameter_passes():
    assert "forced_output_token_ids" in SamplingParams.__struct_fields__
    assert generate_contract_error(body(FORCED)) is None
    assert generate_contract_error(body([FORCED, SAMPLING])) is None


def test_symmetric_pd_and_session_fields_pass():
    tagged = body(SAMPLING, kv_export={"transfer_id": 1}, kv_import={"transfer_id": 1},
                  kv_session={"session_id": "s", "version": 1}, late_d={"mode": "choose"})
    assert generate_contract_error(tagged) is None


def test_other_sampling_parameters_the_schema_lacks_are_still_refused():
    for name in ("no_such_param", "forced_output_seed"):
        assert f"unknown field `{name}`" in generate_contract_error(body({**SAMPLING, name: 1}))
