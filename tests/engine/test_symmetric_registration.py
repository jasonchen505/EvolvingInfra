# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Native registration and bounded page copies without CUDA import dependencies.

Compile the GPU-independent definitions from their source files. The address
math and copy engine are the production implementations, not test replicas.
"""
import ast
import enum
import logging
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import typing

import pytest

msgspec = pytest.importorskip("msgspec")
SRT = Path(__file__).resolve().parents[2] / "evolving_infra/engine/decoupled_engine/srt"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == names
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def native():
    ns = dict(vars(typing), msgspec=msgspec, math=math, enum=enum, threading=threading,
              PROTOCOL_VERSION=1, logger=logging.getLogger(__name__))
    definitions(SRT / "mem_cache/memory_pool.py", {"KvBufferDesc"}, ns)
    definitions(SRT / "disaggregation/pull/protocol.py", {
        "ProtocolError", "PoolRegion", "ItemRegion", "PoolLayout", "Registration",
        "_pool_region_from_layout", "_contiguous_runs", "build_page_read_plan",
    }, ns)
    ns.update(POOL_FULL="full", POOL_SWA="swa", POOL_C4="c4", POOL_C128="c128")
    definitions(SRT / "disaggregation/symmetric/export_table.py",
                {"_is_dsv4_pool", "registration_pools_for"}, ns)
    definitions(SRT / "disaggregation/pull/read_engine.py", {"FakeMemory", "FakeReadEngine"}, ns)
    return SimpleNamespace(**ns)


@pytest.mark.parametrize("page_size", [1, 4, 64, 128])
@pytest.mark.parametrize("pool_classes", [
    ("MHATokenToKVPool",),
    ("MHATokenToKVPool", "MHATokenToKVPoolFP4"),
    ("MLATokenToKVPool",),
    # Wrappers whose get_contiguous_buf_infos returns their inner pool's values.
    ("HybridLinearKVPool",),
    ("SWAKVPool",),
    ("MiniMaxSparseKVPool",),
], ids=lambda names: names[-1])
@pytest.mark.parametrize(("source_pages", "target_pages"), [
    ([1], [8]), ([1, 2, 4, 5], [5, 6, 1, 2]), ([8, 3], [1, 7]),
])
def test_registration_copies_pages_within_the_registered_span(
        native, page_size, pool_classes, source_pages, target_pages):
    cls = object
    for name in pool_classes:
        cls = type(name, (cls,), {})
    pool = cls()
    row_bytes, size = 2, 8 * page_size
    descriptor = native.KvBufferDesc("k", (size + page_size, 1), row_bytes=row_bytes, tokens_per_row=1)
    span = descriptor.final_span_bytes(size, page_size)
    stride = descriptor.item_len_bytes(page_size)

    def register(engine, fill):
        ptrs = [engine.memory.alloc(span, fill=fill) for _ in range(2)]
        args = SimpleNamespace(
            page_size=page_size, kv_data_ptrs=ptrs, kv_item_lens=[stride, stride],
            kv_layer_ids=[0, 0], aux_data_ptrs=[], aux_item_lens=[],
            state_types=[], state_data_ptrs=[], state_item_lens=[], engine_rank=0,
        )
        layouts, extra = native.registration_pools_for(kvcache=pool, kv_args=args)
        assert not extra
        registration = native.Registration.from_kv_args(
            kv_args=args, session_id=engine.session_id, attn_tp_size=1, pool_layouts=layouts)
        return registration.pools["full"], ptrs

    owner = native.FakeReadEngine(session_id="owner")
    reader = native.FakeReadEngine(session_id="reader")
    try:
        remote, source_ptrs = register(owner, 0)
        local, target_ptrs = register(reader, 255)
        for entry, ptr in enumerate(source_ptrs):
            for page in range(9):
                owner.memory.view(ptr + page * stride, stride)[:] = bytes([16 * entry + page]) * stride
        local_ptrs, remote_ptrs, lengths = native.build_page_read_plan(
            local=local, remote=remote, local_pages=target_pages, remote_pages=source_pages)
        assert sum(lengths) == len(source_pages) * page_size * row_bytes * 2
        assert reader.batch_read("owner", local_ptrs, remote_ptrs, lengths) == 0
        mapping = dict(zip(target_pages, source_pages))
        for entry, ptr in enumerate(target_ptrs):
            for page in range(9):
                value = 16 * entry + mapping[page] if page in mapping else 255
                assert bytes(reader.memory.view(ptr + page * stride, stride)) == bytes([value]) * stride
    finally:
        owner.close()
        reader.close()
