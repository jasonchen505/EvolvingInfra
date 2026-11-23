# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""CPU tests of the SWA window a symmetric export keeps at a branching point
(ExportTable.trim_swa), on a FULL+SWA tree (DSv4). Needs the engine
(``decoupled-engine``, with torch).

Rig: the real unified FULL+SWA tree, SWA allocator and req-to-token pool on
CPU, page 4 and window 2 (window < page, as DSv4: window 128, page 256), and
the real ExportTable SWA methods over that allocator.

A first prompt is exported and released: the tree keeps FULL for [0, 68) and
SWA for its tail only. A second prompt shares its first 38 tokens, so its
match hits FULL up to 36 with no SWA there: prefix 0, branching point 36,
branch window [28, 36). While it exports, the trim frees its SWA below the
tail, except that window.

Every case ends with check(): each SWA and FULL slot is free or a tree
node's, and no node holds the padding slot as SWA.
"""

from array import array
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="needs the engine (decoupled-engine)")
pytest.importorskip("decoupled_engine", reason="needs the engine (decoupled-engine)")

import torch  # noqa: E402

from decoupled_engine.srt.disaggregation.symmetric.export_table import (  # noqa: E402
    RELEASED,
    RELEASED_BEFORE_EXPORT,
    ExportTable,
)
from decoupled_engine.srt.disaggregation.symmetric.scheduler_mixin import (  # noqa: E402
    SchedulerSymmetricMixin,
)
from decoupled_engine.srt.environ import InvariantCheckLevel, envs  # noqa: E402
from decoupled_engine.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
)
from decoupled_engine.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator  # noqa: E402
from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from decoupled_engine.srt.mem_cache.common import (  # noqa: E402
    checkpoint_kv_cache,
    match_kv_cache,
    release_kv_cache,
)
from decoupled_engine.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool  # noqa: E402
from decoupled_engine.srt.mem_cache.unified_cache.components.base import ComponentType  # noqa: E402
from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from decoupled_engine.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from decoupled_engine.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)

FULL, SWA = ComponentType.FULL, ComponentType.SWA
PAGE = 4
WINDOW = 2
CHUNK = 16
SIZE = 1024
FIRST = list(range(1, 71))
SHARED = 38  # FULL hit 36 = the second prompt's branching point


@pytest.fixture(autouse=True)
def strict_invariants():
    """The allocator refuses a free of an unmapped SWA page, and a FULL-only
    free of a page that still has one, where it happens."""
    with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
        yield


def second(n):
    return FIRST[:SHARED] + list(range(1001, 1001 + n - SHARED))


class Rig:
    def __init__(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE)
        )
        self.pool = ReqToTokenPool(
            size=8, max_context_len=256, device="cpu", enable_memory_saver=False
        )
        kv = SWAKVPool(
            size=SIZE,
            size_swa=SIZE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[2],
            device="cpu",
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=SIZE,
            size_swa=SIZE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv,
            need_sort=False,
        )
        self.cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.alloc,
                page_size=PAGE,
                disable=False,
                sliding_window_size=WINDOW,
                tree_components=(FULL, SWA),
            )
        )
        self.table = ExportTable.__new__(ExportTable)
        self.table.swa_tail_window = WINDOW
        self.table.page_size = PAGE
        self.table.req_to_token_pool = self.pool
        self.table.token_to_kv_pool_allocator = self.alloc
        self.table.stats_swa_trimmed_tokens = 0
        self.sched = SimpleNamespace(export_table=self.table, tree_cache=self.cache)
        self.full_total = self.alloc.full_available_size()
        self.swa_total = self.alloc.swa_available_size()

    def admit(self, prompt, rid):
        """Match the prompt and give the request FULL and SWA slots for the rest."""
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = array("q", prompt)
        match_kv_cache(req, self.cache, req.full_untruncated_fill_ids)
        req.kv.cache_protected_len = req.prefix_len
        req.lock = self.cache.lock(req.last_node)
        self.pool.alloc([req])
        n = len(prompt)
        aligned = -(-n // PAGE) * PAGE
        row = self.pool.req_to_token[req.kv.req_pool_idx]
        row[: req.prefix_len] = self.cache.path_device_indices(req.last_node)
        full = self.alloc.full_attn_allocator.alloc(aligned - req.prefix_len)
        swa = self.alloc.swa_attn_allocator.alloc(aligned - req.prefix_len)
        self.alloc.full_to_swa_index_mapping[full] = swa
        row[req.prefix_len : aligned] = full
        req.kv.kv_allocated_len = req.kv.kv_committed_len = n
        return req

    def chunks(self, req):
        n = len(req.origin_input_ids)
        bounds = list(range(req.kv.cache_protected_len, n, CHUNK)) + [n]
        return list(zip(bounds[:-1], bounds[1:]))

    def export(self, req, *, stashes=None):
        """The SWA-tail trims of an export: one at each chunk-boundary stash,
        and one after the last chunk. ``stashes`` stops after that many."""
        chunks = self.chunks(req)
        for start, end in chunks[:-1][:stashes]:
            req.prefix_len, req.extend_end = start, end
            self.table.trim_swa(req, start)
            req.prefix_len = end
        if stashes is None:
            req.prefix_len, req.extend_end = chunks[-1]
            self.table.trim_swa(req, len(req.origin_input_ids))

    def finish(self, req, reason=None):
        req.extend_end = len(req.origin_input_ids)
        req.output_ids = array("q", [9001])
        req.finished_reason = reason or FINISH_LENGTH(length=1)
        req.finished_len = 1

    def free_exported(self, req, reason=RELEASED):
        SchedulerSymmetricMixin._symmetric_free_exported(self.sched, req, reason)

    def release_first(self):
        req = self.admit(FIRST, "first")
        self.export(req)
        self.finish(req)
        self.free_exported(req)

    def nodes(self):
        core, stack, out = self.cache.tree_core, [self.cache.tree_core.root_node], []
        while stack:
            node = stack.pop()
            if node is not core.root_node:
                out.append(node)
            stack.extend(node.children.values())
        return out

    def shape(self):
        """(start, end, has SWA) of every node, sorted."""
        out = []

        def walk(node, pos):
            for child in node.children.values():
                end = pos + len(child.key)
                out.append((pos, end, child.component_data[SWA].value is not None))
                walk(child, end)

        walk(self.cache.tree_core.root_node, 0)
        return sorted(out)

    def check(self):
        mapping = self.alloc.full_to_swa_index_mapping
        for node in self.nodes():
            swa = node.component_data[SWA].value
            if swa is not None:
                full = node.component_data[FULL].value
                assert bool((swa > 0).all()), f"node {node.id} holds the padding slot"
                assert torch.equal(mapping[full], swa.to(mapping.dtype))
        assert (
            self.alloc.swa_available_size()
            + self.cache.swa_evictable_size()
            + self.cache.swa_protected_size()
            == self.swa_total
        )
        assert (
            self.alloc.full_available_size()
            + self.cache.full_evictable_size()
            + self.cache.full_protected_size()
            == self.full_total
        )


def exported_past_the_branch_window(rig, n=70, stashes=None):
    rig.release_first()
    req = rig.admit(second(n), "second")
    assert (req.prefix_len, req.swa_branching_seqlen) == (0, 36)
    rig.export(req, stashes=stashes)
    # The cursor is past the window: the SWA above it is freed, the window is not.
    mapping = rig.alloc.full_to_swa_index_mapping
    row = rig.pool.req_to_token[req.kv.req_pool_idx]
    assert req.kv.swa_evicted_seqlen >= 44
    assert bool((mapping[row[28:36]] > 0).all())
    assert bool((mapping[row[36:44]] == 0).all())
    return req


def test_release_caches_the_branch_window_and_tombstones_the_trimmed_range():
    """The release insert must not take the SWA that the trim freed above the
    branch window: the tree would count it and serve the padding slot there."""
    rig = Rig()
    req = exported_past_the_branch_window(rig)
    rig.finish(req)
    rig.free_exported(req)
    assert rig.shape() == [
        (0, 28, False),
        (28, 36, True),  # the branch window
        (36, 64, False),  # the first prompt
        (36, 64, False),  # the second prompt: freed by the trim
        (64, 68, True),
        (64, 68, True),
    ]
    rig.check()


def test_abort_frees_the_branch_window():
    """An aborted export frees its KV without an insert; the kept window lies
    below the SWA cursor, where the release frees the FULL side only."""
    rig = Rig()
    req = exported_past_the_branch_window(rig)
    rig.finish(req, FINISH_ABORT("aborted by client"))
    rig.free_exported(req)
    rig.check()


def test_reader_gone_before_export_frees_the_branch_window():
    """The P-leg runs on as a plain chunked prefill after its reader left: its
    chunk and finish inserts must not free again what the trim freed."""
    rig = Rig()
    req = exported_past_the_branch_window(rig, n=86, stashes=4)
    req.export_mark, req.transfer_id = object(), 7
    rig.free_exported(req, RELEASED_BEFORE_EXPORT)
    chunks = rig.chunks(req)
    req.prefix_len, req.extend_end = chunks[4]
    checkpoint_kv_cache(req, rig.cache)
    rig.finish(req)
    release_kv_cache(req, rig.cache, checkpoint=True)
    rig.check()
