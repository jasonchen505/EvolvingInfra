# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The compressed gather of DSV4 sparse prefill when a chunk runs as request groups.

Needs the engine (``decoupled-engine``).
"""
import pytest

torch = pytest.importorskip("torch", reason="needs the engine (decoupled-engine)")
spu = pytest.importorskip(
    "decoupled_engine.srt.layers.attention.dsv4.sparse_prefill_utils",
    reason="needs the engine (decoupled-engine)",
)

SEQ_LENS = [1500, 40, 900, 300, 2100, 7]
EXTEND_LENS = [64, 40, 33, 20, 70, 7]
SWA_WINDOW = 128
RATIO = 4
C_PAGE = 64  # tokens per page of the compressed cache


def make_cache(budget_rows):
    """A chunk cache with the fields the compressed gather reads, on the CPU."""
    n = len(SEQ_LENS)
    query_start_loc = torch.zeros(n + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(torch.tensor(EXTEND_LENS), 0)
    num_q = int(query_start_loc[-1])
    swa_lens = [min(s, e + SWA_WINDOW - 1) for s, e in zip(SEQ_LENS, EXTEND_LENS)]
    swa_offsets = torch.zeros(n + 1, dtype=torch.int32)
    swa_offsets[1:] = torch.cumsum(torch.tensor(swa_lens), 0)
    return spu.SparsePrefillChunkCache(
        num_reqs=n,
        num_qo_tokens=num_q,
        max_seq_len=max(SEQ_LENS),
        swa_window_size=SWA_WINDOW,
        swa_page_size=256,
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_pos=torch.zeros(num_q, dtype=torch.int32),
        swa_token_ids=torch.zeros(sum(swa_lens), dtype=torch.int32),
        swa_first_pos=torch.zeros(n, dtype=torch.int32),
        swa_gather_lens=torch.tensor(swa_lens, dtype=torch.int32),
        swa_offsets=swa_offsets,
        seq_lens_cpu=list(SEQ_LENS),
        extend_seq_lens_cpu=list(EXTEND_LENS),
        swa_gather_lens_cpu=swa_lens,
        workspace_budget_rows=budget_rows,
    )


def first_page(request):
    return 1000 * (request + 1)


def page_table():
    """(query rows, blocks): every row of request r holds r's own page ids. The
    width is that of the longest request, as the engine's page table."""
    blocks = max(SEQ_LENS) // RATIO // C_PAGE + 1
    rows = []
    for r, extend in enumerate(EXTEND_LENS):
        rows += [torch.arange(blocks, dtype=torch.int32) + first_page(r)] * extend
    return torch.stack(rows)


def expected_token_ids():
    """Request r's first seq_len // RATIO compressed slots, packed in request order."""
    ids = []
    for r, seq_len in enumerate(SEQ_LENS):
        ids += [(first_page(r) + k // C_PAGE) * C_PAGE + k % C_PAGE
                for k in range(seq_len // RATIO)]
    return torch.tensor(ids, dtype=torch.int32)


def test_one_group_gathers_each_request_from_its_own_pages():
    cache = make_cache(budget_rows=0)
    gather = cache.ensure_compressed(RATIO, page_table(), C_PAGE)
    assert len(cache.groups_for(RATIO, gather.flat_token_ids.shape[0])) == 1
    assert torch.equal(gather.flat_token_ids, expected_token_ids())


def test_request_groups_gather_the_same_token_ids_with_group_relative_bases():
    cache = make_cache(budget_rows=700)
    gather = cache.ensure_compressed(RATIO, page_table(), C_PAGE)
    groups = cache.groups_for(RATIO, gather.flat_token_ids.shape[0])
    assert len(groups) > 1, "the budget splits this chunk"
    # The packed token ids do not depend on the groups: group g reads rows
    # [c_start, c_end) of them.
    assert torch.equal(gather.flat_token_ids, expected_token_ids())
    counts = [s // RATIO for s in SEQ_LENS]
    swa_lens = cache.swa_gather_lens_cpu
    compressed_base, swa_base = spu.sparse_prefill_group_bases(groups, counts, swa_lens)
    assert gather.compressed_base.tolist() == compressed_base
    assert gather.swa_base.tolist() == swa_base
    # Every request's rows lie inside its group's workspace.
    for g in groups:
        for r in range(g.req_start, g.req_end):
            assert compressed_base[r] + counts[r] <= g.num_compressed
            assert g.num_compressed <= swa_base[r] and swa_base[r] + swa_lens[r] <= g.num_rows
