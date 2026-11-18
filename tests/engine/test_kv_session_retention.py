# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Bound of the engine's KV session registry, without CUDA import dependencies.

Load session_ref_tracker.py and retention_order.py from their source files with
the scheduler imports stubbed, and drive KvSessionRetention through admit() and
control() with a small bound.
"""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

SRT = Path(__file__).resolve().parents[2] / "evolving_infra/engine/decoupled_engine/srt"
PKG = "decoupled_engine.srt"
BOOT = "boot-1"
CAP = 4


class FinishAbort:
    pass


def stub(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)


def load(monkeypatch, name, rel):
    spec = importlib.util.spec_from_file_location(name, SRT / rel)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tracker(monkeypatch):
    for name in ("decoupled_engine", PKG, f"{PKG}.managers",
                 f"{PKG}.managers.scheduler_components", f"{PKG}.mem_cache",
                 f"{PKG}.mem_cache.unified_cache"):
        stub(monkeypatch, name)
    stub(monkeypatch, f"{PKG}.managers.scheduler_components.phase_timers",
         timed_phase=lambda _name: (lambda f: f))
    stub(monkeypatch, f"{PKG}.managers.schedule_batch", FINISH_ABORT=FinishAbort)
    load(monkeypatch, f"{PKG}.mem_cache.unified_cache.retention_order",
         "mem_cache/unified_cache/retention_order.py")
    return load(monkeypatch, f"{PKG}.mem_cache.unified_cache.session_ref_tracker",
                "mem_cache/unified_cache/session_ref_tracker.py")


class Tree:
    """Tree whose retention prefix is always the root: no node counters move."""

    page_size = 16
    root_node = SimpleNamespace(host_session_anchors=None)

    def locate_retention_prefix(self, key, start_node=None, start_offset=0):
        return self.root_node, []


class Req:
    to_finish = None
    finished_reason = None

    def __init__(self, sid):
        self.rid = f"r-{sid}"
        self.kv_session = {"session_id": sid, "version": 1}


@pytest.fixture
def registry(tracker):
    return tracker.KvSessionRetention(Tree(), lambda actions: None, max_sessions=CAP)


def control(registry, op, sid):
    return registry.control(
        {"op": op, "session_id": sid, "version": 1, "expected_boot_id": BOOT})


def fill_retired(registry, op):
    for i in range(CAP):
        control(registry, op, f"old-{i}")
    assert len(registry.sessions) == CAP


@pytest.mark.parametrize("op", ["end", "release"])
def test_admit_new_session_when_full_of_retired(registry, op):
    fill_retired(registry, op)

    entry = registry.admit(Req("new"), input_key=("k",))

    assert entry is registry.sessions["new"]
    assert list(registry.sessions) == ["new"]


def test_admit_new_session_refused_when_full_of_live(registry):
    for i in range(CAP):
        registry.admit(Req(f"live-{i}"), input_key=("k",))

    with pytest.raises(ValueError, match="capacity exceeded"):
        registry.admit(Req("new"), input_key=("k",))
    assert len(registry.sessions) == CAP


@pytest.mark.parametrize("fill", ["end", "release"])
@pytest.mark.parametrize("op", ["end", "release"])
def test_control_unseen_session_when_full_of_retired(registry, fill, op):
    fill_retired(registry, fill)

    reply = control(registry, op, "unseen")

    assert reply["status"] == {"end": "ended", "release": "released"}[op]
    assert list(registry.sessions) == ["unseen"]


def test_ended_fence_kept_below_bound(registry):
    control(registry, "end", "done")

    assert registry.admit(Req("done"), input_key=("k",)) is None
    assert registry.sessions["done"].state == "ended"


def test_admit_new_session_after_live_session_released_at_bound(registry):
    for i in range(CAP):
        registry.admit(Req(f"live-{i}"), input_key=("k",))
    with pytest.raises(ValueError, match="capacity exceeded"):
        registry.admit(Req("first"), input_key=("k",))

    control(registry, "release", "live-0")
    entry = registry.admit(Req("second"), input_key=("k",))

    assert entry is registry.sessions["second"]
    assert "live-0" not in registry.sessions
