# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Session retention on FULL+SWA trees (DeepSeek-V4, other hybrid-SWA models).

The retention counters (host_session_count / host_session_active) sit on tree
nodes, so they cover a node's SWA copies as well as its FULL ones. These CPU
tests check that every host eviction follows the retention order for SWA too.

- N1-N8: non-uniform SWA (the default DSv4 tree), where the SWA host pool
  holds trailing windows. SWA host pressure frees SWA host copies in the
  retention order (_drive_retention_swa_host_eviction): class 0 first, then
  the other class-1 windows, then the resume windows (the window before a
  live registration's anchor or input end); idle before active under the
  soft pin, CPU recency inside each. The walk ends on SWA tokens; live
  sessions keep their FULL host copies. FULL pressure keeps the whole-node
  order and counts the SWA tokens of its victims. anchor_tokens follows the
  anchor.
- U1-U2: --dsv4-uniform-cache-lifecycle, where SWA host goes with FULL: the
  walk ends on the pool that asked (a node without an SWA host copy frees no
  SWA), and the joint victims follow the classes.
- W1-W4: --dsv4-uniform-swa-residency window. No walk releases a duplicate's
  host copy (W8). The host headroom (device_drop) also deletes backed-up
  device leaves whole (W8b) in the retention walk, class before kind, so a
  class-0 device leaf goes before a class-1 host-only leaf. With nothing
  tagged its victims equal the base headroom's (host-only leaves, then
  drop_backed_device_leaves), and they do not depend on candidate order.
- R1-R2: a READY status needs the SWA a resume needs: the trailing window at
  the end (non-uniform), every node's SWA (uniform, either residency).
- S1: the KV_SESSION_STATS swa_* and window drop fields.
- D1: the SWA victims do not depend on the candidates' insertion order or
  the node hashes (TP lockstep).
- T1: UnifiedRadixCache accepts retention on FULL+SWA, window SWA residency
  included, and still refuses Mamba.
"""

import contextlib
import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("torch", reason="needs the engine (decoupled-engine)")
pytest.importorskip("decoupled_engine", reason="needs the engine (decoupled-engine)")

import torch  # noqa: E402

from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams
from decoupled_engine.srt.mem_cache.radix_cache import RadixKey
from decoupled_engine.srt.mem_cache.unified_cache.components.full import FullComponent
from decoupled_engine.srt.mem_cache.unified_cache.components.swa import SWAComponent
from decoupled_engine.srt.mem_cache.unified_cache.components import base as tree_component
from decoupled_engine.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    TreeComponent,
)
from decoupled_engine.srt.mem_cache.unified_cache.session_ref_tracker import KvSessionRetention
from decoupled_engine.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)


FULL, SWA = ComponentType.FULL, ComponentType.SWA
PAGE = 2
WINDOW = 4  # sliding_window_size: two pages


class CpuSWAComponent(SWAComponent):
    """A real SWAComponent without the device allocator check."""

    def __init__(self, cache, params, *, uniform=False, window=False, host_pool=None):
        TreeComponent.__init__(self, cache, params)
        self._session_leaf_covered_len = {}
        self.uniform_lifecycle = uniform
        self.device_window = window
        self.sliding_window_size = params.sliding_window_size
        self.full_window_pages = params.sliding_window_size // params.page_size
        self._swa_kv_pool_host = host_pool


@contextlib.contextmanager
def pinned(node_base=70_000_000, tick_base=3_000_000.0):
    """Fix the process-wide node-id and access-tick counters."""
    saved = tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT
    tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT = tick_base
    try:
        with patch.object(UnifiedTreeNode, "counter", node_base):
            yield
    finally:
        tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT = saved


class SaltedNode(UnifiedTreeNode):
    """A node that hashes by (salt, id). A set of these iterates in an order
    that the salt picks; plain nodes hash by object address."""

    def __hash__(self):
        return hash((self._salt, self.id))


def counter_errors(tracker):
    """Nodes whose (host_session_count, host_session_active) differ from a
    recount over the registry: (node id, counters, expected)."""
    tree = tracker.tree
    expected = {}
    for entry in tracker.sessions.values():
        node = entry.anchor
        while node is not None and node is not tree.root_node:
            count, active = expected.get(node.id, (0, 0))
            expected[node.id] = (count + 1, active + int(bool(entry.request_ids)))
            node = node.parent
    errors = []
    for node_id, node in sorted(tree._node_arena.items()):
        if node is tree.root_node:
            continue
        got = (node.host_session_count, node.host_session_active)
        want = expected.pop(node_id, (0, 0))
        if got != want or node.host_session_active < 0:
            errors.append((node_id, got, want))
    # An anchor path through a node that left the arena.
    errors.extend((node_id, None, want) for node_id, want in sorted(expected.items()))
    return errors


class SwaTree:
    """A FULL+SWA UnifiedTreeCore in write-through mode with the retention
    registry. Values are CPU tensors (no allocator, no host pool).

    ``uniform``: --dsv4-uniform-cache-lifecycle (full residency).
    ``window``: the uniform lifecycle with window SWA residency.
    ``victims``: ("node", id) for a whole host release (host leaf,
    duplicate or dropped device leaf), ("swa", id) for an SWA-only host
    release, in order.
    """

    def __init__(self, *, uniform=False, window=False, soft_pin=True, salt=None):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=PAGE,
            tree_components=(FULL, SWA),
            sliding_window_size=WINDOW,
        )
        cache = SimpleNamespace(enable_session_radix_cache=False)
        full = FullComponent(cache, params)
        swa = CpuSWAComponent(
            cache, params, uniform=uniform or window, window=window, host_pool=object()
        )
        self.core = UnifiedTreeCore(params, {FULL: full, SWA: swa})
        self.core.is_write_back = False
        self.core.enable_hicache = True
        self.core.has_swa_host_pool = True
        self.tracker = KvSessionRetention(
            self.core, lambda actions: None, active_soft_pin=soft_pin
        )
        self.core.kv_session_retention = self.tracker
        self.salt = salt
        self.next_token = 1
        self.nodes = {}
        self.victims = []
        push_node = self.core._peer_push_removed
        push_swa = self.core._peer_push_swa_removed

        def record_node(node):
            self.victims.append(("node", node.id))
            return push_node(node)

        def record_swa(node):
            self.victims.append(("swa", node.id))
            return push_swa(node)

        self.core._peer_push_removed = record_node
        self.core._peer_push_swa_removed = record_swa

    def add(
        self,
        name,
        parent,
        *,
        tick,
        pages=2,
        full_device=False,
        swa_device=False,
        swa_host=False,
    ):
        """A child of ``parent`` (a name or None: the root) with a FULL host
        copy; both ticks are ``tick``. Each flag adds that copy of the whole
        edge."""
        core = self.core
        parent_node = core.root_node if parent is None else self.nodes[parent]
        n = pages * PAGE
        tokens = list(range(self.next_token, self.next_token + n))
        self.next_token += n
        node = core._new_node()
        if self.salt is not None:
            node.__class__ = SaltedNode
            node._salt = self.salt
        node.parent = parent_node
        node.key = RadixKey(array("q", tokens))
        node.last_access_time = node.host_last_access_time = tick
        parent_node.children[node.key.child_key(PAGE)] = node
        values = torch.arange(tokens[0], tokens[0] + n)
        full, swa = node.component_data[FULL], node.component_data[SWA]
        if full_device:
            full.value = values.clone()
            core.component_evictable_size_[FULL] += n
        full.host_value = values.clone()
        if swa_device:
            swa.value = values.clone()
            core.component_evictable_size_[SWA] += n
            core.lru_lists[SWA].insert_mru(node)
        if swa_host:
            swa.host_value = values.clone()
            if not swa_device:
                core.host_lru_lists[SWA].insert_mru(node)
        core._update_evictable_leaf_sets(node)
        core._update_evictable_leaf_sets(parent_node)
        core._update_duplicate_tracking(node)
        self.nodes[name] = node

    def id(self, name):
        return self.nodes[name].id

    def path_tokens(self, name):
        node, out = self.nodes[name], []
        while node is not self.core.root_node:
            out = list(node.key) + out
            node = node.parent
        return out

    def admit(self, sid, version, name, trim=0):
        """Admit the path to ``name`` less its last ``trim`` tokens (an
        admission that ends inside the edge splits it)."""
        tokens = self.path_tokens(name)
        ids = array("q", tokens[: len(tokens) - trim])
        req = SimpleNamespace(
            rid=f"{sid}-{version}",
            kv_session={"session_id": sid, "version": version},
            origin_input_ids=ids,
            extra_key=None,
            cache_salt=None,
        )
        self.tracker.admit(req, RadixKey(ids))
        return req

    def idle(self, sid, name):
        """A finished request: a registration without a live request."""
        req = self.admit(sid, 1, name)
        self.tracker.refresh_request(req, finished=True)

    def control(self, op, sid):
        return self.tracker.control({"op": op, "session_id": sid, "version": 1})

    def evict(self, component, tokens, device_drop=False):
        """One drive_host_eviction call; the freed host tokens by pool."""
        result = self.core.drive_host_eviction(
            component, tokens, device_drop=device_drop
        )
        freed = (result.tracker[FULL], result.tracker[SWA])
        result.device_frees.clear()
        result.host_frees.clear()
        return freed

    def has(self, name, component, host=True):
        node = self.nodes[name]
        if node.deleted:
            return False
        cd = node.component_data[component]
        return (cd.host_value if host else cd.value) is not None


def anchor_token_errors(tracker):
    """Registrations whose anchor_tokens differs from their anchor's path."""
    errors = []
    for sid, entry in sorted(tracker.sessions.items()):
        want = sum(len(node.key) for node in tracker._path(entry.anchor))
        if entry.anchor_tokens != want:
            errors.append((sid, entry.anchor_tokens, want))
    return errors


def mixed_tree(**kw):
    """Non-uniform: an ended session d and an idle session a."""
    t = SwaTree(**kw)
    # Ended session d: an interior window D1 and a leaf D2 whose window an
    # earlier SWA eviction already took (class 0 after END).
    t.add("D1", None, tick=1, swa_host=True)
    t.add("D2", "D1", tick=2)
    t.admit("d", 1, "D2")
    t.control("end", "d")
    # Idle session a: A2 is its anchor and window (class 1, active 0).
    t.add("A1", None, tick=3, pages=4)
    t.add("A2", "A1", tick=4, swa_host=True)
    t.idle("a", "A2")
    return t


class TestNonUniformSwaRetention(unittest.TestCase):
    def test_n1_swa_pressure_takes_class0_swa_before_live_sessions(self):
        t = mixed_tree()
        # D1's window is class 0; A2's is class 1: D1 loses only its SWA
        # host copy and keeps FULL, and nothing of session a goes.
        self.assertEqual(t.evict(SWA, WINDOW), (0, WINDOW))
        self.assertEqual(t.victims, [("swa", t.id("D1"))])
        self.assertTrue(t.has("D1", FULL))
        self.assertFalse(t.has("D1", SWA))
        for name in ("A1", "A2"):
            self.assertTrue(t.has(name, FULL), name)
        self.assertTrue(t.has("A2", SWA))
        # Only class-1 SWA is left: A2 is a host leaf, so it goes whole.
        self.assertEqual(t.evict(SWA, WINDOW), (WINDOW, WINDOW))
        self.assertEqual(t.victims[1:], [("node", t.id("A2"))])
        st = t.tracker.stats
        self.assertEqual(st.swa_evicted_tokens, {0: WINDOW, 1: WINDOW})
        self.assertEqual(st.evicted_tokens, {(1, "host"): WINDOW})
        self.assertEqual(counter_errors(t.tracker), [])
        # Nothing with an SWA host copy is left; the FULL-only D2 stays.
        self.assertEqual(t.evict(SWA, WINDOW), (0, 0))
        self.assertTrue(t.has("D2", FULL))

    def test_n2_soft_pin_takes_idle_windows_before_active_ones(self):
        for soft_pin in (True, False):
            with self.subTest(soft_pin=soft_pin):
                t = SwaTree(soft_pin=soft_pin)
                # The active session's window is older.
                t.add("B", None, tick=1, swa_host=True)
                t.add("I", None, tick=2, swa_host=True)
                t.admit("busy", 1, "B")  # queued or running
                t.idle("idle", "I")
                t.evict(SWA, WINDOW)
                first = "I" if soft_pin else "B"
                self.assertEqual(t.victims, [("node", t.id(first))])
                active = 0 if soft_pin else WINDOW
                self.assertEqual(t.tracker.stats.swa_evicted_active_tokens, active)

    def test_n3_windows_above_an_anchor_go_before_its_own(self):
        t = SwaTree()
        # A chunk boundary window W1 above the anchor's window W3. A prefix
        # match touches ancestors before the anchor, so W1 is older.
        t.add("W1", None, tick=1, swa_host=True)
        t.add("M", "W1", tick=2)
        t.add("W3", "M", tick=3, swa_host=True)
        t.idle("a", "W3")
        self.assertEqual(t.control("status", "a")["status"], "ready")
        self.assertEqual(t.evict(SWA, 1), (0, WINDOW))
        self.assertEqual(t.victims, [("swa", t.id("W1"))])
        # The anchor's window still serves a resume there.
        reply = t.control("status", "a")
        self.assertEqual(reply["status"], "ready")
        self.assertEqual(reply["ready_tokens"], 3 * WINDOW)

    def test_n4_full_pressure_keeps_whole_node_order_and_counts_swa(self):
        t = mixed_tree()
        # Class 0: the host leaf D2 (no SWA), then D1 once it is a leaf
        # (its window goes with it); A2 (class 1) stays.
        self.assertEqual(t.evict(FULL, 2 * WINDOW), (2 * WINDOW, WINDOW))
        self.assertEqual(t.victims, [("node", t.id("D2")), ("node", t.id("D1"))])
        self.assertTrue(t.has("A2", SWA))
        st = t.tracker.stats
        self.assertEqual(st.swa_evicted_tokens, {0: WINDOW})
        self.assertEqual(st.evicted_tokens, {(0, "host"): 2 * WINDOW})
        # A FULL duplicate whose SWA lives only on the host keeps both host
        # copies: the SWA host copy is its only one.
        t.add("G", None, tick=0, full_device=True, swa_host=True)
        before = len(t.victims)
        t.evict(FULL, 1)
        self.assertNotIn(("node", t.id("G")), t.victims[before:])
        self.assertTrue(t.has("G", FULL) and t.has("G", SWA))
        self.assertEqual(counter_errors(t.tracker), [])

    def test_n6_resume_windows_go_after_other_class1_windows(self):
        t = SwaTree()
        # Two idle sessions, each a chunk-boundary window above its anchor.
        t.add("A1", None, tick=1, swa_host=True)
        t.add("A2", "A1", tick=2, swa_host=True)
        t.idle("a", "A2")
        t.add("B1", None, tick=3, swa_host=True)
        t.add("B2", "B1", tick=4, swa_host=True)
        t.idle("b", "B2")
        self.assertEqual(t.tracker.swa_resume_nodes(WINDOW), {t.id("A2"), t.id("B2")})
        # By CPU tick alone, a's anchor window (tick 2) would go before b's
        # boundary window B1 (tick 3); both anchors keep their windows.
        t.evict(SWA, 2 * WINDOW)
        self.assertEqual(t.victims, [("swa", t.id("A1")), ("swa", t.id("B1"))])
        for sid in ("a", "b"):
            self.assertEqual(t.control("status", sid)["status"], "ready")
        self.assertEqual(t.tracker.stats.swa_evicted_resume_tokens, 0)
        # Then the resume windows, oldest first.
        t.evict(SWA, 1)
        self.assertEqual(t.victims[2:], [("node", t.id("A2"))])
        self.assertEqual(t.tracker.stats.swa_evicted_resume_tokens, WINDOW)

    def test_n7_input_end_window_is_a_resume_window(self):
        t = SwaTree()
        # One turn: a prompt in two chunks (C1, C2), then the output O.
        t.add("C1", None, tick=1, swa_host=True)
        t.add("C2", "C1", tick=2, swa_host=True)
        t.add("O", "C2", tick=3, swa_host=True)
        req = t.admit("s", 1, "C2")
        # The final insert ends at O; the input (prompt) still ends at C2.
        key = RadixKey(array("q", t.path_tokens("O")))
        t.tracker.refresh_request(req, finished=True, key=key, node_id=t.id("O"))
        self.assertIs(t.tracker.sessions["s"].anchor, t.nodes["O"])
        self.assertEqual(t.tracker.swa_resume_nodes(WINDOW), {t.id("C2"), t.id("O")})
        t.evict(SWA, 1)
        self.assertEqual(t.victims, [("swa", t.id("C1"))])

    def test_n8_short_edges_and_anchor_tokens(self):
        t = SwaTree()
        # One-page edges: the two-page window spans the anchor and its parent.
        t.add("X", None, tick=1, pages=1, swa_host=True)
        t.add("Y", "X", tick=2, pages=1, swa_host=True)
        t.add("Z", "Y", tick=3, pages=1, swa_host=True)
        t.idle("s", "Z")
        entry = t.tracker.sessions["s"]
        self.assertEqual(entry.anchor_tokens, 3 * PAGE)
        self.assertEqual(t.tracker.swa_resume_nodes(WINDOW), {t.id("Y"), t.id("Z")})
        # X (outside the window) first, then the window oldest first: Y loses
        # its SWA host copy, Z (a host leaf) goes whole and moves the anchor.
        t.evict(SWA, 3 * PAGE)
        self.assertEqual(
            t.victims,
            [("swa", t.id("X")), ("swa", t.id("Y")), ("node", t.id("Z"))],
        )
        self.assertIs(entry.anchor, t.nodes["Y"])
        self.assertEqual(entry.anchor_tokens, 2 * PAGE)
        self.assertEqual(anchor_token_errors(t.tracker), [])
        # Released and ended registrations need no window.
        t.control("end", "s")
        self.assertEqual(t.tracker.swa_resume_nodes(WINDOW), set())

    def test_n5_walk_ends_on_swa_tokens(self):
        t = SwaTree()
        for i in range(4):
            t.add(f"C{i}", None, tick=i, swa_host=True, full_device=True)
        # Class 0 windows, all SWA-only (FULL stays on device): the walk
        # stops after the second window.
        self.assertEqual(t.evict(SWA, WINDOW + 1), (0, 2 * WINDOW))
        self.assertEqual(t.victims, [("swa", t.id("C0")), ("swa", t.id("C1"))])


class TestUniformSwaRetention(unittest.TestCase):
    def test_u1_swa_request_ends_on_swa_tokens(self):
        t = SwaTree(uniform=True)
        # X has a FULL host copy without an SWA one, so it frees FULL only.
        # Full residency never builds X (backups and prefetches carry both);
        # window residency can, until the node's SWA copy is acked.
        t.add("X", None, tick=0)
        t.add("Y", None, tick=1, swa_host=True)
        t.add("Z", None, tick=2, swa_host=True)
        t.idle("z", "Z")
        self.assertEqual(t.evict(SWA, 1), (2 * WINDOW, WINDOW))
        self.assertEqual(t.victims, [("node", t.id("X")), ("node", t.id("Y"))])
        self.assertTrue(t.has("Z", FULL) and t.has("Z", SWA))
        st = t.tracker.stats
        self.assertEqual(st.swa_evicted_tokens, {0: WINDOW})
        self.assertEqual(st.evicted_tokens, {(0, "host"): 2 * WINDOW})

    def test_u2_joint_victims_follow_the_classes(self):
        t = SwaTree(uniform=True)
        t.add("BUSY", None, tick=0, swa_host=True)
        t.add("IDLE", None, tick=1, swa_host=True)
        t.add("FREE", None, tick=2, swa_host=True)
        t.admit("busy", 1, "BUSY")
        t.idle("idle", "IDLE")
        t.evict(SWA, 3 * WINDOW)
        self.assertEqual(
            t.victims,
            [("node", t.id(n)) for n in ("FREE", "IDLE", "BUSY")],
        )
        st = t.tracker.stats
        self.assertEqual(st.swa_evicted_tokens, {0: WINDOW, 1: 2 * WINDOW})
        self.assertEqual(st.swa_evicted_active_tokens, WINDOW)
        self.assertEqual(st.evicted_active_tokens, WINDOW)
        self.assertEqual(counter_errors(t.tracker), [])


def window_pair(**kw):
    """D: a class-0 device leaf with FULL and SWA on both tiers (a duplicate
    under full residency); H: an idle session's host-only leaf (class 1)."""
    t = SwaTree(**kw)
    t.add("D", None, tick=0, full_device=True, swa_device=True, swa_host=True)
    t.add("H", None, tick=1, swa_host=True)
    t.idle("h", "H")
    return t


def build_window(seed, salt=None, tagged=False):
    """A seeded window residency tree: FULL on the device for a prefix-closed
    subset, every node backed up with its whole SWA, device SWA on some
    device nodes, GPU ticks independent of CPU ticks. ``tagged`` adds six
    registrations, half of them idle."""
    rng = random.Random(seed)
    t = SwaTree(window=True, salt=salt)
    names = []
    for i in range(40):
        parent = rng.choice([None] + names) if names else None
        on_device = parent is None or t.has(parent, FULL, host=False)
        full_device = on_device and rng.random() < 0.5
        name = f"n{i}"
        t.add(
            name,
            parent,
            tick=rng.randint(0, 5),
            pages=rng.randint(1, 3),
            full_device=full_device,
            swa_device=full_device and rng.random() < 0.3,
            swa_host=True,
        )
        names.append(name)
    for name in names:
        t.nodes[name].last_access_time = rng.randint(0, 5)
    if tagged:
        for i in range(6):
            req = t.admit(f"s{i}", 1, rng.choice(names))
            if rng.random() < 0.5:
                t.tracker.refresh_request(req, finished=True)  # idle
    return t


class TestWindowSwaRetention(unittest.TestCase):
    def test_w1_no_duplicate_release_under_window(self):
        # Full residency releases D's host copy first: class 0 before 1.
        u = window_pair(uniform=True)
        self.assertEqual(u.evict(FULL, WINDOW), (WINDOW, WINDOW))
        self.assertEqual(u.victims, [("node", u.id("D"))])
        self.assertTrue(u.has("D", FULL, host=False))
        self.assertFalse(u.has("D", FULL))
        # Window residency keeps every duplicate (W8): only H can go.
        w = window_pair(window=True)
        self.assertEqual(w.evict(FULL, WINDOW), (WINDOW, WINDOW))
        self.assertEqual(w.victims, [("node", w.id("H"))])
        for ct in (FULL, SWA):
            self.assertTrue(w.has("D", ct) and w.has("D", ct, host=False), ct)
        self.assertEqual(w.tracker.stats.evicted_tokens, {(1, "host"): WINDOW})

    def test_w2_headroom_drops_class0_device_leaf_first(self):
        t = window_pair(window=True)
        # The headroom walk deletes D whole, GPU copy too; H stays.
        self.assertEqual(t.evict(FULL, WINDOW, device_drop=True), (WINDOW, WINDOW))
        self.assertTrue(t.nodes["D"].deleted)
        self.assertTrue(t.has("H", FULL) and t.has("H", SWA))
        st = t.tracker.stats
        self.assertEqual(st.evicted_tokens, {(0, "drop"): WINDOW})
        fields = dict(part.split("=", 1) for part in t.tracker.stats_line().split())
        self.assertEqual(fields["evicted_c0_drop"], str(WINDOW))
        self.assertEqual(fields["evicted_c1_drop"], "0")
        # Inside class 1, a host-only leaf goes before a device leaf.
        t.add("L", None, tick=2, full_device=True, swa_host=True)
        t.idle("l", "L")
        t.evict(FULL, WINDOW, device_drop=True)
        self.assertFalse(t.has("H", FULL))
        self.assertTrue(t.has("L", FULL) and t.has("L", FULL, host=False))
        t.evict(FULL, WINDOW, device_drop=True)
        self.assertTrue(t.nodes["L"].deleted)
        self.assertEqual(st.evicted_tokens[(1, "drop")], WINDOW)
        self.assertEqual(counter_errors(t.tracker), [])
        self.assertEqual(anchor_token_errors(t.tracker), [])

    def test_w3_untagged_headroom_matches_base(self):
        dropped = 0
        for seed in range(12):
            with self.subTest(seed=seed):
                runs = []
                for retention in (True, False):
                    with pinned():
                        t = build_window(seed)
                        if not retention:
                            t.core.kv_session_retention = None
                        steps = random.Random(seed)
                        for _ in range(4):
                            # Large enough to reach the device leaves.
                            want = PAGE * steps.randint(4, 24)
                            if retention:
                                t.evict(FULL, want, device_drop=True)
                            else:
                                # maintain_swa_host_headroom without retention
                                freed = t.evict(FULL, want)[0]
                                if freed < want:
                                    result = t.core.drop_backed_device_leaves(
                                        want - freed
                                    )
                                    result.device_frees.clear()
                                    result.host_frees.clear()
                            t.core.sanity_check([], [])
                        runs.append((t.victims, t.core.host_release_digest))
                        if retention:
                            dropped += sum(
                                v
                                for (
                                    _,
                                    kind,
                                ), v in t.tracker.stats.evicted_tokens.items()
                                if kind == "drop"
                            )
                self.assertEqual(runs[0], runs[1])
        self.assertGreater(dropped, 0)

    def test_w4_window_victims_do_not_depend_on_candidate_order(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                runs = []
                for salt in (1, 2, 3):
                    with pinned():
                        t = build_window(seed, salt=salt, tagged=True)
                        steps = random.Random(seed)
                        for _ in range(6):
                            want = PAGE * steps.randint(1, 6)
                            t.evict(FULL, want, device_drop=steps.random() < 0.6)
                            t.core.sanity_check([], [])
                        self.assertEqual(counter_errors(t.tracker), [])
                        self.assertEqual(anchor_token_errors(t.tracker), [])
                        evicted = sorted(t.tracker.stats.evicted_tokens.items())
                        runs.append((t.victims, t.core.host_release_digest, evicted))
                self.assertEqual(runs[0], runs[1])
                self.assertEqual(runs[0], runs[2])


class TestSwaReadyStatus(unittest.TestCase):
    def test_r1_nonuniform_ready_needs_the_trailing_window(self):
        t = SwaTree()
        t.add("P1", None, tick=1)
        t.add("P2", "P1", tick=2, swa_host=True)
        t.idle("s", "P2")
        reply = t.control("status", "s")
        self.assertEqual((reply["status"], reply["ready_tokens"]), ("ready", 8))
        # A write-through in flight on the anchor: its window is not acked.
        t.nodes["P2"].write_through_pending_id = 7
        reply = t.control("status", "s")
        self.assertEqual((reply["status"], reply["ready_tokens"]), ("pending", 0))
        t.nodes["P2"].write_through_pending_id = None

        u = SwaTree()
        u.add("P1", None, tick=1)
        # The anchor keeps FULL on the device: not a host leaf, so SWA
        # pressure takes only its window.
        u.add("P2", "P1", tick=2, full_device=True, swa_host=True)
        u.idle("s", "P2")
        self.assertEqual(u.control("status", "s")["status"], "ready")
        u.evict(SWA, 1)
        self.assertEqual(u.victims, [("swa", u.id("P2"))])
        # FULL is still on the host along the whole path, but no window
        # ends anywhere on it: nothing is resumable from the host.
        reply = u.control("status", "s")
        self.assertEqual(
            (reply["status"], reply["ready_tokens"], reply["required_tokens"]),
            ("pending", 0, 8),
        )

    def test_r1b_gapless_run_from_root_is_resumable(self):
        t = SwaTree()
        # Shorter than the window, with SWA on every page from the root.
        t.add("S", None, tick=1, pages=1, swa_host=True)
        t.idle("s", "S")
        reply = t.control("status", "s")
        self.assertEqual((reply["status"], reply["ready_tokens"]), ("ready", PAGE))

    def test_r2_uniform_ready_needs_every_swa_host_copy(self):
        for window in (False, True):
            with self.subTest(window=window):
                t = SwaTree(uniform=True, window=window)
                t.add("P1", None, tick=1, swa_host=True)
                t.add("P2", "P1", tick=2)  # FULL host only
                t.add("P3", "P2", tick=3, swa_host=True)
                t.idle("s", "P3")
                reply = t.control("status", "s")
                self.assertEqual(
                    (reply["status"], reply["ready_tokens"], reply["required_tokens"]),
                    ("pending", WINDOW, 3 * WINDOW),
                )
                swa = t.nodes["P2"].component_data[SWA]
                swa.host_value = torch.arange(WINDOW)
                reply = t.control("status", "s")
                self.assertEqual(
                    (reply["status"], reply["ready_tokens"]), ("ready", 12)
                )


class TestSwaRetentionStats(unittest.TestCase):
    def test_s1_stats_line_swa_fields(self):
        t = mixed_tree()
        t.evict(SWA, WINDOW)
        fields = dict(part.split("=", 1) for part in t.tracker.stats_line().split())
        # D1's window went; A2's (class 1) is left.
        self.assertEqual(fields["swa_host_tokens"], str(WINDOW))
        self.assertEqual(fields["swa_class0_host_tokens"], "0")
        self.assertEqual(fields["swa_evicted_c0"], str(WINDOW))
        self.assertEqual(fields["swa_evicted_c1"], "0")
        self.assertEqual(fields["swa_evicted_active_tokens"], "0")
        self.assertEqual(fields["swa_evicted_resume_tokens"], "0")
        names = list(fields)
        self.assertLess(names.index("evicted_active_tokens"), names.index("scan_ms"))
        self.assertLess(names.index("swa_evicted_c1"), names.index("scan_ms"))
        # The drop fields are for window residency only.
        self.assertNotIn("evicted_c0_drop", fields)

    def test_s1b_full_only_line_has_no_swa_fields(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=PAGE,
            tree_components=(FULL,),
        )
        cache = SimpleNamespace(enable_session_radix_cache=False)
        core = UnifiedTreeCore(params, {FULL: FullComponent(cache, params)})
        tracker = KvSessionRetention(core, lambda actions: None)
        core.kv_session_retention = tracker
        self.assertNotIn("swa_", tracker.stats_line())
        self.assertIsNone(core.retention_host_scan()[3])


def build_random(seed, salt, lru_rng, uniform=False):
    """A seeded tree; ``salt`` sets node hashes, ``lru_rng`` the SWA host
    LRU insertion order (as different ranks may build it). Non-uniform: SWA
    copies on either tier or both. Uniform: SWA follows FULL, except a few
    nodes without an SWA host copy (the state window residency has before an
    SWA ack; full residency never builds it)."""
    rng = random.Random(seed)
    t = SwaTree(salt=salt, uniform=uniform)
    names = []
    for i in range(40):
        parent = rng.choice([None] + names) if names else None
        # FULL on the device only under the root or a device-resident parent.
        on_device = parent is None or t.has(parent, FULL, host=False)
        full_device = on_device and rng.random() < 0.3
        if uniform:
            swa_device, swa_host = full_device, rng.random() < 0.9
        else:
            swa_device = full_device and rng.random() < 0.4
            swa_host = rng.random() < 0.6
        name = f"n{i}"
        t.add(
            name,
            parent,
            tick=rng.randint(0, 5),
            pages=rng.randint(1, 3),
            full_device=full_device,
            swa_device=swa_device,
            swa_host=swa_host,
        )
        names.append(name)
    lru = t.core.host_lru_lists[SWA]
    members = sorted(lru.cache.values(), key=lambda n: n.id)
    for node in members:
        lru.remove_node(node)
    lru_rng.shuffle(members)
    for node in members:
        lru.insert_mru(node)
    for i in range(6):
        name = rng.choice(names)
        # Some admissions end a page inside the edge and split it.
        trim = PAGE if len(t.nodes[name].key) > PAGE and rng.random() < 0.3 else 0
        req = t.admit(f"s{i}", 1, name, trim=trim)
        if rng.random() < 0.5:
            t.tracker.refresh_request(req, finished=True)  # idle
    return t


class TestSwaEvictionDeterminism(unittest.TestCase):
    def test_d1_same_victims_for_any_candidate_order(self):
        victims = {False: set(), True: set()}
        for uniform, seed in [(u, s) for u in (False, True) for s in range(12)]:
            with self.subTest(uniform=uniform, seed=seed):
                runs = []
                for salt in (1, 2, 3):
                    with pinned():
                        lru_rng = random.Random(salt * 31)
                        t = build_random(seed, salt, lru_rng, uniform)
                        t.core.sanity_check([], [])
                        steps = random.Random(seed)
                        for _ in range(8):
                            ct = SWA if steps.random() < 0.7 else FULL
                            t.evict(ct, PAGE * steps.randint(1, 4))
                            # Leaf sets, the SWA host LRU and the sizes
                            # still match the tree after every walk.
                            t.core.sanity_check([], [])
                        self.assertEqual(counter_errors(t.tracker), [])
                        self.assertEqual(anchor_token_errors(t.tracker), [])
                        runs.append((t.victims, t.core.host_release_digest))
                self.assertEqual(runs[0], runs[1])
                self.assertEqual(runs[0], runs[2])
                victims[uniform].update(kind for kind, _ in runs[0][0])
        # Both walks ran: SWA-only releases (non-uniform) and whole nodes.
        self.assertEqual(victims[False], {"node", "swa"})
        self.assertEqual(victims[True], {"node"})


class TestStartupComponents(unittest.TestCase):
    def build(self, components, *, residency="full"):
        from decoupled_engine.srt.mem_cache import unified_radix_cache as urc

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=PAGE,
            tree_components=components,
            sliding_window_size=WINDOW,
            enable_kv_session_retention=True,
            component_registry_override={SWA: CpuSWAComponent},
        )
        memory = SimpleNamespace(
            dsv4_uniform_cache_lifecycle=residency != "off",
            dsv4_uniform_swa_residency=residency,
        )
        context = SimpleNamespace(
            is_config_namespace_published=lambda name: name == "memory"
        )
        with patch.object(urc, "get_context", return_value=context), patch.object(
            urc, "get_memory", return_value=memory
        ):
            return urc.UnifiedRadixCache(params)

    def test_t1_full_swa_accepted_with_any_residency_mamba_refused(self):
        for residency in ("off", "full", "window"):
            with self.subTest(residency=residency):
                cache = self.build((FULL, SWA), residency=residency)
                self.assertEqual(cache.tree_components, (FULL, SWA))
                self.assertIsNotNone(cache.kv_sessions)
                self.assertIs(cache.tree_core.kv_session_retention, cache.kv_sessions)
        with self.assertRaisesRegex(ValueError, "FULL\\+SWA"):
            self.build((FULL, ComponentType.MAMBA), residency="off")


if __name__ == "__main__":
    unittest.main()
