from __future__ import annotations

from copy import deepcopy
import time

from asterlm.data.hf_stream import (
    ASTERLM_SEQUENTIAL_CURSOR,
    BOUNDED_RESHARD_LAYOUT,
    LEGACY_NEXT_SHARD_LAYOUT,
    LEGACY_SEQUENTIAL_LAYOUT,
    SequentializedHfDataset,
    convert_sequential_cursor_to_next_shard,
    extract_legacy_resume_plan,
    is_legacy_multistream_cursor,
    open_resumable_hf_stream,
)


class FakeTable:
    def __init__(self, row):
        self.row = row

    def to_pylist(self):
        return [deepcopy(self.row)]


class FakeChild:
    def __init__(self, values):
        self.values = [dict(value) for value in values]
        self._state_dict = {"position": 0}

    def _init_state_dict(self):
        self._state_dict = {"position": 0}
        return self._state_dict

    @staticmethod
    def _position_from_state(state):
        if state.get("type") == "RebatchedArrowExamplesIterable":
            previous = state.get("previous_state")
            base = previous if isinstance(previous, dict) else state.get("examples_iterable", {})
            return int(base.get("position", 0)) + int(state.get("num_chunks_since_previous_state", 0))
        return int(state.get("position", 0))

    def load_state_dict(self, state):
        self._state_dict = {"position": self._position_from_state(state)}

    def state_dict(self):
        return deepcopy(self._state_dict)

    def iter_arrow(self):
        while self._state_dict["position"] < len(self.values):
            index = self._state_dict["position"]
            self._state_dict["position"] += 1
            yield str(index), FakeTable(self.values[index])


class FakeCycling:
    def __init__(self, children):
        self.ex_iterables = children
        self._state_dict = None

    def _init_state_dict(self):
        for child in self.ex_iterables:
            child._init_state_dict()
        # This is the real datasets==5.0.1 durable shape. Child iterable
        # objects/states are not serialized in an ``ex_iterables`` key.
        self._state_dict = {
            "ex_iterable_idx": 0,
            "previous_states": [None] * len(self.ex_iterables),
            "is_exhausted": [False] * len(self.ex_iterables),
            "type": "CyclingMultiSourcesExamplesIterable",
        }
        return self._state_dict


class FakeBuffer:
    def __init__(self, cycling):
        self.ex_iterable = cycling
        self._state_dict = None

    def _init_state_dict(self):
        self._state_dict = self.ex_iterable._init_state_dict()
        return self._state_dict


class FakeLegacyDataset:
    def __init__(self, children=None):
        if children is None:
            children = [
                FakeChild([{"text": "a0"}, {"text": "a1"}, {"text": "a2"}]),
                FakeChild([{"text": "b0"}, {"text": "b1"}, {"text": "b2"}]),
            ]
        self._ex_iterable = FakeBuffer(FakeCycling(children))


class FakeBoundedDataset:
    def __init__(self):
        self.resharded = False
        self.shuffle_args = None
        self.loaded = None

    def reshard(self):
        self.resharded = True
        return self

    def shuffle(self, **kwargs):
        self.shuffle_args = kwargs
        return self

    def load_state_dict(self, state):
        self.loaded = state


def cycling_state(*, positions=(1, 1), index=0):
    return {
        "ex_iterable_idx": index,
        "previous_states": [{"position": position} for position in positions],
        "is_exhausted": [False] * len(positions),
        "type": "CyclingMultiSourcesExamplesIterable",
    }


def legacy_cursor():
    return {
        "examples_iterable": cycling_state(positions=(1, 1), index=1),
        "epoch": 0,
    }


def exact_outer_rebatched_cursor():
    # Mirrors the uploaded datasets==5.0.1 checkpoint: the nested current state
    # is far behind, while the outer previous_state is authoritative and two
    # one-row chunks have already been emitted after it.
    stale_current = cycling_state(positions=(0, 0), index=1)
    authoritative = cycling_state(positions=(1, 1), index=0)
    return {
        "examples_iterable": {
            "examples_iterable": stale_current,
            "previous_state": authoritative,
            "batch_idx": 4,
            "num_chunks_since_previous_state": 2,
            "cropped_chunk_length": 0,
            "type": "RebatchedArrowExamplesIterable",
        },
        "epoch": 0,
    }


def user_cursor_shape(streams=10):
    def child_state(position):
        return {
            "examples_iterable": {
                "shard_idx": 0,
                "shard_example_idx": 28_000,
                "type": "ArrowExamplesIterable",
            },
            "previous_state": {
                "shard_idx": 0,
                "shard_example_idx": position,
                "type": "ArrowExamplesIterable",
            },
            "batch_idx": position + 999,
            "num_chunks_since_previous_state": 999,
            "cropped_chunk_length": 0,
            "type": "RebatchedArrowExamplesIterable",
        }

    current = {
        "ex_iterable_idx": 9,
        "previous_states": [child_state(27_000) for _ in range(streams)],
        "is_exhausted": [False] * streams,
        "type": "CyclingMultiSourcesExamplesIterable",
    }
    previous = {
        "ex_iterable_idx": 0,
        "previous_states": [child_state(413_000) for _ in range(streams)],
        "is_exhausted": [False] * streams,
        "type": "CyclingMultiSourcesExamplesIterable",
    }
    return {
        "examples_iterable": {
            "examples_iterable": current,
            "previous_state": previous,
            "batch_idx": 4_139_989,
            "num_chunks_since_previous_state": 2,
            "cropped_chunk_length": 0,
            "type": "RebatchedArrowExamplesIterable",
        },
        "epoch": 0,
    }


def texts(stream):
    return [row["text"] for row in stream]


def test_real_datasets_5_cursor_detection_without_ex_iterables_key():
    cursor = user_cursor_shape()
    assert is_legacy_multistream_cursor(cursor)
    plan = extract_legacy_resume_plan(cursor)
    assert plan is not None
    assert plan.stream_count == 10
    assert plan.pending_chunks == 2
    assert plan.cycling_state["previous_states"][0]["previous_state"]["shard_example_idx"] == 413_000
    assert plan.source == "$.examples_iterable.previous_state"


def test_outer_rebatched_previous_state_is_authoritative():
    plan = extract_legacy_resume_plan(exact_outer_rebatched_cursor())
    assert plan is not None
    assert plan.cycling_state["previous_states"] == [{"position": 1}, {"position": 1}]
    assert plan.pending_chunks == 2


def test_sequentializes_direct_legacy_cursor_without_duplicates_or_skips():
    stream = SequentializedHfDataset(FakeLegacyDataset(), legacy_cursor(), legacy_policy="exact")
    assert texts(stream) == ["a1", "a2", "b1", "b2"]
    cursor = stream.state_dict()
    assert cursor["_asterlm_cursor"] == ASTERLM_SEQUENTIAL_CURSOR
    assert cursor["exhausted"] == [True, True]


def test_outer_pending_chunks_are_discarded_before_sequential_drain():
    stream = SequentializedHfDataset(FakeLegacyDataset(), exact_outer_rebatched_cursor(), legacy_policy="exact")
    # Old round-robin continuation would yield a1,b1 first, but those two rows
    # were already emitted by the outer wrapper and must be skipped.
    assert texts(stream) == ["a2", "b2"]


def test_sequential_cursor_round_trip_after_exact_migration():
    first = SequentializedHfDataset(FakeLegacyDataset(), exact_outer_rebatched_cursor(), legacy_policy="exact")
    iterator = iter(first)
    assert next(iterator)["text"] == "a2"
    saved = first.state_dict()
    assert saved["pending_cycle_skips"] == 0

    resumed = SequentializedHfDataset(FakeLegacyDataset(), saved, legacy_policy="exact")
    assert texts(resumed) == ["b2"]


def test_new_stream_is_resharded_and_uses_one_input_shard():
    created = []

    def factory():
        dataset = FakeBoundedDataset()
        created.append(dataset)
        return dataset

    dataset, layout = open_resumable_hf_stream(
        factory,
        seed=1701,
        cursor=None,
        fallback_skip=0,
        layout=None,
    )
    assert layout == BOUNDED_RESHARD_LAYOUT
    assert dataset.resharded
    assert dataset.shuffle_args == {
        "seed": 1701,
        "buffer_size": 1,
        "max_buffer_input_shards": 1,
    }


def test_existing_multistream_cursor_can_use_exact_sequential_migration(monkeypatch):
    monkeypatch.setenv("ASTERLM_LEGACY_RESUME_POLICY", "exact")
    stream, layout = open_resumable_hf_stream(
        FakeLegacyDataset,
        seed=None,
        cursor=exact_outer_rebatched_cursor(),
        fallback_skip=0,
        layout=None,
    )
    assert layout == LEGACY_SEQUENTIAL_LAYOUT
    assert texts(stream) == ["a2", "b2"]


class FakeShardChild:
    def __init__(self, shards):
        self.shards = [[dict(row) for row in shard] for shard in shards]
        self._state_dict = None

    @property
    def num_shards(self):
        return len(self.shards)

    def _init_state_dict(self):
        self._state_dict = {
            "examples_iterable": {
                "shard_idx": 0,
                "shard_example_idx": 0,
                "type": "ArrowExamplesIterable",
            },
            "previous_state": None,
            "batch_idx": 0,
            "num_chunks_since_previous_state": 0,
            "cropped_chunk_length": 0,
            "type": "RebatchedArrowExamplesIterable",
        }
        return self._state_dict

    def load_state_dict(self, state):
        self._state_dict = deepcopy(state)

    def state_dict(self):
        return deepcopy(self._state_dict)

    def iter_arrow(self):
        base = self._state_dict.get("previous_state") or self._state_dict["examples_iterable"]
        shard_idx = int(base["shard_idx"])
        row_idx = int(base.get("shard_example_idx", 0))
        while shard_idx < len(self.shards):
            shard = self.shards[shard_idx]
            while row_idx < len(shard):
                row = shard[row_idx]
                row_idx += 1
                self._state_dict["examples_iterable"] = {
                    "shard_idx": shard_idx,
                    "shard_example_idx": row_idx,
                    "type": "ArrowExamplesIterable",
                }
                yield f"{shard_idx}:{row_idx - 1}", FakeTable(row)
            shard_idx += 1
            row_idx = 0
            self._state_dict["examples_iterable"] = {
                "shard_idx": shard_idx,
                "shard_example_idx": 0,
                "type": "ArrowExamplesIterable",
            }


def test_default_legacy_migration_skips_only_partial_files_without_remote_replay():
    children = [
        FakeShardChild(
            [
                [{"text": f"old-{i}"}],
                [{"text": f"new-{i}-a"}, {"text": f"new-{i}-b"}],
                [{"text": f"later-{i}"}],
            ]
        )
        for i in range(10)
    ]
    stream = SequentializedHfDataset(
        FakeLegacyDataset(children),
        user_cursor_shape(),
        legacy_policy="next-shard",
    )
    rows = texts(stream)
    assert not any(value.startswith("old-") for value in rows)
    assert rows[:3] == ["new-0-a", "new-0-b", "later-0"]
    saved = stream.state_dict()
    assert saved["migration_policy"] == "next-shard"
    assert len(saved["skipped_partial_shards"]) == 10
    assert saved["skipped_partial_shards"][0]["rows_already_committed_or_prefetched"] == 414_000
    assert saved["skipped_partial_shards"][1]["rows_already_committed_or_prefetched"] == 414_000
    assert saved["skipped_partial_shards"][2]["rows_already_committed_or_prefetched"] == 413_999


def test_open_resumable_defaults_to_next_shard_policy(monkeypatch):
    monkeypatch.delenv("ASTERLM_LEGACY_RESUME_POLICY", raising=False)
    children = [FakeShardChild([[{"text": "old"}], [{"text": f"new-{i}"}]]) for i in range(10)]

    def factory():
        return FakeLegacyDataset(children)

    stream, layout = open_resumable_hf_stream(
        factory,
        seed=None,
        cursor=user_cursor_shape(),
        fallback_skip=0,
        layout=None,
    )
    assert layout == LEGACY_NEXT_SHARD_LAYOUT
    assert texts(stream)[0] == "new-0"


def sequential_v3_cursor(*, rows=413_000, pending=0, current_child=0):
    states = []
    for _ in range(10):
        states.append(
            {
                "examples_iterable": {
                    "shard_idx": 0,
                    "shard_example_idx": rows,
                    "type": "ArrowExamplesIterable",
                },
                "previous_state": {
                    "shard_idx": 0,
                    "shard_example_idx": rows,
                    "type": "ArrowExamplesIterable",
                },
                "batch_idx": rows,
                "num_chunks_since_previous_state": 0,
                "cropped_chunk_length": 0,
                "type": "RebatchedArrowExamplesIterable",
            }
        )
    return {
        "_asterlm_cursor": ASTERLM_SEQUENTIAL_CURSOR,
        "current_child": current_child,
        "exhausted": [False] * 10,
        "child_states": states,
        "legacy_stream_count": 10,
        "pending_cycle_skips": pending,
        "cycle_index": 0,
    }


def test_v3_v4_sequential_cursor_upgrades_offline_to_next_shard():
    converted = convert_sequential_cursor_to_next_shard(sequential_v3_cursor(pending=2))
    assert converted["migration_policy"] == "next-shard"
    assert converted["pending_cycle_skips"] == 0
    assert len(converted["skipped_partial_shards"]) == 10
    for state in converted["child_states"]:
        base = state["examples_iterable"]
        assert base["shard_idx"] == 1
        assert base["shard_example_idx"] == 0
        assert state["previous_state"] is None
    assert converted["skipped_partial_shards"][0]["rows_already_committed_or_prefetched"] == 413_001
    assert converted["skipped_partial_shards"][1]["rows_already_committed_or_prefetched"] == 413_001
    assert converted["skipped_partial_shards"][2]["rows_already_committed_or_prefetched"] == 413_000


def test_sequential_upgrade_does_not_skip_an_untouched_boundary_shard():
    cursor = sequential_v3_cursor(rows=0)
    converted = convert_sequential_cursor_to_next_shard(cursor)
    assert converted["migration_policy"] == "next-shard"
    assert converted["skipped_partial_shards"] == []
    for state in converted["child_states"]:
        assert state["examples_iterable"]["shard_idx"] == 0
        assert state["examples_iterable"]["shard_example_idx"] == 0


def test_sequential_upgrade_preserves_completed_children():
    cursor = sequential_v3_cursor(rows=413_000, current_child=2)
    cursor["exhausted"][0] = True
    cursor["exhausted"][1] = True
    converted = convert_sequential_cursor_to_next_shard(cursor)
    assert converted["current_child"] == 2
    assert len(converted["skipped_partial_shards"]) == 8
    assert converted["child_states"][0] == cursor["child_states"][0]
    assert converted["child_states"][1] == cursor["child_states"][1]
    assert converted["child_states"][2]["examples_iterable"]["shard_idx"] == 1


def test_next_shard_policy_is_reapplied_on_every_sequential_restart():
    # Simulate a cursor that was already migrated, then made progress partway
    # through the next remote file before checkpointing again.
    cursor = sequential_v3_cursor(rows=123_456)
    cursor["migration_policy"] = "next-shard"
    cursor["skipped_partial_shards"] = [{"stream_index": 0, "partial_shard_index": 0, "next_shard_index": 1}]

    children = [
        FakeShardChild(
            [
                [{"text": f"old0-{i}"}],
                [{"text": f"partial1-{i}"}],
                [{"text": f"fresh2-{i}"}],
            ]
        )
        for i in range(10)
    ]
    # Make the stored file index 1 to represent progress after the first migration.
    for state in cursor["child_states"]:
        state["examples_iterable"]["shard_idx"] = 1
        state["previous_state"]["shard_idx"] = 1

    stream = SequentializedHfDataset(FakeLegacyDataset(children), cursor, legacy_policy="next-shard")
    rows = texts(stream)
    assert rows[0] == "fresh2-0"
    saved = stream.state_dict()
    latest = saved["last_resume_skipped_partial_shards"]
    assert len(latest) == 10
    assert latest[0]["partial_shard_index"] == 1
    assert latest[0]["next_shard_index"] == 2
    assert len(saved["skipped_partial_shards"]) == 11


def test_next_shard_restart_does_not_skip_clean_boundary_again():
    cursor = sequential_v3_cursor(rows=0)
    cursor["migration_policy"] = "next-shard"
    for state in cursor["child_states"]:
        state["examples_iterable"]["shard_idx"] = 1
        state["previous_state"]["shard_idx"] = 1
    children = [FakeShardChild([[{"text": "old"}], [{"text": f"fresh-{i}"}]]) for i in range(10)]
    stream = SequentializedHfDataset(FakeLegacyDataset(children), cursor, legacy_policy="next-shard")
    assert texts(stream)[0] == "fresh-0"
    assert stream.state_dict()["last_resume_skipped_partial_shards"] == []


def test_parallel_prefetch_checkpoint_ignores_unemitted_lookahead(monkeypatch):
    monkeypatch.setenv("ASTERLM_HF_PARALLEL_STREAMS", "2")
    cursor = {
        "examples_iterable": cycling_state(positions=(0, 0), index=0),
        "epoch": 0,
    }
    stream = SequentializedHfDataset(FakeLegacyDataset(), cursor, legacy_policy="exact")
    iterator = iter(stream)

    assert next(iterator)["text"] == "a0"

    # Give both background readers time to fetch speculative lookahead. The live
    # child states can now be ahead of what AsterLM has actually emitted.
    time.sleep(0.05)
    saved = stream.state_dict()

    assert saved["parallel_streams"] == 2
    # a0 crossed the iterator boundary, but b0 and a1 must remain uncommitted
    # even if their futures have already completed.
    assert saved["child_states"][0]["position"] == 1
    assert saved["child_states"][1]["position"] == 0

    iterator.close()

    monkeypatch.setenv("ASTERLM_HF_PARALLEL_STREAMS", "1")
    resumed = SequentializedHfDataset(FakeLegacyDataset(), saved, legacy_policy="exact")
    assert texts(resumed) == ["a1", "a2", "b0", "b1", "b2"]


def test_parallel_prefetch_round_robins_bounded_children(monkeypatch):
    monkeypatch.setenv("ASTERLM_HF_PARALLEL_STREAMS", "2")
    cursor = {
        "examples_iterable": cycling_state(positions=(0, 0), index=0),
        "epoch": 0,
    }
    stream = SequentializedHfDataset(FakeLegacyDataset(), cursor, legacy_policy="exact")
    assert texts(stream) == ["a0", "b0", "a1", "b1", "a2", "b2"]
    saved = stream.state_dict()
    assert saved["exhausted"] == [True, True]
    assert saved["parallel_streams"] == 2
