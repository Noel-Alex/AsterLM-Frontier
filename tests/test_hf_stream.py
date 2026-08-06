from __future__ import annotations

from copy import deepcopy

from asterlm.data.hf_stream import (
    ASTERLM_SEQUENTIAL_CURSOR,
    BOUNDED_RESHARD_LAYOUT,
    LEGACY_SEQUENTIAL_LAYOUT,
    SequentializedHfDataset,
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
    stream = SequentializedHfDataset(FakeLegacyDataset(), legacy_cursor())
    assert texts(stream) == ["a1", "a2", "b1", "b2"]
    cursor = stream.state_dict()
    assert cursor["_asterlm_cursor"] == ASTERLM_SEQUENTIAL_CURSOR
    assert cursor["exhausted"] == [True, True]


def test_outer_pending_chunks_are_discarded_before_sequential_drain():
    stream = SequentializedHfDataset(FakeLegacyDataset(), exact_outer_rebatched_cursor())
    # Old round-robin continuation would yield a1,b1 first, but those two rows
    # were already emitted by the outer wrapper and must be skipped.
    assert texts(stream) == ["a2", "b2"]


def test_sequential_cursor_round_trip_after_exact_migration():
    first = SequentializedHfDataset(FakeLegacyDataset(), exact_outer_rebatched_cursor())
    iterator = iter(first)
    assert next(iterator)["text"] == "a2"
    saved = first.state_dict()
    assert saved["pending_cycle_skips"] == 0

    resumed = SequentializedHfDataset(FakeLegacyDataset(), saved)
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


def test_existing_multistream_cursor_uses_sequential_migration():
    stream, layout = open_resumable_hf_stream(
        FakeLegacyDataset,
        seed=None,
        cursor=exact_outer_rebatched_cursor(),
        fallback_skip=0,
        layout=None,
    )
    assert layout == LEGACY_SEQUENTIAL_LAYOUT
    assert texts(stream) == ["a2", "b2"]
