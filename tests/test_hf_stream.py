from __future__ import annotations

from copy import deepcopy

from asterlm.data.hf_stream import (
    ASTERLM_SEQUENTIAL_CURSOR,
    BOUNDED_RESHARD_LAYOUT,
    LEGACY_SEQUENTIAL_LAYOUT,
    SequentializedHfDataset,
    is_legacy_multistream_cursor,
    open_resumable_hf_stream,
)


class FakeChild:
    def __init__(self, values):
        self.values = list(values)
        self._state_dict = {"position": 0}

    def _init_state_dict(self):
        self._state_dict = {"position": 0}
        return self._state_dict

    def load_state_dict(self, state):
        self._state_dict.clear()
        self._state_dict.update(deepcopy(state))

    def state_dict(self):
        return deepcopy(self._state_dict)

    def __iter__(self):
        while self._state_dict["position"] < len(self.values):
            index = self._state_dict["position"]
            self._state_dict["position"] += 1
            yield str(index), self.values[index]


class FakeCycling:
    def __init__(self, children):
        self.ex_iterables = children
        self._state_dict = None

    def _init_state_dict(self):
        self._state_dict = {
            "ex_iterable_idx": 0,
            "ex_iterables": [child._init_state_dict() for child in self.ex_iterables],
            "previous_states": [None] * len(self.ex_iterables),
            "is_exhausted": [False] * len(self.ex_iterables),
        }
        return self._state_dict


class FakeBuffer:
    def __init__(self, cycling):
        self.ex_iterable = cycling
        self._state_dict = None

    def _init_state_dict(self):
        self._state_dict = self.ex_iterable._init_state_dict()
        return self._state_dict

    def load_state_dict(self, state):
        self._state_dict.clear()
        self._state_dict.update(deepcopy(state))
        for child, child_state in zip(self.ex_iterable.ex_iterables, self._state_dict["ex_iterables"]):
            child.load_state_dict(child_state)
        self.ex_iterable._state_dict = self._state_dict


class FakeLegacyDataset:
    def __init__(self):
        self._ex_iterable = FakeBuffer(FakeCycling([FakeChild(["a0", "a1", "a2"]), FakeChild(["b0", "b1"])]))


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


def legacy_cursor():
    return {
        "examples_iterable": {
            "ex_iterable_idx": 1,
            "ex_iterables": [{"position": 2}, {"position": 1}],
            # Cycling has looked one item ahead from child 0. Rewind to position 1.
            "previous_states": [{"position": 1}, None],
            "is_exhausted": [False, False],
        },
        "epoch": 0,
    }


def test_legacy_cursor_detection():
    assert is_legacy_multistream_cursor(legacy_cursor())


def test_sequentializes_legacy_cursor_without_duplicates_or_skips():
    stream = SequentializedHfDataset(FakeLegacyDataset(), legacy_cursor())
    assert list(stream) == ["a1", "a2", "b1"]
    cursor = stream.state_dict()
    assert cursor["_asterlm_cursor"] == ASTERLM_SEQUENTIAL_CURSOR
    assert cursor["exhausted"] == [True, True]


def test_sequential_cursor_round_trip():
    first = SequentializedHfDataset(FakeLegacyDataset(), legacy_cursor())
    iterator = iter(first)
    assert next(iterator) == "a1"
    saved = first.state_dict()

    resumed = SequentializedHfDataset(FakeLegacyDataset(), saved)
    assert list(resumed) == ["a2", "b1"]


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
        cursor=legacy_cursor(),
        fallback_skip=0,
        layout=None,
    )
    assert layout == LEGACY_SEQUENTIAL_LAYOUT
    assert list(stream) == ["a1", "a2", "b1"]
