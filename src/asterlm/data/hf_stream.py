from __future__ import annotations

import gc
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASTERLM_SEQUENTIAL_CURSOR = "asterlm-hf-sequential-v1"
BOUNDED_RESHARD_LAYOUT = "bounded-reshard-max1-v1"
LEGACY_SINGLE_LAYOUT = "legacy-single-max1-v1"
LEGACY_SEQUENTIAL_LAYOUT = "legacy-multistream-sequential-v1"


@dataclass(frozen=True, slots=True)
class LegacyResumePlan:
    """Authoritative resume point extracted from a Hugging Face 5.x cursor."""

    cycling_state: dict[str, Any]
    pending_chunks: int
    cropped_chunk_length: int
    stream_count: int
    source: str


def _is_cycling_state(value: Any) -> bool:
    """Recognize datasets 5.x CyclingMultiSources state dictionaries.

    Hugging Face does not serialize the child iterable objects in this state.
    The durable shape contains only ``ex_iterable_idx``, ``previous_states``,
    ``is_exhausted`` and ``type``. Older AsterLM tests incorrectly required an
    ``ex_iterables`` key that real datasets==5.0.1 cursors never contain.
    """

    if not isinstance(value, dict):
        return False
    previous = value.get("previous_states")
    exhausted = value.get("is_exhausted")
    if not isinstance(previous, list) or not isinstance(exhausted, list):
        return False
    if len(previous) <= 1 or len(previous) != len(exhausted):
        return False
    if not isinstance(value.get("ex_iterable_idx"), int):
        return False
    kind = value.get("type")
    return kind == "CyclingMultiSourcesExamplesIterable" or (
        kind is None and all(item is None or isinstance(item, dict) for item in previous)
    )


def extract_legacy_resume_plan(cursor: Any) -> LegacyResumePlan | None:
    """Extract the exact legacy multi-stream resume point.

    A shuffled Arrow stream has an outer ``RebatchedArrowExamplesIterable``.
    On restore, Hugging Face loads that wrapper's ``previous_state`` into the
    underlying cycling iterable and then skips
    ``num_chunks_since_previous_state`` chunks. Therefore the nested current
    ``examples_iterable`` snapshot is *not* authoritative. Selecting it can
    replay millions of rows. This function mirrors those wrapper semantics.
    """

    seen: set[int] = set()

    def visit(value: Any, path: str) -> LegacyResumePlan | None:
        if isinstance(value, (dict, list, tuple)):
            marker = id(value)
            if marker in seen:
                return None
            seen.add(marker)

        if isinstance(value, dict):
            if value.get("type") == "RebatchedArrowExamplesIterable":
                previous = value.get("previous_state")
                current = value.get("examples_iterable")
                if _is_cycling_state(previous):
                    state = previous
                    return LegacyResumePlan(
                        cycling_state=state,
                        pending_chunks=max(0, int(value.get("num_chunks_since_previous_state", 0))),
                        cropped_chunk_length=max(0, int(value.get("cropped_chunk_length", 0))),
                        stream_count=len(state["previous_states"]),
                        source=f"{path}.previous_state",
                    )
                if previous is None and _is_cycling_state(current):
                    state = current
                    return LegacyResumePlan(
                        cycling_state=state,
                        pending_chunks=0,
                        cropped_chunk_length=0,
                        stream_count=len(state["previous_states"]),
                        source=f"{path}.examples_iterable",
                    )

            if _is_cycling_state(value):
                return LegacyResumePlan(
                    cycling_state=value,
                    pending_chunks=0,
                    cropped_chunk_length=0,
                    stream_count=len(value["previous_states"]),
                    source=path,
                )

            # Traverse wrapper paths before arbitrary metadata. The first outer
            # Rebatched wrapper is the authoritative one.
            preferred = ("examples_iterable", "previous_state", "state", "dataset_state")
            for key in preferred:
                if key in value:
                    found = visit(value[key], f"{path}.{key}")
                    if found is not None:
                        return found
            for key, child in value.items():
                if key not in preferred:
                    found = visit(child, f"{path}.{key}")
                    if found is not None:
                        return found
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found = visit(child, f"{path}[{index}]")
                if found is not None:
                    return found
        return None

    return visit(cursor, "$")


def find_legacy_multistream_state(cursor: Any) -> dict[str, Any] | None:
    plan = extract_legacy_resume_plan(cursor)
    return plan.cycling_state if plan is not None else None


def is_sequential_cursor(cursor: Any) -> bool:
    return isinstance(cursor, dict) and cursor.get("_asterlm_cursor") == ASTERLM_SEQUENTIAL_CURSOR


def is_legacy_multistream_cursor(cursor: Any) -> bool:
    return find_legacy_multistream_state(cursor) is not None


def _walk_iterable_objects(root: Any) -> Iterator[Any]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        child = getattr(value, "ex_iterable", None)
        if child is not None:
            stack.append(child)
        children = getattr(value, "ex_iterables", None)
        if isinstance(children, list):
            stack.extend(reversed(children))


def _find_cycling_iterable(root: Any) -> Any | None:
    for value in _walk_iterable_objects(root):
        children = getattr(value, "ex_iterables", None)
        state = getattr(value, "_state_dict", None)
        if (
            isinstance(children, list)
            and children
            and isinstance(state, dict)
            and isinstance(state.get("previous_states"), list)
            and isinstance(state.get("is_exhausted"), list)
        ):
            return value
    return None


def current_rss_gib() -> float | None:
    """Return current process RSS without requiring psutil."""

    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        pass
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return None


def total_ram_gib() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
        return values["MemTotal"] * 1024 / 2**30
    except Exception:
        return None


def available_ram_gib() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
        return values["MemAvailable"] * 1024 / 2**30
    except Exception:
        return None


def default_max_rss_gib() -> float | None:
    total = total_ram_gib()
    if total is None:
        return None
    # On a 32 GiB laptop this resolves to about 20 GiB. Leave enough room for
    # Fedora, the desktop/browser, filesystem cache and the parent downloader.
    return max(2.0, min(total * 0.65, total - 8.0))


def default_min_available_gib() -> float | None:
    total = total_ram_gib()
    if total is None:
        return None
    return max(2.0, min(6.0, total * 0.15))


def release_arrow_memory() -> None:
    """Best-effort release after closing a failed/replaced HF iterator."""

    gc.collect()
    try:
        import pyarrow as pa

        release = getattr(pa.default_memory_pool(), "release_unused", None)
        if callable(release):
            release()
    except Exception:
        pass


@dataclass(slots=True)
class MemoryGuard:
    max_rss_gib: float | None = None
    min_available_gib: float | None = None
    check_every_records: int = 100

    def __post_init__(self) -> None:
        if self.max_rss_gib is None:
            self.max_rss_gib = default_max_rss_gib()
        if self.min_available_gib is None:
            self.min_available_gib = default_min_available_gib()

    def sample(self, records: int) -> tuple[float | None, float | None, str | None]:
        if self.check_every_records <= 0 or records % self.check_every_records:
            return None, None, None
        rss = current_rss_gib()
        available = available_ram_gib()
        if rss is not None and self.max_rss_gib is not None and rss >= self.max_rss_gib:
            return rss, available, "process_rss"
        if available is not None and self.min_available_gib is not None and available <= self.min_available_gib:
            return rss, available, "system_available"
        return rss, available, None


class MemoryPressureError(RuntimeError):
    pass


def memory_status(rss: float | None, available: float | None) -> str:
    parts = []
    if rss is not None:
        parts.append(f"rss={rss:.1f}GiB")
    if available is not None:
        parts.append(f"available={available:.1f}GiB")
    return " ".join(parts) if parts else "memory=unknown"


class SequentializedHfDataset:
    """Drain a legacy multi-stream Hugging Face cursor one child at a time.

    Hugging Face 5.x creates one prefetch thread per interleaved input shard.
    This adapter never iterates the CyclingMultiSources object. It restores the
    child cursors directly, applies the outer Rebatched wrapper's small pending
    skip exactly, and then drains one child only. This removes ten-way remote
    prefetch while preserving every uncommitted source record.
    """

    def __init__(self, legacy_dataset: Any, cursor: dict[str, Any]) -> None:
        self._dataset = legacy_dataset
        self._children: list[Any]
        self._exhausted: list[bool]
        self._current_child: int
        self._pending_cycle_skips = 0
        self._cycle_index = 0

        top = getattr(legacy_dataset, "_ex_iterable", None)
        if top is None:
            raise RuntimeError("Hugging Face iterable internals are unavailable for cursor migration")
        init = getattr(top, "_init_state_dict", None)
        if not callable(init):
            raise RuntimeError("Hugging Face iterable does not expose checkpoint initialization")
        init()

        cycling = _find_cycling_iterable(top)
        if cycling is None:
            raise RuntimeError(
                "Legacy cursor says multiple streams were active, but the recreated dataset "
                "does not expose the expected CyclingMultiSources layout. Keep datasets==5.0.1."
            )
        children = getattr(cycling, "ex_iterables", None)
        if not isinstance(children, list) or not children:
            raise RuntimeError("Legacy Hugging Face cycling stream has no child streams")
        self._children = children

        if is_sequential_cursor(cursor):
            states = cursor.get("child_states")
            exhausted = cursor.get("exhausted")
            if not isinstance(states, list) or len(states) != len(children):
                raise RuntimeError("Sequential Hugging Face cursor has an incompatible child count")
            if not isinstance(exhausted, list) or len(exhausted) != len(children):
                raise RuntimeError("Sequential Hugging Face cursor has invalid exhaustion state")
            for child, state in zip(children, states):
                load = getattr(child, "load_state_dict", None)
                if not callable(load):
                    raise RuntimeError("Hugging Face child stream cannot restore its cursor")
                load(state)
            self._exhausted = [bool(value) for value in exhausted]
            self._current_child = int(cursor.get("current_child", 0))
            self._pending_cycle_skips = max(0, int(cursor.get("pending_cycle_skips", 0)))
            self._cycle_index = int(cursor.get("cycle_index", 0)) % len(children)
            return

        plan = extract_legacy_resume_plan(cursor)
        if plan is None:
            raise RuntimeError("Legacy Hugging Face multi-stream cursor structure was not found")
        if plan.stream_count != len(children):
            raise RuntimeError(
                f"Legacy cursor contains {plan.stream_count} streams but the recreated dataset "
                f"contains {len(children)}. Keep the pinned dataset revision and datasets==5.0.1."
            )
        if plan.cropped_chunk_length:
            raise RuntimeError(
                "The legacy outer Rebatched cursor stopped inside a multi-row Arrow chunk. "
                "Automatic migration refuses to guess; preserve this cursor for manual recovery."
            )

        cycling_state = plan.cycling_state
        previous_states = cycling_state.get("previous_states")
        exhausted = cycling_state.get("is_exhausted")
        if not isinstance(previous_states, list) or len(previous_states) != len(children):
            raise RuntimeError("Legacy Hugging Face cursor has invalid child lookahead state")
        if not isinstance(exhausted, list) or len(exhausted) != len(children):
            raise RuntimeError("Legacy Hugging Face cursor has invalid exhaustion state")

        # CyclingMultiSources reads ahead from every child. Its durable
        # ``previous_states`` entries are the exact states from which the next
        # record of each child must be produced. Do not load the nested current
        # child snapshots: those are already ahead of the committed output.
        for child, previous in zip(children, previous_states):
            if previous is None:
                continue
            load = getattr(child, "load_state_dict", None)
            if not callable(load):
                raise RuntimeError("Hugging Face child stream cannot restore its cursor")
            load(previous)

        self._exhausted = [bool(value) for value in exhausted]
        self._current_child = 0
        self._pending_cycle_skips = plan.pending_chunks
        self._cycle_index = int(cycling_state.get("ex_iterable_idx", 0)) % len(children)

    @property
    def resume_mode(self) -> str:
        return LEGACY_SEQUENTIAL_LAYOUT

    @staticmethod
    def _row_from_arrow_item(item: Any) -> dict[str, Any]:
        if not (isinstance(item, tuple) and len(item) == 2):
            if isinstance(item, dict):
                return item
            raise RuntimeError(f"Unexpected Hugging Face Arrow item type: {type(item).__name__}")
        table = item[1]
        to_pylist = getattr(table, "to_pylist", None)
        if not callable(to_pylist):
            if isinstance(table, dict):
                return table
            raise RuntimeError("Legacy Hugging Face child did not yield an Arrow table")
        rows = to_pylist()
        if len(rows) != 1:
            raise RuntimeError(
                "Legacy migration expected one-row Rebatched Arrow chunks but received "
                f"{len(rows)} rows. Refusing a checkpoint-unsafe conversion."
            )
        row = rows[0]
        if not isinstance(row, dict):
            raise RuntimeError("Legacy Hugging Face Arrow row is not a mapping")
        return row

    def _iter_child_rows(self, index: int) -> Iterator[dict[str, Any]]:
        child = self._children[index]
        arrow_method = getattr(child, "iter_arrow", None)
        if callable(arrow_method):
            iterator = arrow_method()
            try:
                for item in iterator:
                    yield self._row_from_arrow_item(item)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            return

        iterator = iter(child)
        try:
            for item in iterator:
                if isinstance(item, tuple) and len(item) == 2:
                    item = item[1]
                if not isinstance(item, dict):
                    raise RuntimeError("Legacy Hugging Face child record is not a mapping")
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def _consume_one(self, index: int) -> bool:
        iterator = self._iter_child_rows(index)
        try:
            next(iterator)
            return True
        except StopIteration:
            self._exhausted[index] = True
            return False
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def _apply_outer_pending_skips(self) -> None:
        """Mirror the outer Rebatched wrapper before changing output order."""

        stream_count = len(self._children)
        while self._pending_cycle_skips > 0:
            attempts = 0
            consumed = False
            while attempts < stream_count:
                index = self._cycle_index
                self._cycle_index = (self._cycle_index + 1) % stream_count
                attempts += 1
                if self._exhausted[index]:
                    continue
                if self._consume_one(index):
                    self._pending_cycle_skips -= 1
                    consumed = True
                    break
            if not consumed:
                raise RuntimeError(
                    "Legacy cursor asks to skip additional outer chunks, but every child stream is exhausted"
                )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._apply_outer_pending_skips()
        while self._current_child < len(self._children):
            index = self._current_child
            if self._exhausted[index]:
                self._current_child += 1
                continue
            yielded = False
            for row in self._iter_child_rows(index):
                yielded = True
                yield row
            self._exhausted[index] = True
            self._current_child += 1
            if not yielded:
                release_arrow_memory()

    def state_dict(self) -> dict[str, Any]:
        states: list[Any] = []
        for child in self._children:
            method = getattr(child, "state_dict", None)
            if not callable(method):
                raise RuntimeError("Hugging Face child stream cannot expose a cursor")
            states.append(method())
        return {
            "_asterlm_cursor": ASTERLM_SEQUENTIAL_CURSOR,
            "current_child": self._current_child,
            "exhausted": list(self._exhausted),
            "child_states": states,
            "legacy_stream_count": len(self._children),
            "pending_cycle_skips": self._pending_cycle_skips,
            "cycle_index": self._cycle_index,
        }

def bounded_shuffle(dataset: Any, *, seed: int | None, reshard: bool) -> Any:
    """Create a deterministic stream with one active remote input shard.

    ``datasets==5.0.1`` defaults to ten input shards in ``IterableDataset.shuffle``.
    Passing ``max_buffer_input_shards=1`` is the critical backpressure control.
    New streams are also resharded by Parquet row group when supported, reducing
    restart replay and the maximum decoded Arrow table retained at once.
    """

    if reshard:
        method = getattr(dataset, "reshard", None)
        if callable(method):
            dataset = method()
    if seed is None:
        return dataset
    try:
        return dataset.shuffle(seed=seed, buffer_size=1, max_buffer_input_shards=1)
    except TypeError:
        # datasets<5 did not interleave ten shards by default.
        return dataset.shuffle(seed=seed, buffer_size=1)


def legacy_shuffle(dataset: Any, *, seed: int | None) -> Any:
    if seed is None:
        return dataset
    try:
        return dataset.shuffle(seed=seed, buffer_size=1, max_buffer_input_shards=10)
    except TypeError:
        return dataset.shuffle(seed=seed, buffer_size=1)


def open_resumable_hf_stream(
    base_factory: Callable[[], Any],
    *,
    seed: int | None,
    cursor: Any | None,
    fallback_skip: int,
    layout: str | None,
) -> tuple[Any, str]:
    """Open a bounded stream while preserving old multi-stream checkpoints."""

    if is_sequential_cursor(cursor) or is_legacy_multistream_cursor(cursor):
        legacy = legacy_shuffle(base_factory(), seed=seed)
        return SequentializedHfDataset(legacy, cursor), LEGACY_SEQUENTIAL_LAYOUT

    if cursor is None and fallback_skip:
        raise RuntimeError(
            "This legacy checkpoint has consumed records but no dataset cursor. Replaying it with a "
            "different bounded layout could duplicate or omit data. Restore the referenced cursor file "
            "or move this source directory aside and restart only this source."
        )

    use_reshard = layout == BOUNDED_RESHARD_LAYOUT or (cursor is None and not fallback_skip)
    dataset = bounded_shuffle(base_factory(), seed=seed, reshard=use_reshard)
    if cursor is not None:
        load = getattr(dataset, "load_state_dict", None)
        if callable(load):
            load(cursor)
        else:
            raise RuntimeError("Hugging Face iterable cannot restore its saved cursor")
    return dataset, BOUNDED_RESHARD_LAYOUT if use_reshard else LEGACY_SINGLE_LAYOUT
