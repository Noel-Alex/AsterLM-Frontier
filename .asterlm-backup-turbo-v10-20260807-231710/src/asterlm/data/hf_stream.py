from __future__ import annotations

import concurrent.futures
import gc
import os
from collections import deque
from copy import deepcopy
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASTERLM_SEQUENTIAL_CURSOR = "asterlm-hf-sequential-v1"
BOUNDED_RESHARD_LAYOUT = "bounded-reshard-max1-v1"
LEGACY_SINGLE_LAYOUT = "legacy-single-max1-v1"
LEGACY_SEQUENTIAL_LAYOUT = "legacy-multistream-sequential-v1"
LEGACY_NEXT_SHARD_LAYOUT = "legacy-multistream-next-shard-v1"
LEGACY_POLICY_EXACT = "exact"
LEGACY_POLICY_NEXT_SHARD = "next-shard"


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

    def __init__(
        self,
        legacy_dataset: Any,
        cursor: dict[str, Any],
        *,
        legacy_policy: str | None = None,
    ) -> None:
        self._dataset = legacy_dataset
        self._children: list[Any]
        self._exhausted: list[bool]
        self._current_child: int
        self._pending_cycle_skips = 0
        self._cycle_index = 0
        self._parallel_streams = max(1, int(os.getenv("ASTERLM_HF_PARALLEL_STREAMS", "1")))
        # In parallel mode child iterators may read ahead before their rows are
        # emitted. These snapshots track only rows that have actually crossed
        # the AsterLM iterator boundary, so state_dict() never checkpoints
        # speculative lookahead.
        self._safe_child_states: list[Any] = []
        requested_policy = (legacy_policy or os.getenv("ASTERLM_LEGACY_RESUME_POLICY", LEGACY_POLICY_NEXT_SHARD)).strip().lower()
        if requested_policy not in {LEGACY_POLICY_EXACT, LEGACY_POLICY_NEXT_SHARD}:
            raise RuntimeError(
                "ASTERLM_LEGACY_RESUME_POLICY must be next-shard or exact; "
                f"got {requested_policy!r}"
            )
        self._migration_policy = requested_policy
        self._skipped_partial_shards: list[dict[str, Any]] = []
        self._last_resume_skipped_partial_shards: list[dict[str, Any]] = []

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
            stored_policy = str(cursor.get("migration_policy", self._migration_policy)).strip().lower()
            if stored_policy in {LEGACY_POLICY_EXACT, LEGACY_POLICY_NEXT_SHARD}:
                self._migration_policy = stored_policy

            # A next-shard cursor is a *policy*, not a one-time conversion. Every
            # later checkpoint can again land partway through a large legacy
            # Parquet file. Reopening that exact row cursor would reread gigabytes
            # before yielding one new record. Refresh it locally on every restart
            # so the active child begins at the next untouched file instead.
            if self._migration_policy == LEGACY_POLICY_NEXT_SHARD:
                cursor = convert_sequential_cursor_to_next_shard(cursor)

            states = cursor.get("child_states")
            exhausted = cursor.get("exhausted")
            if not isinstance(states, list) or len(states) != len(children):
                raise RuntimeError("Sequential Hugging Face cursor has an incompatible child count")
            if not isinstance(exhausted, list) or len(exhausted) != len(children):
                raise RuntimeError("Sequential Hugging Face cursor has invalid exhaustion state")

            self._exhausted = [bool(value) for value in exhausted]
            for index, (child, state) in enumerate(zip(children, states)):
                if self._exhausted[index]:
                    continue
                # If the policy advanced past the last shard in a legacy child,
                # mark it exhausted rather than asking datasets to open it.
                try:
                    shard_idx, _ = _state_partial_rows(state)
                except RuntimeError:
                    shard_idx = -1
                total_shards = getattr(child, "num_shards", None)
                if isinstance(total_shards, int) and shard_idx >= total_shards:
                    self._exhausted[index] = True
                    continue
                load = getattr(child, "load_state_dict", None)
                if not callable(load):
                    raise RuntimeError("Hugging Face child stream cannot restore its cursor")
                load(state)
            self._current_child = int(cursor.get("current_child", 0))
            self._pending_cycle_skips = max(0, int(cursor.get("pending_cycle_skips", 0)))
            self._cycle_index = int(cursor.get("cycle_index", 0)) % len(children)
            skipped = cursor.get("skipped_partial_shards")
            if isinstance(skipped, list):
                self._skipped_partial_shards = deepcopy(skipped)
            latest = cursor.get("last_resume_skipped_partial_shards")
            self._last_resume_skipped_partial_shards = deepcopy(latest) if isinstance(latest, list) else []
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

        self._exhausted = [bool(value) for value in exhausted]

        if self._migration_policy == LEGACY_POLICY_NEXT_SHARD:
            self._initialize_next_shard_migration(previous_states, plan)
            return

        # Exact legacy recovery: CyclingMultiSources reads ahead from every
        # child. Its durable ``previous_states`` entries are the exact states
        # from which the next record of each child must be produced. This can
        # require rereading a large partial Parquet file because the HF cursor
        # stores row counts rather than a byte/row-group offset.
        for child, previous in zip(children, previous_states):
            if previous is None:
                continue
            load = getattr(child, "load_state_dict", None)
            if not callable(load):
                raise RuntimeError("Hugging Face child stream cannot restore its cursor")
            load(previous)

        self._current_child = 0
        self._pending_cycle_skips = plan.pending_chunks
        self._cycle_index = int(cycling_state.get("ex_iterable_idx", 0)) % len(children)

    @staticmethod
    def _pending_skips_per_child(plan: LegacyResumePlan, exhausted: list[bool]) -> list[int]:
        counts = [0] * len(exhausted)
        remaining = plan.pending_chunks
        index = int(plan.cycling_state.get("ex_iterable_idx", 0)) % len(exhausted)
        while remaining > 0:
            attempts = 0
            while attempts < len(exhausted) and exhausted[index]:
                index = (index + 1) % len(exhausted)
                attempts += 1
            if attempts >= len(exhausted):
                raise RuntimeError("Legacy cursor has pending chunks but every stream is exhausted")
            counts[index] += 1
            remaining -= 1
            index = (index + 1) % len(exhausted)
        return counts

    @staticmethod
    def _next_shard_state(state: dict[str, Any], outer_pending: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if state.get("type") == "RebatchedArrowExamplesIterable":
            base = state.get("previous_state")
            if not isinstance(base, dict):
                base = state.get("examples_iterable")
            if not isinstance(base, dict):
                raise RuntimeError("Legacy child cursor has no underlying Arrow state")
            shard_idx = int(base.get("shard_idx", 0))
            rows_before_wrapper = int(base.get("shard_example_idx", 0))
            wrapper_rows = int(state.get("num_chunks_since_previous_state", 0))
            next_base = deepcopy(base)
            next_base["shard_idx"] = shard_idx + 1
            next_base["shard_example_idx"] = 0
            converted = deepcopy(state)
            converted["examples_iterable"] = deepcopy(next_base)
            converted["previous_state"] = None
            converted["batch_idx"] = 0
            converted["num_chunks_since_previous_state"] = 0
            converted["cropped_chunk_length"] = 0
        else:
            shard_idx = int(state.get("shard_idx", 0))
            rows_before_wrapper = int(state.get("shard_example_idx", 0))
            wrapper_rows = 0
            converted = deepcopy(state)
            converted["shard_idx"] = shard_idx + 1
            converted["shard_example_idx"] = 0

        audit = {
            "partial_shard_index": shard_idx,
            "next_shard_index": shard_idx + 1,
            "rows_already_committed_or_prefetched": rows_before_wrapper + wrapper_rows + outer_pending,
            "outer_pending_rows_discarded_with_partial_shard": outer_pending,
        }
        return converted, audit

    def _initialize_next_shard_migration(
        self,
        previous_states: list[Any],
        plan: LegacyResumePlan,
    ) -> None:
        """Skip each partially consumed legacy file and continue at its next file."""

        converted_cursor = convert_legacy_cursor_to_next_shard(
            {
                "examples_iterable": {
                    "previous_state": plan.cycling_state,
                    "num_chunks_since_previous_state": plan.pending_chunks,
                    "cropped_chunk_length": plan.cropped_chunk_length,
                    "type": "RebatchedArrowExamplesIterable",
                },
                "epoch": 0,
            }
        )
        states = converted_cursor["child_states"]
        exhausted = converted_cursor["exhausted"]
        for index, (child, state) in enumerate(zip(self._children, states)):
            if exhausted[index]:
                continue
            total_shards = getattr(child, "num_shards", None)
            audit = converted_cursor["skipped_partial_shards"][index]
            if isinstance(total_shards, int):
                audit["child_total_shards"] = total_shards
                if int(audit["next_shard_index"]) >= total_shards:
                    exhausted[index] = True
                    audit["child_exhausted_after_skip"] = True
                    continue
            load = getattr(child, "load_state_dict", None)
            if not callable(load):
                raise RuntimeError("Hugging Face child stream cannot restore its cursor")
            load(state)

        self._exhausted = [bool(value) for value in exhausted]
        self._current_child = 0
        self._pending_cycle_skips = 0
        self._cycle_index = 0
        self._skipped_partial_shards = deepcopy(converted_cursor["skipped_partial_shards"])

    @property
    def resume_mode(self) -> str:
        if self._migration_policy == LEGACY_POLICY_NEXT_SHARD:
            return LEGACY_NEXT_SHARD_LAYOUT
        return LEGACY_SEQUENTIAL_LAYOUT

    @property
    def migration_audit(self) -> dict[str, Any] | None:
        if self._migration_policy != LEGACY_POLICY_NEXT_SHARD:
            return None
        return {
            "policy": self._migration_policy,
            "skipped_partial_shards": deepcopy(self._skipped_partial_shards),
            "last_resume_skipped_partial_shards": deepcopy(self._last_resume_skipped_partial_shards),
        }

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

    def _snapshot_live_child_states(self) -> list[Any]:
        states: list[Any] = []
        for child in self._children:
            method = getattr(child, "state_dict", None)
            if not callable(method):
                raise RuntimeError("Hugging Face child stream cannot expose a cursor")
            states.append(deepcopy(method()))
        return states

    def _advance_current_child(self) -> None:
        while self._current_child < len(self._children) and self._exhausted[self._current_child]:
            self._current_child += 1

    def _iter_parallel(self) -> Iterator[dict[str, Any]]:
        """Round-robin a bounded number of legacy child readers concurrently.

        Hugging Face's original datasets==5.0.1 shuffle opened ten children at
        once and was fast, but each child could retain a decoded Parquet row
        group. Here the active-reader count is explicit and configurable.

        Each worker returns the child's state *after* fetching one row. We only
        promote that state to ``_safe_child_states`` immediately before yielding
        that row. Futures for other children are therefore speculative lookahead
        and never leak into a committed checkpoint.
        """

        self._advance_current_child()
        candidates = [
            index
            for index in range(self._current_child, len(self._children))
            if not self._exhausted[index]
        ]
        if not candidates:
            return

        width = min(self._parallel_streams, len(candidates))
        waiting = deque(candidates)
        active = deque()
        iterators: dict[int, Iterator[dict[str, Any]]] = {}
        futures: dict[int, concurrent.futures.Future[tuple[bool, dict[str, Any] | None, Any]]] = {}

        if not self._safe_child_states:
            self._safe_child_states = self._snapshot_live_child_states()

        def fetch_one(index: int) -> tuple[bool, dict[str, Any] | None, Any]:
            iterator = iterators[index]
            try:
                row = next(iterator)
            except StopIteration:
                state = deepcopy(self._children[index].state_dict())
                return True, None, state
            state = deepcopy(self._children[index].state_dict())
            return False, row, state

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=width,
            thread_name_prefix="asterlm-hf-prefetch",
        )

        def activate_one() -> bool:
            if not waiting:
                return False
            index = waiting.popleft()
            iterator = self._iter_child_rows(index)
            iterators[index] = iterator
            futures[index] = executor.submit(fetch_one, index)
            active.append(index)
            return True

        for _ in range(width):
            activate_one()

        try:
            while active:
                index = active.popleft()
                future = futures[index]
                ended, row, after_state = future.result()

                if ended:
                    self._safe_child_states[index] = deepcopy(after_state)
                    self._exhausted[index] = True
                    futures.pop(index, None)
                    iterator = iterators.pop(index, None)
                    if iterator is not None:
                        close = getattr(iterator, "close", None)
                        if callable(close):
                            close()
                    self._advance_current_child()
                    activate_one()
                    continue

                if row is None:
                    raise RuntimeError("Parallel Hugging Face reader returned an empty non-terminal row")

                # Commit the row's exact child cursor before exposing it to the
                # caller. The next fetch may now run in parallel, but its live
                # state stays speculative until that future is selected.
                self._safe_child_states[index] = deepcopy(after_state)
                futures[index] = executor.submit(fetch_one, index)
                active.append(index)
                yield row
        finally:
            # Do not wait indefinitely for a blocked remote read during Ctrl-C.
            # The outer process-group supervisor has a bounded stop path, and
            # _safe_child_states already represents the last rows actually
            # emitted to AsterLM.
            for future in futures.values():
                future.cancel()
            for index, iterator in list(iterators.items()):
                future = futures.get(index)
                if future is not None and not future.done():
                    continue
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except (RuntimeError, ValueError):
                        pass
            executor.shutdown(wait=False, cancel_futures=True)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._apply_outer_pending_skips()

        if self._parallel_streams > 1:
            yield from self._iter_parallel()
            return

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
        states = (
            deepcopy(self._safe_child_states)
            if self._parallel_streams > 1 and self._safe_child_states
            else self._snapshot_live_child_states()
        )
        return {
            "_asterlm_cursor": ASTERLM_SEQUENTIAL_CURSOR,
            "current_child": self._current_child,
            "exhausted": list(self._exhausted),
            "child_states": states,
            "legacy_stream_count": len(self._children),
            "pending_cycle_skips": self._pending_cycle_skips,
            "cycle_index": self._cycle_index,
            "migration_policy": self._migration_policy,
            "parallel_streams": self._parallel_streams,
            "skipped_partial_shards": deepcopy(self._skipped_partial_shards),
            "last_resume_skipped_partial_shards": deepcopy(self._last_resume_skipped_partial_shards),
        }

def _sequential_pending_skips_per_child(cursor: dict[str, Any], exhausted: list[bool]) -> list[int]:
    """Distribute any still-pending outer one-row chunks over sequential children."""

    counts = [0] * len(exhausted)
    remaining = max(0, int(cursor.get("pending_cycle_skips", 0)))
    if not counts:
        return counts
    index = int(cursor.get("cycle_index", 0)) % len(counts)
    while remaining > 0:
        attempts = 0
        while attempts < len(counts) and exhausted[index]:
            index = (index + 1) % len(counts)
            attempts += 1
        if attempts >= len(counts):
            raise RuntimeError("Sequential cursor has pending chunks but every stream is exhausted")
        counts[index] += 1
        remaining -= 1
        index = (index + 1) % len(counts)
    return counts


def _state_partial_rows(state: dict[str, Any], outer_pending: int = 0) -> tuple[int, int]:
    """Return (shard_index, rows_into_current_shard) for a real HF child cursor."""

    if state.get("type") == "RebatchedArrowExamplesIterable":
        base = state.get("previous_state")
        if not isinstance(base, dict):
            base = state.get("examples_iterable")
        if not isinstance(base, dict) or "shard_idx" not in base:
            raise RuntimeError("Sequential child cursor has no underlying Arrow shard state")
        shard_idx = int(base.get("shard_idx", 0))
        rows = int(base.get("shard_example_idx", 0))
        rows += int(state.get("num_chunks_since_previous_state", 0))
        rows += outer_pending
        return shard_idx, rows
    if "shard_idx" not in state:
        raise RuntimeError("Sequential child cursor has no Arrow shard index")
    return int(state.get("shard_idx", 0)), int(state.get("shard_example_idx", 0)) + outer_pending


def convert_sequential_cursor_to_next_shard(cursor: Any) -> dict[str, Any]:
    """Upgrade a v3/v4 exact sequential cursor to the bounded next-shard policy.

    Older AsterLM patches could already rewrite the original Hugging Face cursor
    into ``asterlm-hf-sequential-v1`` before the offline migration was added.
    Those cursors have no ``migration_policy`` field and still point inside the
    same large partial Parquet files. This conversion is entirely local: it
    advances only child states that are actually partway through a shard and
    leaves untouched shard-boundary states unchanged.
    """

    if not is_sequential_cursor(cursor):
        raise RuntimeError("Cursor is not an AsterLM sequential checkpoint")
    states = cursor.get("child_states")
    exhausted = cursor.get("exhausted")
    if not isinstance(states, list) or not states:
        raise RuntimeError("Sequential cursor has no child states")
    if not isinstance(exhausted, list) or len(exhausted) != len(states):
        raise RuntimeError("Sequential cursor has invalid exhaustion state")

    exhausted_flags = [bool(value) for value in exhausted]
    current_child = max(0, min(int(cursor.get("current_child", 0)), len(states)))
    pending_counts = _sequential_pending_skips_per_child(cursor, exhausted_flags)
    converted_states: list[Any] = []
    audit: list[dict[str, Any]] = []

    for index, state in enumerate(states):
        if not isinstance(state, dict):
            raise RuntimeError(f"Sequential stream {index} has no durable child state")
        if exhausted_flags[index] or index < current_child:
            converted_states.append(deepcopy(state))
            continue

        shard_idx, partial_rows = _state_partial_rows(state, pending_counts[index])
        if partial_rows <= 0:
            # Already at the start of an untouched shard: do not skip it.
            converted_states.append(deepcopy(state))
            continue

        converted, item = SequentializedHfDataset._next_shard_state(state, pending_counts[index])
        item["stream_index"] = index
        item["conversion_source"] = "asterlm-sequential-v3-v4"
        item["partial_rows_detected"] = partial_rows
        converted_states.append(converted)
        audit.append(item)

    result = deepcopy(cursor)
    previous_audit = cursor.get("skipped_partial_shards")
    history = deepcopy(previous_audit) if isinstance(previous_audit, list) else []
    history.extend(deepcopy(audit))
    result["child_states"] = converted_states
    result["exhausted"] = exhausted_flags
    result["current_child"] = current_child
    result["pending_cycle_skips"] = 0
    result["cycle_index"] = 0
    result["migration_policy"] = LEGACY_POLICY_NEXT_SHARD
    result["skipped_partial_shards"] = history
    result["last_resume_skipped_partial_shards"] = deepcopy(audit)
    result["legacy_resume_state_from"] = "asterlm-sequential-v3-v4"
    result["legacy_outer_pending_chunks"] = max(0, int(cursor.get("pending_cycle_skips", 0)))
    result["previous_migration_policy"] = cursor.get("migration_policy")
    return result


def convert_legacy_cursor_to_next_shard(cursor: Any) -> dict[str, Any]:
    """Convert a legacy ten-stream cursor without opening or reading remote data.

    The returned cursor preserves all already committed AsterLM output and starts
    each legacy child at the file after its partially consumed current file. The
    unconsumed tails of those few files are intentionally omitted and replaced by
    later untouched files from the same source.
    """

    plan = extract_legacy_resume_plan(cursor)
    if plan is None:
        raise RuntimeError("Legacy Hugging Face multi-stream cursor structure was not found")
    previous_states = plan.cycling_state.get("previous_states")
    exhausted = plan.cycling_state.get("is_exhausted")
    if not isinstance(previous_states, list) or len(previous_states) != plan.stream_count:
        raise RuntimeError("Legacy Hugging Face cursor has invalid child lookahead state")
    if not isinstance(exhausted, list) or len(exhausted) != plan.stream_count:
        raise RuntimeError("Legacy Hugging Face cursor has invalid exhaustion state")
    if plan.cropped_chunk_length:
        raise RuntimeError(
            "The legacy cursor stopped inside a multi-row Arrow chunk; refusing an offline migration"
        )

    exhausted_flags = [bool(value) for value in exhausted]
    outer_counts = SequentializedHfDataset._pending_skips_per_child(plan, exhausted_flags)
    child_states: list[Any] = []
    audit: list[dict[str, Any]] = []
    for index, previous in enumerate(previous_states):
        if not isinstance(previous, dict):
            raise RuntimeError(f"Legacy stream {index} has no durable child state")
        converted, item = SequentializedHfDataset._next_shard_state(previous, outer_counts[index])
        item["stream_index"] = index
        child_states.append(converted)
        audit.append(item)

    return {
        "_asterlm_cursor": ASTERLM_SEQUENTIAL_CURSOR,
        "current_child": 0,
        "exhausted": exhausted_flags,
        "child_states": child_states,
        "legacy_stream_count": plan.stream_count,
        "pending_cycle_skips": 0,
        "cycle_index": 0,
        "migration_policy": LEGACY_POLICY_NEXT_SHARD,
        "skipped_partial_shards": audit,
        "legacy_resume_state_from": plan.source,
        "legacy_outer_pending_chunks": plan.pending_chunks,
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
        adapter = SequentializedHfDataset(
            legacy,
            cursor,
            legacy_policy=os.getenv("ASTERLM_LEGACY_RESUME_POLICY", LEGACY_POLICY_NEXT_SHARD),
        )
        return adapter, adapter.resume_mode

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
