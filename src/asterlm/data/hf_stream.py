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


def _examples_state(cursor: Any) -> Any:
    if isinstance(cursor, dict) and "examples_iterable" in cursor:
        return cursor.get("examples_iterable")
    return cursor


def is_sequential_cursor(cursor: Any) -> bool:
    return isinstance(cursor, dict) and cursor.get("_asterlm_cursor") == ASTERLM_SEQUENTIAL_CURSOR


def is_legacy_multistream_cursor(cursor: Any) -> bool:
    state = _examples_state(cursor)
    return (
        isinstance(state, dict)
        and isinstance(state.get("ex_iterables"), list)
        and isinstance(state.get("previous_states"), list)
        and isinstance(state.get("is_exhausted"), list)
    )


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
    """Drain a legacy 10-stream Hugging Face shuffle one stream at a time.

    datasets 5.x may interleave up to ten Parquet input shards even when
    ``buffer_size=1``. Each stream can retain an Arrow table, causing extreme RAM
    use and a large offline backlog. This adapter preserves every child cursor,
    rewinds CyclingMultiSources' one-record lookahead, and then drains children
    sequentially. Output order changes, but no source record is duplicated or
    omitted. Subsequent checkpoints use a small AsterLM-owned cursor format.
    """

    def __init__(self, legacy_dataset: Any, cursor: dict[str, Any]) -> None:
        self._dataset = legacy_dataset
        self._children: list[Any]
        self._exhausted: list[bool]
        self._current_child: int

        top = getattr(legacy_dataset, "_ex_iterable", None)
        if top is None:
            raise RuntimeError("Hugging Face iterable internals are unavailable for cursor migration")
        init = getattr(top, "_init_state_dict", None)
        if not callable(init):
            raise RuntimeError("Hugging Face iterable does not expose checkpoint initialization")
        init()

        inner = getattr(top, "ex_iterable", None)
        children = getattr(inner, "ex_iterables", None)
        if not isinstance(children, list) or not children:
            raise RuntimeError(
                "Legacy cursor says multiple streams were active, but the recreated dataset "
                "does not expose the expected CyclingMultiSources layout. Keep datasets==5.0.1."
            )
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
            return

        examples_state = _examples_state(cursor)
        load_top = getattr(top, "load_state_dict", None)
        if not callable(load_top):
            raise RuntimeError("Hugging Face iterable cannot restore a legacy cursor")
        load_top(examples_state)

        cycling_state = getattr(inner, "_state_dict", None)
        if not isinstance(cycling_state, dict):
            raise RuntimeError("Legacy Hugging Face multi-stream state was not initialized")
        previous_states = cycling_state.get("previous_states")
        exhausted = cycling_state.get("is_exhausted")
        if not isinstance(previous_states, list) or len(previous_states) != len(children):
            raise RuntimeError("Legacy Hugging Face cursor has invalid lookahead state")
        if not isinstance(exhausted, list) or len(exhausted) != len(children):
            raise RuntimeError("Legacy Hugging Face cursor has invalid exhaustion state")

        # CyclingMultiSources reads one record ahead from every child. Rewind each
        # child to its recorded previous state before abandoning the interleaver.
        for child, previous in zip(children, previous_states):
            if previous is not None:
                child.load_state_dict(previous)
        self._exhausted = [bool(value) for value in exhausted]
        self._current_child = 0

    @property
    def resume_mode(self) -> str:
        return LEGACY_SEQUENTIAL_LAYOUT

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while self._current_child < len(self._children):
            index = self._current_child
            if self._exhausted[index]:
                self._current_child += 1
                continue
            child = self._children[index]
            for item in child:
                if isinstance(item, tuple) and len(item) == 2:
                    yield item[1]
                else:
                    yield item
            self._exhausted[index] = True
            self._current_child += 1

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
