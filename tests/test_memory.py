"""MemoryStore: upsert semantics, search (FTS5 and LIKE fallback), thread safety."""

from __future__ import annotations

import threading

import pytest

from lai.agent.memory import MemoryEntry, MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.db") as s:
        yield s


# -- basic remember / recall ----------------------------------------------


def test_remember_without_key_always_inserts(store):
    store.remember("fact one", kind="fact")
    store.remember("fact two", kind="fact")
    assert store.stats()["total"] == 2


def test_remember_with_key_upserts_instead_of_duplicating(store):
    first = store.remember("Xed saves via Name: field", kind="app", key="xed.save_field")
    second = store.remember("Xed saves via Filename: field", kind="app", key="xed.save_field")
    assert first.id == second.id
    assert store.stats()["total"] == 1
    [entry] = store.all(kind="app")
    assert entry.content == "Xed saves via Filename: field"


def test_upsert_is_scoped_to_kind_and_key_together(store):
    store.remember("a", kind="fact", key="shared")
    store.remember("b", kind="preference", key="shared")
    assert store.stats()["total"] == 2


def test_remember_rejects_empty_content(store):
    with pytest.raises(ValueError):
        store.remember("", kind="fact")
    with pytest.raises(ValueError):
        store.remember("   ", kind="fact")


def test_remember_rejects_empty_kind(store):
    with pytest.raises(ValueError):
        store.remember("x", kind="")


def test_to_dict_round_trips_fields():
    entry = MemoryEntry(id=1, kind="fact", content="hello", key="k", tags=("a", "b"))
    data = entry.to_dict()
    assert data["id"] == 1 and data["kind"] == "fact" and data["tags"] == ["a", "b"]


# -- forget ----------------------------------------------------------------


def test_forget_by_numeric_id(store):
    entry = store.remember("temp fact", kind="fact")
    assert store.forget(entry.id) == 1
    assert store.stats()["total"] == 0


def test_forget_by_numeric_id_as_string(store):
    entry = store.remember("temp fact", kind="fact")
    assert store.forget(str(entry.id)) == 1


def test_forget_by_key(store):
    store.remember("v1", kind="app", key="notepad.quirk")
    assert store.forget("notepad.quirk") == 1
    assert store.stats()["total"] == 0


def test_forget_nonexistent_returns_zero(store):
    assert store.forget(999999) == 0
    assert store.forget("no-such-key") == 0


# -- search: FTS5 and LIKE fallback -----------------------------------


def test_recall_finds_matching_content(store):
    store.remember("The save dialog field is labelled Name:", kind="app", key="xed")
    store.remember("Unrelated fact about something else", kind="fact")
    results = store.recall("save dialog")
    assert any("save dialog" in r.content for r in results)


def test_recall_ranks_stronger_matches_higher(store):
    store.remember("banana banana banana", kind="fact", key="a")
    store.remember("banana appears once here", kind="fact", key="b")
    results = store.recall("banana")
    assert results[0].key == "a"


def test_recall_filters_by_kind(store):
    store.remember("apple pie recipe", kind="fact")
    store.remember("apple pie preference", kind="preference")
    results = store.recall("apple", kind="preference")
    assert results and all(r.kind == "preference" for r in results)


def test_recall_respects_limit(store):
    for i in range(5):
        store.remember(f"widget number {i}", kind="fact")
    results = store.recall("widget", limit=2)
    assert len(results) == 2


def test_recall_with_no_matches_returns_empty(store):
    store.remember("something entirely unrelated", kind="fact")
    assert store.recall("zzzznomatchzzzz") == []


def test_recall_empty_query_returns_recent_entries(store):
    store.remember("first", kind="fact")
    store.remember("second", kind="fact")
    results = store.recall("")
    assert len(results) == 2


def test_recall_increments_hits(store):
    entry = store.remember("hit counter test unique term", kind="fact")
    assert entry.hits == 0
    store.recall("unique term")
    [reloaded] = store.all(kind="fact")
    assert reloaded.hits == 1


def test_like_fallback_matches_the_same_as_fts(tmp_path):
    """Force the LIKE path by disabling FTS5 and check it still finds things."""
    store = MemoryStore(tmp_path / "memory.db")
    store._fts_enabled = False  # simulate a build without FTS5
    store.remember("the save dialog names its field Name:", kind="app", key="xed")
    store.remember("completely unrelated content", kind="fact")
    results = store.recall("save dialog")
    assert any("save dialog" in r.content for r in results)
    store.close()


def test_like_fallback_ranks_by_term_frequency(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store._fts_enabled = False
    store.remember("gizmo gizmo gizmo", kind="fact", key="a")
    store.remember("gizmo appears once", kind="fact", key="b")
    results = store.recall("gizmo")
    assert results[0].key == "a"
    store.close()


# -- all / stats -------------------------------------------------------


def test_all_orders_by_most_recently_updated(store):
    store.remember("older", kind="fact", key="k1")
    store.remember("newer", kind="fact", key="k2")
    entries = store.all()
    assert entries[0].key == "k2"


def test_all_filters_by_kind(store):
    store.remember("a fact", kind="fact")
    store.remember("a preference", kind="preference")
    assert all(e.kind == "preference" for e in store.all(kind="preference"))


def test_stats_reports_counts_by_kind(store):
    store.remember("f1", kind="fact")
    store.remember("f2", kind="fact")
    store.remember("p1", kind="preference")
    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_kind"]["fact"] == 2
    assert stats["by_kind"]["preference"] == 1
    assert "fts5" in stats and "path" in stats


# -- context_block ------------------------------------------------------


def test_context_block_is_empty_when_nothing_relevant(store):
    assert store.context_block("nothing has ever been saved") == ""


def test_context_block_renders_relevant_entries(store):
    store.remember("Xed's save dialog names its field 'Name:'", kind="app", key="xed.save")
    block = store.context_block("xed save dialog")
    assert block.startswith("## Memory")
    assert "Xed's save dialog" in block


def test_context_block_empty_store_is_empty(tmp_path):
    with MemoryStore(tmp_path / "memory.db") as s:
        assert s.context_block("anything") == ""


# -- thread safety smoke ------------------------------------------------


def test_concurrent_remember_and_recall_does_not_crash_or_corrupt(tmp_path):
    with MemoryStore(tmp_path / "memory.db") as s:
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(20):
                    s.remember(f"thread {n} fact {i}", kind="fact", key=f"t{n}-{i}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(20):
                    s.recall("thread")
                    s.all(limit=5)
                    s.stats()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert s.stats()["total"] == 80


# -- open() / close() ----------------------------------------------------


def test_open_creates_db_under_home(tmp_path):
    home = tmp_path / "lai-home"
    store = MemoryStore.open(home)
    try:
        assert (home / "memory.db").is_file()
    finally:
        store.close()


def test_context_manager_closes_on_exit(tmp_path):
    with MemoryStore(tmp_path / "memory.db") as s:
        s.remember("x", kind="fact")
    with pytest.raises(Exception):
        s.stats()  # connection is closed
