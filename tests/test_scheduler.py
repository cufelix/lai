"""Cron parsing, TaskStore persistence, and the Scheduler daemon."""

from __future__ import annotations

from datetime import datetime

import pytest

from lai.scheduler import (
    Scheduler,
    TaskStore,
    describe_schedule,
    make_task,
    next_run_after,
    parse_cron,
)

# -- cron field syntax -----------------------------------------------------


def test_star_matches_every_value():
    matcher = parse_cron("* * * * *")
    assert matcher.matches(datetime(2026, 1, 1, 0, 0))
    assert matcher.matches(datetime(2026, 6, 15, 13, 47))


def test_exact_numbers():
    matcher = parse_cron("30 9 1 6 *")
    assert matcher.matches(datetime(2026, 6, 1, 9, 30))
    assert not matcher.matches(datetime(2026, 6, 1, 9, 31))
    assert not matcher.matches(datetime(2026, 6, 2, 9, 30))
    assert not matcher.matches(datetime(2026, 7, 1, 9, 30))


def test_comma_list():
    matcher = parse_cron("0,15,45 * * * *")
    assert matcher.matches(datetime(2026, 1, 1, 5, 0))
    assert matcher.matches(datetime(2026, 1, 1, 5, 15))
    assert matcher.matches(datetime(2026, 1, 1, 5, 45))
    assert not matcher.matches(datetime(2026, 1, 1, 5, 30))


def test_range():
    matcher = parse_cron("0 9-17 * * *")
    assert matcher.matches(datetime(2026, 1, 1, 9, 0))
    assert matcher.matches(datetime(2026, 1, 1, 17, 0))
    assert not matcher.matches(datetime(2026, 1, 1, 8, 0))
    assert not matcher.matches(datetime(2026, 1, 1, 18, 0))


def test_step_on_minutes_every_15():
    matcher = parse_cron("*/15 * * * *")
    matching = {0, 15, 30, 45}
    for minute in range(60):
        dt = datetime(2026, 1, 1, 0, minute)
        assert matcher.matches(dt) == (minute in matching), minute


def test_range_with_step():
    matcher = parse_cron("0 0-12/4 * * *")
    matching_hours = {0, 4, 8, 12}
    for hour in range(24):
        assert matcher.matches(datetime(2026, 1, 1, hour, 0)) == (hour in matching_hours)


def test_shorthands():
    assert parse_cron("@hourly").matches(datetime(2026, 1, 1, 5, 0))
    assert not parse_cron("@hourly").matches(datetime(2026, 1, 1, 5, 1))
    assert parse_cron("@daily").matches(datetime(2026, 1, 1, 0, 0))
    assert not parse_cron("@daily").matches(datetime(2026, 1, 1, 1, 0))
    assert parse_cron("@weekly").matches(datetime(2026, 1, 4, 0, 0))  # a Sunday
    assert not parse_cron("@weekly").matches(datetime(2026, 1, 5, 0, 0))  # a Monday
    assert parse_cron("@monthly").matches(datetime(2026, 3, 1, 0, 0))
    assert not parse_cron("@monthly").matches(datetime(2026, 3, 2, 0, 0))


# -- day-of-month vs day-of-week ---------------------------------------


def test_dow_only_restricted_matches_any_matching_weekday():
    # Every Monday, dom left as '*'.
    matcher = parse_cron("0 8 * * 1")
    assert matcher.matches(datetime(2026, 1, 5, 8, 0))  # Monday
    assert not matcher.matches(datetime(2026, 1, 6, 8, 0))  # Tuesday


def test_dom_only_restricted_ignores_weekday():
    matcher = parse_cron("0 8 15 * *")
    assert matcher.matches(datetime(2026, 3, 15, 8, 0))
    assert not matcher.matches(datetime(2026, 3, 16, 8, 0))


def test_dom_and_dow_both_restricted_are_ored():
    # Classic cron semantics: 1st of the month OR any Friday.
    matcher = parse_cron("0 0 1 * 5")
    assert matcher.matches(datetime(2026, 2, 1, 0, 0))  # a Sunday, but the 1st
    assert matcher.matches(datetime(2026, 1, 2, 0, 0))  # a Friday, not the 1st
    assert not matcher.matches(datetime(2026, 1, 3, 0, 0))  # neither


def test_sunday_alias_seven_normalises_to_zero():
    matcher = parse_cron("0 0 * * 7")
    assert matcher.matches(datetime(2026, 1, 4, 0, 0))  # a Sunday


# -- next_after boundaries ------------------------------------------------


def test_next_after_within_the_same_hour():
    matcher = parse_cron("*/15 * * * *")
    nxt = matcher.next_after(datetime(2026, 1, 1, 10, 5))
    assert nxt == datetime(2026, 1, 1, 10, 15)


def test_next_after_rolls_over_an_hour_boundary():
    matcher = parse_cron("0 * * * *")
    nxt = matcher.next_after(datetime(2026, 1, 1, 10, 30))
    assert nxt == datetime(2026, 1, 1, 11, 0)


def test_next_after_rolls_over_a_day_boundary():
    matcher = parse_cron("0 0 * * *")
    nxt = matcher.next_after(datetime(2026, 1, 1, 23, 59))
    assert nxt == datetime(2026, 1, 2, 0, 0)


def test_next_after_rolls_over_a_month_boundary():
    matcher = parse_cron("0 0 1 * *")
    nxt = matcher.next_after(datetime(2026, 1, 15, 12, 0))
    assert nxt == datetime(2026, 2, 1, 0, 0)


def test_next_after_rolls_over_a_year_boundary():
    matcher = parse_cron("0 0 1 1 *")
    nxt = matcher.next_after(datetime(2026, 6, 1, 0, 0))
    assert nxt == datetime(2027, 1, 1, 0, 0)


def test_next_after_never_returns_the_reference_time_itself():
    matcher = parse_cron("*/15 * * * *")
    nxt = matcher.next_after(datetime(2026, 1, 1, 10, 15))
    assert nxt > datetime(2026, 1, 1, 10, 15)
    assert nxt == datetime(2026, 1, 1, 10, 30)


def test_impossible_schedule_raises_rather_than_hanging():
    # 30 February never exists.
    matcher = parse_cron("0 0 30 2 *")
    with pytest.raises(ValueError):
        matcher.next_after(datetime(2026, 1, 1, 0, 0))


# -- malformed cron -----------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    ["* * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "* * * 13 *", "* * * * 8", "a * * * *"],
)
def test_invalid_cron_raises(expr):
    with pytest.raises(ValueError):
        parse_cron(expr)


# -- every:<seconds> --------------------------------------------------


def test_interval_schedule_advances_by_seconds():
    nxt = next_run_after("every:60", datetime(2026, 1, 1, 0, 0, 0))
    assert nxt == datetime(2026, 1, 1, 0, 1, 0)


def test_interval_schedule_rejects_non_positive():
    with pytest.raises(ValueError):
        next_run_after("every:0", datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        next_run_after("every:-5", datetime(2026, 1, 1))


# -- make_task -----------------------------------------------------------


def test_make_task_computes_next_run():
    task = make_task(name="daily backup", task="run the backup script", schedule="@daily",
                      at=datetime(2026, 1, 1, 10, 0))
    assert task.next_run == datetime(2026, 1, 2, 0, 0).timestamp()
    assert task.enabled and task.runs == 0 and task.failures == 0
    assert task.id


def test_make_task_rejects_bad_schedule():
    with pytest.raises(ValueError):
        make_task(name="x", task="y", schedule="not a cron", at=datetime(2026, 1, 1))


def test_make_task_rejects_empty_name_or_task():
    with pytest.raises(ValueError):
        make_task(name="", task="y", schedule="@daily")
    with pytest.raises(ValueError):
        make_task(name="x", task="  ", schedule="@daily")


# -- TaskStore ------------------------------------------------------------


def test_store_add_list_remove_roundtrip(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="a", task="do a", schedule="@daily", at=datetime(2026, 1, 1))
    store.add(task)
    assert [t.id for t in store.list()] == [task.id]
    assert store.get(task.id) == task
    assert store.remove(task.id) is True
    assert store.list() == []
    assert store.remove(task.id) is False


def test_store_update_roundtrips_fields(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="a", task="do a", schedule="@daily", at=datetime(2026, 1, 1))
    store.add(task)
    updated = store.update(task.__class__(**{**task.to_dict(), "runs": 3, "failures": 1}))
    assert store.get(task.id).runs == 3
    assert store.get(task.id).failures == 1
    assert updated.runs == 3


def test_store_missing_file_returns_empty(tmp_path):
    store = TaskStore(tmp_path / "does-not-exist.json")
    assert store.list() == []


def test_store_corrupt_file_is_tolerated(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text("{not valid json at all", encoding="utf-8")
    store = TaskStore(path)
    assert store.list() == []
    # And it should still be usable afterward — writing repairs the file.
    task = make_task(name="a", task="do a", schedule="@daily", at=datetime(2026, 1, 1))
    store.add(task)
    assert [t.id for t in store.list()] == [task.id]


def test_store_tolerates_one_corrupt_entry_among_valid_ones(tmp_path):
    import json

    path = tmp_path / "schedule.json"
    good = make_task(name="good", task="x", schedule="@daily", at=datetime(2026, 1, 1))
    path.write_text(
        json.dumps({"tasks": {good.id: good.to_dict(), "bad-id": {"name": "incomplete"}}}),
        encoding="utf-8",
    )
    store = TaskStore(path)
    ids = [t.id for t in store.list()]
    assert ids == [good.id]


# -- Scheduler --------------------------------------------------------


def test_due_now_finds_tasks_whose_next_run_has_passed(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    due_task = make_task(name="due", task="x", schedule="every:60", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(due_task)
    scheduler = Scheduler(store, callback=lambda t: None)
    found = scheduler.due_now(datetime(2026, 1, 1, 0, 1, 1))
    assert [t.id for t in found] == [due_task.id]


def test_due_now_excludes_tasks_not_yet_due(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    future_task = make_task(name="future", task="x", schedule="every:3600", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(future_task)
    scheduler = Scheduler(store, callback=lambda t: None)
    assert scheduler.due_now(datetime(2026, 1, 1, 0, 1, 0)) == []


def test_due_now_excludes_disabled_tasks(tmp_path):
    from dataclasses import replace

    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="off", task="x", schedule="every:60", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(replace(task, enabled=False))
    scheduler = Scheduler(store, callback=lambda t: None)
    assert scheduler.due_now(datetime(2026, 1, 1, 1, 0, 0)) == []


def test_tick_fires_due_tasks_and_advances_next_run(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="t", task="x", schedule="every:60", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(task)
    fired = []
    scheduler = Scheduler(store, callback=fired.append)
    scheduler.tick(datetime(2026, 1, 1, 0, 1, 0))
    assert [t.id for t in fired] == [task.id]
    updated = store.get(task.id)
    assert updated.runs == 1
    assert updated.failures == 0
    assert updated.last_run == datetime(2026, 1, 1, 0, 1, 0).timestamp()
    assert updated.next_run == datetime(2026, 1, 1, 0, 2, 0).timestamp()


def test_tick_does_not_fire_tasks_not_yet_due(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="t", task="x", schedule="every:3600", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(task)
    fired = []
    scheduler = Scheduler(store, callback=fired.append)
    scheduler.tick(datetime(2026, 1, 1, 0, 1, 0))
    assert fired == []
    assert store.get(task.id).runs == 0


def test_a_raising_callback_is_recorded_as_a_failure_and_does_not_propagate(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="boom", task="x", schedule="every:60", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(task)

    def bad_callback(_task):
        raise RuntimeError("callback exploded")

    scheduler = Scheduler(store, callback=bad_callback)
    scheduler.tick(datetime(2026, 1, 1, 0, 1, 0))  # must not raise
    updated = store.get(task.id)
    assert updated.runs == 1
    assert updated.failures == 1
    # next_run still advanced, so a broken task doesn't fire in a tight loop.
    assert updated.next_run == datetime(2026, 1, 1, 0, 2, 0).timestamp()


def test_one_raising_task_does_not_block_a_healthy_one(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    bad = make_task(name="bad", task="x", schedule="every:60", at=datetime(2026, 1, 1, 0, 0, 0))
    good = make_task(name="good", task="y", schedule="every:60", at=datetime(2026, 1, 1, 0, 0, 0))
    store.add(bad)
    store.add(good)
    fired = []

    def callback(task):
        if task.id == bad.id:
            raise RuntimeError("boom")
        fired.append(task.id)

    scheduler = Scheduler(store, callback=callback)
    scheduler.tick(datetime(2026, 1, 1, 0, 1, 0))
    assert fired == [good.id]
    assert store.get(bad.id).failures == 1
    assert store.get(good.id).failures == 0


def test_start_and_stop_run_the_background_thread(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    task = make_task(name="t", task="x", schedule="every:60", at=datetime.now())
    # Make it due immediately by back-dating next_run.
    import time as _time
    from dataclasses import replace as _replace

    store.add(_replace(task, next_run=_time.time() - 1))

    fired = []
    scheduler = Scheduler(store, callback=fired.append, interval=0.05)
    scheduler.start()
    try:
        deadline = _time.monotonic() + 3.0
        while not fired and _time.monotonic() < deadline:
            _time.sleep(0.02)
    finally:
        scheduler.stop(timeout=2.0)
    assert fired
    assert not scheduler.running


def test_stop_is_safe_to_call_when_never_started(tmp_path):
    store = TaskStore(tmp_path / "schedule.json")
    scheduler = Scheduler(store, callback=lambda t: None)
    scheduler.stop()  # must not raise


# -- human-readable schedules --------------------------------------------


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("every:60", "every minute"),
        ("every:120", "every 2 minutes"),
        ("every:3600", "every hour"),
        ("every:7200", "every 2 hours"),
        ("every:86400", "every day"),
        ("every:20", "every 20s"),
        ("every:90", "every 90s"),
        ("@daily", "every day at midnight"),
        ("0 * * * *", "every hour"),
        ("* * * * *", "every minute"),
    ],
)
def test_describe_schedule(schedule, expected):
    assert describe_schedule(schedule) == expected


def test_describe_schedule_echoes_what_it_cannot_phrase():
    """Better to show the raw expression than to describe it wrongly."""
    assert describe_schedule("15 3 * * 1-5") == "15 3 * * 1-5"
    assert describe_schedule("every:notanumber") == "every:notanumber"
