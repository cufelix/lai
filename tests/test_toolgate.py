"""Which tools the model is shown, and what it costs.

Connect a handful of MCP servers and the registry grows to several hundred
tools whose schemas are re-sent in full on every turn. The gate is what stops
an ordinary desktop question from paying sixty thousand tokens for a database
API it was never going to touch.
"""

from __future__ import annotations

import json

import pytest

from lai.agent import relevance
from lai.agent.toolgate import DEFAULT_LIMIT, ToolGate, is_extension, server_of
from lai.safety.policy import Risk
from lai.tools.base import ToolRegistry, ToolResult, ToolSpec


def spec(name: str, description: str = "", group: str = "core") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={"properties": {}},
        handler=lambda ctx, args: ToolResult.text("ok"),
        risk=Risk.READ,
        group=group,
    )


@pytest.fixture
def registry():
    reg = ToolRegistry()
    for name in ("ui_click", "window_list", "shell_run", "screenshot"):
        reg.register(spec(name, f"{name} does a thing"))
    return reg


def add_server(reg, server: str, tools: dict[str, str]) -> None:
    for name, description in tools.items():
        reg.register(spec(f"mcp__{server}__{name}", description, group=f"mcp:{server}"))


# -- what counts as an extension ----------------------------------------


def test_an_mcp_group_marks_a_tool_as_an_extension():
    assert is_extension(spec("x", group="mcp:github"))
    assert not is_extension(spec("ui_click"))
    assert server_of(spec("x", group="mcp:token-optimizer")) == "token-optimizer"


# -- the common case: nothing relevant is connected ----------------------


def test_without_extensions_every_tool_is_shown(registry):
    shown, withheld = ToolGate(registry).choose("open the calculator")
    assert len(shown) == 4 and withheld == []


def test_irrelevant_servers_are_withheld_entirely(registry):
    add_server(registry, "stripe", {
        "create_charge": "Charge a customer's saved card",
        "refund": "Refund a payment",
        "list_invoices": "List invoices for a customer",
    })
    shown, withheld = ToolGate(registry).choose("how many windows do I have open?")
    assert [s.name for s in shown] == ["screenshot", "shell_run", "ui_click", "window_list"]
    assert len(withheld) == 3


def test_the_core_tools_are_never_withheld(registry):
    add_server(registry, "stripe", {f"t{i}": "Charge a card" for i in range(40)})
    shown, _ = ToolGate(registry).choose("refund every charge")
    assert {s.name for s in shown} >= {"ui_click", "window_list", "shell_run", "screenshot"}


# -- naming the service --------------------------------------------------


def test_naming_a_server_brings_its_tools_in(registry):
    add_server(registry, "stripe", {
        "create_charge": "Charge a customer's saved card",
        "refund": "Refund a payment",
    })
    shown, withheld = ToolGate(registry).choose("refund the last stripe payment")
    assert "mcp__stripe__refund" in {s.name for s in shown}
    # Naming the service does not buy the whole service: charging a card is
    # still not what was asked for.
    assert [s.name for s in withheld] == ["mcp__stripe__create_charge"]


def test_a_named_server_with_nothing_else_to_go_on_still_offers_tools(registry):
    """"use stripe" says which service and nothing about which tool — the
    server's own name is in all of them, so it cannot be the discriminator."""
    add_server(registry, "stripe", {f"tool_{i}": "Does a stripe thing" for i in range(20)})
    shown, withheld = ToolGate(registry).choose("use stripe")
    offered = [s for s in shown if is_extension(s)]
    assert len(offered) == DEFAULT_LIMIT
    assert len(withheld) == 20 - DEFAULT_LIMIT


def test_a_named_server_is_capped(registry):
    add_server(registry, "stripe", {f"charge_{i}": "Charge a card" for i in range(40)})
    shown, _ = ToolGate(registry).choose("stripe: charge the card")
    assert sum(1 for s in shown if is_extension(s)) <= DEFAULT_LIMIT


def test_a_hyphenated_server_is_matched_by_either_half(registry):
    add_server(registry, "token-optimizer", {"smart_logs": "Read logs efficiently"})
    shown, _ = ToolGate(registry).choose("use the optimizer to read logs")
    assert "mcp__token-optimizer__smart_logs" in {s.name for s in shown}


# -- earning a place without being named ---------------------------------


def test_a_description_hit_alone_is_not_enough(registry):
    """Two hundred tool descriptions share a lot of words. One coincidence in
    prose must not buy a schema on every turn."""
    add_server(registry, "stripe", {"refund": "Undo a payment for a customer window"})
    shown, withheld = ToolGate(registry).choose("close the window")
    assert not [s for s in shown if is_extension(s)]
    assert len(withheld) == 1


def test_a_name_hit_is_enough(registry):
    add_server(registry, "stripe", {"invoice_pdf": "Fetch an invoice as a PDF"})
    shown, _ = ToolGate(registry).choose("download the invoice")
    assert "mcp__stripe__invoice_pdf" in {s.name for s in shown}


# -- unlocking -----------------------------------------------------------


def test_unlocking_keeps_a_tool_for_the_rest_of_the_run(registry):
    add_server(registry, "stripe", {"refund": "Undo a payment"})
    gate = ToolGate(registry)
    assert not [s for s in gate.choose("close the window")[0] if is_extension(s)]
    assert gate.unlock(["mcp__stripe__refund"]) == ["mcp__stripe__refund"]
    shown, withheld = gate.choose("close the window")
    assert "mcp__stripe__refund" in {s.name for s in shown}
    assert withheld == []


def test_unlocking_something_that_does_not_exist_takes_nothing(registry):
    gate = ToolGate(registry)
    assert gate.unlock(["mcp__nope__nope"]) == []
    assert gate.unlocked == set()


def test_unlocked_tools_do_not_eat_the_whole_budget(registry):
    add_server(registry, "stripe", {f"tool_{i}": "Does a stripe thing" for i in range(40)})
    gate = ToolGate(registry, limit=4)
    gate.unlock([f"mcp__stripe__tool_{i}" for i in range(3)])
    shown, _ = gate.choose("use stripe")
    assert sum(1 for s in shown if is_extension(s)) == 4


# -- telling the model what it cannot see --------------------------------


def test_withheld_tools_are_named_by_server(registry):
    add_server(registry, "stripe", {"refund": "Undo a payment"})
    add_server(registry, "github", {"pr": "Open a pull request", "issue": "File an issue"})
    text = ToolGate(registry).describe_withheld("how many windows are open?")
    assert "3 further tools" in text
    assert "github (2)" in text and "stripe (1)" in text
    assert "tool_find" in text


def test_nothing_withheld_says_nothing(registry):
    assert ToolGate(registry).describe_withheld("open the calculator") == ""


# -- the point of the exercise -------------------------------------------


def test_the_gate_actually_cuts_the_prompt(registry):
    """The measurement that motivated all of this."""
    add_server(registry, "stripe", {
        f"tool_{i}": "Charge, refund or reconcile a payment. " * 20 for i in range(200)
    })
    everything = len(json.dumps(registry.to_anthropic()))
    gated = len(json.dumps(ToolGate(registry).schemas("how many windows do I have open?")))
    assert gated < everything / 10


def test_schemas_render_in_the_dialect_asked_for(registry):
    anthropic = ToolGate(registry).schemas("anything", dialect="anthropic")
    openai = ToolGate(registry).schemas("anything", dialect="openai")
    assert "input_schema" in anthropic[0]
    assert openai[0]["type"] == "function"


# -- the shared ranking --------------------------------------------------


def test_a_word_in_most_of_the_corpus_carries_no_signal():
    items = [("page_create", ""), ("page_delete", ""), ("page_get", ""), ("unrelated", "")]
    _, matched = relevance.rank(
        items * 5, "page", name_of=lambda i: i[0], text_of=lambda i: i[1]
    )
    assert matched == 0, "15 of 20 entries contain it; it cannot choose between them"


def test_the_noise_ceiling_holds_on_a_large_corpus():
    """A quarter of two hundred is fifty — far too permissive to mean anything."""
    items = [(f"page_{i}", "") for i in range(20)] + [(f"other_{i}", "") for i in range(200)]
    _, matched = relevance.rank(items, "page", name_of=lambda i: i[0], text_of=lambda i: i[1])
    assert matched == 0


def test_scores_separate_a_name_hit_from_a_prose_hit():
    items = [("refund", "does a thing"), ("charge", "issue a refund later")]
    scored = relevance.rank_scored(
        items, "refund the payment", name_of=lambda i: i[0], text_of=lambda i: i[1]
    )
    assert scored[0][1][0] == "refund"
    assert scored[0][0] > scored[1][0]
    assert scored[1][0] == 1
