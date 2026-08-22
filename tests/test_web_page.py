"""The browser interface, as a file.

It is one self-contained page served verbatim, so the things worth guarding are
structural: that every element the script reaches for exists, that nothing
reintroduces `innerHTML` (the page renders model output and runs under a strict
CSP), and that the ways somebody can get locked out stay closed.
"""

from __future__ import annotations

import re

import pytest

from lai.web import PAGE


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def ids_used(page: str) -> set[str]:
    return set(re.findall(r'\$\("([\w-]+)"\)', page))


def ids_defined(page: str) -> set[str]:
    return set(re.findall(r'id="([\w-]+)"', page))


# -- structure -----------------------------------------------------------


def test_every_element_the_script_reaches_for_exists(page):
    missing = ids_used(page) - ids_defined(page)
    assert missing == set(), f"$(...) with no matching element: {sorted(missing)}"


def test_the_page_is_self_contained(page):
    """A strict CSP blocks external hosts, and the wheel ships one file."""
    assert "<script" in page and page.count("<script") == page.count("</script>")
    assert not re.search(r'<script[^>]+src=', page)
    assert not re.search(r'<link[^>]+href="https?://', page)


def test_nothing_renders_model_output_as_html(page):
    """The feed shows text the model produced. `innerHTML` there is an
    injection waiting for a task whose result happens to contain a tag."""
    code = [line for line in page.splitlines() if not line.strip().startswith("//")]
    assignments = [line for line in code if re.search(r"\.innerHTML\s*=", line)]
    assert assignments == [], assignments


# -- getting in ----------------------------------------------------------


def test_a_rejected_token_is_not_reported_as_a_missing_one(page):
    """Telling somebody who is looking at a token in their own URL bar that
    they have no token sends them nowhere."""
    assert "token rejected" in page
    assert "was refused" in page


def test_a_rejected_token_can_be_replaced_without_clearing_site_data(page):
    """The bad token is remembered, so without this there is no way out."""
    assert "tokenfield" in page
    assert re.search(r'sessionStorage\.setItem\(KEY, typed\)', page)


def test_the_hand_written_spelling_of_the_fragment_works(page):
    """`#<token>` is what `lai web` writes; `#token=<token>` is what anybody
    reconstructing the URL types."""
    assert re.search(r'replace\(/\^token=/i, ""\)', page)


def test_a_fragment_pasted_into_an_open_page_is_noticed(page):
    """Same-document navigation does not re-run the script."""
    assert 'addEventListener("hashchange"' in page


def test_the_token_never_goes_into_a_url_the_server_sees(page):
    """A query string reaches logs and proxies; a fragment does not."""
    assert "location.hash" in page
    assert "?token=" not in page.split("/screen?token=")[0]


# -- the settings that changed --------------------------------------------


def test_which_screen_the_agent_works_on_is_settable(page):
    """It is the defining behaviour, and it was unreachable from the browser."""
    for element in ("ownDisplaySelect", "watchSelect", "handoverSelect"):
        assert f'id="{element}"' in page
    assert '"/desktop"' in page


def test_the_screen_panel_says_whose_screen_it_is(page):
    assert "The agent's screen" in page
    assert "Your desktop" in page


def test_the_screen_opens_by_default_only_when_there_is_a_separate_one(page):
    assert "separate" in page
    assert "lai.screen" in page, "and a viewer who closed it is remembered"


# -- readable by somebody who is not a programmer -------------------------


def test_the_feed_speaks_plainly_by_default(page):
    """`ui_click {"ref":114}` is precise for whoever is debugging the call, and
    useless to the person watching their own computer being used."""
    assert "let plain = true" in page
    assert "event.plain" in page


def test_the_detail_is_one_button_away(page):
    """The moment something goes wrong the tool name is what matters."""
    assert 'id="detail"' in page
    assert "lai.detail" in page, "and the choice is remembered"


def test_a_step_that_worked_does_not_repeat_itself_in_the_feed(page):
    assert "if (event.ok) sharpen(row, summary);" in page


def test_a_click_by_reference_is_named_from_its_result(page):
    """The request could not say which button; the result can."""
    assert "function sharpen" in page
    assert "an item on screen" in page
