from __future__ import annotations

from lol_ai.pipeline.cblol import should_include_row


def test_cblol_2026_incluido():
    assert should_include_row("CBLOL", "2026") is True


def test_cblol_academy_incluido():
    assert should_include_row("CBLOLA", "2026") is True


def test_lta_sul_2025_incluido():
    assert should_include_row("LTA S", "2025") is True


def test_lta_norte_excluido():
    assert should_include_row("LTA N", "2025") is False


def test_outra_liga_excluida():
    assert should_include_row("LCK", "2025") is False


def test_lta_sul_2026_excluido():
    assert should_include_row("LTA S", "2026") is False


from lol_ai.pipeline.cblol import compute_gold_diff  # noqa: E402


def test_gold_diff_perspectiva_azul():
    assert compute_gold_diff({"totalgold": "65405"}, {"totalgold": "60000"}) == "5405"
    assert compute_gold_diff({"totalgold": "60000"}, {"totalgold": "65405"}) == "-5405"


def test_gold_diff_ausente_fica_vazio():
    assert compute_gold_diff({"totalgold": ""}, {"totalgold": "60000"}) == ""
    assert compute_gold_diff({}, {}) == ""
