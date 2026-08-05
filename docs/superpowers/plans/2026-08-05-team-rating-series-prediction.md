# Rating por Adversário + Previsão de Séries — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rating de força por time (estilo Elo) ponderado pelo adversário, com ajuste de roster, previsão de probabilidade por jogo e por série (Bo3/Bo5), backtest walk-forward com métricas completas e CLI atualizado.

**Architecture:** Novo módulo `rating.py` (motor Elo com virada de temporada e ajuste de roster) alimentado pelo dataset contextual já existente, que passa a incluir LTA Sul 2025. Um módulo `rating_backtest.py` roda walk-forward, calibra hiperparâmetros na janela de validação e gera relatórios/gráficos. `series.py` converte probabilidades por jogo em probabilidade de série. `prediction.py` ganha `predict_series` que combina rating + modelo de draft (LogReg existente) em logit, exposto pelo CLI `predict_matchup.py`.

**Tech Stack:** Python 3, pandas, scikit-learn, xgboost, matplotlib, pytest. Sem dependências novas além de pytest.

**Spec:** `docs/superpowers/specs/2026-08-05-team-rating-series-prediction-design.md`

## Global Constraints

- Mensagens de CLI e erros em português (padrão do projeto).
- Todo módulo novo começa com `from __future__ import annotations` e usa dataclasses (padrão do projeto).
- Walk-forward estrito: a probabilidade de um jogo é calculada ANTES de o resultado atualizar o rating; o lookup de impact score usado na calibração/avaliação corta em `cutoff_date` (início da validação) para não vazar futuro.
- Rating inicial de todo time: `1500.0`. Expectativa Elo: `E = 1 / (1 + 10^(-((Ra + adv) - Rb)/400))`.
- Hiperparâmetros calibrados por busca em grade minimizando log loss na janela de validação de `chronological_series_split` (frações padrão 0.7/0.15).
- Artefatos novos: `artifacts/reports/team_ratings.csv`, `artifacts/reports/team_rating_history.csv`, `artifacts/reports/rating_model_metrics.json`, gráficos PNG em `artifacts/reports/`, e `artifacts/models/rating_config.json`.
- Testes com pytest em `tests/`; rodar com `python3 -m pytest tests/ -v` a partir da raiz do projeto.
- Commits pequenos ao final de cada task, mensagens em português com prefixo `feat:`/`test:`/`chore:`.
- Antes de escrever código de gráficos (Task 7), invocar a skill `dataviz`.

---

### Task 1: Baseline no git, pytest e inclusão da LTA Sul 2025

**Files:**
- Modify: `.gitignore`
- Modify: `src/lol_ai/pipeline/cblol.py` (função `filter_cblol_matches`, ~linha 118)
- Modify: `src/lol_ai/modeling/player_impact.py` (função `_resolve_filtered_path`, ~linha 43)
- Create: `tests/conftest.py`
- Create: `tests/test_pipeline_filter.py`

**Interfaces:**
- Produces: `should_include_row(league: str, year: str) -> bool` em `lol_ai.pipeline.cblol`; dataset processado `data/processed/cblol_game_context_dataset.csv` reconstruído com LTA S 2025 + CBLOL 2026; modelos re-treinados nesse dataset.

- [ ] **Step 1: Commitar o código existente como baseline e ignorar dados/artefatos**

```bash
cd "/Users/luizmathias/Desktop/LoL AI"
cat > .gitignore <<'EOF'
__pycache__/
*.pkl
.venv/
.DS_Store
data/
artifacts/
saida/
/*.csv
.pytest_cache/
EOF
git add -A
git commit -m "chore: baseline do projeto antes do sistema de rating"
```

- [ ] **Step 2: Instalar pytest**

Run: `python3 -m pip install pytest`
Expected: instalação sem erro; `python3 -m pytest --version` funciona.

- [ ] **Step 3: Criar conftest para colocar src/ no path**

`tests/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

- [ ] **Step 4: Escrever teste que falha para o predicado de liga**

`tests/test_pipeline_filter.py`:

```python
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
```

- [ ] **Step 5: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_pipeline_filter.py -v`
Expected: FAIL — `ImportError: cannot import name 'should_include_row'`.

- [ ] **Step 6: Implementar o predicado e usá-lo no filtro**

Em `src/lol_ai/pipeline/cblol.py`, adicionar após `is_truthy` (~linha 33):

```python
def should_include_row(league: str, year: str) -> bool:
    normalized_league = normalize_text(league).lower()
    normalized_year = normalize_text(year)
    if "cblol" in normalized_league and normalized_year in {"2025", "2026"}:
        return True
    return normalized_league == "lta s" and normalized_year == "2025"
```

E dentro de `filter_cblol_matches`, substituir o bloco:

```python
                league = normalize_text(row.get("league")).lower()
                year = normalize_text(row.get("year"))
                if "cblol" in league and year in {"2025", "2026"}:
                    filtered_rows.append(row)
```

por:

```python
                if should_include_row(row.get("league", ""), row.get("year", "")):
                    filtered_rows.append(row)
```

- [ ] **Step 7: Rodar e ver passar**

Run: `python3 -m pytest tests/test_pipeline_filter.py -v`
Expected: 6 PASS.

- [ ] **Step 8: Fazer o player_impact ler o arquivo filtrado novo (interim) em vez do legado**

Em `src/lol_ai/modeling/player_impact.py`, substituir `_resolve_filtered_path` por:

```python
def _resolve_filtered_path(input_path: Path | None = None) -> Path:
    if input_path is not None:
        return input_path
    interim = INTERIM_DATA_DIR / "cblol_esports_matches_data.csv"
    if interim.exists():
        return interim
    if LEGACY_FILTERED_FILE.exists():
        return LEGACY_FILTERED_FILE
    raise FileNotFoundError(f"Arquivo filtrado não encontrado: {interim} nem {LEGACY_FILTERED_FILE}")
```

E ajustar o import na linha 10 para incluir `INTERIM_DATA_DIR`:

```python
from lol_ai.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, LEGACY_FILTERED_FILE, PROCESSED_DATA_DIR
```

- [ ] **Step 9: Reconstruir dataset e re-treinar modelos com o histórico ampliado**

```bash
python3 scripts/filter_cblol_matches.py
python3 scripts/build_cblol_game_context_dataset.py
python3 scripts/train_cblol_models.py
```

Expected: sem erros; verificar que o dataset cresceu (antes: 191 linhas):

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/cblol_game_context_dataset.csv')
print(len(df), 'jogos'); print(df.groupby(['league','year']).size())
"
```

Expected: ~380+ jogos, com grupos `LTA S/2025` e `CBLOL/2026`.

- [ ] **Step 10: Commit**

```bash
git add src/lol_ai/pipeline/cblol.py src/lol_ai/modeling/player_impact.py tests/
git commit -m "feat: incluir LTA Sul 2025 no histórico e preparar pytest"
```

---

### Task 2: Núcleo Elo (`rating.py`)

**Files:**
- Create: `src/lol_ai/modeling/rating.py`
- Create: `tests/test_rating.py`

**Interfaces:**
- Produces: `EloConfig` (dataclass congelada com `k`, `initial_rating`, `season_carry`, `roster_regression_per_player`, `impact_scale`, `side_advantage`), `expected_score(rating_a, rating_b, advantage_a=0.0) -> float`, `elo_delta(k, result, expected) -> float`.

- [ ] **Step 1: Escrever testes que falham**

`tests/test_rating.py`:

```python
from __future__ import annotations

import pytest

from lol_ai.modeling.rating import EloConfig, elo_delta, expected_score


def test_expectativa_igual_para_ratings_iguais():
    assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)


def test_expectativa_soma_um():
    assert expected_score(1600.0, 1450.0) + expected_score(1450.0, 1600.0) == pytest.approx(1.0)


def test_favorito_tem_expectativa_maior():
    assert expected_score(1700.0, 1300.0) > 0.9


def test_vantagem_de_lado_aumenta_expectativa():
    assert expected_score(1500.0, 1500.0, advantage_a=30.0) > 0.5


def test_vitoria_esperada_rende_pouco_e_derrota_inesperada_custa_caro():
    expected = expected_score(1700.0, 1300.0)  # ~0.91
    gain_expected_win = elo_delta(32.0, 1.0, expected)
    loss_unexpected = elo_delta(32.0, 0.0, expected)
    assert 0 < gain_expected_win < 5
    assert loss_unexpected < -25


def test_config_padrao():
    config = EloConfig()
    assert config.initial_rating == 1500.0
    assert config.k > 0
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_rating.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lol_ai.modeling.rating'`.

- [ ] **Step 3: Implementar o núcleo**

`src/lol_ai/modeling/rating.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloConfig:
    k: float = 32.0
    initial_rating: float = 1500.0
    season_carry: float = 0.6
    roster_regression_per_player: float = 0.10
    impact_scale: float = 1.5
    side_advantage: float = 0.0


def expected_score(rating_a: float, rating_b: float, advantage_a: float = 0.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((rating_a + advantage_a) - rating_b) / 400.0))


def elo_delta(k: float, result: float, expected: float) -> float:
    return k * (result - expected)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python3 -m pytest tests/test_rating.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lol_ai/modeling/rating.py tests/test_rating.py
git commit -m "feat: núcleo Elo com expectativa e delta ponderado pelo adversário"
```

---

### Task 3: RatingEngine — virada de temporada e ajuste de roster

**Files:**
- Modify: `src/lol_ai/modeling/rating.py`
- Test: `tests/test_rating.py` (acrescentar)

**Interfaces:**
- Consumes: `EloConfig`, `expected_score`, `elo_delta` (Task 2).
- Produces: classe `RatingEngine(config: EloConfig, impact_lookup: dict[tuple[str, str], float] | None = None)` com:
  - `rating(team: str) -> float`
  - `process_game(*, date, league, year: int, blue_team: str, red_team: str, blue_lineup: dict[str, str], red_lineup: dict[str, str], blue_win: bool) -> float` — retorna a probabilidade pré-jogo do lado azul e atualiza o estado.
  - `history: list[dict]` — duas entradas por jogo (uma por time) com chaves: `date, league, team, opponent, side, result, rating_before, expected, delta, rating_after, season_adjustment, roster_changes, roster_adjustment`.
  - `current_ratings() -> dict[str, float]`
- O `impact_lookup` é indexado por `(nome_do_jogador_minusculo, posicao)` → impact score (0–100, neutro 50).

- [ ] **Step 1: Escrever testes que falham (acrescentar em `tests/test_rating.py`)**

```python
from lol_ai.modeling.rating import RatingEngine

LINEUP_A = {"top": "a1", "jng": "a2", "mid": "a3", "bot": "a4", "sup": "a5"}
LINEUP_B = {"top": "b1", "jng": "b2", "mid": "b3", "bot": "b4", "sup": "b5"}


def _play(engine, blue, red, blue_win, year=2026, blue_lineup=None, red_lineup=None):
    return engine.process_game(
        date="2026-01-01",
        league="CBLOL",
        year=year,
        blue_team=blue,
        red_team=red,
        blue_lineup=blue_lineup or LINEUP_A,
        red_lineup=red_lineup or LINEUP_B,
        blue_win=blue_win,
    )


def test_probabilidade_retornada_e_pre_jogo():
    engine = RatingEngine(EloConfig())
    prob = _play(engine, "FURIA", "LEV", blue_win=True)
    assert prob == pytest.approx(0.5)  # ambos 1500 antes do jogo


def test_vitorias_sobem_o_rating_do_vencedor():
    engine = RatingEngine(EloConfig())
    for _ in range(5):
        _play(engine, "FURIA", "LEV", blue_win=True)
    assert engine.rating("FURIA") > 1500.0 > engine.rating("LEV")


def test_derrota_inesperada_penaliza_mais_que_vitoria_esperada_premia():
    engine = RatingEngine(EloConfig())
    for _ in range(10):
        _play(engine, "FURIA", "LEV", blue_win=True)
    rating_before = engine.rating("FURIA")
    _play(engine, "FURIA", "LEV", blue_win=True)
    small_gain = engine.rating("FURIA") - rating_before
    rating_before = engine.rating("FURIA")
    _play(engine, "FURIA", "LEV", blue_win=False)
    big_loss = rating_before - engine.rating("FURIA")
    assert big_loss > small_gain


def test_virada_de_temporada_regride_a_media():
    engine = RatingEngine(EloConfig(season_carry=0.5))
    for _ in range(10):
        _play(engine, "FURIA", "LEV", blue_win=True, year=2025)
    rating_2025 = engine.rating("FURIA")
    _play(engine, "FURIA", "LEV", blue_win=True, year=2026)
    entry = [h for h in engine.history if h["team"] == "FURIA"][-1]
    expected_after_carry = 1500.0 + (rating_2025 - 1500.0) * 0.5
    assert entry["season_adjustment"] == pytest.approx(expected_after_carry - rating_2025)


def test_troca_de_jogador_regride_e_ajusta_por_impacto():
    lookup = {("a2", "jng"): 40.0, ("novo", "jng"): 70.0}
    engine = RatingEngine(
        EloConfig(roster_regression_per_player=0.10, impact_scale=1.0),
        impact_lookup=lookup,
    )
    for _ in range(10):
        _play(engine, "FURIA", "LEV", blue_win=True)
    rating_before = engine.rating("FURIA")
    new_lineup = dict(LINEUP_A, jng="novo")
    _play(engine, "FURIA", "LEV", blue_win=True, blue_lineup=new_lineup)
    entry = [h for h in engine.history if h["team"] == "FURIA"][-1]
    assert entry["roster_changes"] == 1
    regressed = 1500.0 + (rating_before - 1500.0) * 0.9
    expected_adjustment = (regressed - rating_before) + 1.0 * (70.0 - 40.0)
    assert entry["roster_adjustment"] == pytest.approx(expected_adjustment)


def test_primeiro_jogo_nao_conta_troca():
    engine = RatingEngine(EloConfig())
    _play(engine, "FURIA", "LEV", blue_win=True)
    entry = [h for h in engine.history if h["team"] == "FURIA"][-1]
    assert entry["roster_changes"] == 0
    assert entry["roster_adjustment"] == 0.0
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_rating.py -v`
Expected: FAIL — `ImportError: cannot import name 'RatingEngine'`.

- [ ] **Step 3: Implementar o RatingEngine (acrescentar em `rating.py`)**

```python
ROSTER_ADJUSTMENT_CAP = 60.0
NEUTRAL_IMPACT = 50.0
TEAM_POSITIONS = ("top", "jng", "mid", "bot", "sup")


class RatingEngine:
    def __init__(self, config: EloConfig, impact_lookup: dict[tuple[str, str], float] | None = None) -> None:
        self.config = config
        self.impact_lookup = impact_lookup or {}
        self.ratings: dict[str, float] = {}
        self.last_lineups: dict[str, dict[str, str]] = {}
        self.last_years: dict[str, int] = {}
        self.history: list[dict[str, object]] = []

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial_rating)

    def current_ratings(self) -> dict[str, float]:
        return dict(self.ratings)

    def _impact(self, player: str, position: str) -> float:
        return self.impact_lookup.get((player.strip().lower(), position), NEUTRAL_IMPACT)

    def _season_adjustment(self, team: str, year: int) -> float:
        last_year = self.last_years.get(team)
        if last_year is None or year <= last_year:
            return 0.0
        mean = self.config.initial_rating
        old = self.rating(team)
        new = mean + (old - mean) * self.config.season_carry
        return new - old

    def _roster_adjustment(self, team: str, lineup: dict[str, str], rating_now: float) -> tuple[int, float]:
        previous = self.last_lineups.get(team)
        if previous is None or not any(lineup.values()):
            return 0, 0.0
        changes = 0
        impact_delta = 0.0
        for position in TEAM_POSITIONS:
            new_player = (lineup.get(position) or "").strip()
            old_player = (previous.get(position) or "").strip()
            if not new_player or not old_player or new_player.lower() == old_player.lower():
                continue
            changes += 1
            impact_delta += self._impact(new_player, position) - self._impact(old_player, position)
        if changes == 0:
            return 0, 0.0
        mean = self.config.initial_rating
        keep = (1.0 - self.config.roster_regression_per_player) ** changes
        regressed = mean + (rating_now - mean) * keep
        adjustment = (regressed - rating_now) + self.config.impact_scale * impact_delta
        adjustment = max(-ROSTER_ADJUSTMENT_CAP, min(ROSTER_ADJUSTMENT_CAP, adjustment))
        return changes, adjustment

    def process_game(
        self,
        *,
        date: object,
        league: str,
        year: int,
        blue_team: str,
        red_team: str,
        blue_lineup: dict[str, str],
        red_lineup: dict[str, str],
        blue_win: bool,
    ) -> float:
        adjustments: dict[str, dict[str, float]] = {}
        for team, lineup in ((blue_team, blue_lineup), (red_team, red_lineup)):
            season_adjustment = self._season_adjustment(team, year)
            rating_after_season = self.rating(team) + season_adjustment
            roster_changes, roster_adjustment = self._roster_adjustment(team, lineup, rating_after_season)
            self.ratings[team] = rating_after_season + roster_adjustment
            adjustments[team] = {
                "season_adjustment": season_adjustment,
                "roster_changes": roster_changes,
                "roster_adjustment": roster_adjustment,
            }

        rating_blue = self.rating(blue_team)
        rating_red = self.rating(red_team)
        expected_blue = expected_score(rating_blue, rating_red, self.config.side_advantage)
        result_blue = 1.0 if blue_win else 0.0
        delta_blue = elo_delta(self.config.k, result_blue, expected_blue)

        for team, opponent, side, rating_before, expected, result, delta in (
            (blue_team, red_team, "Blue", rating_blue, expected_blue, result_blue, delta_blue),
            (red_team, blue_team, "Red", rating_red, 1.0 - expected_blue, 1.0 - result_blue, -delta_blue),
        ):
            self.ratings[team] = rating_before + delta
            self.history.append(
                {
                    "date": date,
                    "league": league,
                    "team": team,
                    "opponent": opponent,
                    "side": side,
                    "result": int(result),
                    "rating_before": round(rating_before, 2),
                    "expected": round(expected, 4),
                    "delta": round(delta, 2),
                    "rating_after": round(rating_before + delta, 2),
                    "season_adjustment": round(adjustments[team]["season_adjustment"], 2),
                    "roster_changes": int(adjustments[team]["roster_changes"]),
                    "roster_adjustment": round(adjustments[team]["roster_adjustment"], 2),
                }
            )

        self.last_lineups[blue_team] = dict(blue_lineup)
        self.last_lineups[red_team] = dict(red_lineup)
        self.last_years[blue_team] = year
        self.last_years[red_team] = year
        return expected_blue
```

Nota: `season_adjustment` e `roster_adjustment` no histórico registram o ajuste aplicado ANTES do jogo; `rating_before` é o rating já com esses ajustes (o teste de temporada compara via `season_adjustment`).

- [ ] **Step 4: Rodar e ver passar**

Run: `python3 -m pytest tests/test_rating.py -v`
Expected: 12 PASS. Se `test_troca_de_jogador_regride_e_ajusta_por_impacto` falhar por arredondamento, comparar com `pytest.approx(..., abs=0.01)` (os valores do histórico são arredondados a 2 casas).

- [ ] **Step 5: Commit**

```bash
git add src/lol_ai/modeling/rating.py tests/test_rating.py
git commit -m "feat: RatingEngine com virada de temporada e ajuste de roster"
```

---

### Task 4: Probabilidade de série (`series.py`)

**Files:**
- Create: `src/lol_ai/modeling/series.py`
- Create: `tests/test_series.py`

**Interfaces:**
- Produces: `series_probabilities(game_probs: list[float], best_of: int) -> dict` com chaves `score_probabilities` (ex.: `{"2x0": 0.3, "2x1": 0.2, "0x2": ...}` na perspectiva do time A), `a_series_win`, `b_series_win`, `most_likely_score`. `game_probs[i]` é a chance do time A no jogo `i+1`; jogos além da lista reutilizam o último valor.

- [ ] **Step 1: Escrever testes que falham**

`tests/test_series.py`:

```python
from __future__ import annotations

import pytest

from lol_ai.modeling.series import series_probabilities


def test_bo3_com_p_meio_a_meio():
    result = series_probabilities([0.5], best_of=3)
    assert result["a_series_win"] == pytest.approx(0.5)
    assert sum(result["score_probabilities"].values()) == pytest.approx(1.0)


def test_bo5_bate_com_formula_fechada():
    p, q = 0.6, 0.4
    result = series_probabilities([p], best_of=5)
    assert result["score_probabilities"]["3x0"] == pytest.approx(p**3)
    assert result["score_probabilities"]["3x1"] == pytest.approx(3 * p**3 * q)
    assert result["score_probabilities"]["3x2"] == pytest.approx(6 * p**3 * q**2)
    assert result["a_series_win"] == pytest.approx(p**3 + 3 * p**3 * q + 6 * p**3 * q**2)


def test_probabilidades_diferentes_por_jogo_sao_usadas():
    uniform = series_probabilities([0.5, 0.5, 0.5], best_of=3)
    strong_late = series_probabilities([0.5, 0.9, 0.9], best_of=3)
    assert strong_late["a_series_win"] > uniform["a_series_win"]


def test_placar_mais_provavel():
    result = series_probabilities([0.9], best_of=3)
    assert result["most_likely_score"] == "2x0"


def test_best_of_invalido():
    with pytest.raises(ValueError):
        series_probabilities([0.5], best_of=4)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_series.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar**

`src/lol_ai/modeling/series.py`:

```python
from __future__ import annotations

from collections import defaultdict


def _game_probability(game_probs: list[float], index: int) -> float:
    if not game_probs:
        raise ValueError("game_probs não pode ser vazio.")
    if index < len(game_probs):
        return game_probs[index]
    return game_probs[-1]


def series_probabilities(game_probs: list[float], best_of: int) -> dict[str, object]:
    if best_of not in {1, 3, 5}:
        raise ValueError(f"best_of inválido: {best_of}. Use 1, 3 ou 5.")
    wins_needed = best_of // 2 + 1
    scores: dict[str, float] = defaultdict(float)

    def walk(a_wins: int, b_wins: int, accumulated: float) -> None:
        if a_wins == wins_needed or b_wins == wins_needed:
            scores[f"{a_wins}x{b_wins}"] += accumulated
            return
        p = _game_probability(game_probs, a_wins + b_wins)
        walk(a_wins + 1, b_wins, accumulated * p)
        walk(a_wins, b_wins + 1, accumulated * (1.0 - p))

    walk(0, 0, 1.0)
    a_series_win = sum(
        probability
        for score, probability in scores.items()
        if int(score.split("x")[0]) == wins_needed
    )
    return {
        "score_probabilities": dict(scores),
        "a_series_win": a_series_win,
        "b_series_win": 1.0 - a_series_win,
        "most_likely_score": max(scores, key=scores.get),
    }
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python3 -m pytest tests/test_series.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lol_ai/modeling/series.py tests/test_series.py
git commit -m "feat: árvore de probabilidades de série Bo1/Bo3/Bo5"
```

---

### Task 5: Backtest walk-forward, vantagem de lado e calibração

**Files:**
- Create: `src/lol_ai/modeling/rating_backtest.py`
- Modify: `src/lol_ai/modeling/player_impact.py` (função `build_player_ratings`, ~linha 63)
- Create: `tests/test_rating_backtest.py`

**Interfaces:**
- Consumes: `RatingEngine`, `EloConfig` (Task 3); `chronological_series_split`, `load_context_dataset` (features.py); `evaluate_predictions` (training.py).
- Produces em `lol_ai.modeling.rating_backtest`:
  - `estimate_side_advantage(blue_win_rate: float) -> float`
  - `run_walk_forward(frame: pd.DataFrame, config: EloConfig, impact_lookup: dict) -> tuple[RatingEngine, pd.Series]` — a Series é indexada como o frame, com a probabilidade pré-jogo do azul.
  - `calibrate_config(frame, validation_index, impact_lookup, side_advantage) -> EloConfig`
  - `series_level_accuracy(frame, probabilities, evaluation_index) -> float | None`
- Produces em `lol_ai.modeling.player_impact`:
  - `build_player_ratings(filtered_path=None, *, frame=None, write_report=True)` (parâmetros novos, comportamento antigo preservado quando chamada sem eles)
  - `build_impact_lookup(filtered_path=None, cutoff_date=None) -> dict[tuple[str, str], float]`

- [ ] **Step 1: Escrever testes que falham**

`tests/test_rating_backtest.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lol_ai.modeling.rating import EloConfig
from lol_ai.modeling.rating_backtest import (
    calibrate_config,
    estimate_side_advantage,
    run_walk_forward,
    series_level_accuracy,
)


def _synthetic_frame(n_games: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n_games):
        strong_is_blue = i % 2 == 0
        rows.append(
            {
                "series_id": f"s{i // 2}",
                "game_number": (i % 2) + 1,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "league": "CBLOL",
                "year": 2026,
                "blue_team": "FORTE" if strong_is_blue else "FRACO",
                "red_team": "FRACO" if strong_is_blue else "FORTE",
                "blue_win": 1 if strong_is_blue else 0,
                "winner_team": "FORTE",
                **{f"blue_{p}_player": f"b_{p}" for p in ("top", "jng", "mid", "bot", "sup")},
                **{f"red_{p}_player": f"r_{p}" for p in ("top", "jng", "mid", "bot", "sup")},
            }
        )
    return pd.DataFrame(rows)


def test_vantagem_de_lado_neutra_e_positiva():
    assert estimate_side_advantage(0.5) == pytest.approx(0.0)
    assert estimate_side_advantage(0.55) > 0.0
    assert estimate_side_advantage(0.45) < 0.0


def test_walk_forward_cobre_todos_os_jogos_e_aprende():
    frame = _synthetic_frame()
    engine, probs = run_walk_forward(frame, EloConfig(), impact_lookup={})
    assert len(probs) == len(frame)
    assert probs.index.equals(frame.index)
    assert engine.rating("FORTE") > engine.rating("FRACO")
    # último jogo com FORTE no azul: o modelo já deve favorecer o FORTE
    last_strong_blue = frame[frame["blue_team"] == "FORTE"].index[-1]
    assert probs.loc[last_strong_blue] > 0.6


def test_probabilidade_e_pre_jogo():
    frame = _synthetic_frame(2)
    _, probs = run_walk_forward(frame, EloConfig(), impact_lookup={})
    assert probs.iloc[0] == pytest.approx(0.5)


def test_calibracao_retorna_config_com_vantagem():
    frame = _synthetic_frame()
    validation_index = frame.index[-10:]
    config = calibrate_config(frame, validation_index, impact_lookup={}, side_advantage=10.0)
    assert isinstance(config, EloConfig)
    assert config.side_advantage == pytest.approx(10.0)


def test_acuracia_por_serie():
    frame = _synthetic_frame(8)
    probs = pd.Series([0.9 if row.blue_team == "FORTE" else 0.1 for row in frame.itertuples()], index=frame.index)
    accuracy = series_level_accuracy(frame, probs, frame.index)
    assert accuracy == pytest.approx(1.0)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_rating_backtest.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Generalizar `build_player_ratings` e criar `build_impact_lookup`**

Em `src/lol_ai/modeling/player_impact.py`, mudar a assinatura (~linha 63) para:

```python
def build_player_ratings(
    filtered_path: Path | None = None,
    *,
    frame: pd.DataFrame | None = None,
    write_report: bool = True,
) -> pd.DataFrame:
    if frame is None:
        source_path = _resolve_filtered_path(filtered_path)
        frame = pd.read_csv(source_path)
    frame = frame[frame["position"].isin(TEAM_POSITIONS)].copy()
```

e envolver a escrita do CSV no final com o flag:

```python
    if write_report:
        PLAYER_RATINGS_REPORT.parent.mkdir(parents=True, exist_ok=True)
        aggregated.sort_values(["impact_score", "games"], ascending=[False, False]).to_csv(PLAYER_RATINGS_REPORT, index=False)
    return aggregated.sort_values(["impact_score", "games"], ascending=[False, False]).reset_index(drop=True)
```

Adicionar no final do arquivo:

```python
def build_impact_lookup(
    filtered_path: Path | None = None,
    cutoff_date: object | None = None,
) -> dict[tuple[str, str], float]:
    source_path = _resolve_filtered_path(filtered_path)
    frame = pd.read_csv(source_path)
    if cutoff_date is not None:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") < pd.Timestamp(cutoff_date)]
    frame = frame[frame["position"].isin(TEAM_POSITIONS)]
    if frame.empty:
        return {}
    ratings = build_player_ratings(frame=frame.copy(), write_report=False)
    return {
        (str(row["playername"]).strip().lower(), str(row["position"]).strip().lower()): float(row["impact_score"])
        for _, row in ratings.iterrows()
    }
```

- [ ] **Step 4: Implementar o backtest**

`src/lol_ai/modeling/rating_backtest.py`:

```python
from __future__ import annotations

import math
from itertools import product

import pandas as pd
from sklearn.metrics import log_loss

from lol_ai.modeling.rating import TEAM_POSITIONS, EloConfig, RatingEngine

K_GRID = (16.0, 24.0, 32.0, 40.0)
SEASON_CARRY_GRID = (0.4, 0.6, 0.8, 1.0)
ROSTER_REGRESSION_GRID = (0.0, 0.05, 0.10, 0.20)
IMPACT_SCALE_GRID = (0.0, 1.0, 2.0)


def estimate_side_advantage(blue_win_rate: float) -> float:
    clamped = min(max(blue_win_rate, 0.05), 0.95)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def _lineup_from_row(row: pd.Series, prefix: str) -> dict[str, str]:
    return {
        position: str(row.get(f"{prefix}_{position}_player") or "").strip()
        for position in TEAM_POSITIONS
    }


def run_walk_forward(
    frame: pd.DataFrame,
    config: EloConfig,
    impact_lookup: dict[tuple[str, str], float],
) -> tuple[RatingEngine, pd.Series]:
    engine = RatingEngine(config, impact_lookup)
    ordered = frame.sort_values(["date", "series_id", "game_number"])
    probabilities: dict[object, float] = {}
    for index, row in ordered.iterrows():
        probabilities[index] = engine.process_game(
            date=row["date"],
            league=str(row.get("league", "")),
            year=int(row["year"]),
            blue_team=str(row["blue_team"]),
            red_team=str(row["red_team"]),
            blue_lineup=_lineup_from_row(row, "blue"),
            red_lineup=_lineup_from_row(row, "red"),
            blue_win=bool(int(row["blue_win"])),
        )
    return engine, pd.Series(probabilities).reindex(frame.index)


def calibrate_config(
    frame: pd.DataFrame,
    validation_index: pd.Index,
    impact_lookup: dict[tuple[str, str], float],
    side_advantage: float,
) -> EloConfig:
    best_config: EloConfig | None = None
    best_loss = float("inf")
    y_validation = frame.loc[validation_index, "blue_win"].astype(int)
    for k, carry, roster, impact in product(K_GRID, SEASON_CARRY_GRID, ROSTER_REGRESSION_GRID, IMPACT_SCALE_GRID):
        config = EloConfig(
            k=k,
            season_carry=carry,
            roster_regression_per_player=roster,
            impact_scale=impact,
            side_advantage=side_advantage,
        )
        _, probabilities = run_walk_forward(frame, config, impact_lookup)
        loss = float(log_loss(y_validation, probabilities.loc[validation_index], labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_config = config
    assert best_config is not None
    return best_config


def series_level_accuracy(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    evaluation_index: pd.Index,
) -> float | None:
    rows = frame.loc[evaluation_index].copy()
    rows["blue_prob"] = probabilities.loc[evaluation_index]
    hits: list[bool] = []
    for _, group in rows.groupby("series_id"):
        ordered = group.sort_values("game_number")
        first = ordered.iloc[0]
        predicted = first["blue_team"] if first["blue_prob"] >= 0.5 else first["red_team"]
        actual = ordered["winner_team"].value_counts().idxmax()
        hits.append(predicted == actual)
    if not hits:
        return None
    return float(sum(hits) / len(hits))
```

Nota: o teste sintético não tem a coluna `winner_team` faltando — já está no frame sintético. `calibrate_config` roda 192 walk-forwards; no dataset real (~400 jogos) isso leva segundos.

- [ ] **Step 5: Rodar e ver passar**

Run: `python3 -m pytest tests/test_rating_backtest.py tests/test_rating.py -v`
Expected: todos PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lol_ai/modeling/rating_backtest.py src/lol_ai/modeling/player_impact.py tests/test_rating_backtest.py
git commit -m "feat: backtest walk-forward com calibração e vantagem de lado"
```

---

### Task 6: Combinação rating + draft em logit

**Files:**
- Modify: `src/lol_ai/modeling/rating_backtest.py`
- Test: `tests/test_rating_backtest.py` (acrescentar)

**Interfaces:**
- Produces: `blend_probabilities(p_rating: float, p_draft: float, weight: float) -> float`, `fit_draft_weight(p_rating: pd.Series, p_draft: pd.Series, y_true: pd.Series) -> float` (grade 0.0–1.0 passo 0.1, minimiza log loss), `draft_model_probabilities(frame: pd.DataFrame, index: pd.Index) -> pd.Series` (usa os artefatos LogReg já treinados).

- [ ] **Step 1: Escrever testes que falham (acrescentar em `tests/test_rating_backtest.py`)**

```python
import numpy as np

from lol_ai.modeling.rating_backtest import blend_probabilities, fit_draft_weight


def test_blend_peso_zero_devolve_rating():
    assert blend_probabilities(0.7, 0.2, 0.0) == pytest.approx(0.7)


def test_blend_draft_neutro_nao_muda():
    assert blend_probabilities(0.7, 0.5, 1.0) == pytest.approx(0.7)


def test_blend_draft_favoravel_aumenta():
    assert blend_probabilities(0.6, 0.8, 1.0) > 0.6


def test_fit_draft_weight_ignora_draft_ruidoso():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.integers(0, 2, size=200))
    p_rating = pd.Series([0.8 if value == 1 else 0.2 for value in y], index=y.index)
    p_draft = pd.Series(rng.uniform(0.05, 0.95, size=200), index=y.index)
    assert fit_draft_weight(p_rating, p_draft, y) == pytest.approx(0.0)


def test_fit_draft_weight_usa_draft_informativo():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.integers(0, 2, size=200))
    p_rating = pd.Series(0.5, index=y.index)
    p_draft = pd.Series([0.85 if value == 1 else 0.15 for value in y], index=y.index)
    assert fit_draft_weight(p_rating, p_draft, y) > 0.5
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_rating_backtest.py -v`
Expected: FAIL — nomes não definidos.

- [ ] **Step 3: Implementar (acrescentar em `rating_backtest.py`)**

```python
def _logit(probability: float) -> float:
    clamped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clamped / (1.0 - clamped))


def blend_probabilities(p_rating: float, p_draft: float, weight: float) -> float:
    combined = _logit(p_rating) + weight * (_logit(p_draft) - _logit(0.5))
    return 1.0 / (1.0 + math.exp(-combined))


def fit_draft_weight(p_rating: pd.Series, p_draft: pd.Series, y_true: pd.Series) -> float:
    best_weight = 0.0
    best_loss = float("inf")
    for step in range(11):
        weight = step / 10.0
        blended = [
            blend_probabilities(rating_prob, draft_prob, weight)
            for rating_prob, draft_prob in zip(p_rating, p_draft)
        ]
        loss = float(log_loss(y_true, blended, labels=[0, 1]))
        if loss < best_loss - 1e-9:
            best_loss = loss
            best_weight = weight
    return best_weight


def draft_model_probabilities(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
    import json
    import pickle

    from lol_ai.config import MODEL_ARTIFACTS_DIR
    from lol_ai.modeling.features import build_feature_frame

    with (MODEL_ARTIFACTS_DIR / "cblol_preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    with (MODEL_ARTIFACTS_DIR / "cblol_logistic_regression.pkl").open("rb") as handle:
        model = pickle.load(handle)
    features = build_feature_frame(frame.loc[index])
    transformed = preprocessor.transform(features)
    return pd.Series(model.predict_proba(transformed)[:, 1], index=index)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python3 -m pytest tests/test_rating_backtest.py -v`
Expected: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lol_ai/modeling/rating_backtest.py tests/test_rating_backtest.py
git commit -m "feat: combinação em logit do rating com o modelo de draft"
```

---

### Task 7: Orquestração do backtest, relatórios, gráficos e script

**Files:**
- Modify: `src/lol_ai/modeling/rating_backtest.py`
- Modify: `src/lol_ai/config.py`
- Modify: `main.py` (acrescentar etapa ao pipeline)
- Create: `scripts/build_team_ratings.py`
- Test: `tests/test_rating_reports.py`

**Interfaces:**
- Consumes: tudo das Tasks 3–6; `evaluate_predictions` de `training.py`; `build_impact_lookup` de `player_impact.py`; `chronological_series_split`/`load_context_dataset` de `features.py`.
- Produces: `run_rating_backtest(data_path: Path | None = None) -> dict` que grava todos os artefatos e devolve o payload de métricas; constante `RATING_CONFIG_FILE` em `config.py` (= `MODEL_ARTIFACTS_DIR / "rating_config.json"`).

- [ ] **Step 1: Invocar a skill `dataviz`** antes de escrever o código dos gráficos, e seguir as diretrizes dela nos PNGs abaixo (pode ajustar cores/estilo do código de exemplo conforme a skill).

- [ ] **Step 2: Escrever teste de integração que falha**

`tests/test_rating_reports.py`:

```python
from __future__ import annotations

import json

import pytest

from lol_ai.config import REPORT_ARTIFACTS_DIR, RATING_CONFIG_FILE
from lol_ai.modeling.rating_backtest import run_rating_backtest


@pytest.mark.integration
def test_backtest_completo_gera_artefatos():
    payload = run_rating_backtest()

    assert (REPORT_ARTIFACTS_DIR / "team_ratings.csv").exists()
    assert (REPORT_ARTIFACTS_DIR / "team_rating_history.csv").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_model_metrics.json").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_confusion_matrix.png").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_metrics_comparison.png").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_calibration.png").exists()
    assert RATING_CONFIG_FILE.exists()

    assert 0.0 <= payload["rating"]["test"]["log_loss"]
    assert payload["rating"]["test"]["accuracy"] > 0.0
    config = json.loads(RATING_CONFIG_FILE.read_text(encoding="utf-8"))
    assert "config" in config and "draft_weight" in config
```

Registrar o marker em `pytest.ini` (Create):

```ini
[pytest]
markers =
    integration: testes que usam o dataset real e escrevem artefatos
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_rating_reports.py -v`
Expected: FAIL — `ImportError` (`RATING_CONFIG_FILE` / `run_rating_backtest` inexistentes).

- [ ] **Step 4: Adicionar constante em `config.py`**

```python
RATING_CONFIG_FILE = MODEL_ARTIFACTS_DIR / "rating_config.json"
```

- [ ] **Step 5: Implementar `run_rating_backtest` (acrescentar em `rating_backtest.py`)**

```python
def run_rating_backtest(data_path=None) -> dict:
    import json
    from dataclasses import asdict

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lol_ai.config import RATING_CONFIG_FILE, REPORT_ARTIFACTS_DIR
    from lol_ai.modeling.features import chronological_series_split, load_context_dataset
    from lol_ai.modeling.player_impact import build_impact_lookup
    from lol_ai.modeling.training import evaluate_predictions

    frame = load_context_dataset(data_path)
    train_index, validation_index, test_index = chronological_series_split(frame)

    train_blue_win_rate = float(frame.loc[train_index, "blue_win"].astype(int).mean())
    side_advantage = estimate_side_advantage(train_blue_win_rate)

    validation_start = frame.loc[validation_index, "date"].min()
    impact_lookup = build_impact_lookup(cutoff_date=validation_start)

    best_config = calibrate_config(frame, validation_index, impact_lookup, side_advantage)
    engine, probabilities = run_walk_forward(frame, best_config, impact_lookup)

    y_validation = frame.loc[validation_index, "blue_win"].astype(int)
    y_test = frame.loc[test_index, "blue_win"].astype(int)
    metrics_asdict = asdict

    p_draft_validation = draft_model_probabilities(frame, validation_index)
    p_draft_test = draft_model_probabilities(frame, test_index)
    draft_weight = fit_draft_weight(probabilities.loc[validation_index], p_draft_validation, y_validation)
    blended_test = pd.Series(
        [
            blend_probabilities(rating_prob, draft_prob, draft_weight)
            for rating_prob, draft_prob in zip(probabilities.loc[test_index], p_draft_test)
        ],
        index=test_index,
    )

    payload = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "test_rows": int(len(test_index)),
        "side_advantage": side_advantage,
        "train_blue_win_rate": train_blue_win_rate,
        "config": asdict(best_config),
        "draft_weight": draft_weight,
        "rating": {
            "validation": metrics_asdict(evaluate_predictions(y_validation, probabilities.loc[validation_index].to_numpy())),
            "test": metrics_asdict(evaluate_predictions(y_test, probabilities.loc[test_index].to_numpy())),
            "test_series_accuracy": series_level_accuracy(frame, probabilities, test_index),
        },
        "rating_plus_draft": {
            "test": metrics_asdict(evaluate_predictions(y_test, blended_test.to_numpy())),
        },
    }

    baseline_file = REPORT_ARTIFACTS_DIR / "cblol_model_metrics.json"
    if baseline_file.exists():
        payload["baselines"] = json.loads(baseline_file.read_text(encoding="utf-8"))

    REPORT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ratings_frame = (
        pd.DataFrame(
            [{"team": team, "rating": round(rating, 1)} for team, rating in engine.current_ratings().items()]
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    ratings_frame.to_csv(REPORT_ARTIFACTS_DIR / "team_ratings.csv", index=False)
    pd.DataFrame(engine.history).to_csv(REPORT_ARTIFACTS_DIR / "team_rating_history.csv", index=False)

    with (REPORT_ARTIFACTS_DIR / "rating_model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    RATING_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RATING_CONFIG_FILE.open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(best_config), "draft_weight": draft_weight}, handle, indent=2, ensure_ascii=False)

    _plot_reports(payload, y_test, probabilities.loc[test_index], REPORT_ARTIFACTS_DIR, plt)
    return payload
```

E a função de gráficos (mesmo arquivo; ajustar estética conforme a skill dataviz):

```python
def _plot_reports(payload, y_test, p_test, output_dir, plt) -> None:
    import numpy as np

    # 1. Matriz de confusão do rating no teste
    matrix = np.array(payload["rating"]["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["Prev. Red", "Prev. Blue"])
    ax.set_yticks([0, 1], ["Red venceu", "Blue venceu"])
    ax.set_title("Rating — matriz de confusão (teste)")
    fig.tight_layout()
    fig.savefig(output_dir / "rating_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2. Comparação de métricas: rating vs rating+draft vs baselines
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"]
    systems = {"rating": payload["rating"]["test"], "rating+draft": payload["rating_plus_draft"]["test"]}
    baselines = payload.get("baselines", {})
    for name in ("logistic_regression", "xgboost"):
        if name in baselines:
            systems[name] = baselines[name]["test"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_names))
    width = 0.8 / len(systems)
    for offset, (label, metrics) in enumerate(systems.items()):
        values = [metrics.get(metric, float("nan")) for metric in metric_names]
        ax.bar(x + offset * width, values, width, label=label)
    ax.set_xticks(x + width * (len(systems) - 1) / 2, metric_names, rotation=20)
    ax.set_title("Métricas no teste — rating vs modelos anteriores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rating_metrics_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Calibração + distribuição de probabilidades
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0.0, 1.0, 6)
    centers = (bins[:-1] + bins[1:]) / 2
    observed = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (p_test >= low) & (p_test < high)
        observed.append(float(y_test[mask].mean()) if mask.any() else float("nan"))
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].plot(centers, observed, marker="o")
    axes[0].set_title("Calibração (teste)")
    axes[0].set_xlabel("Probabilidade prevista")
    axes[0].set_ylabel("Frequência observada")
    axes[1].hist(p_test, bins=20)
    axes[1].set_title("Distribuição das probabilidades")
    fig.tight_layout()
    fig.savefig(output_dir / "rating_calibration.png", dpi=150)
    plt.close(fig)
```

- [ ] **Step 6: Rodar o teste de integração e ver passar**

Run: `python3 -m pytest tests/test_rating_reports.py -v -m integration`
Expected: PASS (leva alguns segundos por causa da grade de calibração).

- [ ] **Step 7: Criar o script CLI e plugar no main.py**

`scripts/build_team_ratings.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_src_on_path

PROJECT_ROOT = ensure_src_on_path()

from lol_ai.config import REPORT_ARTIFACTS_DIR  # noqa: E402
from lol_ai.modeling.rating_backtest import run_rating_backtest  # noqa: E402


def main() -> None:
    payload = run_rating_backtest()
    print("Backtest de rating concluído.")
    print(f"Config calibrada: {payload['config']}")
    print(f"Peso do draft: {payload['draft_weight']}")
    print(f"Vantagem de lado (azul): {payload['side_advantage']:+.1f} pontos")
    print("Métricas no teste (rating):")
    for key, value in payload["rating"]["test"].items():
        print(f"- {key}: {value}")
    print(f"Acerto por série (teste): {payload['rating']['test_series_accuracy']}")
    print(f"Relatórios em: {REPORT_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
```

Em `main.py`, acrescentar ao final de `run_pipeline()` (depois dos gráficos):

```python
    from lol_ai.modeling.rating_backtest import run_rating_backtest

    rating_payload = run_rating_backtest(processed_output)
    print("Backtest de rating concluído")
    print(rating_payload["rating"]["test"])
```

- [ ] **Step 8: Rodar de verdade e auditar o resultado**

```bash
python3 scripts/build_team_ratings.py
python3 -c "
import pandas as pd
print(pd.read_csv('artifacts/reports/team_ratings.csv'))
history = pd.read_csv('artifacts/reports/team_rating_history.csv')
print(history[history.team == 'FURIA'].tail(12)[['date','opponent','result','expected','delta','rating_after','roster_changes']])
"
```

Expected: ranking plausível (LEV/LØS embaixo se estiverem mal na liga); nas linhas da FURIA, vitória contra time fraco com `expected` alto → `delta` pequeno; derrota com `expected` alto → `delta` bem negativo. Verificar também o exemplo do spec: derrota para o time de menor rating deve ter o maior `|delta|` da tabela.

- [ ] **Step 9: Rodar todos os testes**

Run: `python3 -m pytest tests/ -v`
Expected: todos PASS.

- [ ] **Step 10: Commit**

```bash
git add src/lol_ai/modeling/rating_backtest.py src/lol_ai/config.py main.py scripts/build_team_ratings.py tests/test_rating_reports.py pytest.ini
git commit -m "feat: backtest completo com relatórios, gráficos e script de ratings"
```

---

### Task 8: `predict_series` e CLI atualizado

**Files:**
- Modify: `src/lol_ai/modeling/prediction.py` (acrescentar; manter `predict_matchup` antigo)
- Modify: `predict_matchup.py` (raiz)
- Modify: `scripts/predict_matchup.py`
- Test: `tests/test_predict_series.py`

**Interfaces:**
- Consumes: `RatingEngine`, `EloConfig` (Task 3); `series_probabilities` (Task 4); `run_walk_forward`, `blend_probabilities`, `draft_model_probabilities` — reusar padrão de carga de artefatos de `_load_artifacts` (prediction.py:47); `RATING_CONFIG_FILE` (Task 7); `build_impact_lookup` (Task 5).
- Produces: dataclass `SeriesPrediction` e função:

```python
def predict_series(
    blue_team: str,
    red_team: str,
    best_of: int = 3,
    blue_picks: list[str] | None = None,
    red_picks: list[str] | None = None,
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    data_path: Path | None = None,
) -> SeriesPrediction
```

- [ ] **Step 1: Escrever testes que falham**

`tests/test_predict_series.py`:

```python
from __future__ import annotations

import pytest

from lol_ai.config import RATING_CONFIG_FILE
from lol_ai.modeling.prediction import predict_series

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _requer_config():
    if not RATING_CONFIG_FILE.exists():
        pytest.skip("rodar scripts/build_team_ratings.py antes")


def test_predict_series_bo3():
    prediction = predict_series("FURIA", "paiN Gaming", best_of=3)
    assert prediction.best_of == 3
    assert prediction.series_win_probability_blue + prediction.series_win_probability_red == pytest.approx(1.0)
    assert len(prediction.game_probabilities) == 3
    assert prediction.game_probabilities[0]["used_draft"] is False
    assert 0.0 < prediction.game_probabilities[0]["blue_win_probability"] < 1.0


def test_predict_series_com_draft_marca_jogo_1():
    picks = ["Aatrox", "Maokai", "Ahri", "Jinx", "Rakan"]
    prediction = predict_series("FURIA", "paiN Gaming", best_of=3, blue_picks=picks, red_picks=picks)
    assert prediction.game_probabilities[0]["used_draft"] is True
    assert prediction.game_probabilities[1]["used_draft"] is False


def test_time_desconhecido_da_erro_claro():
    with pytest.raises(ValueError, match="Time desconhecido"):
        predict_series("Time Inventado", "FURIA")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python3 -m pytest tests/test_predict_series.py -v`
Expected: FAIL — `ImportError: cannot import name 'predict_series'`.

- [ ] **Step 3: Implementar em `prediction.py`**

Acrescentar imports no topo (junto aos existentes):

```python
from lol_ai.config import RATING_CONFIG_FILE
from lol_ai.modeling.rating import EloConfig, RatingEngine, expected_score
from lol_ai.modeling.rating_backtest import blend_probabilities, run_walk_forward
from lol_ai.modeling.series import series_probabilities
from lol_ai.modeling.player_impact import build_impact_lookup
```

E ao final do arquivo:

```python
@dataclass(frozen=True)
class SeriesPrediction:
    blue_team: str
    red_team: str
    blue_rating: float
    red_rating: float
    best_of: int
    game_probabilities: list[dict[str, Any]]
    series_win_probability_blue: float
    series_win_probability_red: float
    score_probabilities: dict[str, float]
    most_likely_score: str
    side_advantage: float
    draft_weight: float


def _load_rating_setup() -> tuple[EloConfig, float]:
    if not RATING_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "Configuração de rating não encontrada. Rode antes: python3 scripts/build_team_ratings.py"
        )
    payload = json.loads(RATING_CONFIG_FILE.read_text(encoding="utf-8"))
    return EloConfig(**payload["config"]), float(payload["draft_weight"])


def _draft_probability(
    frame: pd.DataFrame,
    blue_team: str,
    red_team: str,
    blue_picks: list[str],
    red_picks: list[str],
    blue_bans: list[str],
    red_bans: list[str],
) -> float:
    hypothetical = _build_hypothetical_row(frame, blue_team, red_team)
    hypothetical.loc[:, "blue_picks"] = "; ".join(blue_picks)
    hypothetical.loc[:, "red_picks"] = "; ".join(red_picks)
    hypothetical.loc[:, "blue_bans"] = "; ".join(blue_bans)
    hypothetical.loc[:, "red_bans"] = "; ".join(red_bans)
    preprocessor, _, logistic_model = _load_artifacts()
    transformed = preprocessor.transform(build_feature_frame(hypothetical))
    return float(logistic_model.predict_proba(transformed)[:, 1][0])


def predict_series(
    blue_team: str,
    red_team: str,
    best_of: int = 3,
    blue_picks: list[str] | None = None,
    red_picks: list[str] | None = None,
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    data_path: Path | None = None,
) -> SeriesPrediction:
    blue_team = normalize_text(blue_team)
    red_team = normalize_text(red_team)
    config, draft_weight = _load_rating_setup()

    frame = load_context_dataset(_resolve_dataset_path(data_path))
    known_teams = sorted(set(frame["blue_team"]) | set(frame["red_team"]))
    for team in (blue_team, red_team):
        if team not in known_teams:
            raise ValueError(f"Time desconhecido: {team}. Times disponíveis: {', '.join(known_teams)}")

    impact_lookup = build_impact_lookup()
    engine, _ = run_walk_forward(frame, config, impact_lookup)
    blue_rating = engine.rating(blue_team)
    red_rating = engine.rating(red_team)

    game1_rating_prob = expected_score(blue_rating, red_rating, config.side_advantage)
    neutral_prob = expected_score(blue_rating, red_rating, 0.0)

    used_draft = bool(blue_picks and red_picks)
    if used_draft:
        p_draft = _draft_probability(
            frame, blue_team, red_team, blue_picks or [], red_picks or [], blue_bans or [], red_bans or []
        )
        game1_prob = blend_probabilities(game1_rating_prob, p_draft, draft_weight)
    else:
        game1_prob = game1_rating_prob

    game_probs = [game1_prob] + [neutral_prob] * (best_of - 1)
    series = series_probabilities(game_probs, best_of)

    game_probabilities = [
        {
            "game": index + 1,
            "blue_win_probability": probability,
            "red_win_probability": 1.0 - probability,
            "used_draft": used_draft and index == 0,
        }
        for index, probability in enumerate(game_probs)
    ]

    return SeriesPrediction(
        blue_team=blue_team,
        red_team=red_team,
        blue_rating=round(blue_rating, 1),
        red_rating=round(red_rating, 1),
        best_of=best_of,
        game_probabilities=game_probabilities,
        series_win_probability_blue=series["a_series_win"],
        series_win_probability_red=series["b_series_win"],
        score_probabilities=series["score_probabilities"],
        most_likely_score=series["most_likely_score"],
        side_advantage=config.side_advantage,
        draft_weight=draft_weight,
    )
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python3 -m pytest tests/test_predict_series.py -v`
Expected: 3 PASS (requer artefatos da Task 7 já gerados).

- [ ] **Step 5: Atualizar o CLI da raiz (`predict_matchup.py`)**

Substituir a função `main` por:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Prever série do CBLOL com rating por adversário e draft opcional.")
    parser.add_argument("blue_team", help="Time no lado azul no jogo 1")
    parser.add_argument("red_team", help="Time no lado vermelho no jogo 1")
    parser.add_argument("--best-of", type=int, choices=(1, 3, 5), default=3, help="Formato da série (padrão: Bo3)")
    parser.add_argument("--blue-picks", help="Picks do azul no jogo 1, separados por vírgula")
    parser.add_argument("--red-picks", help="Picks do vermelho no jogo 1, separados por vírgula")
    parser.add_argument("--blue-bans", help="Bans do azul no jogo 1, separados por vírgula")
    parser.add_argument("--red-bans", help="Bans do vermelho no jogo 1, separados por vírgula")
    args = parser.parse_args()

    def split_champions(value: str | None) -> list[str] | None:
        if not value:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]

    from lol_ai.modeling.prediction import predict_series

    prediction = predict_series(
        args.blue_team,
        args.red_team,
        best_of=args.best_of,
        blue_picks=split_champions(args.blue_picks),
        red_picks=split_champions(args.red_picks),
        blue_bans=split_champions(args.blue_bans),
        red_bans=split_champions(args.red_bans),
    )

    print(f"Confronto (Bo{prediction.best_of}): {prediction.blue_team} vs {prediction.red_team}")
    print(f"Ratings: {prediction.blue_team} {prediction.blue_rating} | {prediction.red_team} {prediction.red_rating}")
    print(f"Vantagem de lado azul: {prediction.side_advantage:+.1f} pontos | Peso do draft: {prediction.draft_weight}")
    print(f"\nChance de vencer a série: {prediction.blue_team} {prediction.series_win_probability_blue:.1%} | {prediction.red_team} {prediction.series_win_probability_red:.1%}")
    print("\nChance por jogo:")
    for game in prediction.game_probabilities:
        draft_note = " (com draft)" if game["used_draft"] else ""
        print(f"- Jogo {game['game']}{draft_note}: {prediction.blue_team} {game['blue_win_probability']:.1%} | {prediction.red_team} {game['red_win_probability']:.1%}")
    print(f"\nPlacar mais provável ({prediction.blue_team} x {prediction.red_team}): {prediction.most_likely_score}")
    print("Distribuição de placares:")
    for score, probability in sorted(prediction.score_probabilities.items(), key=lambda item: -item[1]):
        print(f"- {score}: {probability:.1%}")


if __name__ == "__main__":
    main()
```

(Remover os imports/funções não usados que sobrarem no arquivo da raiz; o import de `predict_matchup` antigo pode sair.)

- [ ] **Step 6: Atualizar `scripts/predict_matchup.py` com o mesmo conteúdo**

Mesmo `main()` do Step 5, mantendo o cabeçalho de bootstrap existente do script (`ensure_src_on_path`).

- [ ] **Step 7: Testar o CLI de ponta a ponta**

```bash
python3 predict_matchup.py "FURIA" "paiN Gaming" --best-of 3
python3 predict_matchup.py "FURIA" "paiN Gaming" --best-of 5 --blue-picks "Aatrox,Maokai,Ahri,Jinx,Rakan" --red-picks "K'Sante,Vi,Azir,Ezreal,Alistar"
python3 predict_matchup.py "Time Inventado" "FURIA"; echo "exit: $?"
```

Expected: dois primeiros imprimem série/jogos/placares coerentes (probabilidades entre 0 e 1, jogo 1 marcado "(com draft)" no segundo); o terceiro falha com "Time desconhecido" listando os times.

- [ ] **Step 8: Rodar todos os testes**

Run: `python3 -m pytest tests/ -v`
Expected: todos PASS.

- [ ] **Step 9: Commit**

```bash
git add src/lol_ai/modeling/prediction.py predict_matchup.py scripts/predict_matchup.py tests/test_predict_series.py
git commit -m "feat: predict_series com rating, draft opcional e CLI de série"
```

---

### Task 9: Verificação final e critério de aceite

**Files:**
- Nenhum arquivo novo (correções pontuais se algo falhar).

- [ ] **Step 1: Rodar a suíte completa**

Run: `python3 -m pytest tests/ -v`
Expected: todos PASS.

- [ ] **Step 2: Rodar o pipeline completo do zero**

Run: `python3 main.py`
Expected: filtro → dataset → treino → SHAP → impacto de jogadores → gráficos → backtest de rating, sem erros.

- [ ] **Step 3: Conferir o critério de aceite do spec**

```bash
python3 -c "
import json
payload = json.load(open('artifacts/reports/rating_model_metrics.json'))
rating = payload['rating']['test']
baseline = payload['baselines']['logistic_regression']['test']
print('rating   :', {k: round(rating[k], 3) for k in ('accuracy','roc_auc','log_loss','brier','f1')})
print('logreg   :', {k: round(baseline[k], 3) for k in ('accuracy','roc_auc','log_loss','brier','f1')})
print('r+draft  :', {k: round(payload['rating_plus_draft']['test'][k], 3) for k in ('accuracy','roc_auc','log_loss','brier','f1')})
print('série acc:', payload['rating']['test_series_accuracy'])
"
```

Expected (critério do spec): métricas do rating ≥ LogReg no teste (foco em log_loss e roc_auc). Se o rating NÃO superar a LogReg, investigar com a skill superpowers:systematic-debugging antes de aceitar — checar vazamento, ordenação por data e grades de calibração — e reportar o resultado real ao usuário sem maquiar.

- [ ] **Step 4: Auditoria qualitativa (exemplo do usuário)**

```bash
python3 -c "
import pandas as pd
history = pd.read_csv('artifacts/reports/team_rating_history.csv')
cb = history[history.league == 'CBLOL']
print(pd.read_csv('artifacts/reports/team_ratings.csv'))
upsets = cb[(cb.result == 0) & (cb.expected > 0.65)]
print('Derrotas mais inesperadas (maior punição):')
print(upsets.sort_values('delta').head(8)[['date','team','opponent','expected','delta']])
"
```

Expected: as maiores punições são derrotas de favoritos para times de rating baixo — conferir se faz sentido com a temporada real (ex.: favorito perdendo para LEV/LØS).

- [ ] **Step 5: Commit final e resumo**

```bash
git add -A
git commit -m "chore: verificação final do sistema de rating e previsão de séries"
```

Reportar ao usuário: tabela final de ratings, métricas comparadas (rating vs LogReg vs XGBoost), peso do draft calibrado e exemplos de pontos ganhos/perdidos (caso FURIA/LEV).
