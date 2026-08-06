# League of Legends AI Predictor (CBLOL)

Sistema de previsão de partidas e séries do CBLOL que combina um **rating de força por
time ponderado pelo adversário** (método estilo Elo) com modelos de machine learning
sobre drafts. Responde três perguntas:

1. **Qual a chance de um time vencer a série?** (Bo1 / Bo3 / Bo5, com distribuição de placares)
2. **Qual a chance em cada jogo (1, 2, 3...)**, usando o draft quando ele já é conhecido?
3. **Quão forte está cada time agora?** — considerando contra quem ganhou/perdeu e as
   trocas de jogadores no elenco.

> Documentação técnica detalhada: [docs/modelo.md](docs/modelo.md)

## Como funciona

### 1. Rating de força ponderado pelo adversário

Cada time tem uma pontuação (todos começam em 1500). Após cada jogo:

```
rating += K × (resultado − expectativa)
expectativa = 1 / (1 + 10^((rating_adversário − rating_time) / 400))
```

Isso produz o comportamento desejado:

- Favorito ganha de time fraco → ganha **pouco** (~+10) — era esperado.
- Favorito **perde** para um time fraco → perde **muito** (−15 a −18) — resultado inesperado.
- Zebra ganha do líder → salta no ranking.

No backtest real, a maior punição da temporada foi a LOUD perdendo para a lanterna
Leviatan com 77% de expectativa: **−18.5 pontos**.

### 1b. Margem de vitória

Nem toda vitória vale o mesmo: o delta de rating é multiplicado pela **dominância do
vencedor** (vantagem de ouro por minuto, em escala logarítmica, com a mediana da janela
de treino como ponto neutro — multiplicador entre 0.25× e 2×). Um stomp de 25 minutos
move o rating até 2× mais que uma vitória raspada de 45. O peso da margem também é
calibrado no backtest (hoje: peso máximo 1.0).

### 2. Parâmetro de troca de jogadores (roster)

O jogo muda e os times trocam de jogadores. Quando a lineup de um jogo difere da
anterior:

- O rating **regride parcialmente à média** (mais trocas = mais incerteza);
- E é ajustado pela diferença de **impact score** entre quem entra e quem sai
  (score 0–100 por jogador, calculado de dano, ouro, visão, vantagens aos 15min etc.).

Na virada de temporada (2025 → 2026) todos os ratings regridem parcialmente à média.

### 2b. Elo individual por jogador

Cada jogador também tem um rating estilo Elo, atualizado jogo a jogo com a mesma
ponderação por adversário do time, **modulado pelo desempenho individual** na partida
(percentil por posição de dano, ouro, visão e vantagens aos 15min): quem carrega ganha
até 1.5× os pontos; quem joga mal numa derrota perde até 1.5×. O rating acompanha o
jogador em trocas de time. Saídas: `player_elo_ratings.csv`, `player_elo_history.csv`
e o gráfico `player_elo_ranking.png`.

No ajuste de roster do time, a calibração escolhe automaticamente entre o impact score
estático e o Elo individual (a fonte com menor log loss na validação vence — registrado
em `roster_source` no relatório de métricas).

### 3. Calibração automática (sem números chutados)

Os hiperparâmetros (K, regressão de temporada, regressão por troca, escala de impacto)
são escolhidos por **busca em grade num backtest walk-forward**: cada jogo é previsto
usando apenas informação anterior a ele, e vence a configuração com menor log loss na
janela de validação. A vantagem de lado (azul/vermelho) é estimada dos próprios dados.

### 4. Draft como ajuste opcional

Com o draft do jogo 1 conhecido, a probabilidade do modelo de draft (regressão
logística sobre picks/bans) é combinada em logit com a do rating, com peso também
calibrado na validação. Se o draft não agregar sinal, o peso calibra para 0 e a
previsão fica igual à do rating — o sistema nunca piora por causa do draft.

### 5. Probabilidade da série

Com as chances por jogo, uma árvore de estados do Bo3/Bo5 gera a probabilidade de cada
placar (2x0, 2x1, 1x2...) e a chance total de vencer a série.

## Resultados (backtest no CBLOL 2026)

Walk-forward estrito, janela de teste cronológica (~76 jogos):

| Sistema | Accuracy | ROC AUC | Log loss | Brier | F1 |
|---|---|---|---|---|---|
| **Rating + margem de vitória** | **0.68** | **0.64** | **0.66** | **0.23** | **0.76** |
| Regressão logística (contexto) | 0.62 | 0.58 | 0.72 | 0.25 | 0.69 |
| XGBoost (contexto) | 0.58 | 0.61 | 0.87 | 0.30 | 0.64 |

Acerto do vencedor da série na janela de teste: **74%**.

## Dados

- Fonte: [Oracle's Elixir](https://oracleselixir.com/) — CSVs de partidas profissionais.
- Ligas usadas: **CBLOL 2026** + **LTA Sul 2025** (mesmos times, usada para "aquecer"
  os ratings antes da temporada 2026) → ~385 jogos.
- Os CSVs brutos ficam em `data/` e não são versionados.

## Como usar

### Pipeline completo (filtro → dataset → modelos → ratings → relatórios)

```bash
python3 main.py
```

### Só recalcular ratings e relatórios

```bash
python3 scripts/build_team_ratings.py
```

### Avaliar as previsões de uma janela recente (tabelas + gráficos)

```bash
# últimos 14 dias do dataset (padrão), ou uma data específica:
python3 scripts/evaluate_recent_window.py
python3 scripts/evaluate_recent_window.py --start 2026-07-20
```

Gera em `artifacts/reports/recent_window/`: tabela jogo a jogo e por série (CSV),
métricas + parâmetros usados (JSON) e três gráficos (previsões com acertos/erros,
métricas vs backtest completo e matriz de confusão). As previsões são honestas:
cada jogo usa apenas informação anterior a ele.

### Prever uma série

```bash
# Bo3 simples
python3 predict_matchup.py "FURIA" "paiN Gaming"

# Bo5 com o draft do jogo 1 conhecido
python3 predict_matchup.py "FURIA" "paiN Gaming" --best-of 5 \
  --blue-picks "Aatrox,Maokai,Ahri,Jinx,Rakan" \
  --red-picks "K'Sante,Vi,Azir,Ezreal,Alistar"
```

Saída: ratings atuais, chance da série, chance por jogo (jogo 1 marcado "(com draft)"
quando informado), placar mais provável e distribuição de placares.

### Testes

```bash
python3 -m pytest tests/ -v
```

## Artefatos gerados (`artifacts/`)

| Arquivo | Conteúdo |
|---|---|
| `reports/team_ratings.csv` | Rating atual de cada time |
| `reports/team_rating_history.csv` | Jogo a jogo: adversário, expectativa, pontos ganhos/perdidos, trocas de lineup |
| `reports/rating_model_metrics.json` | Métricas completas + configuração calibrada + comparação com baselines |
| `reports/rating_confusion_matrix.png` | Matriz de confusão (falsos positivos/negativos) |
| `reports/rating_metrics_comparison.png` | Barras comparando rating vs modelos de draft |
| `reports/rating_calibration.png` | Curva de calibração + distribuição de probabilidades |
| `reports/player_impact_ratings.csv` | Impact score por jogador |
| `models/rating_config.json` | Hiperparâmetros calibrados usados pelo CLI |

## Estrutura do código

```
src/lol_ai/
├── config.py                  # caminhos do projeto
├── pipeline/cblol.py          # filtro das ligas + dataset contextual por jogo
└── modeling/
    ├── rating.py              # núcleo Elo + RatingEngine (temporada, roster)
    ├── rating_backtest.py     # walk-forward, calibração, blend de draft, relatórios
    ├── series.py              # árvore de probabilidades Bo1/Bo3/Bo5
    ├── prediction.py          # predict_series (API usada pelo CLI)
    ├── player_impact.py       # impact score por jogador e lookup para o roster
    ├── features.py            # features tabulares + split cronológico por série
    ├── training.py            # baselines: regressão logística e XGBoost sobre drafts
    ├── explain.py             # explicação SHAP dos baselines
    └── visualization.py       # gráficos dos baselines

scripts/                       # CLIs individuais de cada etapa
tests/                         # testes unitários e de integração (pytest)
docs/modelo.md                 # documentação técnica do modelo
```

## Requisitos

Python 3.9+, com `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `scipy`, `shap` e
`pytest`.

## Limitações conhecidas

- O modelo de draft atual não agrega sinal preditivo (peso calibrado em 0) — com mais
  dados de picks/bans isso pode mudar; o blend já está pronto para aproveitá-lo.
- Jogos de uma série são tratados como independentes (sem momentum intra-série).
- Jogadores sem histórico entram com impact score neutro (50).
