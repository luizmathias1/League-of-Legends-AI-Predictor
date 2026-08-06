# Como o modelo funciona

Este documento descreve o funcionamento interno do sistema de previsão. Para
instruções de uso, veja o [README](../README.md).

## Fluxo de dados

```
CSVs Oracle's Elixir (data/)
  └─ filtro CBLOL 2026 + LTA Sul 2025          (pipeline/cblol.py)
      └─ dataset contextual por jogo            (data/processed/)
          ├─ rating de times + jogadores        (modeling/rating.py, player_elo.py)
          ├─ backtest walk-forward + calibração (modeling/rating_backtest.py)
          ├─ modelos de contexto LogReg/XGBoost (modeling/training.py)
          └─ previsão de séries + CLI           (modeling/prediction.py)
```

## Rating de times (estilo Elo)

Cada time tem uma pontuação, inicial 1500. Após cada jogo:

```
expectativa = 1 / (1 + 10^((R_adversário − R_time) / 400))
R_time     += K × (resultado − expectativa)
```

Consequências práticas:

- Favorito ganhar de time fraco rende pouco (o resultado era esperado).
- Favorito perder para time fraco pune muito (resultado inesperado).
- O rating se ajusta sozinho quando um time muda de nível — após 2–3 séries o
  novo patamar já aparece nas previsões.

Ajustes aplicados **antes** de cada jogo:

- **Virada de temporada** (2025→2026): `R = 1500 + (R − 1500) × season_carry`.
- **Troca de lineup**: regressão parcial à média por jogador trocado + ajuste
  pela diferença de qualidade entre quem entra e quem sai (fonte escolhida por
  calibração — ver abaixo).
- **Vantagem de lado**: offset em pontos para o lado azul, estimado da taxa de
  vitória azul na janela de treino (hoje ~+3 pontos, efeito desprezível).

## Elo individual por jogador

Cada jogador tem rating próprio (inicial 1500) que acompanha o jogador mesmo
em trocas de time. Atualização por jogo:

```
base          = K_jogador × (resultado_do_time − expectativa_do_time)
multiplicador = 0.5 + desempenho/100        (vitória)
                1.5 − desempenho/100        (derrota)
delta         = base × multiplicador        (multiplicador ∈ [0.5, 1.5])
```

`desempenho` é o percentil (0–100) do jogador na posição naquele jogo, combinando
dano, ouro, DPM, visão, CS e vantagens aos 15 minutos — **sem** a componente de
resultado, que já entra no delta base. Quem carrega ganha mais; quem joga mal
numa derrota perde mais; e tudo continua ponderado pela força do adversário.

## Ajuste de roster: duas fontes, a validação decide

Quando a lineup muda, o ajuste de qualidade pode vir de duas fontes:

1. **Impact score estático** (`player_impact.py`) — média histórica 0–100 por jogador;
2. **Elo individual ao vivo** (`player_elo.py`) — o rating do jogador naquele momento.

O backtest calibra as duas e escolhe a de menor log loss na validação. A escolha
fica registrada em `roster_source` nos relatórios. (Hoje: impact estático.)

## Probabilidade de jogo e de série

- Jogo: expectativa Elo entre os dois ratings (+ vantagem de lado no jogo 1).
- Série: árvore de estados do Bo1/Bo3/Bo5 com a probabilidade de cada jogo →
  P(2x0), P(2x1), ..., e P(vencer a série) (`modeling/series.py`).
- Draft: quando os picks do jogo 1 são conhecidos, o modelo de contexto pode
  ajustar a probabilidade em logit, com peso calibrado na validação. Hoje o
  peso calibra em **0.0** — o modelo de contexto não agrega sinal além do
  rating, então o draft não altera a previsão (o mecanismo fica pronto para
  quando features de draft melhores existirem).

## Calibração e validação (nenhum número chutado)

Split cronológico por série: ~70% treino, ~15% validação, ~15% teste.

- **Walk-forward estrito**: a probabilidade de cada jogo é calculada antes de o
  resultado atualizar qualquer rating; nenhuma informação futura vaza.
- **Busca em grade** na validação: K do time, carry de temporada, regressão por
  troca, escala do ajuste de qualidade (e a fonte do roster). Vence a menor
  log loss.
- **Métricas finais** calculadas na janela de teste, nunca vista na calibração:
  accuracy, precision, recall, F1, ROC AUC, Brier, log loss, matriz de confusão
  e acerto por série — sempre comparadas com os baselines LogReg/XGBoost.

## Relatórios gerados (`artifacts/reports/`)

| Arquivo | Conteúdo |
|---|---|
| `team_ratings.csv` | Rating atual de cada time |
| `team_rating_history.csv` | Por jogo: expectativa, pontos ganhos/perdidos, trocas de lineup |
| `player_elo_ratings.csv` | Ranking de jogadores por Elo individual |
| `player_elo_history.csv` | Histórico jogo a jogo por jogador |
| `player_elo_ranking.png` | Top 15 jogadores (mín. 8 jogos) |
| `rating_model_metrics.json` | Métricas + parâmetros calibrados + comparação com baselines |
| `rating_confusion_matrix.png` / `rating_metrics_comparison.png` / `rating_calibration.png` | Gráficos do backtest |
| `recent_window/` | Avaliação de janela recente (`scripts/evaluate_recent_window.py`) |
| `player_impact_ratings.csv` | Impact score estático por jogador |

## Limitações conhecidas

- Jogos de uma série são tratados como independentes (sem momentum intra-série).
- O modelo de contexto/draft não agrega sinal (peso 0); as features de campeões
  específicos foram removidas por causarem overfitting com poucos jogos.
- Jogadores sem histórico entram com valores neutros (Elo 1500 / score 50).
- Nomes de jogadores são a chave do Elo individual (sem id único entre ligas).

## Próximos passos planejados

- Modelos por jogo específico da série (jogo 1 vs 2 vs 3: escolha de lado,
  fearless draft, momentum).
- Features de draft transferíveis (winrate do campeão no patch, conforto do
  time com o pick) para o peso do draft sair do zero.
