# Design — Rating de força por adversário + previsão de séries CBLOL

**Data:** 2026-08-05
**Status:** Aprovado pelo usuário (conversa de brainstorming)

## Problema

O pipeline atual prevê vitória por jogo com LogReg/XGBoost sobre 191 jogos do CBLOL 2026.
Métricas fracas (ROC AUC teste: 0.59 LogReg, 0.47 XGBoost). As features de winrate
(last5/last10) tratam todos os adversários como iguais: ganhar do último colocado vale o
mesmo que ganhar do líder, e perder de um time fraco não penaliza. Times também trocam de
jogadores no meio do split e o sistema não reage a isso.

## Objetivos

1. Rating de força por time que pondera cada resultado pela força do adversário
   (vitória esperada rende pouco; derrota inesperada penaliza muito).
2. Parâmetro de mudança de roster: o rating reage quando a lineup muda.
3. Previsão de probabilidade por jogo (1/2/3...) e da série (Bo3/Bo5), usando draft
   quando conhecido.
4. Backtest com métricas completas (matriz de confusão, precision, recall, F1, ROC AUC,
   Brier, log loss) comparando com os modelos atuais.

## Decisões tomadas (com o usuário)

- **Rating estilo Elo como base da previsão** — não é "elo de ranqueada"; é o método de
  rating ajustado por expectativa. O modelo de draft entra como ajuste em cima.
- **Roster: regressão + ajuste por jogador** — regressão parcial à média por jogador
  trocado + delta de `impact_score` entre quem entra e quem sai.
- **Incluir LTA Sul 2025** como histórico para semear os ratings de 2026, com peso menor
  (decaimento temporal + regressão à média na virada de temporada).
- **Entregáveis**: CLI de previsão, relatório de ratings por time com histórico
  partida a partida, e arquivo de métricas + gráficos. Sem dashboard HTML.

## Arquitetura

### Componente 1 — Motor de rating (`src/lol_ai/modeling/rating.py`)

Processa todos os jogos em ordem cronológica (LTA Sul 2025 → CBLOL 2026).

- Rating inicial: 1500 para todo time.
- Expectativa: `E = 1 / (1 + 10^((R_adv − R_time) / 400))`.
- Atualização pós-jogo: `R += K × (resultado − E)`, jogo a jogo (não por série).
- Virada de temporada (2025→2026): `R = média + (R − média) × fator_temporada`.
- Times sem histórico entram com 1500.
- Hiperparâmetros (`K`, `fator_temporada`, parâmetros de roster abaixo) ficam em um
  dataclass de configuração e são calibrados por busca em grade no backtest
  (janela de validação), não fixados a mão.

### Componente 2 — Ajuste de roster

A lineup por jogo já existe no dataset (`{side}_{pos}_player`). Para cada jogo:

1. Compara a lineup com a última lineup usada pelo mesmo time.
2. Por posição trocada: `R = média + (R − média) × (1 − p_troca)` (regressão parcial,
   `p_troca` por jogador trocado, acumulável).
3. Ajuste de qualidade: `R += escala × (impact_score_entrante − impact_score_sainte)`,
   usando `player_impact_ratings.csv` já gerado pelo projeto. Jogador sem histórico
   usa score neutro (50).

### Componente 3 — Previsão de jogo e série (`src/lol_ai/modeling/prediction.py`)

- **Probabilidade base do jogo**: expectativa Elo + vantagem de lado (estimada dos dados
  como um offset em pontos de rating para o lado azul).
- **Ajuste de draft (quando o draft é conhecido)**: combinação em logit —
  `logit(p) = logit(p_rating) + w × (logit(p_draft) − logit(0.5))`, onde `p_draft` vem do
  modelo atual (LogReg) e `w` é calibrado na janela de validação. Se o draft não agrega,
  `w → 0` e a previsão fica igual à do rating.
- **Probabilidade da série**: árvore de estados do Bo3/Bo5 assumindo jogos independentes,
  usando a probabilidade base por jogo (com draft apenas nos jogos em que ele for
  informado). Saída: P(vencer série), P(2-0), P(2-1), etc.

### Componente 4 — Backtest e relatórios

Walk-forward sobre o CBLOL 2026: cada jogo é previsto usando apenas informação anterior
a ele; depois o rating é atualizado. Saídas em `artifacts/reports/`:

- `team_ratings.csv` — rating atual por time.
- `team_rating_history.csv` — por jogo: data, adversário, resultado, expectativa,
  pontos ganhos/perdidos, mudanças de lineup aplicadas.
- `rating_model_metrics.json` — accuracy, precision, recall, F1, ROC AUC, Brier,
  log loss, matriz de confusão (jogo a jogo) + acerto por série, lado a lado com
  LogReg/XGBoost atuais.
- Gráficos PNG: matriz de confusão, comparação de métricas, calibração/distribuição
  de probabilidades (mesmo estilo dos gráficos existentes em `visualization.py`).

### Componente 5 — CLI (`predict_matchup.py`)

Entrada: dois times; opcionais: draft do próximo jogo, lineups. Saída: chance da série,
chance dos jogos 1/2/3 (com nota de quais usaram draft), placar mais provável e os
ratings atuais dos dois times.

## Tratamento de erros

- Time desconhecido no CLI → erro claro listando os times disponíveis.
- Jogador sem rating → score neutro 50, reportado como aviso.
- Draft ausente/parcial → previsão cai para a probabilidade base do rating.
- Dados brutos ausentes → mensagens de erro já padronizadas no pipeline atual.

## Testes

- Unitários (pytest): atualização Elo (ganho pequeno vs. grande conforme expectativa),
  detecção de troca de lineup, regressão de temporada, árvore de série (Bo3/Bo5 somam 1),
  combinação em logit.
- Integração: backtest completo roda de ponta a ponta e gera todos os artefatos.
- Critério de aceite: métricas do rating ≥ LogReg atual no conjunto de teste
  cronológico (mesma janela usada hoje).

## Fora de escopo

Dashboard HTML, momentum intra-série, rating individual por jogador como base do time,
outras ligas além de LTA Sul 2025 + CBLOL 2026.
