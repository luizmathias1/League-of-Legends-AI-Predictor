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
