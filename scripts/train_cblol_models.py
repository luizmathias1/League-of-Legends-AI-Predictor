from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_src_on_path


PROJECT_ROOT = ensure_src_on_path()

from lol_ai.modeling.training import train_models  # noqa: E402


def main() -> None:
    result = train_models()
    metrics = result["metrics"]
    print("Treino concluído")
    print(metrics["logistic_regression"]["test"])
    print(metrics["xgboost"]["test"])


if __name__ == "__main__":
    main()
