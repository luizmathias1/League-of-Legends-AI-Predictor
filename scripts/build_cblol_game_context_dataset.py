from __future__ import annotations

import sys

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_src_on_path


PROJECT_ROOT = ensure_src_on_path()

from lol_ai.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from lol_ai.pipeline.cblol import build_context_dataset, filter_cblol_matches  # noqa: E402


def main() -> None:
    interim_path = INTERIM_DATA_DIR / "cblol_esports_matches_data.csv"
    if not interim_path.exists():
        interim_path = filter_cblol_matches(output_path=interim_path)

    output_path = build_context_dataset(input_path=interim_path, output_path=PROCESSED_DATA_DIR / "cblol_game_context_dataset.csv")
    print(f"Arquivo gerado: {output_path}")


if __name__ == "__main__":
    main()
