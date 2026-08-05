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
