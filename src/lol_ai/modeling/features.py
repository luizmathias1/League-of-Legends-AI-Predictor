from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from lol_ai.config import LEGACY_PROCESSED_FILE, PROCESSED_DATA_DIR


TEAM_POSITIONS = ("top", "jng", "mid", "bot", "sup")

NUMERIC_COLUMNS = [
    "game_number",
    "series_game_count",
    "best_of",
    "series_games_played_before",
    "series_score_blue_before",
    "series_score_red_before",
    "blue_last5_winrate",
    "blue_last10_winrate",
    "red_last5_winrate",
    "red_last10_winrate",
    "blue_h2h_last10_winrate",
    "red_h2h_last10_winrate",
    *[f"blue_{position}_last10_winrate" for position in TEAM_POSITIONS],
    *[f"red_{position}_last10_winrate" for position in TEAM_POSITIONS],
]

BASE_CATEGORICAL_COLUMNS = [
    "split",
    "playoffs",
    "patch",
    "blue_team",
    "red_team",
    "first_pick_side",
]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def split_bans_or_picks(value: object, expected_items: int = 5) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return [""] * expected_items
    items = [item.strip() for item in normalized.split(";")]
    if len(items) < expected_items:
        items.extend([""] * (expected_items - len(items)))
    return items[:expected_items]


def resolve_processed_input(input_path: Path | None = None) -> Path:
    if input_path is not None:
        return input_path
    preferred = PROCESSED_DATA_DIR / "cblol_game_context_dataset.csv"
    if preferred.exists():
        return preferred
    return LEGACY_PROCESSED_FILE


def load_context_dataset(input_path: Path | None = None) -> pd.DataFrame:
    resolved_input = resolve_processed_input(input_path)
    if not resolved_input.exists():
        raise FileNotFoundError(f"Dataset processado não encontrado: {resolved_input}")

    frame = pd.read_csv(resolved_input)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["blue_win"] = pd.to_numeric(frame["blue_win"], errors="coerce").fillna(0).astype(int)
    frame["series_games_played_before"] = pd.to_numeric(frame["series_games_played_before"], errors="coerce")
    frame["game_number"] = pd.to_numeric(frame["game_number"], errors="coerce")
    frame["series_game_count"] = pd.to_numeric(frame["series_game_count"], errors="coerce")
    frame["best_of"] = pd.to_numeric(frame["best_of"], errors="coerce")
    return frame


def add_draft_slot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for prefix in ("blue_bans", "red_bans", "blue_picks", "red_picks"):
        slots = enriched[prefix].apply(split_bans_or_picks)
        for index in range(5):
            enriched[f"{prefix[:-1]}{index + 1}"] = slots.apply(lambda values, idx=index: values[idx])
    return enriched


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_draft_slot_columns(frame)
    feature_columns = [
        *NUMERIC_COLUMNS,
        *BASE_CATEGORICAL_COLUMNS,
        *[f"blue_ban{i}" for i in range(1, 6)],
        *[f"red_ban{i}" for i in range(1, 6)],
        *[f"blue_pick{i}" for i in range(1, 6)],
        *[f"red_pick{i}" for i in range(1, 6)],
    ]

    for column in feature_columns:
        if column not in enriched.columns:
            enriched[column] = ""

    feature_frame = enriched[feature_columns].copy()
    for column in NUMERIC_COLUMNS:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    for column in feature_frame.columns.difference(NUMERIC_COLUMNS):
        feature_frame[column] = feature_frame[column].fillna("").astype(str)
    return feature_frame


def get_target(frame: pd.DataFrame) -> pd.Series:
    return frame["blue_win"].astype(int)


def get_group_series(frame: pd.DataFrame) -> pd.Series:
    return frame["series_id"].astype(str)


def get_dates(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["date"], errors="coerce")


def chronological_series_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    if "series_id" not in frame.columns or "date" not in frame.columns:
        raise ValueError("A tabela precisa conter as colunas series_id e date.")

    series_order = (
        frame.groupby("series_id", as_index=True)["date"]
        .min()
        .sort_values()
    )
    series_ids = list(series_order.index)
    total_series = len(series_ids)
    if total_series == 0:
        raise ValueError("Não foi possível identificar nenhuma série no dataset.")

    train_end = max(1, int(total_series * train_fraction))
    validation_end = max(train_end + 1, int(total_series * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, total_series - 1) if total_series > 2 else total_series

    train_series = set(series_ids[:train_end])
    validation_series = set(series_ids[train_end:validation_end])
    test_series = set(series_ids[validation_end:])

    if not validation_series:
        validation_series = set(series_ids[train_end:train_end + 1])
        test_series = set(series_ids[train_end + 1:])
    if not test_series:
        test_series = set(series_ids[-1:])
        if len(series_ids) > 1:
            validation_series = set(series_ids[-2:-1])

    train_index = frame.index[frame["series_id"].isin(train_series)]
    validation_index = frame.index[frame["series_id"].isin(validation_series)]
    test_index = frame.index[frame["series_id"].isin(test_series)]
    return train_index, validation_index, test_index


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index


def create_dataset_split(frame: pd.DataFrame) -> DatasetSplit:
    feature_frame = build_feature_frame(frame)
    target = get_target(frame)
    train_index, validation_index, test_index = chronological_series_split(frame)
    return DatasetSplit(
        X_train=feature_frame.loc[train_index],
        X_validation=feature_frame.loc[validation_index],
        X_test=feature_frame.loc[test_index],
        y_train=target.loc[train_index],
        y_validation=target.loc[validation_index],
        y_test=target.loc[test_index],
        train_index=train_index,
        validation_index=validation_index,
        test_index=test_index,
    )
