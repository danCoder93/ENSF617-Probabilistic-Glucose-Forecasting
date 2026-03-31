from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PAST_NUMERIC_COLUMNS = [
    "glucose_mg_dl",
    "basal_insulin_u",
    "bolus_insulin_u",
    "correction_insulin_u",
    "meal_insulin_u",
    "carbs_g",
]

SPARSE_NUMERIC_COLUMNS = [
    "basal_insulin_u",
    "bolus_insulin_u",
    "correction_insulin_u",
    "meal_insulin_u",
    "carbs_g",
]

DEVICE_MODE_CATEGORIES = ["none", "sleep", "exercise", "other"]
DEVICE_MODE_COLUMNS = [f"device_mode_{name}" for name in DEVICE_MODE_CATEGORIES]

BOLUS_TYPE_CATEGORIES = [
    "none",
    "automatic",
    "standard",
    "standard_correction",
    "ble_standard",
    "ble_standard_correction",
    "quick",
    "extended",
    "extended_correction",
    "other",
]
BOLUS_TYPE_COLUMNS = [f"bolus_type_{name}" for name in BOLUS_TYPE_CATEGORIES]

FUTURE_KNOWN_COLUMNS = [
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
]


@dataclass(frozen=True)
class WindowConfig:
    history_steps: int = 24
    horizon_steps: int = 6
    expected_step_minutes: int = 5
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    batch_size: int = 64
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.expected_step_minutes <= 0:
            raise ValueError("expected_step_minutes must be positive")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must contain three values")
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")


class Azt1dWindowDataset(Dataset):
    def __init__(self, samples: list[dict[str, object]], subject_to_index: dict[str, int]) -> None:
        self.samples = samples
        self.subject_to_index = subject_to_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        subject_id = str(sample["subject_id"])

        return {
            "static_covariates": torch.tensor(
                [self.subject_to_index[subject_id]],
                dtype=torch.long,
            ),
            "past_observed_inputs": torch.tensor(
                sample["past_observed_inputs"],
                dtype=torch.float32,
            ),
            "future_known_inputs": torch.tensor(
                sample["future_known_inputs"],
                dtype=torch.float32,
            ),
            "target": torch.tensor(sample["target"], dtype=torch.float32),
            "metadata": sample["metadata"],
        }


def create_dataloaders(
    csv_path: str | Path,
    config: WindowConfig = WindowConfig(),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataframe = _load_dataframe(csv_path)
    subject_ids = sorted(dataframe["subject_id"].astype(str).unique())
    subject_to_index = {subject_id: index for index, subject_id in enumerate(subject_ids)}

    train_samples: list[dict[str, object]] = []
    val_samples: list[dict[str, object]] = []
    test_samples: list[dict[str, object]] = []

    for subject_id in subject_ids:
        subject_frame = (
            dataframe[dataframe["subject_id"] == subject_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        subject_train, subject_val, subject_test = _build_subject_samples(subject_frame, config)
        train_samples.extend(subject_train)
        val_samples.extend(subject_val)
        test_samples.extend(subject_test)

    train_dataset = Azt1dWindowDataset(train_samples, subject_to_index)
    val_dataset = Azt1dWindowDataset(val_samples, subject_to_index)
    test_dataset = Azt1dWindowDataset(test_samples, subject_to_index)

    return (
        _make_loader(train_dataset, config.batch_size, True, config.num_workers),
        _make_loader(val_dataset, config.batch_size, False, config.num_workers),
        _make_loader(test_dataset, config.batch_size, False, config.num_workers),
    )


def _make_loader(
    dataset: Azt1dWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and len(dataset) > 0,
        num_workers=num_workers,
    )


def _load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe["glucose_mg_dl"] = pd.to_numeric(dataframe["glucose_mg_dl"], errors="coerce")
    dataframe = dataframe.dropna(subset=["subject_id", "timestamp", "glucose_mg_dl"]).copy()
    dataframe["timestamp"] = dataframe["timestamp"].dt.round("min")
    dataframe = dataframe.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

    for column in SPARSE_NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)

    dataframe["device_mode"] = _prepare_text_column(dataframe["device_mode"])
    dataframe["device_mode"] = dataframe["device_mode"].replace({"": pd.NA, "0": pd.NA})
    dataframe["device_mode"] = dataframe.groupby("subject_id")["device_mode"].ffill()
    dataframe["device_mode"] = dataframe["device_mode"].fillna("none")
    dataframe["device_mode"] = dataframe["device_mode"].apply(_normalize_device_mode)

    dataframe["bolus_type"] = _prepare_text_column(dataframe["bolus_type"])
    dataframe["bolus_type"] = dataframe["bolus_type"].replace({"": pd.NA, "0": pd.NA})
    dataframe["bolus_type"] = dataframe.groupby("subject_id")["bolus_type"].ffill()
    dataframe["bolus_type"] = dataframe["bolus_type"].fillna("none")
    dataframe["bolus_type"] = dataframe["bolus_type"].apply(_normalize_bolus_type)

    minute_of_day = dataframe["timestamp"].dt.hour * 60 + dataframe["timestamp"].dt.minute
    day_of_week = dataframe["timestamp"].dt.dayofweek

    dataframe["minute_of_day_sin"] = _sin_from_period(minute_of_day, 1440.0)
    dataframe["minute_of_day_cos"] = _cos_from_period(minute_of_day, 1440.0)
    dataframe["day_of_week_sin"] = _sin_from_period(day_of_week, 7.0)
    dataframe["day_of_week_cos"] = _cos_from_period(day_of_week, 7.0)
    dataframe["is_weekend"] = (day_of_week >= 5).astype("float32")

    dataframe["device_mode"] = pd.Categorical(
        dataframe["device_mode"],
        categories=DEVICE_MODE_CATEGORIES,
    )
    dataframe["bolus_type"] = pd.Categorical(
        dataframe["bolus_type"],
        categories=BOLUS_TYPE_CATEGORIES,
    )

    device_dummies = pd.get_dummies(dataframe["device_mode"], prefix="device_mode", dtype="float32")
    bolus_dummies = pd.get_dummies(dataframe["bolus_type"], prefix="bolus_type", dtype="float32")
    dataframe = pd.concat([dataframe, device_dummies, bolus_dummies], axis=1)

    for column in DEVICE_MODE_COLUMNS + BOLUS_TYPE_COLUMNS:
        if column not in dataframe:
            dataframe[column] = 0.0

    return dataframe


def _prepare_text_column(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _build_subject_samples(
    subject_frame: pd.DataFrame,
    config: WindowConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    total_rows = len(subject_frame)
    if total_rows < config.history_steps + config.horizon_steps:
        return [], [], []

    train_ratio, val_ratio, _ = config.split_ratios
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    train_frame = subject_frame.iloc[:train_end].reset_index(drop=True)
    val_frame = subject_frame.iloc[train_end:val_end].reset_index(drop=True)
    test_frame = subject_frame.iloc[val_end:].reset_index(drop=True)

    train_samples = _build_windows(train_frame, config)
    val_samples = _build_windows(val_frame, config)
    test_samples = _build_windows(test_frame, config)

    total_windows = len(train_samples) + len(val_samples) + len(test_samples)
    if total_windows < 3:
        return _build_windows(subject_frame.reset_index(drop=True), config), [], []

    return train_samples, val_samples, test_samples


def _build_windows(subject_frame: pd.DataFrame, config: WindowConfig) -> list[dict[str, object]]:
    total_steps = config.history_steps + config.horizon_steps
    if len(subject_frame) < total_steps:
        return []

    past_columns = PAST_NUMERIC_COLUMNS + DEVICE_MODE_COLUMNS + BOLUS_TYPE_COLUMNS
    timestamps = subject_frame["timestamp"].reset_index(drop=True)
    deltas = (
        timestamps.diff()
        .dt.total_seconds()
        .div(60)
        .fillna(config.expected_step_minutes)
        .to_numpy()
    )

    past_values = subject_frame[past_columns].to_numpy(dtype="float32")
    future_values = subject_frame[FUTURE_KNOWN_COLUMNS].to_numpy(dtype="float32")
    target_values = subject_frame["glucose_mg_dl"].to_numpy(dtype="float32")
    timestamp_strings = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    subject_id = str(subject_frame.iloc[0]["subject_id"])

    samples: list[dict[str, object]] = []
    for start in range(len(subject_frame) - total_steps + 1):
        end = start + total_steps
        if not (deltas[start + 1:end] == config.expected_step_minutes).all():
            continue

        encoder_end = start + config.history_steps
        samples.append(
            {
                "subject_id": subject_id,
                "past_observed_inputs": past_values[start:encoder_end],
                "future_known_inputs": future_values[encoder_end:end],
                "target": target_values[encoder_end:end],
                "metadata": {
                    "subject_id": subject_id,
                    "encoder_start": timestamp_strings[start],
                    "encoder_end": timestamp_strings[encoder_end - 1],
                    "decoder_start": timestamp_strings[encoder_end],
                    "decoder_end": timestamp_strings[end - 1],
                },
            }
        )

    return samples


def _normalize_device_mode(value: object) -> str:
    if pd.isna(value):
        return "none"

    value = str(value).strip().lower()
    if value in {"", "0", "none"}:
        return "none"
    if value == "sleepsleep":
        return "sleep"
    if value in {"sleep", "exercise"}:
        return value
    return "other"


def _normalize_bolus_type(value: object) -> str:
    if pd.isna(value):
        return "none"

    value = str(value).strip().lower()
    if value in {"", "0", "none"}:
        return "none"
    if "automatic bolus" in value:
        return "automatic"
    if "ble standard bolus/correction" in value:
        return "ble_standard_correction"
    if "ble standard bolus" in value:
        return "ble_standard"
    if value == "standard/correction":
        return "standard_correction"
    if value == "standard":
        return "standard"
    if value == "quick":
        return "quick"
    if value.startswith("extended/correction"):
        return "extended_correction"
    if value.startswith("extended"):
        return "extended"
    return "other"


def _sin_from_period(values: pd.Series, period: float) -> pd.Series:
    angles = values.astype("float64") * (2.0 * math.pi / period)
    return angles.apply(math.sin).astype("float32")


def _cos_from_period(values: pd.Series, period: float) -> pd.Series:
    angles = values.astype("float64") * (2.0 * math.pi / period)
    return angles.apply(math.cos).astype("float32")
