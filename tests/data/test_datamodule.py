"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D DataModule behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from data.datamodule import AZT1DDataModule
from data.downloader import DownloadResult
from data.schema import (
    DEFAULT_KNOWN_CONTINUOUS_COLUMNS,
    DEFAULT_OBSERVED_CATEGORICAL_COLUMNS,
    DEFAULT_OBSERVED_CONTINUOUS_COLUMNS,
)
from tests.support import BuildDataConfig, WriteProcessedCsv
from config import Config, DataConfig, TCNConfig, TFTConfig


# ============================================================
# DataModule tests
# ============================================================
# Purpose:
#   Verify the top-level orchestration layer that connects
#   download/build/setup behavior to dataset and dataloader
#   creation.
# ============================================================

def test_datamodule_setup_builds_datasets_and_loader_batches(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv(steps_per_subject=10, subject_ids=("subject_a", "subject_b"))
    config = build_data_config(csv_path, batch_size=2)
    datamodule = AZT1DDataModule(config)

    # `setup()` is the Lightning lifecycle hook that should materialize the
    # cleaned dataframe view, build sample indices, and instantiate datasets.
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    target = batch["target"]

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert "subject_id" in datamodule.categorical_cardinalities
    assert isinstance(target, torch.Tensor)
    assert tuple(target.shape)[0] <= config.batch_size


def test_datamodule_prepare_data_skips_download_when_processed_file_exists(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = write_processed_csv()
    config = build_data_config(csv_path, dataset_url="https://example.com/data.zip")
    datamodule = AZT1DDataModule(config)

    # If the processed CSV already exists and rebuild is disabled, prepare_data
    # should be a no-op. The monkeypatched sentinels make that contract explicit.
    def _unexpected_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("prepare_data should not download when processed data already exists")

    monkeypatch.setattr("data.datamodule.AZT1DDownloader", _unexpected_call)
    monkeypatch.setattr("data.datamodule.AZT1DPreprocessor", _unexpected_call)

    datamodule.prepare_data()


def test_datamodule_prepare_data_uses_downloader_and_preprocessor_when_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeDownloader:
        def __init__(self, raw_dir: Path, cache_dir: Path, extract_dir: Path) -> None:
            calls["downloader_init"] = (raw_dir, cache_dir, extract_dir)

        def download(self, *, url: str, filename: str, extract: bool, force: bool) -> DownloadResult:
            extracted_path = tmp_path / "downloaded" / "dataset"
            extracted_path.mkdir(parents=True, exist_ok=True)
            calls["download"] = (url, filename, extract, force)
            file_path = tmp_path / "downloaded" / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"zip-bytes")
            return DownloadResult(
                url=url,
                file_path=file_path,
                extracted_path=extracted_path,
                from_cache=False,
                content_type="application/zip",
                size_bytes=len(b"zip-bytes"),
            )

    class FakePreprocessor:
        def __init__(self, dataset_dir: Path, output_file: Path) -> None:
            calls["preprocessor_init"] = (dataset_dir, output_file)
            self.output_file = output_file

        def build(self, force: bool = False) -> Path:
            calls["build_force"] = force
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_text("subject_id,timestamp,glucose_mg_dl\n", encoding="utf-8")
            return self.output_file

    monkeypatch.setattr("data.datamodule.AZT1DDownloader", FakeDownloader)
    monkeypatch.setattr("data.datamodule.AZT1DPreprocessor", FakePreprocessor)

    # This test isolates orchestration behavior from the real filesystem/network:
    # we care that the DataModule wires the downloader and preprocessor together
    # correctly, not that those lower layers re-prove their own unit tests here.
    data_config = DataConfig(
        dataset_url="https://example.com/data.zip",
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        extracted_dir=tmp_path / "extracted",
        processed_dir=tmp_path / "processed",
        processed_file_name="azt1d_processed.csv",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    datamodule = AZT1DDataModule(data_config)
    datamodule.prepare_data()

    assert "downloader_init" in calls
    assert "download" in calls
    assert "preprocessor_init" in calls
    assert calls["build_force"] is False
    assert data_config.processed_file_path.exists()


def test_datamodule_exposes_tft_categorical_cardinalities_in_feature_group_order(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    # The bridge method should expose categorical lens metadata in the exact
    # order expected by TFTConfig. Ordering matters because embedding lists are
    # positional rather than keyed by column name.
    csv_path = write_processed_csv(subject_ids=("subject_a", "subject_b"))
    config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(config)

    datamodule.setup()
    cardinalities = datamodule.get_tft_categorical_cardinalities()

    assert cardinalities["static_categorical_inp_lens"] == [2]
    assert cardinalities["temporal_known_categorical_inp_lens"] == []
    assert cardinalities["temporal_observed_categorical_inp_lens"] == [4, 10]


def test_datamodule_bind_model_config_aligns_tft_metadata_with_data_contract(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    # This test protects the contract-closing step: once setup has discovered
    # vocab sizes and fallback feature groups, the convenience binder should
    # produce a TFTConfig that reflects that exact runtime view of the data.
    csv_path = write_processed_csv(subject_ids=("subject_a", "subject_b"))
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)

    datamodule.setup()

    base_config = Config(
        data=data_config,
        tft=TFTConfig(),
        tcn=TCNConfig(num_inputs=1, num_channels=[4]),
    )

    bound_config = datamodule.bind_model_config(base_config)

    # The binder returns a new config rather than mutating the original one so
    # callers can keep the declarative config and the runtime-bound config
    # separate if they want to inspect or log both.
    assert base_config.tft.static_categorical_inp_lens == []
    assert bound_config.tft.static_categorical_inp_lens == [2]
    assert bound_config.tft.temporal_known_categorical_inp_lens == []
    assert bound_config.tft.temporal_observed_categorical_inp_lens == [4, 10]

    # When the data config did not declare FeatureSpec entries explicitly, the
    # binder should synthesize an equivalent schema from the fallback feature
    # groups so the model sees the same contract as the dataset.
    assert bound_config.tft.temporal_known_continuous_inp_size == len(
        DEFAULT_KNOWN_CONTINUOUS_COLUMNS
    )
    assert bound_config.tft.temporal_observed_continuous_inp_size == len(
        DEFAULT_OBSERVED_CONTINUOUS_COLUMNS
    )
    assert len(bound_config.tft.temporal_observed_categorical_inp_lens) == len(
        DEFAULT_OBSERVED_CATEGORICAL_COLUMNS
    )
    assert bound_config.tft.temporal_target_size == 1
    assert bound_config.tft.encoder_length == data_config.encoder_length
    assert bound_config.tft.example_length == (
        data_config.encoder_length + data_config.prediction_length
    )
