"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D Dataset batch contract.
"""

from __future__ import annotations

from data.dataset import AZT1DSequenceDataset
from data.indexing import build_sequence_index
from data.schema import build_feature_groups
from data.transforms import build_category_maps, load_processed_frame
from tests.support import BuildDataConfig, WriteProcessedCsv


# ============================================================
# Dataset tests
# ============================================================
# Purpose:
#   Verify that one legal sequence index entry becomes one
#   correctly structured batch item.
# ============================================================

def test_sequence_dataset_returns_the_refactored_batch_contract(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv(steps_per_subject=8)
    config = build_data_config(csv_path)
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)
    category_maps = build_category_maps(frame, groups)

    # The sequence index is built separately on purpose. This mirrors the actual
    # production flow and keeps the test aligned with the architecture we want
    # readers to understand.
    sample_index = build_sequence_index(frame, config, groups)

    dataset = AZT1DSequenceDataset(frame, sample_index, groups, category_maps)
    item = dataset[0]

    # The shapes should be derivable from config + feature groups rather than
    # being hardcoded magic numbers in the test.
    assert set(item) == {
        "static_categorical",
        "static_continuous",
        "encoder_continuous",
        "encoder_categorical",
        "decoder_known_continuous",
        "decoder_known_categorical",
        "target",
        "metadata",
    }
    assert tuple(item["static_categorical"].shape) == (len(groups.static_categorical),)
    assert tuple(item["static_continuous"].shape) == (len(groups.static_continuous),)
    assert tuple(item["encoder_continuous"].shape) == (
        config.encoder_length,
        len(groups.encoder_continuous),
    )
    assert tuple(item["encoder_categorical"].shape) == (
        config.encoder_length,
        len(groups.encoder_categorical),
    )
    assert tuple(item["decoder_known_continuous"].shape) == (
        config.prediction_length,
        len(groups.decoder_known_continuous),
    )
    assert tuple(item["decoder_known_categorical"].shape) == (
        config.prediction_length,
        len(groups.decoder_known_categorical),
    )
    assert tuple(item["target"].shape) == (config.prediction_length,)
    assert item["metadata"]["subject_id"] == "subject_a"
