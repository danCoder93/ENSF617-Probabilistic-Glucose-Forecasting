"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates refactored AZT1D split and sequence-indexing
behavior.
"""

from __future__ import annotations

from data.indexing import build_sequence_index, split_processed_frame
from data.schema import build_feature_groups
from data.transforms import load_processed_frame
from tests.support import BuildDataConfig, WriteProcessedCsv


# ============================================================
# Indexing tests
# ============================================================
# Purpose:
#   Verify split policies and sequence-window construction without
#   mixing those concerns into dataset tensor assembly.
# ============================================================

def test_split_processed_frame_uses_explicit_test_ratio(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    # This test protects the recent change that made `test_ratio` explicit in the
    # split helpers rather than treating test as only "whatever rows remain".
    csv_path = write_processed_csv(steps_per_subject=10)
    config = build_data_config(
        csv_path,
        split_by_subject=False,
        split_within_subject=False,
        train_ratio=0.5,
        val_ratio=0.2,
        test_ratio=0.3,
    )
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)

    # Splitting is intentionally tested at the dataframe stage, before any
    # sample indices are built, because split policy and sequence legality are
    # separate responsibilities in the refactored architecture.
    split_frames = split_processed_frame(frame, config, groups)

    assert len(split_frames["train"]) == 5
    assert len(split_frames["val"]) == 2
    assert len(split_frames["test"]) == 3


def test_build_sequence_index_drops_windows_across_time_gaps(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    # A gap in timestamps should break the timeline into separate legal segments.
    # Windows are only allowed within a continuous segment, never across the gap.
    csv_path = write_processed_csv(steps_per_subject=8, gap_after_step=4)
    config = build_data_config(csv_path, encoder_length=3, prediction_length=2)
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)

    # The indexing layer owns continuity checks. The dataset should never need to
    # wonder whether a slice is legal; it should simply trust the index entries
    # it receives.
    sample_index = build_sequence_index(frame, config, groups)

    # Without the gap there would be 4 windows for length=8 and total_steps=5.
    # The inserted gap leaves only one legal segment of length 5, so exactly one
    # window should survive.
    assert len(sample_index) == 1
    assert sample_index[0].encoder_end - sample_index[0].encoder_start == config.encoder_length
    assert sample_index[0].decoder_end - sample_index[0].decoder_start == config.prediction_length
