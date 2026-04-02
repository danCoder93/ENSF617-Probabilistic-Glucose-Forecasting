"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D transform layer.
"""

from __future__ import annotations

from data.schema import DEVICE_MODE_CATEGORIES, build_feature_groups
from data.transforms import build_category_maps, load_processed_frame
from tests.support import BuildDataConfig, WriteProcessedCsv


# ============================================================
# Transform tests
# ============================================================
# Purpose:
#   Verify that dataframe-level cleanup produces a stable, model-
#   ready table contract before indexing or tensor assembly begins.
# ============================================================

def test_load_processed_frame_normalizes_sparse_values_and_categories(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv()
    config = build_data_config(csv_path)
    groups = build_feature_groups(config)

    # `load_processed_frame` is intentionally the one place where dataframe-wide
    # cleanup happens. The rest of the pipeline should be able to trust the
    # resulting frame as numerically and categorically normalized.
    frame = load_processed_frame(csv_path, config, groups)

    # Sparse operational values should be numeric and safe for tensor conversion.
    assert frame["basal_insulin_u"].dtype.kind in {"f", "i"}
    assert frame["carbs_g"].dtype.kind in {"f", "i"}

    # Normalization should collapse messy raw strings into the controlled vocab.
    assert frame.loc[0, "device_mode"] == "sleep"
    assert frame.loc[1, "bolus_type"] == "automatic"

    # Time-derived known features are part of the cleaned dataframe contract.
    assert "minute_of_day_sin" in frame.columns
    assert "day_of_week_cos" in frame.columns
    assert "is_weekend" in frame.columns


def test_build_category_maps_uses_declared_order_for_known_vocabularies(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv(subject_ids=("subject_b", "subject_a"))
    config = build_data_config(csv_path)
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)

    # Category maps are fit after cleaning because the cleaned dataframe is the
    # first point where we know labels have already been normalized into their
    # stable vocabulary.
    category_maps = build_category_maps(frame, groups)

    # Declared vocabularies should preserve the schema-defined ordering exactly
    # so categorical IDs remain stable across runs and model initializations.
    assert category_maps["device_mode"] == DEVICE_MODE_CATEGORIES

    # Subject IDs are not hardcoded, so their vocabulary is discovered from data.
    assert category_maps["subject_id"] == ("subject_a", "subject_b")
