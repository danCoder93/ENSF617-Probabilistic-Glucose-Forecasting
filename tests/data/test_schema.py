"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D schema contract.
"""

from __future__ import annotations

from data.schema import (
    DEFAULT_KNOWN_CONTINUOUS_COLUMNS,
    DEFAULT_OBSERVED_CATEGORICAL_COLUMNS,
    DEFAULT_OBSERVED_CONTINUOUS_COLUMNS,
    build_feature_groups,
)
from tests.support import BuildDataConfig, WriteProcessedCsv
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


# ============================================================
# Schema tests
# ============================================================
# Purpose:
#   Verify that semantic feature grouping is derived correctly from
#   either the shared FeatureSpec schema or the documented AZT1D
#   fallback defaults.
# ============================================================

def test_build_feature_groups_uses_azt1d_defaults_when_feature_schema_missing(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    # The refactor intentionally keeps a documented fallback path so the data
    # layer can run before the whole project has fully migrated to `config.features`.
    csv_path = write_processed_csv()
    config = build_data_config(csv_path)

    groups = build_feature_groups(config)

    # These assertions make the fallback contract explicit. If someone changes
    # the default grouping later, this test will force that change to be a
    # deliberate decision instead of an unnoticed drift.
    assert groups.static_categorical == (config.subject_id_column,)
    assert groups.known_continuous == DEFAULT_KNOWN_CONTINUOUS_COLUMNS
    assert groups.observed_continuous == DEFAULT_OBSERVED_CONTINUOUS_COLUMNS
    assert groups.observed_categorical == DEFAULT_OBSERVED_CATEGORICAL_COLUMNS


def test_build_feature_groups_prefers_declared_feature_schema(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    # Once a caller supplies FeatureSpec entries, the data layer should stop
    # relying on AZT1D-specific defaults and derive its groups directly from the
    # shared schema contract.
    csv_path = write_processed_csv()
    config = build_data_config(
        csv_path,
        features=[
            FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
            FeatureSpec("minute_of_day_sin", InputTypes.KNOWN, DataTypes.CONTINUOUS),
            FeatureSpec("device_mode", InputTypes.OBSERVED, DataTypes.CATEGORICAL),
            FeatureSpec("carbs_g", InputTypes.OBSERVED, DataTypes.CONTINUOUS),
        ],
    )

    groups = build_feature_groups(config)

    # When a shared schema is present, the data layer should act as a faithful
    # projection of that schema rather than mixing in extra AZT1D defaults.
    assert groups.static_categorical == ("subject_id",)
    assert groups.known_continuous == ("minute_of_day_sin",)
    assert groups.observed_categorical == ("device_mode",)
    assert groups.observed_continuous == ("carbs_g",)
