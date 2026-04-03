# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
# work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
# DataModule docs/tutorial
# (https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
# original AZT1D dataset release on Mendeley Data
# (https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to support
# the cleaner DataModule-oriented architecture required by the fused TCN + TFT
# training pipeline.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from data.dataset import AZT1DSequenceDataset, BatchItem
from data.downloader import AZT1DDownloader
from data.indexing import build_sequence_index, split_processed_frame
from data.preprocessor import AZT1DPreprocessor
from data.schema import FeatureGroups, build_feature_groups
from data.statistics import describe_clean_frame
from data.transforms import build_category_maps, load_processed_frame
from config import Config, DataConfig
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes

# This import block is intentionally a little more complex than a normal import.
#
# Why:
# - At runtime, we want to subclass the real LightningDataModule when
#   `pytorch_lightning` is installed.
# - During static analysis, Pylance/pyright can report assignment-type conflicts
#   if the same symbol is conditionally rebound to both a third-party class and a
#   local fallback shim.
#
# The `TYPE_CHECKING` split avoids that confusion:
# - type checkers see only the real Lightning class
# - runtime still gets a minimal fallback so this file remains importable even in
#   lightweight environments where Lightning is not installed yet
if TYPE_CHECKING:
    from pytorch_lightning import LightningDataModule as _LightningDataModuleBase
else:
    try:
        from pytorch_lightning import LightningDataModule as _LightningDataModuleBase
    except ImportError:  # pragma: no cover - lightweight compatibility fallback
        class _LightningDataModuleBase:
            # The fallback base is intentionally minimal. Its purpose is not to
            # emulate Lightning fully, but simply to keep local imports and basic
            # scripts from crashing immediately when Lightning is absent.
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()


# ============================================================
# Lightning DataModule
# ============================================================
# Purpose:
#   Orchestrate download, preprocessing, split construction,
#   dataset creation, and DataLoader creation in the Lightning
#   lifecycle.
#
# Lifecycle split:
#   prepare_data() -> disk side effects
#   setup()        -> in-memory objects
#   dataloaders    -> batching only
# ============================================================
class AZT1DDataModule(_LightningDataModuleBase):
    """
    LightningDataModule that orchestrates AZT1D download, build, split, and loaders.

    Design intent:
    - `prepare_data()` owns side effects on disk.
    - `setup()` owns in-memory dataset construction.
    - loader methods only wrap datasets in DataLoaders.

    This keeps the data path aligned with Lightning conventions and prevents the
    old dataset module from becoming a "god file" again.
    """

    def __init__(self, config: DataConfig) -> None:
        """
        Bind the shared data config and initialize lazy dataset/runtime metadata fields.

        Context:
        the DataModule derives semantic feature groups immediately, but leaves
        category vocabularies and split datasets unset until `setup()` has seen
        the cleaned dataframe.
        """
        super().__init__()
        self.config = config

        # Derive feature groups once from the shared config contract so every
        # downstream layer consumes the same semantic grouping.
        self.feature_groups: FeatureGroups = build_feature_groups(config)

        self.category_maps: dict[str, tuple[str, ...]] = {}
        self.categorical_cardinalities: dict[str, int] = {}
        # Cardinalities are cached on the DataModule because they are model-facing
        # metadata derived from data preparation. Keeping them here makes it easy
        # for training code to configure embeddings after setup has run.
        self.cleaned_dataframe: Any | None = None

        self.train_dataset: AZT1DSequenceDataset | None = None
        self.val_dataset: AZT1DSequenceDataset | None = None
        self.test_dataset: AZT1DSequenceDataset | None = None

    def prepare_data(self) -> None:
        """
        Materialize the processed dataset on disk if it is not already available.

        Context:
        this is the DataModule's side-effecting stage: download raw bytes if
        needed, extract the archive, and rebuild the canonical processed CSV.
        """
        # Keep this method idempotent because Lightning may call it more than
        # once in distributed settings.
        processed_path = Path(self.config.processed_file_path)
        if processed_path.exists() and not self.config.rebuild_processed:
            return

        if not self.config.dataset_url:
            raise ValueError(
                "DataConfig.dataset_url is required when the processed dataset "
                "does not already exist on disk."
            )

        downloader = AZT1DDownloader(
            raw_dir=self.config.raw_dir,
            cache_dir=self.config.cache_dir,
            extract_dir=self.config.extracted_dir,
        )
        download_result = downloader.download(
            url=self.config.dataset_url,
            filename=f"{self.config.dataset_name}.zip",
            extract=True,
            force=self.config.redownload,
        )

        raw_dataset_dir = download_result.extracted_path
        if raw_dataset_dir is None:
            raise ValueError("Expected the AZT1D archive to extract into a dataset directory.")

        preprocessor = AZT1DPreprocessor(
            dataset_dir=raw_dataset_dir,
            output_file=processed_path,
        )
        preprocessor.build(force=self.config.rebuild_processed)

    def setup(self, stage: str | None = None) -> None:
        """
        Build in-memory split datasets and categorical metadata from the processed CSV.

        Context:
        once `prepare_data()` has handled disk state, `setup()` owns the pure
        in-memory objects shared by training, validation, and test dataloaders.
        """
        # Once we reach setup, raw data should already be materialized on disk.
        # From this point onward the job is to create in-memory training objects.
        #
        # `stage` is accepted to match the LightningDataModule interface, but we
        # currently build all splits eagerly because the same cleaned dataframe
        # and fitted vocabularies are shared across train/val/test construction.
        dataframe = load_processed_frame(
            self.config.processed_file_path,
            self.config,
            self.feature_groups,
        )
        self.cleaned_dataframe = dataframe

        # Fit category vocabularies before split datasets are created so every
        # split shares the same ID mapping and embedding cardinalities.
        self.category_maps = build_category_maps(dataframe, self.feature_groups)
        self.categorical_cardinalities = {
            column: len(categories)
            for column, categories in self.category_maps.items()
        }
        # Splitting the cleaned dataframe first and building indices second keeps
        # the responsibilities crisp:
        # - split helpers decide which rows belong to each split
        # - indexing helpers decide which contiguous windows inside those rows are
        #   legal samples for the model
        #
        # That separation makes it much easier to swap split policies without
        # rewriting sequence-boundary logic.

        split_frames = split_processed_frame(dataframe, self.config, self.feature_groups)
        train_index = build_sequence_index(split_frames["train"], self.config, self.feature_groups)
        val_index = build_sequence_index(split_frames["val"], self.config, self.feature_groups)
        test_index = build_sequence_index(split_frames["test"], self.config, self.feature_groups)

        # Small subject timelines can leave a split with no valid windows. Rather
        # than silently producing an unusable training setup, we fall back to
        # building the train split from the full cleaned dataframe if needed.
        if not train_index:
            split_frames["train"] = dataframe.reset_index(drop=True)
            train_index = build_sequence_index(split_frames["train"], self.config, self.feature_groups)

        self.train_dataset = AZT1DSequenceDataset(
            dataframe=split_frames["train"],
            sample_index=train_index,
            feature_groups=self.feature_groups,
            category_maps=self.category_maps,
        )
        self.val_dataset = AZT1DSequenceDataset(
            dataframe=split_frames["val"],
            sample_index=val_index,
            feature_groups=self.feature_groups,
            category_maps=self.category_maps,
        )
        self.test_dataset = AZT1DSequenceDataset(
            dataframe=split_frames["test"],
            sample_index=test_index,
            feature_groups=self.feature_groups,
            category_maps=self.category_maps,
        )

    def train_dataloader(self) -> DataLoader[BatchItem]:
        """Wrap the prepared training dataset in a DataLoader using the shared loader policy."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader().")

        # Loader methods should stay boring on purpose: batching, shuffling, and
        # worker settings only. All preprocessing belongs earlier in the stack.
        dataloader_kwargs = self._shared_dataloader_kwargs()
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=len(self.train_dataset) > 0,
            drop_last=self.config.drop_last_train,
            **dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader[BatchItem]:
        """Wrap the prepared validation dataset in a non-shuffled DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader[BatchItem]:
        """Wrap the prepared test dataset in a non-shuffled DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("setup() must be called before test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def _shared_dataloader_kwargs(self) -> dict[str, Any]:
        """
        Build the DataLoader keyword arguments shared by train/val/test loaders.

        Context:
        worker, pin-memory, persistent-worker, and prefetch semantics should
        stay aligned across split loaders, so the policy is centralized here.
        """
        # The DataLoader policy is intentionally centralized here so train/val/
        # test loaders stay aligned on worker semantics.
        #
        # Two subtle rules matter:
        # - `persistent_workers` is only meaningful when worker processes exist
        # - `prefetch_factor` should only be passed when multiprocessing is in
        #   play, otherwise the repo would expose a knob that has no effect
        kwargs: dict[str, Any] = {
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": (
                self.config.persistent_workers and self.config.num_workers > 0
            ),
        }
        if self.config.num_workers > 0 and self.config.prefetch_factor is not None:
            # Passing this conditionally keeps the runtime summary and the real
            # DataLoader behavior aligned. If workers are disabled, prefetching
            # is effectively a non-concept for this loader.
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return kwargs

    def get_tft_categorical_cardinalities(self) -> dict[str, list[int]]:
        """
        Return categorical embedding sizes in the exact order expected by TFTConfig.

        Context:
        - vocabulary sizes are discovered from the cleaned dataframe at runtime
        - the DataModule is the layer that owns that dataframe and fits category
          maps during `setup()`
        - the model should consume this metadata, not discover it for itself

        Returning plain lists keeps the API flexible. Training scripts, notebooks,
        and tests can either inject these values into config manually or use the
        convenience `bind_model_config()` helper below.
        """
        if not self.categorical_cardinalities:
            raise RuntimeError(
                "setup() must be called before reading TFT categorical cardinalities."
            )

        return {
            "static_categorical_inp_lens": [
                self.categorical_cardinalities[column]
                for column in self.feature_groups.static_categorical
            ],
            "temporal_known_categorical_inp_lens": [
                self.categorical_cardinalities[column]
                for column in self.feature_groups.known_categorical
            ],
            "temporal_observed_categorical_inp_lens": [
                self.categorical_cardinalities[column]
                for column in self.feature_groups.observed_categorical
            ],
        }

    def describe_data(self) -> dict[str, Any]:
        """
        Return descriptive statistics for the cleaned dataset held by this DataModule.

        Context:
        once `setup()` has built the cleaned dataframe and split metadata, the
        DataModule is the natural place for callers to ask high-level questions
        about dataset size, feature distributions, and split/window counts.
        """
        if self.cleaned_dataframe is None:
            raise RuntimeError("setup() must be called before describing the dataset.")

        return describe_clean_frame(
            self.cleaned_dataframe,
            self.config,
            self.feature_groups,
        )

    def bind_model_config(self, config: Config) -> Config:
        """
        Return a new top-level Config whose TFT section matches this DataModule.

        Why return a new config instead of mutating in place:
        - configuration objects are easier to reason about when updates are
          explicit and local
        - callers can keep both the original declarative config and the runtime
          bound config if they want to log or compare them
        - the DataModule provides the discovered metadata, but the training
          bootstrap still owns the final model-construction step
        """
        categorical_lens = self.get_tft_categorical_cardinalities()
        feature_specs = tuple(config.data.features) or self._build_fallback_feature_specs()

        bound_tft = replace(
            config.tft,
            features=feature_specs,
            static_categorical_inp_lens=categorical_lens["static_categorical_inp_lens"],
            temporal_known_categorical_inp_lens=(
                categorical_lens["temporal_known_categorical_inp_lens"]
            ),
            temporal_observed_categorical_inp_lens=(
                categorical_lens["temporal_observed_categorical_inp_lens"]
            ),
            encoder_length=config.data.encoder_length,
            example_length=config.data.encoder_length + config.data.prediction_length,
        )
        return replace(config, tft=bound_tft)

    def _build_fallback_feature_specs(self) -> tuple[FeatureSpec, ...]:
        """
        Synthesize `FeatureSpec` entries from the documented fallback feature groups.

        Context:
        this preserves one shared model/data contract during the migration period
        where some call sites still rely on AZT1D-specific fallback group
        definitions instead of a fully populated `config.data.features`.
        """
        # The long-term design is for `config.data.features` to be the single
        # source of truth. During the transition period, the DataModule may still
        # be operating from documented fallback feature groups. In that case we
        # synthesize equivalent FeatureSpec entries here so the model config sees
        # the same contract as the dataset.
        feature_specs: list[FeatureSpec] = []

        feature_specs.extend(
            FeatureSpec(name, InputTypes.STATIC, DataTypes.CATEGORICAL)
            for name in self.feature_groups.static_categorical
        )
        feature_specs.extend(
            FeatureSpec(name, InputTypes.STATIC, DataTypes.CONTINUOUS)
            for name in self.feature_groups.static_continuous
        )
        feature_specs.extend(
            FeatureSpec(name, InputTypes.KNOWN, DataTypes.CATEGORICAL)
            for name in self.feature_groups.known_categorical
        )
        feature_specs.extend(
            FeatureSpec(name, InputTypes.KNOWN, DataTypes.CONTINUOUS)
            for name in self.feature_groups.known_continuous
        )
        feature_specs.extend(
            FeatureSpec(name, InputTypes.OBSERVED, DataTypes.CATEGORICAL)
            for name in self.feature_groups.observed_categorical
        )
        feature_specs.extend(
            FeatureSpec(name, InputTypes.OBSERVED, DataTypes.CONTINUOUS)
            for name in self.feature_groups.observed_continuous
        )
        feature_specs.append(
            FeatureSpec(self.feature_groups.target_column, InputTypes.TARGET, DataTypes.CONTINUOUS)
        )
        return tuple(feature_specs)
