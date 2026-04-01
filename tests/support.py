"""
AI-assisted implementation note:
This helper module was drafted with AI assistance and then reviewed/adapted for
this project. It provides typed helpers for the refactored AZT1D data tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from config import DataConfig


# These Protocols describe the callable fixtures provided by `conftest.py`.
# Giving them names has two benefits:
# 1. Pylance can understand fixture return shapes inside test functions.
# 2. The tests read more like documentation because the fixture roles are named
#    after what they do rather than appearing as untyped callables.
class WriteProcessedCsv(Protocol):
    def __call__(
        self,
        *,
        filename: str = "processed.csv",
        subject_ids: tuple[str, ...] = ("subject_a",),
        steps_per_subject: int = 12,
        gap_after_step: int | None = None,
    ) -> Path: ...


class BuildDataConfig(Protocol):
    def __call__(self, processed_csv_path: Path, **overrides: Any) -> DataConfig: ...
