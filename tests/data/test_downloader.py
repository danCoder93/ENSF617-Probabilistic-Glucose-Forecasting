"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D downloader behavior.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import TracebackType
from typing import Iterator

import pytest
from data.downloader import AZT1DDownloader


# ============================================================
# Downloader tests
# ============================================================
# Purpose:
#   Verify raw-file acquisition behavior without using the real
#   network.
# ============================================================

class FakeResponse:
    """
    Small response stub used to test the downloader without real network access.
    """

    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        yield self._payload


def test_downloader_uses_cache_dir_for_temporary_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    downloader = AZT1DDownloader(raw_dir=raw_dir, cache_dir=cache_dir, extract_dir=tmp_path / "extracted")

    monkeypatch.setattr(
        downloader.session,
        "get",
        lambda *args, **kwargs: FakeResponse(
            b"payload",
            headers={"Content-Type": "application/octet-stream"},
        ),
    )

    # The important behavior here is not just that a file downloads, but that
    # partial artifacts are staged in `cache_dir` and cleaned up afterward.
    result = downloader.download("https://example.com/file.bin", filename="sample.bin")

    assert result.file_path == raw_dir / "sample.bin"
    assert result.file_path.read_bytes() == b"payload"
    assert not any(cache_dir.glob("*.part"))


def test_downloader_extracts_zip_archives(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    extract_dir = tmp_path / "extracted"
    downloader = AZT1DDownloader(raw_dir=raw_dir, cache_dir=tmp_path / "cache", extract_dir=extract_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/file.txt", "hello")

    # Extraction is intentionally a downloader concern because archive handling
    # belongs to raw-file acquisition rather than preprocessing.
    extracted_path = downloader._extract_if_needed(archive_path)

    assert extracted_path == extract_dir / "dataset"
    assert (extract_dir / "dataset" / "nested" / "file.txt").read_text() == "hello"
