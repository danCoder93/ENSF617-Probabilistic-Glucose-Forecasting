# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
# work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
# DataModule docs/tutorial
# (https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
# original AZT1D dataset release on Mendeley Data
# (https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to keep raw
# data download and extraction separate from preprocessing, indexing, and
# dataset assembly.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib
import mimetypes
import shutil
import zipfile

import requests


# ============================================================
# Download result contract
# ============================================================
# Purpose:
#   Describe the outcome of downloading and optionally extracting
#   the raw AZT1D archive.
#
# Context:
#   The downloader should return structured metadata instead of
#   leaking HTTP/archive details into the DataModule.
# ============================================================
@dataclass
class DownloadResult:
    """
    Structured description of one download/extraction outcome.

    Context:
    the DataModule and setup code care about where the file landed, whether the
    result came from cache, and whether an extracted directory exists. Returning
    those facts in one object keeps downloader-specific logic out of callers.
    """
    url: str
    file_path: Path
    extracted_path: Optional[Path]
    from_cache: bool
    content_type: Optional[str]
    size_bytes: int


class AZT1DDownloader:
    """
    Download the raw AZT1D archive and optionally extract it.

    Purpose:
    isolate remote file acquisition and optional archive extraction from the
    rest of the data pipeline.

    Context:
    the downloader intentionally knows nothing about:
    - dataframe columns
    - subject splits
    - tensors

    Its only job is to move bytes from the remote source into the local raw-data
    cache and expose where the extracted files landed.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        cache_dir: str | Path | None = None,
        extract_dir: str | Path = "data/extracted",
        user_agent: str = "Mozilla/5.0",
        timeout: int = 60,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        # `cache_dir` now has a real job: it stores temporary download artifacts
        # such as partial files. If the caller does not provide one, we fall back
        # to placing those artifacts next to the raw archive.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.raw_dir
        self.extract_dir = Path(extract_dir)
        self.timeout = timeout

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        # A persistent Session reuses headers and connections across downloads,
        # which is simpler and a bit more efficient than constructing a fresh
        # requests call for every operation.
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "*/*",
                "Connection": "keep-alive",
            }
        )

    def download(
        self,
        url: str,
        filename: Optional[str] = None,
        extract: bool = False,
        force: bool = False,
    ) -> DownloadResult:
        """
        Download a file from a direct URL.

        Args:
            url: Direct dataset/file URL.
            filename: Optional custom filename.
            extract: If True, auto-extract ZIP files.
            force: If True, re-download even if cached.

        Returns:
            DownloadResult
        """
        file_path = self._build_file_path(url=url, filename=filename)

        # Returning cached results early makes repeated prepare-data runs cheap
        # and keeps download behavior idempotent.
        if file_path.exists() and not force:
            extracted_path = self._extract_if_needed(file_path) if extract else None
            return DownloadResult(
                url=url,
                file_path=file_path,
                extracted_path=extracted_path,
                from_cache=True,
                content_type=None,
                size_bytes=file_path.stat().st_size,
            )

        with self.session.get(url, stream=True, allow_redirects=True, timeout=self.timeout) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type")
            resolved_file_path = self._resolve_filename(
                initial_path=file_path,
                response=response,
                explicit_filename=filename,
                url=url,
            )

            if resolved_file_path.exists() and not force:
                extracted_path = self._extract_if_needed(resolved_file_path) if extract else None
                return DownloadResult(
                    url=url,
                    file_path=resolved_file_path,
                    extracted_path=extracted_path,
                    from_cache=True,
                    content_type=content_type,
                    size_bytes=resolved_file_path.stat().st_size,
                )

            # Partial downloads go into `cache_dir` first, then move atomically
            # into `raw_dir` once the transfer completes. This gives DataConfig's
            # cache directory a concrete responsibility in the pipeline.
            tmp_path = self.cache_dir / f"{resolved_file_path.name}.part"

            try:
                # Stream to a temporary file first so interrupted downloads do
                # not leave a partially written archive at the final path.
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                shutil.move(str(tmp_path), str(resolved_file_path))
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        extracted_path = self._extract_if_needed(resolved_file_path) if extract else None

        return DownloadResult(
            url=url,
            file_path=resolved_file_path,
            extracted_path=extracted_path,
            from_cache=False,
            content_type=content_type,
            size_bytes=resolved_file_path.stat().st_size,
        )

    def _build_file_path(self, url: str, filename: Optional[str]) -> Path:
        # A stable URL hash gives us deterministic cache keys even when the
        # caller does not provide a filename explicitly.
        if filename:
            return self.raw_dir / filename

        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.raw_dir / f"{url_hash}.bin"

    def _resolve_filename(
        self,
        initial_path: Path,
        response: requests.Response,
        explicit_filename: Optional[str],
        url: str,
    ) -> Path:
        # Filename resolution order is:
        # 1. explicit caller-provided filename
        # 2. HTTP Content-Disposition filename
        # 3. basename from the URL
        # 4. extension guessed from Content-Type
        #
        # This order keeps caller intent strongest while still producing readable
        # cached filenames when the server provides enough metadata.
        if explicit_filename:
            return initial_path

        cd = response.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            name = cd.split("filename=")[-1].strip().strip('"')
            if name:
                return self.raw_dir / name

        guessed_name = Path(url).name
        if guessed_name and guessed_name != "file_downloaded":
            return self.raw_dir / guessed_name

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        ext = mimetypes.guess_extension(content_type) if content_type else None
        if ext:
            return initial_path.with_suffix(ext)

        return initial_path

    def _extract_if_needed(self, file_path: Path) -> Optional[Path]:
        # Extraction stays inside the downloader because archive handling is part
        # of raw file acquisition, not preprocessing.
        if not zipfile.is_zipfile(file_path):
            return None

        target_dir = self.extract_dir / file_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(target_dir)

        return target_dir
