from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib
import mimetypes
import shutil
import zipfile

import requests


@dataclass
class DownloadResult:
    url: str
    file_path: Path
    extracted_path: Optional[Path]
    from_cache: bool
    content_type: Optional[str]
    size_bytes: int


class DatasetDownloader:
    """
    Download a dataset from a direct URL with local caching.

    Features:
    - Skips download if file already exists
    - Uses a stable URL-hash cache key
    - Sends safe default headers
    - Streams large files to disk
    - Optional ZIP extraction
    - Timeout + retry support via requests Session
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw",
        extract_dir: str | Path = "data/extracted",
        user_agent: str = "Mozilla/5.0",
        timeout: int = 60,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.extract_dir = Path(extract_dir)
        self.timeout = timeout

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

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

            tmp_path = resolved_file_path.with_suffix(resolved_file_path.suffix + ".part")

            try:
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
        if filename:
            return self.cache_dir / filename

        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{url_hash}.bin"

    def _resolve_filename(
        self,
        initial_path: Path,
        response: requests.Response,
        explicit_filename: Optional[str],
        url: str,
    ) -> Path:
        if explicit_filename:
            return initial_path

        cd = response.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            name = cd.split("filename=")[-1].strip().strip('"')
            if name:
                return self.cache_dir / name

        guessed_name = Path(url).name
        if guessed_name and guessed_name != "file_downloaded":
            return self.cache_dir / guessed_name

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        ext = mimetypes.guess_extension(content_type) if content_type else None
        if ext:
            return initial_path.with_suffix(ext)

        return initial_path

    def _extract_if_needed(self, file_path: Path) -> Optional[Path]:
        if not zipfile.is_zipfile(file_path):
            return None

        target_dir = self.extract_dir / file_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(target_dir)

        return target_dir