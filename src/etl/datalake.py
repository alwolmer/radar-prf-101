from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "data"


def _normalize_subpath(subpath: str) -> str:
    normalized = Path(subpath).as_posix().strip("/")
    if not normalized:
        raise ValueError("Datalake subpath must not be empty")
    return normalized


class DatalakeAdapter(ABC):
    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> DatalakeAdapter:
        project_root = project_root or PROJECT_ROOT
        load_dotenv(project_root / ".env")

        adapter_config = dict(config or {})
        backend = str(
            adapter_config.get("backend") or os.getenv("DATALAKE_BACKEND", "local")
        ).strip().lower()

        if backend == "local":
            local_root = Path(
                adapter_config.get("local_root")
                or os.getenv("DATALAKE_LOCAL_ROOT")
                or DEFAULT_LOCAL_ROOT
            )
            return LocalDatalakeAdapter(root=local_root)

        if backend == "s3":
            bucket = str(
                adapter_config.get("bucket")
                or os.getenv("DATALAKE_S3_BUCKET")
                or os.getenv("AWS_S3_BUCKET")
                or ""
            ).strip()
            if not bucket:
                raise ValueError(
                    "S3 datalake backend requires DATALAKE_S3_BUCKET or config['bucket']"
                )

            prefix = str(
                adapter_config.get("prefix") or os.getenv("DATALAKE_S3_PREFIX", "")
            ).strip().strip("/")
            region_name = str(
                adapter_config.get("region_name")
                or os.getenv("DATALAKE_S3_REGION")
                or os.getenv("AWS_REGION")
                or os.getenv("AWS_DEFAULT_REGION")
                or ""
            ).strip() or None
            endpoint_url = str(
                adapter_config.get("endpoint_url") or os.getenv("DATALAKE_S3_ENDPOINT", "")
            ).strip() or None

            return S3DatalakeAdapter(
                bucket=bucket,
                prefix=prefix,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )

        raise ValueError(f"Unsupported datalake backend: {backend}")

    @abstractmethod
    def stage_directory(self, subpath: str, target_root: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def stage_file(self, subpath: str, target_root: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def persist_directory(self, local_dir: Path, subpath: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def persist_file(self, local_file: Path, subpath: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def uri_for(self, subpath: str) -> str:
        raise NotImplementedError


class LocalDatalakeAdapter(DatalakeAdapter):
    def __init__(self, *, root: Path) -> None:
        self.root = Path(root)

    def stage_directory(self, subpath: str, target_root: Path) -> Path:
        source_dir = self.root / _normalize_subpath(subpath)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Datalake directory not found: {source_dir}")
        return source_dir

    def stage_file(self, subpath: str, target_root: Path) -> Path:
        source_file = self.root / _normalize_subpath(subpath)
        if not source_file.is_file():
            raise FileNotFoundError(f"Datalake file not found: {source_file}")
        return source_file

    def persist_directory(self, local_dir: Path, subpath: str) -> str:
        source_dir = Path(local_dir)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Local directory to persist not found: {source_dir}")

        target_dir = self.root / _normalize_subpath(subpath)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
            else:
                target_dir.unlink()
        shutil.move(str(source_dir), str(target_dir))
        return str(target_dir)

    def persist_file(self, local_file: Path, subpath: str) -> str:
        source_file = Path(local_file)
        if not source_file.is_file():
            raise FileNotFoundError(f"Local file to persist not found: {source_file}")

        target_file = self.root / _normalize_subpath(subpath)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if target_file.exists():
            if target_file.is_dir():
                shutil.rmtree(target_file)
            else:
                target_file.unlink()
        shutil.move(str(source_file), str(target_file))
        return str(target_file)

    def uri_for(self, subpath: str) -> str:
        return str(self.root / _normalize_subpath(subpath))


class S3DatalakeAdapter(DatalakeAdapter):
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "S3 datalake backend requires boto3 to be installed"
            ) from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self._client = boto3.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        )

    def _key(self, subpath: str) -> str:
        normalized_subpath = _normalize_subpath(subpath)
        if self.prefix:
            return f"{self.prefix}/{normalized_subpath}"
        return normalized_subpath

    def _directory_prefix(self, subpath: str) -> str:
        return f"{self._key(subpath).rstrip('/')}/"

    def _list_objects(self, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        keys: list[str] = []
        for page in page_iterator:
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith("/"):
                    keys.append(key)
        return keys

    def _delete_prefix(self, prefix: str) -> None:
        keys = self._list_objects(prefix)
        if not keys:
            return

        for index in range(0, len(keys), 1000):
            batch = keys[index : index + 1000]
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in batch]},
            )

    def stage_directory(self, subpath: str, target_root: Path) -> Path:
        prefix = self._directory_prefix(subpath)
        keys = self._list_objects(prefix)
        if not keys:
            raise FileNotFoundError(f"S3 datalake directory not found: {self.uri_for(subpath)}")

        target_dir = Path(target_root) / Path(_normalize_subpath(subpath))
        for key in keys:
            relative_key = key[len(prefix) :]
            destination = target_dir / relative_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, key, str(destination))
        return target_dir

    def stage_file(self, subpath: str, target_root: Path) -> Path:
        key = self._key(subpath)
        target_file = Path(target_root) / Path(_normalize_subpath(subpath))
        target_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, key, str(target_file))
        except Exception as exc:
            raise FileNotFoundError(f"S3 datalake file not found: {self.uri_for(subpath)}") from exc
        return target_file

    def persist_directory(self, local_dir: Path, subpath: str) -> str:
        source_dir = Path(local_dir)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Local directory to persist not found: {source_dir}")

        prefix = self._directory_prefix(subpath)
        self._delete_prefix(prefix)

        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(source_dir).as_posix()
            key = f"{prefix}{relative_path}"
            self._client.upload_file(str(path), self.bucket, key)

        return self.uri_for(subpath)

    def persist_file(self, local_file: Path, subpath: str) -> str:
        source_file = Path(local_file)
        if not source_file.is_file():
            raise FileNotFoundError(f"Local file to persist not found: {source_file}")

        key = self._key(subpath)
        self._client.upload_file(str(source_file), self.bucket, key)
        return self.uri_for(subpath)

    def uri_for(self, subpath: str) -> str:
        key = self._key(subpath)
        return f"s3://{self.bucket}/{key}"
