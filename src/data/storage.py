"""
GCS (Google Cloud Storage) Integration Module.

Provides storage abstraction for the Crypto Quant Ecosystem:
- Bot log retrieval from GCS
- Model storage and loading
- Processed data sync

Part of: crypto-quant-system -> crypto-bot -> crypto-regime-classifier-ml
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from google.cloud.storage import Bucket, Client  # type: ignore[import-untyped]

logger = get_logger(__name__)


class GCSStorageError(Exception):
    """Exception raised for GCS storage errors."""


def _get_gcs_client() -> Client:
    """Get GCS client with lazy import."""
    try:
        from google.cloud import storage  # type: ignore[import-untyped]

        return storage.Client()
    except ImportError as e:
        raise GCSStorageError(
            "google-cloud-storage not installed. Install with: pip install google-cloud-storage"
        ) from e
    except (OSError, ValueError, TypeError, RuntimeError) as e:
        raise GCSStorageError(f"Failed to create GCS client: {e}") from e
    except Exception as e:
        # google.auth.exceptions.DefaultCredentialsError and similar auth errors
        if "google.auth" in type(e).__module__:
            raise GCSStorageError(f"GCS credentials not configured: {e}") from e
        raise


class GCSStorage:
    """
    Google Cloud Storage interface for the Crypto Quant Ecosystem.

    Bucket structure:
        gs://{bucket}/
        ├── logs/
        │   └── {account}/
        │       ├── trades_{date}.csv
        │       └── positions.json
        ├── models/
        │   └── regime_classifier_v1.pkl
        └── data/
            └── processed/
                ├── BTC_1d.parquet
                └── ETH_1d.parquet
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        project: str | None = None,
    ) -> None:
        """
        Initialize GCS storage.

        Args:
            bucket_name: GCS bucket name (reads from GCS_BUCKET env if not provided)
            project: GCP project ID (optional, uses default if not provided)
        """
        import os

        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET")
        if not self.bucket_name:
            raise GCSStorageError(
                "GCS bucket not specified. Set GCS_BUCKET environment variable "
                "or pass bucket_name parameter."
            )

        self.project = project
        self._client: Client | None = None
        self._bucket: Bucket | None = None

    @property
    def client(self) -> Client:
        """Lazy-load GCS client."""
        if self._client is None:
            self._client = _get_gcs_client()
        return self._client

    @property
    def bucket(self) -> Bucket:
        """Get bucket reference."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    # =========================================================================
    # Bot Logs
    # =========================================================================

    def get_bot_logs(
        self,
        date: str | datetime,
        account: str = "Main",
    ) -> pd.DataFrame:
        """
        Get bot trade logs for a specific date.

        Args:
            date: Date string (YYYY-MM-DD) or datetime object
            account: Account name (default: "Main")

        Returns:
            DataFrame with trade logs
        """
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date

        blob_path = f"logs/{account}/trades_{date_str}.csv"

        try:
            blob = self.bucket.blob(blob_path)

            if not blob.exists():
                logger.warning(f"No logs found for {account} on {date_str}")
                return pd.DataFrame()

            content = blob.download_as_text()
            from io import StringIO

            return pd.read_csv(StringIO(content))

        except (GCSStorageError, OSError, ValueError, KeyError) as e:
            logger.error(f"Error reading bot logs: {e}")
            raise GCSStorageError(f"Failed to read bot logs: {e}") from e

    def get_bot_positions(self, account: str = "Main") -> dict[str, Any]:
        """
        Get current bot positions.

        Args:
            account: Account name (default: "Main")

        Returns:
            Dictionary with current positions
        """
        blob_path = f"logs/{account}/positions.json"

        try:
            blob = self.bucket.blob(blob_path)

            if not blob.exists():
                logger.warning(f"No positions found for {account}")
                return {}

            content = blob.download_as_text()
            result: dict[str, Any] = json.loads(content)
            return result

        except (GCSStorageError, OSError, ValueError, KeyError) as e:
            logger.error(f"Error reading positions: {e}")
            raise GCSStorageError(f"Failed to read positions: {e}") from e

    def list_bot_log_dates(
        self,
        account: str = "Main",
        limit: int = 30,
    ) -> list[str]:
        """
        List available log dates for an account.

        Args:
            account: Account name
            limit: Maximum number of dates to return

        Returns:
            List of date strings (YYYY-MM-DD), most recent first
        """
        prefix = f"logs/{account}/trades_"

        try:
            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix=prefix,
                max_results=limit * 2,  # Some buffer for filtering
            )

            dates = []
            for blob in blobs:
                # Extract date from trades_YYYY-MM-DD.csv
                name = blob.name.split("/")[-1]
                if name.startswith("trades_") and name.endswith(".csv"):
                    date_str = name[7:-4]  # Remove "trades_" and ".csv"
                    dates.append(date_str)

            # Sort descending (most recent first)
            dates.sort(reverse=True)
            return dates[:limit]

        except (GCSStorageError, OSError, ValueError) as e:
            logger.error(f"Error listing log dates: {e}")
            return []

    def list_accounts(self) -> list[str]:
        """
        List available bot accounts.

        Returns:
            List of account names
        """
        try:
            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix="logs/",
                delimiter="/",
            )

            # Trigger the request to populate prefixes
            list(blobs)

            accounts = []
            for prefix in blobs.prefixes:
                # prefix looks like "logs/Main/"
                account = prefix.rstrip("/").split("/")[-1]
                accounts.append(account)

            return sorted(accounts)

        except (GCSStorageError, OSError, ValueError) as e:
            logger.error(f"Error listing accounts: {e}")
            return []

    # =========================================================================
    # Models
    # =========================================================================

    def download_model(
        self,
        model_name: str,
        local_path: Path | str,
    ) -> Path:
        """
        Download a model from GCS.

        Args:
            model_name: Model filename (e.g., "regime_classifier_v1.pkl")
            local_path: Local directory or file path to save

        Returns:
            Path to downloaded model
        """
        blob_path = f"models/{model_name}"
        local_path = Path(local_path)

        if local_path.is_dir():
            local_path = local_path / model_name

        try:
            blob = self.bucket.blob(blob_path)

            if not blob.exists():
                raise GCSStorageError(f"Model not found: {model_name}")

            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))

            logger.info(f"Downloaded model: {model_name} -> {local_path}")
            return local_path

        except GCSStorageError:
            raise
        except (OSError, ValueError, KeyError) as e:
            raise GCSStorageError(f"Failed to download model: {e}") from e

    def upload_model(
        self,
        local_path: Path | str,
        model_name: str | None = None,
    ) -> str:
        """
        Upload a model to GCS.

        Args:
            local_path: Local path to model file
            model_name: Target name in GCS (uses local filename if not provided)

        Returns:
            GCS URI of uploaded model
        """
        local_path = Path(local_path)
        model_name = model_name or local_path.name
        blob_path = f"models/{model_name}"

        try:
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))

            uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded model: {local_path} -> {uri}")
            return uri

        except (GCSStorageError, OSError, ValueError) as e:
            raise GCSStorageError(f"Failed to upload model: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        """
        List available models.

        Returns:
            List of model info dictionaries
        """
        try:
            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix="models/",
            )

            models = []
            for blob in blobs:
                if blob.name == "models/":
                    continue

                name = blob.name.split("/")[-1]
                models.append(
                    {
                        "name": name,
                        "path": blob.name,
                        "size_bytes": blob.size,
                        "updated": blob.updated,
                    }
                )

            return models

        except (GCSStorageError, OSError, ValueError) as e:
            logger.error(f"Error listing models: {e}")
            return []

    # =========================================================================
    # Processed Data
    # =========================================================================

    def upload_data(
        self,
        local_path: Path | str,
        filename: str | None = None,
    ) -> str:
        """
        Upload processed data to GCS.

        Args:
            local_path: Local path to data file
            filename: Target filename (uses local filename if not provided)

        Returns:
            GCS URI of uploaded data
        """
        local_path = Path(local_path)
        filename = filename or local_path.name
        blob_path = f"data/processed/{filename}"

        try:
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))

            uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded data: {local_path} -> {uri}")
            return uri

        except (GCSStorageError, OSError, ValueError) as e:
            raise GCSStorageError(f"Failed to upload data: {e}") from e

    def download_data(
        self,
        filename: str,
        local_path: Path | str,
    ) -> Path:
        """
        Download processed data from GCS.

        Args:
            filename: Data filename (e.g., "BTC_1d.parquet")
            local_path: Local directory or file path to save

        Returns:
            Path to downloaded data
        """
        blob_path = f"data/processed/{filename}"
        local_path = Path(local_path)

        if local_path.is_dir():
            local_path = local_path / filename

        try:
            blob = self.bucket.blob(blob_path)

            if not blob.exists():
                raise GCSStorageError(f"Data not found: {filename}")

            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))

            logger.info(f"Downloaded data: {filename} -> {local_path}")
            return local_path

        except GCSStorageError:
            raise
        except (OSError, ValueError, KeyError) as e:
            raise GCSStorageError(f"Failed to download data: {e}") from e

    def list_data(self) -> list[dict[str, Any]]:
        """
        List available processed data files.

        Returns:
            List of data file info dictionaries
        """
        try:
            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix="data/processed/",
            )

            data_files = []
            for blob in blobs:
                if blob.name == "data/processed/":
                    continue

                name = blob.name.split("/")[-1]
                data_files.append(
                    {
                        "name": name,
                        "path": blob.name,
                        "size_bytes": blob.size,
                        "updated": blob.updated,
                    }
                )

            return data_files

        except (GCSStorageError, OSError, ValueError) as e:
            logger.error(f"Error listing data: {e}")
            return []


@lru_cache(maxsize=1)
def get_gcs_storage() -> GCSStorage | None:
    """
    Get cached GCS storage instance.

    Returns None if GCS is not configured.

    Returns:
        GCSStorage instance or None
    """
    import os

    if not os.getenv("GCS_BUCKET"):
        return None

    try:
        return GCSStorage()
    except GCSStorageError:
        return None


def is_gcs_available() -> bool:
    """Check if GCS storage is available and configured."""
    return get_gcs_storage() is not None
