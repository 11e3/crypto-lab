"""Tests for GCS storage module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.storage import (
    GCSStorage,
    GCSStorageError,
    _get_gcs_client,
    get_gcs_storage,
    is_gcs_available,
)

# =========================================================================
# _get_gcs_client
# =========================================================================


class TestGetGCSClient:
    """Tests for _get_gcs_client helper."""

    def test_import_error_raises_storage_error(self) -> None:
        with (
            patch.dict("sys.modules", {"google.cloud": None, "google.cloud.storage": None}),
            pytest.raises(GCSStorageError, match="not installed"),
        ):
            _get_gcs_client()

    def test_os_error_raises_storage_error(self) -> None:
        mock_storage_mod = MagicMock()
        mock_storage_mod.Client.side_effect = OSError("connection failed")
        mock_google_cloud = MagicMock()
        mock_google_cloud.storage = mock_storage_mod
        with (
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.cloud": mock_google_cloud,
                    "google.cloud.storage": mock_storage_mod,
                },
            ),
            pytest.raises(GCSStorageError, match="Failed to create GCS client"),
        ):
            _get_gcs_client()

    def test_auth_error_raises_storage_error(self) -> None:
        """Google auth errors are caught by the generic Exception handler."""

        class FakeAuthError(Exception):
            __module__ = "google.auth.exceptions"

        mock_storage_mod = MagicMock()
        mock_storage_mod.Client.side_effect = FakeAuthError("no credentials")
        mock_google_cloud = MagicMock()
        mock_google_cloud.storage = mock_storage_mod
        with (
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.cloud": mock_google_cloud,
                    "google.cloud.storage": mock_storage_mod,
                },
            ),
            pytest.raises(GCSStorageError, match="credentials not configured"),
        ):
            _get_gcs_client()


# =========================================================================
# GCSStorage.__init__
# =========================================================================


class TestGCSStorageInit:
    """Tests for GCSStorage initialization."""

    def test_raises_without_bucket(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(GCSStorageError, match="bucket not specified"),
        ):
            GCSStorage(bucket_name=None)

    def test_uses_env_variable(self) -> None:
        with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
            storage = GCSStorage()
            assert storage.bucket_name == "test-bucket"

    def test_explicit_bucket_name(self) -> None:
        storage = GCSStorage(bucket_name="my-bucket")
        assert storage.bucket_name == "my-bucket"

    def test_lazy_client_not_initialized(self) -> None:
        storage = GCSStorage(bucket_name="my-bucket")
        assert storage._client is None
        assert storage._bucket is None


# =========================================================================
# GCSStorage.client / bucket properties
# =========================================================================


class TestGCSStorageProperties:
    """Tests for lazy-loaded client and bucket properties."""

    def test_client_lazy_loaded(self) -> None:
        storage = GCSStorage(bucket_name="test-bucket")
        mock_client = MagicMock()
        with patch("src.data.storage._get_gcs_client", return_value=mock_client):
            result = storage.client
            assert result is mock_client

    def test_bucket_lazy_loaded(self) -> None:
        storage = GCSStorage(bucket_name="test-bucket")
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        with patch("src.data.storage._get_gcs_client", return_value=mock_client):
            result = storage.bucket
            assert result is mock_bucket
            mock_client.bucket.assert_called_once_with("test-bucket")


# =========================================================================
# Bot Logs
# =========================================================================


class TestBotLogs:
    """Tests for bot log operations."""

    def _make_storage(self) -> tuple[GCSStorage, MagicMock]:
        storage = GCSStorage(bucket_name="test-bucket")
        mock_bucket = MagicMock()
        storage._bucket = mock_bucket
        storage._client = MagicMock()
        return storage, mock_bucket

    def test_get_bot_logs_returns_dataframe(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = "col1,col2\na,b\nc,d"
        mock_bucket.blob.return_value = mock_blob

        result = storage.get_bot_logs("2024-01-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_bucket.blob.assert_called_once_with("logs/sh/trades_2024-01-01.csv")

    def test_get_bot_logs_with_datetime(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = "col1\nval1"
        mock_bucket.blob.return_value = mock_blob

        storage.get_bot_logs(datetime(2024, 3, 15))
        mock_bucket.blob.assert_called_once_with("logs/sh/trades_2024-03-15.csv")

    def test_get_bot_logs_not_found(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob

        result = storage.get_bot_logs("2024-01-01")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_bot_logs_error(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_bucket.blob.side_effect = OSError("network error")

        with pytest.raises(GCSStorageError, match="Failed to read bot logs"):
            storage.get_bot_logs("2024-01-01")

    def test_get_bot_positions(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = '{"BTC": 0.5}'
        mock_bucket.blob.return_value = mock_blob

        result = storage.get_bot_positions()
        assert result == {"BTC": 0.5}

    def test_get_bot_positions_not_found(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob

        result = storage.get_bot_positions()
        assert result == {}

    def test_get_bot_positions_error(self) -> None:
        storage, mock_bucket = self._make_storage()
        mock_bucket.blob.side_effect = OSError("fail")

        with pytest.raises(GCSStorageError, match="Failed to read positions"):
            storage.get_bot_positions()


# =========================================================================
# List operations
# =========================================================================


class TestListOperations:
    """Tests for list_bot_log_dates, list_accounts."""

    def _make_storage(self) -> tuple[GCSStorage, MagicMock]:
        storage = GCSStorage(bucket_name="test-bucket")
        mock_client = MagicMock()
        storage._client = mock_client
        storage._bucket = MagicMock()
        return storage, mock_client

    def test_list_bot_log_dates(self) -> None:
        storage, mock_client = self._make_storage()
        blob1 = MagicMock()
        blob1.name = "logs/sh/trades_2024-01-02.csv"
        blob2 = MagicMock()
        blob2.name = "logs/sh/trades_2024-01-01.csv"
        mock_client.list_blobs.return_value = [blob1, blob2]

        result = storage.list_bot_log_dates()
        assert result == ["2024-01-02", "2024-01-01"]

    def test_list_bot_log_dates_empty(self) -> None:
        storage, mock_client = self._make_storage()
        mock_client.list_blobs.return_value = []

        result = storage.list_bot_log_dates()
        assert result == []

    def test_list_bot_log_dates_error(self) -> None:
        storage, mock_client = self._make_storage()
        mock_client.list_blobs.side_effect = OSError("fail")

        result = storage.list_bot_log_dates()
        assert result == []

    def test_list_accounts(self) -> None:
        storage, mock_client = self._make_storage()
        blobs_result = MagicMock()
        blobs_result.__iter__ = MagicMock(return_value=iter([]))
        blobs_result.prefixes = ["logs/Main/", "logs/Sub/"]
        mock_client.list_blobs.return_value = blobs_result

        result = storage.list_accounts()
        assert result == ["Main", "Sub"]

    def test_list_accounts_error(self) -> None:
        storage, mock_client = self._make_storage()
        mock_client.list_blobs.side_effect = OSError("fail")

        result = storage.list_accounts()
        assert result == []


# =========================================================================
# Model operations
# =========================================================================


class TestModelOperations:
    """Tests for model download/upload/list."""

    def _make_storage(self) -> tuple[GCSStorage, MagicMock]:
        storage = GCSStorage(bucket_name="test-bucket")
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        storage._bucket = mock_bucket
        storage._client = mock_client
        return storage, mock_bucket

    def test_download_model(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob

        result = storage.download_model("model.pkl", tmp_path)
        assert result == tmp_path / "model.pkl"
        mock_blob.download_to_filename.assert_called_once()

    def test_download_model_to_file(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob

        target = tmp_path / "my_model.pkl"
        result = storage.download_model("model.pkl", target)
        assert result == target

    def test_download_model_not_found(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob

        with pytest.raises(GCSStorageError, match="Model not found"):
            storage.download_model("missing.pkl", tmp_path)

    def test_download_model_error(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_to_filename.side_effect = OSError("disk full")
        mock_bucket.blob.return_value = mock_blob

        with pytest.raises(GCSStorageError, match="Failed to download model"):
            storage.download_model("model.pkl", tmp_path)

    def test_upload_model(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        model_file = tmp_path / "model.pkl"
        model_file.write_text("data")
        result = storage.upload_model(model_file)
        assert result == "gs://test-bucket/models/model.pkl"

    def test_upload_model_custom_name(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        model_file = tmp_path / "model.pkl"
        model_file.write_text("data")
        result = storage.upload_model(model_file, model_name="v2.pkl")
        assert result == "gs://test-bucket/models/v2.pkl"

    def test_upload_model_error(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_bucket.blob.side_effect = OSError("fail")

        with pytest.raises(GCSStorageError, match="Failed to upload model"):
            storage.upload_model(tmp_path / "model.pkl")

    def test_list_models(self) -> None:
        storage, mock_bucket = self._make_storage()
        blob = MagicMock()
        blob.name = "models/classifier.pkl"
        blob.size = 1024
        blob.updated = "2024-01-01"
        storage._client.list_blobs.return_value = [blob]

        result = storage.list_models()
        assert len(result) == 1
        assert result[0]["name"] == "classifier.pkl"

    def test_list_models_skips_prefix(self) -> None:
        storage, _ = self._make_storage()
        blob = MagicMock()
        blob.name = "models/"
        storage._client.list_blobs.return_value = [blob]

        result = storage.list_models()
        assert result == []

    def test_list_models_error(self) -> None:
        storage, _ = self._make_storage()
        storage._client.list_blobs.side_effect = OSError("fail")

        result = storage.list_models()
        assert result == []


# =========================================================================
# Data operations
# =========================================================================


class TestDataOperations:
    """Tests for processed data upload/download/list."""

    def _make_storage(self) -> tuple[GCSStorage, MagicMock]:
        storage = GCSStorage(bucket_name="test-bucket")
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        storage._bucket = mock_bucket
        storage._client = mock_client
        return storage, mock_bucket

    def test_upload_data(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        data_file = tmp_path / "BTC_1d.parquet"
        data_file.write_text("data")
        result = storage.upload_data(data_file)
        assert result == "gs://test-bucket/data/processed/BTC_1d.parquet"

    def test_upload_data_error(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_bucket.blob.side_effect = OSError("fail")

        with pytest.raises(GCSStorageError, match="Failed to upload data"):
            storage.upload_data(tmp_path / "data.parquet")

    def test_download_data(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob

        result = storage.download_data("BTC_1d.parquet", tmp_path)
        assert result == tmp_path / "BTC_1d.parquet"

    def test_download_data_not_found(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob

        with pytest.raises(GCSStorageError, match="Data not found"):
            storage.download_data("missing.parquet", tmp_path)

    def test_download_data_error(self, tmp_path: Path) -> None:
        storage, mock_bucket = self._make_storage()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_to_filename.side_effect = OSError("fail")
        mock_bucket.blob.return_value = mock_blob

        with pytest.raises(GCSStorageError, match="Failed to download data"):
            storage.download_data("data.parquet", tmp_path)

    def test_list_data(self) -> None:
        storage, _ = self._make_storage()
        blob = MagicMock()
        blob.name = "data/processed/BTC_1d.parquet"
        blob.size = 2048
        blob.updated = "2024-01-01"
        storage._client.list_blobs.return_value = [blob]

        result = storage.list_data()
        assert len(result) == 1
        assert result[0]["name"] == "BTC_1d.parquet"

    def test_list_data_skips_prefix(self) -> None:
        storage, _ = self._make_storage()
        blob = MagicMock()
        blob.name = "data/processed/"
        storage._client.list_blobs.return_value = [blob]

        result = storage.list_data()
        assert result == []

    def test_list_data_error(self) -> None:
        storage, _ = self._make_storage()
        storage._client.list_blobs.side_effect = OSError("fail")

        result = storage.list_data()
        assert result == []


# =========================================================================
# Module-level functions
# =========================================================================


class TestModuleFunctions:
    """Tests for get_gcs_storage and is_gcs_available."""

    def test_get_gcs_storage_returns_none_without_env(self) -> None:
        get_gcs_storage.cache_clear()
        with patch.dict("os.environ", {}, clear=True):
            result = get_gcs_storage()
            assert result is None
        get_gcs_storage.cache_clear()

    def test_is_gcs_available_false_without_env(self) -> None:
        get_gcs_storage.cache_clear()
        with patch.dict("os.environ", {}, clear=True):
            assert is_gcs_available() is False
        get_gcs_storage.cache_clear()

    def test_get_gcs_storage_returns_none_on_error(self) -> None:
        get_gcs_storage.cache_clear()
        with (
            patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}, clear=False),
            patch("src.data.storage.GCSStorage.__init__", side_effect=GCSStorageError("fail")),
        ):
            result = get_gcs_storage()
            assert result is None
        get_gcs_storage.cache_clear()
