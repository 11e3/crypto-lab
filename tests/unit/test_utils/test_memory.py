"""Tests for utils.memory module."""

import numpy as np
import pandas as pd

from src.utils.memory import get_memory_usage_mb, optimize_dtypes, use_float32_for_arrays


class TestMemoryUtilities:
    """Test memory utilities."""

    def test_get_memory_usage_mb_numpy(self) -> None:
        """Test get_memory_usage_mb with numpy array."""
        arr = np.zeros((1000, 1000), dtype=np.float64)
        usage_mb = get_memory_usage_mb(arr)
        assert isinstance(usage_mb, float)
        assert usage_mb > 0
        # ~8MB for 1000x1000 float64
        assert 7 < usage_mb < 10

    def test_get_memory_usage_mb_dataframe(self) -> None:
        """Test get_memory_usage_mb with pandas DataFrame."""
        df = pd.DataFrame({"a": np.arange(1000), "b": np.arange(1000)})
        usage_mb = get_memory_usage_mb(df)
        assert isinstance(usage_mb, float)
        assert usage_mb > 0

    def test_get_memory_usage_mb_list(self) -> None:
        """Test get_memory_usage_mb with list."""
        data = [0] * 100000
        usage_mb = get_memory_usage_mb(data)
        assert isinstance(usage_mb, float)
        assert usage_mb >= 0

    def test_get_memory_usage_mb_dict(self) -> None:
        """Test get_memory_usage_mb with dict."""
        data = {f"key_{i}": i for i in range(1000)}
        usage_mb = get_memory_usage_mb(data)
        assert isinstance(usage_mb, float)
        assert usage_mb >= 0

    def test_optimize_dtypes(self) -> None:
        """Test optimize_dtypes reduces DataFrame memory."""
        df = pd.DataFrame(
            {
                "open": np.random.rand(1000),
                "high": np.random.rand(1000),
                "low": np.random.rand(1000),
                "close": np.random.rand(1000),
                "volume": np.arange(1000, dtype=np.int64),
            }
        )

        original_memory = get_memory_usage_mb(df)
        optimized_df = optimize_dtypes(df)
        optimized_memory = get_memory_usage_mb(optimized_df)

        # Optimized version should be smaller or equal
        assert optimized_memory <= original_memory

    def test_use_float32_for_arrays(self) -> None:
        """Test use_float32_for_arrays returns boolean."""
        result = use_float32_for_arrays()
        assert isinstance(result, bool)
