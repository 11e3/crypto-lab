"""
Data format conversion utilities.

Provides functions to convert between different data formats (CSV, Parquet, etc.).
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Interval mapping: directory name -> upbit interval name
INTERVAL_MAP: dict[str, str] = {
    "1d": "day",
    "1w": "week",
    "4h": "minute240",
}


def convert_ticker_format(filename: str) -> str:
    """
    Convert ticker from CSV filename format to Upbit format.

    Args:
        filename: CSV filename like 'BTC_KRW.csv'

    Returns:
        Upbit ticker format like 'KRW-BTC'

    Raises:
        ValueError: If filename format is unexpected
    """
    # Remove .csv extension and split
    base_name = filename.replace(".csv", "")
    parts = base_name.split("_")

    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {filename}")

    # BTC_KRW -> KRW-BTC
    coin, market = parts
    return f"{market}-{coin}"


def csv_to_parquet(
    csv_path: Path,
    output_dir: Path | None = None,
    interval: str | None = None,
) -> Path:
    """
    Convert CSV file to parquet format.

    Args:
        csv_path: Path to source CSV file
        output_dir: Directory to save parquet file (defaults to ``data/raw/``)
        interval: Upbit interval name (day, week, minute240). If None, tries to infer from path

    Returns:
        Path to created parquet file

    Raises:
        ValueError: If interval cannot be determined
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert ticker format
    ticker = convert_ticker_format(csv_path.name)

    # Determine interval if not provided
    if interval is None:
        # Try to infer from parent directory
        parent_name = csv_path.parent.name
        if parent_name in INTERVAL_MAP:
            interval = INTERVAL_MAP[parent_name]
        else:
            raise ValueError(
                f"Cannot determine interval. Provide interval parameter or ensure "
                f"CSV is in a directory named one of: {list(INTERVAL_MAP.keys())}"
            )

    # Read CSV with datetime index
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Ensure index name is set
    df.index.name = "datetime"

    # Create output filename
    output_filename = f"{ticker}_{interval}.parquet"
    output_path = output_dir / output_filename

    # Save as parquet
    df.to_parquet(output_path, engine="pyarrow")

    logger.info(f"Converted {csv_path.name} -> {output_path.name}")
    return output_path


def convert_csv_directory(
    source_dir: Path,
    output_dir: Path | None = None,
) -> int:
    """
    Convert all CSV files in a directory structure to parquet format.

    Expected structure:
        source_dir/
            interval_dir/  (e.g., "1d", "1w", "4h")
                *.csv

    Args:
        source_dir: Root directory containing interval subdirectories
        output_dir: Directory to save parquet files (defaults to data/raw/)

    Returns:
        Number of files converted
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    # Process each interval directory
    for interval_dir in source_dir.iterdir():
        if not interval_dir.is_dir():
            continue

        dir_name = interval_dir.name

        if dir_name not in INTERVAL_MAP:
            logger.warning(f"Skipping unknown interval directory: {dir_name}")
            continue

        upbit_interval = INTERVAL_MAP[dir_name]

        # Process all CSV files in the interval directory
        for csv_file in interval_dir.glob("*.csv"):
            try:
                csv_to_parquet(csv_file, output_dir, upbit_interval)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}", exc_info=True)

    logger.info(f"Total files converted: {processed_count}")
    return processed_count
