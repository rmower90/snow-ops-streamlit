# csv_to_parquet.py
from pathlib import Path
import pandas as pd

def csv_to_parquet(
    csv_path: str,
    parquet_path: str,
    **read_csv_kwargs,
):
    """
    Convert a CSV file to Parquet format.

    Parameters
    ----------
    csv_path : str
        Absolute path to the input CSV file.
    parquet_path : str
        Absolute path to the output Parquet file.
    **read_csv_kwargs
        Optional keyword arguments passed directly to pandas.read_csv()
        (e.g., parse_dates, dtype, usecols).

    Returns
    -------
    None
    """

    csv_path = Path(csv_path).expanduser().resolve()
    parquet_path = Path(parquet_path).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path, **read_csv_kwargs)

    # Write Parquet
    df.to_parquet(parquet_path, index=False)

    print(f"Saved Parquet file to: {parquet_path}")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a CSV file to Parquet format."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Absolute path to the input CSV file.",
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Absolute path to the output Parquet file.",
    )
    args = parser.parse_args()

    csv_to_parquet(
        csv_path=args.csv_path,
        parquet_path=args.parquet_path,
    )