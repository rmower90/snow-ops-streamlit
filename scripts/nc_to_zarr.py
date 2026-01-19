from pathlib import Path
import xarray as xr


def netcdf_to_zarr(
    netcdf_path: str,
    zarr_path: str,
    *,
    mode: str = "w",
    consolidated: bool = True,
    chunks: dict = None,
    **open_dataset_kwargs,
):
    """
    Convert a NetCDF file to Zarr format.

    Parameters
    ----------
    netcdf_path : str
        Absolute path to the input NetCDF file.
    zarr_path : str
        Absolute path to the output Zarr directory.
    mode : str, optional
        Zarr write mode ("w" = overwrite, "a" = append). Default is "w".
    consolidated : bool, optional
        Whether to write consolidated Zarr metadata. Default is True.
    chunks : dict, optional
        Chunking scheme passed to xarray.open_dataset (e.g., {"time": 1}).
        If None, uses on-disk chunking or loads unchunked.
    **open_dataset_kwargs
        Optional keyword arguments passed directly to xarray.open_dataset()
        (e.g., decode_times=False, engine="netcdf4").

    Returns
    -------
    None
    """

    netcdf_path = Path(netcdf_path).expanduser().resolve()
    zarr_path = Path(zarr_path).expanduser().resolve()

    if not netcdf_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")

    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    # Open NetCDF
    ds = xr.open_dataset(
        netcdf_path,
        chunks=chunks,
        **open_dataset_kwargs,
    )

    # Write Zarr
    ds.to_zarr(
        zarr_path,
        mode=mode,
        consolidated=consolidated,
    )

    ds.close()

    print(f"Saved Zarr dataset to: {zarr_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a NetCDF file to Zarr format."
    )
    parser.add_argument(
        "netcdf_path",
        type=str,
        help="Absolute path to the input NetCDF file.",
    )
    parser.add_argument(
        "zarr_path",
        type=str,
        help="Absolute path to the output Zarr directory.",
    )
    parser.add_argument(
        "--chunk",
        action="append",
        help="Chunk spec like dim=size (can be used multiple times)",
    )

    args = parser.parse_args()

    # Parse chunk arguments like: --chunk time=1 --chunk y=256 --chunk x=256
    chunks = None
    if args.chunk:
        chunks = {}
        for c in args.chunk:
            dim, size = c.split("=")
            chunks[dim] = int(size)

    netcdf_to_zarr(
        netcdf_path=args.netcdf_path,
        zarr_path=args.zarr_path,
        chunks=chunks,
    )
