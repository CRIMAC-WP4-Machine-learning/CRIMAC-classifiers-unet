import xarray as xr
import shutil
import os
from numcodecs import Blosc
import zarr
from rechunker import rechunk
from dask.diagnostics import ProgressBar
import argparse


def rechunk_zarr(zarr_path, target_chunks):
    ds_zarr = xr.open_zarr(zarr_path)

    # Rechunk predictions so that we one chunk covers the entire range and one category
    rechunked_ds = ds_zarr.chunk(target_chunks).unify_chunks()
    coordinates = rechunked_ds.coords.dims

    # Get dictionary with optimal chunk sizes for each coordinate
    opt_chunk_sizes = {coord: rechunked_ds.chunks[coord][0] for coord_idx, coord in enumerate(coordinates)}

    # Prepare encoding and chunks parameters for rechunking
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {var: {"compressor": compressor} for var in ds_zarr.data_vars}
    newchunks = {var: {xi: opt_chunk_sizes[xi] for xi in ds_zarr[var].coords.dims} for var in ds_zarr.data_vars}

    # Need to unify chunk first
    ds_unify = ds_zarr.chunk(target_chunks)

    # Needed because a bug in rechunk-xarray
    for var in ds_unify.variables:
        if "chunks" in ds_unify[var].encoding:
            del ds_unify[var].encoding['chunks']
        if "preferred_chunks" in ds_unify[var].encoding:
            del ds_unify[var].encoding['preferred_chunks']

    # Do rechunk
    directory = os.path.split(zarr_path)[0]
    target_store = os.path.join(directory, 'target.zarr')
    temp_store = os.path.join(directory, 'tmp.zarr')
    rechunked = rechunk(ds_unify, target_chunks=newchunks, max_mem='300MB', temp_store=temp_store,
                        target_store=target_store, target_options=encoding)
    with ProgressBar():
        rechunked.execute()

    # Remove old file and temporary file
    shutil.rmtree(zarr_path)  # Remove previous zarr
    shutil.rmtree(temp_store)  # Remove temp file
    shutil.move(target_store, zarr_path)  # Move new rechunked file into output directory

    # Consolidate metadata
    zarr.convenience.consolidate_metadata(zarr_path)


if __name__ == '__main__':
    from glob import glob

    zarr_paths = []
    chunk_size_sv = {'frequency': 1, 'ping_time': 1000, 'range': 1000}
    chunk_size_annot = {'category': 1, 'ping_time': 1000, 'range': 1000}
    chunk_size_bottom = {'ping_time': 1000, 'range': 1000}

    for zarr_path_dir in zarr_paths:
        zarr_files = glob(os.path.join(zarr_path_dir, "*.zarr"))
        for filepath in zarr_files:
            if "sv" in filepath:
                rechunk_zarr(filepath, chunk_size_sv)
            elif "labels" in filepath:
                rechunk_zarr(filepath, chunk_size_annot)
            elif "bottom" in filepath:
                rechunk_zarr(filepath, chunk_size_bottom)
