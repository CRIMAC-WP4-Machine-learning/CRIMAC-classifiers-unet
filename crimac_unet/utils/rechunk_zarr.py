import xarray as xr
import shutil
import os
from numcodecs import Blosc
import zarr
from rechunker import rechunk


def rechunk_predictions(output):
    output_dir = os.path.split(output)[0]
    predictions = xr.open_zarr(output)

    # Rechunk predictions so that we one chunk covers the entire range and one category
    rechunked_predictions = predictions.annotation.chunk({'category': 1, 'ping_time': 'auto', 'range': -1})
    coordinates = rechunked_predictions.coords.dims

    # Get dictionary with optimal chunk sizes for each coordinate
    opt_chunk_sizes = {coord: rechunked_predictions.chunks[coord_idx][0] for coord_idx, coord in enumerate(coordinates)}

    # Prepare encoding and chunks parameters for rechunking
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {var: {"compressor": compressor} for var in predictions.data_vars}
    newchunks = {var: {xi: opt_chunk_sizes[xi] for xi in predictions[var].coords.dims} for var in predictions.data_vars}

    # Need to unify chunk first
    predictions_unified = predictions.chunk(newchunks['annotation'])

    # Needed because a bug in rechunk-xarray
    for var in predictions_unified.variables:
        if "chunks" in predictions_unified[var].encoding:
            del predictions_unified[var].encoding['chunks']
        if "preferred_chunks" in predictions_unified[var].encoding:
            del predictions_unified[var].encoding['preferred_chunks']

    # Do rechunk
    tmp_file = os.path.join(output_dir, "temp.zarr")
    predictions_rechunked_file = os.path.join(output_dir, "pred_rechunked.zarr")
    rechunked = rechunk(predictions_unified, target_chunks=newchunks, max_mem='300MB', temp_store=tmp_file,
                        target_store=predictions_rechunked_file, target_options=encoding)
    rechunked.execute()

    # Remove old file and temporary file
    shutil.rmtree(output)  # Remove previous zarr
    shutil.rmtree(tmp_file)  # Remove temp file
    shutil.move(predictions_rechunked_file, output)  # Move new rechunked file into output directory

    # Consolidate metadata
    zarr.convenience.consolidate_metadata(output)
