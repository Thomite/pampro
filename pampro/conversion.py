def timestamps_to_offsets(timestamps):

    # Start is the first time stamp, and thus the reference point for the rest
    start = timestamps[0]

    # Express each timestamp as number of milliseconds since "start"
    offsets = ((timestamps - start)/timedelta(microseconds=1000)).astype("uint32")

    return (start,offsets)

def interpolate_offsets(offsets, data_length):
    """
    Expand a sparse list of offsets into an exhaustive list which effectively gives the offset of every data entry.
    Called when data is only timestamped at page level in a file.
    """

    # Essentially equates to the number of observations per page in a raw file
    data_to_offset_ratio = data_length / len(offsets)

    # Append a final offset value so the iteration below fills the final values
    offsets = np.concatenate((offsets, [offsets[-1] + (offsets[-1]-offsets[-2])]))

    full_offsets = np.empty(data_length, dtype="uint32")

    for i,a,b in zip(range(len(offsets)), offsets, offsets[1:]):

        diff = (b-a) / data_to_offset_ratio

        for n in range(data_to_offset_ratio):
            full_offsets[i*data_to_offset_ratio+n] = a+diff*n

    return full_offsets

def convert(ts, output_filename):

    f = h5py.File(output_filename, "w")

    f.attrs["start"] = ts.time_period[0].strftime("%d/%m/%Y %H:%M:%S")

    start, offsets = timestamps_to_offsets(ts.channels[0].timestamps)

    # If the timestamps are sparse, expand them
    if len(offsets) < len(ts.channels[0].data):
        offsets = interpolate_offsets(offsets, len(ts.channels[0].data))

    # Create a HDF5 "dataset" for each channel of data
    for channel in ts:
        dset = f.create_dataset(channel.name, (len(channel.data),), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="float64")
        dset[...] = channel.data

    offsets_dset = f.create_dataset("offsets", (len(offsets),), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
    offsets_dset[...] = offsets

    f.close()
