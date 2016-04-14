from datetime import datetime, timedelta
from pampro import Time_Series, Channel, Bout

import h5py
import numpy as np
import math


def load_time_series(hdf5_group):

    ts = Time_Series.Time_Series("")

    for dataset_name in hdf5_group:

        start = datetime.strptime(hdf5_group.attrs["start"], "%d/%m/%Y %H:%M:%S")

        timestamps = hdf5_group["timestamps"][:]

        if dataset_name != "timestamps":
            d = hdf5_group[dataset_name]

            chan = Channel.Channel(dataset_name)
            chan.start = start
            chan.set_contents(d, timestamps, timestamp_policy="offset")

            ts.add_channel(chan)

    return ts

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
    data_to_offset_ratio = math.ceil(data_length / len(offsets))

    # Append a final offset value so the iteration below fills the final values
    offsets = np.concatenate((offsets, [offsets[-1] + (offsets[-1]-offsets[-2])]))

    full_offsets = np.empty(data_length, dtype="uint32")


    for i,a,b in zip(range(len(offsets)), offsets, offsets[1:]):

        diff = (b-a) / data_to_offset_ratio
        try:
            for n in range(data_to_offset_ratio):
                full_offsets[i*data_to_offset_ratio+n] = a+diff*n
        except:
            pass

    return full_offsets

def convert(ts,  output_filename, groups=[("Raw", ["X", "Y", "Z"])]):
    """

    """

    f = h5py.File(output_filename, "w")

    # Each tuple in the variable "groups" becomes a HDF5 group inside the container
    # For example, the default groups=[("Raw", ["X", "Y", "Z"])] means create 1 group called Raw, with 3 channels
    # We group them to indicate that they share a common set of timestamps to save storage
    for group_name, channels in groups:

        group = f.create_group(group_name)

        first_channel = ts[channels[0]]
        timestamps = first_channel.timestamps
        data_length = len(first_channel.data)
        timestamp_length = len(timestamps)

        group.attrs["start"] = timestamps[0].strftime("%d/%m/%Y %H:%M:%S")

        # Convert timestamps to offsets from the first timestamp - makes storing them easier as ints
        start, offsets = timestamps_to_offsets(timestamps)

        # If the timestamps are sparse, expand them to 1 per observation
        if timestamp_length < data_length:
            offsets = interpolate_offsets(offsets, data_length)

        # When we have page-level timestamps from a file, a timestamp points at the first observation in the page
        # This leaves some data at the end of a file without timestamps
        # So the data_loading function infers a final timestamp that points at the last observation
        # This means there is 1 extra timestamp than the page level data, which we want to ignore here
        if timestamp_length == data_length+1:
            offsets = offsets[:-1]

        # Each channel's data array becomes a HDF5 dataset inside the group
        for channel_name in channels:

            channel = ts[channel_name]
            dset = group.create_dataset(channel.name, (data_length,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="float64")
            dset[...] = channel.data

        offsets_dset = group.create_dataset("timestamps", (data_length,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
        offsets_dset[...] = offsets

    f.close()
