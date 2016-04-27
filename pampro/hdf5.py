from datetime import datetime, timedelta
from pampro import Time_Series, Channel, Bout

import h5py
import numpy as np
import math


def load_time_series(hdf5_group):
    """
    Given a HDF5 group reference, load the contained data as a Time_Series.
    """

    #if hdf5_group.attrs["pampro_type"] == "channels":

    ts = Time_Series.Time_Series("")

    for dataset_name in hdf5_group:

        # The start attribute is the anchor that timestamps are expressed relative to
        start = datetime.strptime(hdf5_group.attrs["start"], "%d/%m/%Y %H:%M:%S")

        # The saving module guarantees there will be a dataset called "timestamps"
        timestamps = hdf5_group["timestamps"][:]

        if dataset_name != "timestamps":
            d = hdf5_group[dataset_name]

            chan = Channel.Channel(dataset_name)
            chan.start = start
            chan.set_contents(d[:], timestamps, timestamp_policy="offset")

            ts.add_channel(chan)

    return ts

    #else:

    #    raise Exception("HDF5 group does not contain a Time_Series.")

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

def save_bouts_to_hdf5_group(bouts, hdf5_group):
    """
    Given a list of bouts and a HDF5 group, save the bouts as 2 separate HDF5 datasets of starts and ends.
    Calculate start anchor time as earliest start_timestamp of bout, calculate the rest as offsets in milliseconds since that time.
    HDF5 datasets to be called start_timestamps and end_timestamps.
    """

    num_bouts = len(bouts)

    # Get the earliest start_timestamp in the list of bouts
    anchor = np.min([bout.start_timestamp for bout in bouts])

    # Express the timestamps as number of milliseconds since "anchor"
    starts = np.empty(num_bouts, dtype="uint32")
    ends = np.empty(num_bouts, dtype="uint32")

    for i,b in enumerate(bouts):

        starts[i] = (b.start_timestamp - anchor)/timedelta(microseconds=1000)
        ends[i] = (b.end_timestamp - anchor)/timedelta(microseconds=1000)

    # Make sure they're unsigned 32 bit ints
    starts = starts.astype("uint32")
    ends = ends.astype("uint32")

    # Save the anchor as an attribute called "start"
    hdf5_group.attrs["start"] = anchor.strftime("%d/%m/%Y %H:%M:%S")

    start = hdf5_group.create_dataset("start_timestamps", (num_bouts,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
    start[...] = starts

    end = hdf5_group.create_dataset("endt_timestamps", (num_bouts,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
    end[...] = ends


def save_bouts(bouts, output, group_name):

    if type(output) is h5py._hl.files.File:

        f = output

    elif type(output) is str:

        if not output.endswith(".hdf5"):
            output += ".hdf5"
            print("Adding .hdf5 extension to filename - file will be saved in " + output)

        f = h5py.File(output, "w")

    else:
        raise Exception("Incompatible type of output supplied: {}".format(str(type(output))))

    group = f.create_group(group_name)
    group.attrs["pampro_type"] = "bouts"
    save_bouts_to_hdf5_group(bouts, group)



def save(ts, output, groups=[("Raw", ["X", "Y", "Z"])], data_type="float64", compression=9):
    """
    Output a Time_Series object to a HDF5 container, for super-fast loading by the data_loading module.
    For information on HDF5: https://www.hdfgroup.org/HDF5/
    For information on the Python HDF5 implementation used here: http://www.h5py.org/
    """

    # If we have been given a h5py file, just use that
    if type(output) is h5py._hl.files.File:

        f = output

    # If it's a string, assume it is a filename to create one
    elif type(output) is str:

        f = h5py.File(output, "w")

    else:
        raise Exception("Incompatible type of output supplied: {}".format(str(type(output))))

    # Each tuple in the variable "groups" becomes a HDF5 group inside the container
    # For example, the default groups=[("Raw", ["X", "Y", "Z"])] means create 1 group called Raw, with 3 channels
    # We group them to indicate that they share a common set of timestamps to save storage
    for group_name, channels in groups:

        group = f.create_group(group_name)

        group.attrs["pampro_type"] = "channels"

        first_channel = ts[channels[0]]
        timestamps = first_channel.timestamps
        data_length = len(first_channel.data)
        timestamp_length = len(timestamps)

        group.attrs["start"] = first_channel.time_period[0].strftime("%d/%m/%Y %H:%M:%S")

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
            dset = group.create_dataset(channel.name, (data_length,), chunks=True, compression="gzip", shuffle=True, compression_opts=compression, dtype=data_type)
            dset[...] = channel.data

        offsets_dset = group.create_dataset("timestamps", (data_length,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
        offsets_dset[...] = offsets

    f.close()
