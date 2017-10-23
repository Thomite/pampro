from datetime import datetime, timedelta
from collections import OrderedDict
import h5py
import numpy as np
import math

from .Time_Series import *
from .Channel import *
from .Bout import *

def list_caches(hdf5_file):
    """
    List the name of any functions that have their results cached in this file.
    """

    if "cache" in hdf5_file:

        return list(hdf5_file["cache"].keys())
    else:
        return "Empty"

def get_appropriate_cache(name, args, parameter_names, hdf5_file):
    """
    Return the cached results of a given method, only if the arguments passed to it are identical to those in the cache.
    """

    #
    if "cache/"+name in hdf5_file:
        for group_name, group in hdf5_file["cache/"+name].items():

            # Start by assuming this cache contains suitable results
            suitable = True

            # But disprove it by checking if each parmeter is the same
            for param in parameter_names:

                if group.attrs[param] != str(args[param]):

                    suitable = False

            if suitable:
                return group

    # None of the caches under "cache/name" were ran with the same parameters
    return None

def make_cache(name, args, parameter_names, hdf5_file):
    """
    Create a HDF5 group suitable for caching the results of a process.
    This saves each setter() function from having to write function parameters and finding a suitable group name.
    """

    if "cache/"+name in hdf5_file:

        # If no cache exists for this function, it will be 0
        num_existing_alternatives = len(hdf5_file["cache/"+name])

    else:

        num_existing_alternatives = 0

    # Label each group numerically
    hdf5_group = hdf5_file.create_group("cache/"+name+"/"+str(num_existing_alternatives))

    for param in parameter_names:

        hdf5_group.attrs[param] = str(args[param])

    return hdf5_group

def do_if_not_cached(name, method, args, parameter_names, getter, setter, hdf5_file):
    """
    This is a wrapper for time-intensive functions. It contains the logic to reuse cached results from previous analyses.
    The method to be executed if no cache is available
    The method that extracts the cached result from the HDF5 file
    The method that saves a cached result to the HDF5 file
    """

    # If the user didn't pass a HDF5 file reference, we can't get or set a cache
    if hdf5_file is None:

        r = method(**args)

    else:
        # (Convention is to store a cache under "cache/name/0", "cache/name/1", etc.)
        # If a cache exists with the right name in the HDF5 file
        # There may be many caches for different parameters, get the right ones
        hdf5_group = get_appropriate_cache(name, args, parameter_names, hdf5_file)
        if hdf5_group is not None:

            # Call the appropriate method to load the cache
            r = getter(hdf5_group)

        else:
        # A cached result was not found

            # So call the nominated function to get the result
            r = method(**args)

            # Create a cache suitable to store the results in
            # And cache the result for when the function is called again
            hdf5_group = make_cache(name, args, parameter_names, hdf5_file)
            setter(r, hdf5_group)

    return r

def delete_cache(hdf5_file):
    """
    Delete the cache folder and its subfolders. Any subsequent query on the cache will conclude nothing has been cached.
    Returns True if there was a cache to delete, False if there was not.
    """

    try:

        del hdf5_file["cache"]
        return True

    except:
        return False

def dictionary_to_attributes(dictionary, hdf5_thing):
    """
    Write each item in the given dictionary as an attribute attached to the HDF5 group or dataset.
    """

    for k,v in dictionary.items():
        hdf5_thing.attrs[k] = v

def dictionary_from_attributes(hdf5_thing):
    """
    Load the attributes of a given HDF5 group or dataset as an ordered dictionary.
    """

    dictionary = OrderedDict()
    for k,v in hdf5_thing.attrs.items():
        dictionary[k] = v

    return dictionary

def load_bouts_from_hdf5_group(hdf5_group):
    """
    Load and return a list of bouts stored in the given HDF5 group.
    Assumes they are saved according to the layout defined in save_bouts_to_hdf5_group().
    """

    num_bouts = hdf5_group.attrs["num_bouts"]
    bouts = []

    if num_bouts > 0:

        start_timestamps = hdf5_group["start_timestamps"][:]
        end_timestamps = hdf5_group["end_timestamps"][:]

        start = datetime.strptime(hdf5_group.attrs["start"], "%d/%m/%Y %H:%M:%S")

        one_ms = timedelta(microseconds=1000)
        for a,b in zip(start_timestamps, end_timestamps):

            bouts.append(Bout.Bout(start + one_ms*a, start + one_ms*b))

    return bouts

def save_bouts_to_hdf5_group(bouts, hdf5_group):
    """
    Given a list of bouts and a HDF5 group, save the bouts as 2 separate HDF5 datasets of starts and ends.
    Calculate start anchor time as earliest start_timestamp of bout, calculate the rest as offsets in milliseconds since that time.
    HDF5 datasets to be called start_timestamps and end_timestamps.
    """

    num_bouts = len(bouts)
    hdf5_group.attrs["num_bouts"] = num_bouts

    if num_bouts > 0:

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

        end = hdf5_group.create_dataset("end_timestamps", (num_bouts,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
        end[...] = ends

def save_bouts(bouts, output, group_name):

    if type(output) is h5py._hl.files.File:

        f = output

    elif type(output) is str:

        f = h5py.File(output, "w")

    else:
        raise Exception("Incompatible type of output supplied: {}".format(str(type(output))))

    group = f.create_group(group_name)
    group.attrs["pampro_type"] = "bouts"
    save_bouts_to_hdf5_group(bouts, group)

def load_time_series(hdf5_group):
    """
    Given a reference to a hdf5_group, assume it is layed out according to pampro conventions and load a Time Series object from it.
    """
    ts = Time_Series("")

    # "timestamps" will be a single HDF5 dataset shared by the rest of the channels
    timestamps = hdf5_group["timestamps"][:]

    # The timestamps will be expressed in milliseconds relative to the start value
    start = datetime.strptime(hdf5_group.attrs["start"], "%d/%m/%Y %H:%M:%S")

    # Each channel of data will be a HDF5 dataset, same length as timestamps
    for dataset_name in hdf5_group:

        if dataset_name != "timestamps":
            d = hdf5_group[dataset_name]

            chan = Channel(dataset_name)
            chan.start = start

            # Any meta data on the dataset is copied to the Channel object
            for attr_name, attr_value in d.attrs.items():
                setattr(chan, attr_name, attr_value)

            chan.set_contents(d[:], timestamps[:], timestamp_policy="offset")

            ts.add_channel(chan)

    return ts

def timestamps_to_offsets(timestamps):
    """
    Express a list of timestamps as a list of millisecond offsets from the first timestamp
    """

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


def save(ts, output_filename, groups=[("Raw", ["X", "Y", "Z"])], meta_candidates=["calibrated", "frequency"], compression=4, data_type="float32"):
    """
    Output a Time_Series object to a HDF5 container, for super-fast loading by the data_loading module.
    For information on HDF5: https://www.hdfgroup.org/HDF5/
    For information on the Python HDF5 implementation used here: http://www.h5py.org/
    """

    if not output_filename.endswith(".hdf5"):
        output_filename += ".hdf5"
        print("Adding .hdf5 extension to filename - file will be saved in " + output_filename)

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

            # If the Channel has any of these properties, preserve them as attrs in HDF5 counterpart
            for mc in meta_candidates:

                if hasattr(channel, mc):
                    dset.attrs[mc] = getattr(channel, mc)

        offsets_dset = group.create_dataset("timestamps", (data_length,), chunks=True, compression="gzip", shuffle=True, compression_opts=9, dtype="uint32")
        offsets_dset[...] = offsets

    f.close()
