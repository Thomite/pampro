# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from datetime import datetime, date, time, timedelta
import copy
import itertools
from struct import *
from math import *
import sys
import re
import pandas as pd
from scipy.io.wavfile import write
import zipfile
from collections import OrderedDict
from scipy.interpolate import interp1d
from bisect import bisect_left, bisect_right

from .Bout  import *
from .Time_Series import *
from .time_utilities import *
from .pampro_utilities import *
from .hdf5 import *
from .Channel import *


def derive_diffs_secs_array(timestamps):
    """Using a timestamps array, return an array of difference in seconds"""
    
    # difference between consecutive timestamps
    diffs = np.diff(timestamps)
    # ... in seconds
    diffs_secs = [d.total_seconds() for d in diffs]
    diffs_secs += [diffs_secs[-1]]
    
    return diffs_secs


def derive_expected_timestamps(timestamps, diffs_secs, allowance):
    """Using a timestamps array and an array of difference in seconds, return an array of expected timestamps"""
    
    expected_uncertainty = np.diff(diffs_secs)
    expected_uncertainty = np.array([eu for eu in expected_uncertainty] + [diffs_secs[len(diffs_secs)-1]]) # TODO FIX
    expected_uncertainty = np.abs(expected_uncertainty)

    normal_diff_secs = np.median(diffs_secs)

    deviation_from_median = np.abs(diffs_secs - normal_diff_secs)  
    deviation_indices = np.where(deviation_from_median > allowance)[0]
    
    for di in deviation_indices:
        diffs_secs[di]= np.nan

    diffs_secs_series = pd.Series(diffs_secs)
    diffs_secs_smooth_mean = diffs_secs_series.rolling(101, min_periods=1, center=True, win_type="gaussian").mean(std=10)
    
    bad_indices = np.where(np.isnan(diffs_secs_smooth_mean))[0]
    
    for i in bad_indices:
        diffs_secs_smooth_mean[i] = normal_diff_secs
    
    # create array of expected timestamps based on 'normal' difference
    expected_timestamp = timestamps[0]
    expected_timestamps_mean = []
    for index in range(len(timestamps)):

        expected_timestamps_mean.append(expected_timestamp)
        expected_timestamp += timedelta(seconds=diffs_secs_smooth_mean[index])
    
    expected_timestamps = np.array(expected_timestamps_mean)
        
    return expected_timestamps, expected_uncertainty
    
    
def derive_discrepancy_array(timestamps,expected_timestamps):
    
    discrepancy = timestamps - expected_timestamps
    discrepancy_array = np.array([d.total_seconds() for d in abs(discrepancy)])

    return discrepancy_array
    
    
def diagnose_fix_anomalies(channels, window_size=timedelta(hours=2), discrepancy_threshold=2, allowance=None):
    """ Examine the channels in 'channel_combinations' , diagnose and fix anomalies in the time series data.
        
        channels = list of one or more Channel objects, sharing common timestamps
        window_size = time window to examine for each anomaly to see if the time series 'recovers'
        discrepancy_threshold = scaling factor applied to difference between the normal difference between timestamps,/
        to allow for variation and noise
        allowance = factor to allow for variance from median frequency, can be passed a value to set as a static amount
        
    """

    for channel in channels:
        if len(channel.data) == len(channels[0].data):
            pass
        else:
            raise Exception("Channel data not the same length")
            
    timestamps = channels[0].timestamps
    
    # difference between consecutive timestamps in seconds
    diffs_secs = derive_diffs_secs_array(timestamps)

    anomalies = []
        
    normal_diff_secs = np.median(diffs_secs)    
    
    if normal_diff_secs < 0:
        anomaly_def = {}
        anomaly_def["anomaly_type"] = "G"
        anomaly_def["normal_diff_secs"] = normal_diff_secs 
        anomalies.append(anomaly_def)

        return anomalies

    # if allowance is to be calculated
    if allowance is None:
        allowance = 0.2*normal_diff_secs
    
    # convert window size to integer number of samples, depending on frequency of signal
    window_length = int(round(window_size.total_seconds()/normal_diff_secs))
    
    # create array of expected timestamps based on diffs_secs array
    expected_timestamps, expected_uncertainty = derive_expected_timestamps(timestamps, diffs_secs, allowance)
    
    # discrepancy array
    discrepancy_array = derive_discrepancy_array(timestamps,expected_timestamps)
    
    ## IS THERE A DEVIATION FROM EXPECTED IN TIMESTAMPS ARRAY?
    while max(discrepancy_array) > discrepancy_threshold:
        
        if len(anomalies) > 10:
            break
        
        # DEVIATION
        start_index = np.where(discrepancy_array > discrepancy_threshold)[0][0]   
        end_index = min(start_index + window_length, len(timestamps)-1)
        
        if start_index == end_index:
            break
        
        discrepancy_ahead = timestamps - expected_timestamps
        discrepancy_behind = expected_timestamps - timestamps
    
        ahead = np.array([d.total_seconds() for d in (discrepancy_ahead)])
        behind = np.array([d.total_seconds() for d in (discrepancy_behind)])
        # make these absolute
        ahead[ahead < 0] = 0
        behind[behind < 0] = 0
        
        anomaly_def = {}
        
        context_start = max(0, start_index-5)
        context_end = min(start_index+5, len(timestamps)-1)
        anomaly_def["timestamp_context"] = [timestamps[t].strftime("%Y-%m-%d %H:%M:%S.%f") for t in range(context_start, context_end, 1)]
        anomaly_def["last_good_index"] = start_index - 1
        anomaly_def["last_good_timestamp"] = timestamps[start_index - 1].strftime("%Y-%m-%d %H:%M:%S.%f")
        
        if min(discrepancy_array[start_index:end_index]) < discrepancy_threshold:
        # timeseries recovers to expected sequence - anomaly A or C
            recovery_point = np.where(discrepancy_array[start_index:end_index] < discrepancy_threshold)[0][0]
            recovery_point += start_index
            anomaly_def["recovery_point"] = recovery_point
            anomaly_def["recovery_point_timestamp"] = timestamps[recovery_point].strftime("%Y-%m-%d %H:%M:%S.%f")
        
            if sum(ahead[start_index:recovery_point]) > sum(behind[start_index:recovery_point]):
                anomaly_def["anomaly_type"] = "A"
                
            elif sum(ahead[start_index:recovery_point]) < sum(behind[start_index:recovery_point]):
                anomaly_def["anomaly_type"] = "C"
            
            else:
                anomaly_def["anomaly_type"] = "UNEXPECTED ANOMALY TYPE"
                break
    
        else:
            # timeseries does not recovers to expected sequence - anomaly B,D E or F
            
            # calculate an array of timestamp diff discrepancies
            diffs_secs_local = derive_diffs_secs_array(timestamps[start_index-1:end_index+1])
            discrepancy_local = np.abs(diffs_secs_local - normal_diff_secs)
            
            # search for indices where discrepancy greater than allowance
            bad_indices = np.where(discrepancy_local > allowance)[0]
            bad_indices += start_index 
            
            if ahead[end_index] > behind[end_index]:
                # Then anomaly B or E
                if len(bad_indices) > 1:
                    anomaly_def["anomaly_type"] = "E"
                    recovery_point = bad_indices[len(bad_indices)-1]+2
                    anomaly_def["recovery_point"] = recovery_point
            
                else:
                    anomaly_def["anomaly_type"] = "B"
                         
            else:
                # anomaly D or F
                if len(bad_indices) > 1:
                    anomaly_def["anomaly_type"] = "F"
                    
                else:
                    anomaly_def["anomaly_type"] = "D"
        
        channels, expected_timestamps, anomaly_def = fix_anomaly(anomaly_def, channels, expected_timestamps)
        anomalies.append(anomaly_def)
        
        timestamps = channels[0].timestamps
        
        # recalculate discrepancy for next loop
        discrepancy_array = derive_discrepancy_array(timestamps,expected_timestamps)
    
    return anomalies


def fix_anomalies(anomalies, channels, missing_value=-111, allowance=None):

    timestamps = channels[0].timestamps
    
    diffs_secs = derive_diffs_secs_array(timestamps)
        
    normal_diff_secs = np.median(diffs_secs)    

    # if allowance is to be calculated
    if allowance is None:
        allowance = 0.2*normal_diff_secs
    
    # create array of expected timestamps based on diffs_secs array
    expected_timestamps_original, expected_uncertainty = derive_expected_timestamps(timestamps, diffs_secs, allowance)
    
    # split the list of channels into two lists, those with 1:1 data:timestamps and those with m:1 data:timestamps
    channels_timestamp_ratios = {}
    for channel in channels:
    
        channel.missing_value = missing_value
        data_timestamp_ratio = int(round(len(channel.data)/len(channel.timestamps)))
        
        if data_timestamp_ratio in channels_timestamp_ratios:
            channels_timestamp_ratios[data_timestamp_ratio].append(channel)
        else:

            channels_timestamp_ratios[data_timestamp_ratio] = [channel]

    fixed_channels = []
    for data_timestamp_ratio, channel_temps in channels_timestamp_ratios.items():

        expected_timestamps_temp = np.array(expected_timestamps_original, copy=True)
    
        for anomaly_def in anomalies:
            
            channels_temps, expected_timestamps_temp, anomaly_def = fix_anomaly(anomaly_def, channel_temps, expected_timestamps_temp, missing_value, data_timestamp_ratio)
        
        fixed_channels += channels_temps
        
    return fixed_channels


def fix_anomaly(anomaly_def, channels, expected_timestamps, missing_value=-111, data_timestamp_ratio=1):
    """ Performs a 'fix' on a anomaly, given the anomaly definition"""

    # last good timestamp index
    last_good_index = anomaly_def["last_good_index"]
    dtr = data_timestamp_ratio   #this will be 1 for page-level data channels
    
    if anomaly_def["anomaly_type"] == "A" or anomaly_def["anomaly_type"] == "C":
        # timestamp index at recovery
        recovery_point = int(anomaly_def["recovery_point"])
        for channel in channels:
            for i in range(last_good_index + 1,recovery_point, 1):
                channel.timestamps[i] = expected_timestamps[i]
            for i in range((last_good_index + 1)*dtr, recovery_point*dtr, 1):    
                if channel.name == "Integrity":
                    channel.data[i] = 1
                else:    
                    channel.data[i] = missing_value
            
    
    elif anomaly_def["anomaly_type"] == "B":
        
        timestamps = np.array(channels[0].timestamps, copy=True)
            
        first_bad_timestamp = timestamps[last_good_index+1]
        last_good_timestamp = timestamps[last_good_index]
        
        normal_time_diff = timestamps[last_good_index-1] - timestamps[last_good_index-2]
        time_jump = first_bad_timestamp - last_good_timestamp - normal_time_diff

        a = last_good_timestamp + timedelta(microseconds=10)
        b = first_bad_timestamp - timedelta(microseconds=10)
        
        # insert a timestamp just after last_good_index and another just before last_good_index+1
        timestamps = np.insert(timestamps, last_good_index+1, np.array([a,b]))
        expected_timestamps = np.insert(expected_timestamps, last_good_index+1, np.array([a,b]))
        expected_timestamps[last_good_index + 3:] += time_jump
        
        anomaly_def["first_index_after_shift"] = last_good_index + 3
        anomaly_def["first_timestamp_after_shift"] = expected_timestamps[last_good_index + 3].strftime("%Y-%m-%d %H:%M:%S.%f")
        
        #insert missing_value into each channel to align with these new timestamps, and update timestamp arrays
        missing_value_array = np.tile(A=missing_value, reps=2*dtr)
        integrity_array = np.tile(A=1, reps=2*dtr)
        for channel in channels:
            # "B" anomalies can be the result of pauses in recording while the device is charging, so retain battery level prior to and after anomaly
            if channel.name == "Battery":
                anomaly_def["Battery_before_anomaly"] = channel.data[last_good_index]
                anomaly_def["Battery_after_anomaly"] = channel.data[last_good_index+2]
            if channel.name == "Integrity":
                channel.data = np.insert(channel.data, (last_good_index+1)*dtr, integrity_array)
            else:    
                channel.data = np.insert(channel.data, (last_good_index+1)*dtr, missing_value_array)
            channel.timestamps = timestamps
            
            
    elif anomaly_def["anomaly_type"] == "E":
        recovery_point = int(anomaly_def["recovery_point"])
        
        timestamps = np.array(channels[0].timestamps, copy=True)
        
        for channel in channels:
            end_point = min(len(timestamps)-1, recovery_point)
            for i in range(last_good_index + 1, end_point, 1):
                channel.timestamps[i] = expected_timestamps[i]
            
            for i in range((last_good_index + 1)*dtr, (end_point+1)*dtr, 1):    
                if channel.name == "Integrity":
                    channel.data[i] = 1
                else:
                    channel.data[i] = missing_value
        
        
        # if recovery point is not the end of the file
        if recovery_point < len(timestamps)-1:
            time_jump = timestamps[recovery_point] - expected_timestamps[recovery_point]
            anomaly_def["time_jump_secs"] = time_jump.total_seconds()
            anomaly_def["recovery_point_timestamp"] = timestamps[recovery_point].strftime("%Y-%m-%d %H:%M:%S.%f")
            expected_timestamps[recovery_point:] += time_jump
       
    
    elif anomaly_def["anomaly_type"] == "D" or anomaly_def["anomaly_type"] == "F":
        # truncate each channel data after last good index 
        for channel in channels:
            channel.data = channel.data[:(last_good_index)*dtr]
            channel.timestamps = channel.timestamps[:last_good_index]
            
        expected_timestamps = expected_timestamps[:last_good_index]

    for channel in channels:
        if channel.name == "Integrity":
            channel.missing_value = "None"
        else:
            channel.missing_value = missing_value
            
    return channels, expected_timestamps, anomaly_def  
    

def diagnose_axes(x, y, z, window_size=timedelta(minutes=10), noise_cutoff_mg=13):
    """Returns a dict of max and min axis values for a set time period (window_size) of stillness"""

    x_std = x.piecewise_statistics(window_size, statistics=[("generic", ["std"])], time_period=x.timeframe)[0]
    y_std = y.piecewise_statistics(window_size, statistics=[("generic", ["std"])], time_period=y.timeframe)[0]
    z_std = z.piecewise_statistics(window_size, statistics=[("generic", ["std"])], time_period=z.timeframe)[0]

    # Find bouts where standard deviation is below threshold for long periods
    x_bouts = x_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    y_bouts = y_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    z_bouts = z_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    
    still_x, std_x = x.build_statistics_channels(x_bouts, [("generic", ["mean", "std"])])
    still_y, std_y = y.build_statistics_channels(y_bouts, [("generic", ["mean", "std"])])
    still_z, std_z = z.build_statistics_channels(z_bouts, [("generic", ["mean", "std"])])
    
    axes_dict = dict()
    
    for chan in [still_x, still_y, still_z]:
        axes_dict["{}_max".format(chan.name.replace("_mean",""))] = max(chan.data)
        axes_dict["{}_min".format(chan.name.replace("_mean",""))] = min(chan.data)
            
    return axes_dict
    
    
def find_stuck_bouts(x, y, z, dynamic_range):
    """Take channels of x, y, z data and a tuple of dynamic range=(low,high)
    return: list of bouts where every axis is stuck exceeding either maximum or minimum"""
    
    # create a list of bouts for each axis where the value is outside the dynamic range
    x_low = x.bouts(-20, dynamic_range[0])
    x_high = x.bouts(dynamic_range[1], 20)
    x_list = bout_list_union(x_low, x_high)
    
    y_low = y.bouts(-20, dynamic_range[0])
    y_high = y.bouts(dynamic_range[1], 20)
    y_list = bout_list_union(y_low, y_high)
    
    z_low = z.bouts(-20, dynamic_range[0])
    z_high = z.bouts(dynamic_range[1], 20)
    z_list = bout_list_union(z_low, z_high)
    
    # the axis is stuck if the bouts of all three axes have any overlap
    xy_list = bout_list_intersection(x_list, y_list)
    xyz_list = bout_list_intersection(xy_list, z_list)

    return xyz_list
    

def diagnose_fix_axes_stuck(x, y, z, integrity, dynamic_range, acc_missing=-111, integrity_missing=1):
    """Takes the x,y,z and integrity channels, finds the stuck bouts and fills the channels with "missing" values"""
    
    missing_list = [acc_missing, acc_missing, acc_missing, integrity_missing]
    
    stuck_bouts = find_stuck_bouts(x, y, z, dynamic_range)
    
    for channel, fill_value in zip([x, y, z, integrity], missing_list):
        channel.fill_windows(stuck_bouts, fill_value)
        
    return stuck_bouts


def find_dynamic_range(header, monitor, sensitivity_dict={"Axivity":0.5, "GeneActiv":0.5}):
    """Function to use the header information, monitor type and sensitivity values per monitor type to calculate a diagnostic dynamic range
    return: dynamic range tuple (low, high)"""
    
    if monitor in sensitivity_dict.keys():
        sensitivity = sensitivity_dict[monitor]
    else:
        sensitivity = 0
                       
    if monitor == "Axivity":
        val = int(header["sample_range"]) - sensitivity
        dynamic_range = (0-val, val)
        
    elif monitor == "GeneActiv":
        val = header["accelerometer_range"].split(" to ")
        dynamic_range = ((int(val[0]) + sensitivity), (int(val[1]) - sensitivity))        
        
    elif monitor == "activPAL":
        val = int(header["resolution"]) - sensitivity
        dynamic_range = (0-val, val)
        
    elif monitor == "GT3X+": 
        dynamic_range = (-6 + sensitivity, 6 - sensitivity)
                       
    else:
        # monitor type not supported, use a default set of values
        dynamic_range = (-8,8)               
    
    return dynamic_range