# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

from datetime import timedelta
import numpy as np
import copy
import math

from .Channel import *
from .Bout  import *
from .Time_Series import *
from .time_utilities import *
from .pampro_fourier import *
from .hdf5 import *


def activpal_classification(pitch):

    transition_sit_stand = 32 # Sit -> Stand if angle >= 32
    transition_stand_sit = 22 # Stand -> Sit if angle <= 22
    ten_seconds_in_samples = int(pitch.frequency * 10) #20Hz

    # This should be faster than pointlessly copying the input array
    classification = Channel("activPAL_class")
    classification.set_contents(np.zeros(len(pitch.data)), pitch.timestamps)

    current_classification = 0 # 0 = sitting, 1 = standing, 2 = stepping
    bout_length = 0
    bout_index = 0
    for index,value in enumerate(pitch.data):

        classification.data[index] = current_classification

        if current_classification == 0:

            if value >= transition_sit_stand:

                if bout_length == 0:
                    bout_length = 1
                    bout_index = index

                else:
                    bout_length += 1
                    if bout_length >= ten_seconds_in_samples:
                        current_classification = 1
                        classification.data[bout_index:index] = 1

            else:
                bout_length = 0
                bout_index = index

        elif current_classification == 1:
            if value <= transition_stand_sit:

                if bout_length == 0:
                    bout_length = 1
                    bout_index = index

                else:
                    bout_length += 1
                    if bout_length >= ten_seconds_in_samples:
                        current_classification = 0
                        classification.data[bout_index:index] = 0

            else:
                bout_length = 0
                bout_index = index

    return classification


def infer_vector_magnitude(x,y,z):

    result = Channel("VM")

    result.set_contents( np.sqrt( np.multiply(x.data,x.data) + np.multiply(y.data,y.data) + np.multiply(z.data,z.data) ), x.timestamps )

    result.inherit_time_properties(x)

    result.draw_properties = {"c":[0.05,0.8,0.05], "lw":2}

    return result


def infer_pitch_roll(x,y,z):
    """
    Infer pitch and roll channels from raw triaxial acceleration channels.
    """

    pitch = Channel("Pitch")
    roll = Channel("Roll")
#    zangle = Channel("Zangle")

# include "+ 0.0000000000001" into the equation to avoid 
#the instances where y and z, or x and z, are both zero and therefore the equation resolves to division by zero.

    pitch_degrees = np.arctan(x.data/np.sqrt((y.data*y.data) + (z.data*z.data) + 0.0000000000001)) * 180.0/np.pi
    roll_degrees = np.arctan(y.data/np.sqrt((x.data*x.data) + (z.data*z.data) + 0.0000000000001)) * 180.0/np.pi

#   z_degrees = np.arctan(z.data/np.sqrt((x.data*x.data)+(y.data*y.data))) * 180.0/np.pi

    pitch.set_contents( pitch_degrees, x.timestamps)
    roll.set_contents( roll_degrees, x.timestamps)
#    zangle.set_contents( z_degrees, x.timestamps)

    pitch.inherit_time_properties(x)
    roll.inherit_time_properties(x)
    
#   zangle.inherit_time_properties(x)

#   return [pitch, roll, zangle]
    return [pitch, roll]

def infer_enmo(vm):
    """
    Subtract 1g from Vector Magnitude signal, truncate results below 0 to 0
    """

    result = Channel("ENMO")

    result.set_contents((vm.data - 1.0)*1000.0, vm.timestamps)

    result.data[np.where(result.data < 0)] = 0

    result.inherit_time_properties(vm)

    result.draw_properties = {"c":[0.05,0.8,0.05], "lw":2}

    return result


def infer_enmo_a(vm):

    result = Channel("ENMOa")

    result.set_contents(np.absolute((vm.data - 1.0)*1000.0), vm.timestamps)

    result.inherit_time_properties(vm)

    return result


def infer_vm_hpf(vm):
    """Apply high pass filter to VM at 0.2 hertz. Absolute resulting data. Returned in mg. """

    vm_hpf = high_pass_filter(vm, 0.2, frequency=vm.frequency, order=4)
    vm_hpf.name = "VM_HPF"
    vm_hpf.data = np.multiply(1000.0, abs(vm_hpf.data))

    vm_hpf.inherit_time_properties(vm)

    vm_hpf.draw_properties = {"c": [0.8, 0.05, 0.8], "lw": 2}

    return vm_hpf


def infer_nonwear_actigraph(counts, zero_minutes=timedelta(minutes=60)):
    """Given an Actigraph counts signal, infer nonwear as consecutive zeros of a given duration. """

    # List all bouts where the signal was <= 0
    nonwear_bouts = counts.bouts(-999999, 0)

    # Limit those bouts to the minimum duration specified in "zero_minutes"
    nonwear_bouts = limit_to_lengths(nonwear_bouts, min_length=zero_minutes)

    # Invert the nonwear bouts to get wear bouts
    wear_bouts = time_period_minus_bouts([counts.timeframe[0], counts.timeframe[1]], nonwear_bouts)

    return nonwear_bouts, wear_bouts


def get_still_bouts_triaxial(hdf5_group):
    """
    Getter method for infer_still_bouts_triaxial
    """

    still_bouts = load_bouts_from_hdf5_group(hdf5_group)
    return still_bouts


def set_still_bouts_triaxial(still_bouts, hdf5_group):
    """
    Setter method for infer_still_bouts_triaxial
    """

    save_bouts_to_hdf5_group(still_bouts, hdf5_group)


def infer_still_bouts_triaxial_method(x, y, z, window_size=timedelta(seconds=10), noise_cutoff_mg=13, minimum_length=timedelta(seconds=10)):
    # Get windows of standard deviation in each axis
    x_std = x.piecewise_statistics(window_size, statistics=[("generic", ["std"])], time_period=x.timeframe)[0]
    y_std = y.piecewise_statistics(window_size, statistics=[("generic", ["std"])], time_period=y.timeframe)[0]
    z_std = z.piecewise_statistics(window_size, statistics=[("generic", ["std"])], time_period=z.timeframe)[0]

    # Find bouts where standard deviation is below threshold for long periods
    x_bouts = x_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    y_bouts = y_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    z_bouts = z_std.bouts(0, float(noise_cutoff_mg)/1000.0)

    x_bouts = limit_to_lengths(x_bouts, min_length=minimum_length)
    y_bouts = limit_to_lengths(y_bouts, min_length=minimum_length)
    z_bouts = limit_to_lengths(z_bouts, min_length=minimum_length)

    # Get the times where those bouts overlap
    x_intersect_y = bout_list_intersection(x_bouts, y_bouts)
    x_intersect_y_intersect_z = bout_list_intersection(x_intersect_y, z_bouts)

    return x_intersect_y_intersect_z


def infer_still_bouts_triaxial(x, y, z, window_size=timedelta(seconds=10), noise_cutoff_mg=13, minimum_length=timedelta(seconds=10), hdf5_file=None):
    """
    Find a list of bouts where the standard deviation of each axis is below a given threshold, and is therefore still.
    """

    args = {"x":x, "y":y, "z":z, "window_size":window_size, "noise_cutoff_mg":noise_cutoff_mg, "minimum_length":minimum_length}
    params = ["minimum_length", "noise_cutoff_mg", "window_size"]
    return do_if_not_cached("still_bouts_triaxial", infer_still_bouts_triaxial_method, args, params, get_still_bouts_triaxial, set_still_bouts_triaxial, hdf5_file)

def infer_nonwear_triaxial_method(x, y, z, minimum_length=timedelta(hours=1), noise_cutoff_mg=13):
    # Get an exhaustive list of bouts where the monitor was still
    x_intersect_y_intersect_z = infer_still_bouts_triaxial(x, y, z, noise_cutoff_mg=noise_cutoff_mg, minimum_length=minimum_length)

    # Restrict those bouts to only those with a length that exceeds the minimum length criterion
    x_intersect_y_intersect_z = limit_to_lengths(x_intersect_y_intersect_z, min_length=minimum_length)

    return x_intersect_y_intersect_z


def get_infer_nonwear_triaxial(hdf5_group):
    """
    Getter method for infer_nonwear_triaxial
    """

    nonwear_bouts = load_bouts_from_hdf5_group(hdf5_group)
    return nonwear_bouts


def set_infer_nonwear_triaxial(nonwear_bouts, hdf5_group):
    """
    Setter method for infer_nonwear_triaxial
    """

    save_bouts_to_hdf5_group(nonwear_bouts, hdf5_group)


def infer_nonwear_triaxial(x, y, z, minimum_length=timedelta(hours=1), noise_cutoff_mg=13, hdf5_file=None):
    """
    Use the 3 channels of triaxial acceleration to infer periods of nonwear
    """

    args = {"x":x, "y":y, "z":z, "minimum_length":minimum_length, "noise_cutoff_mg":noise_cutoff_mg}
    params = ["minimum_length", "noise_cutoff_mg"]
    return do_if_not_cached("infer_nonwear_triaxial", infer_nonwear_triaxial_method, args, params, get_infer_nonwear_triaxial, set_infer_nonwear_triaxial, hdf5_file)


def infer_valid_days(channel, wear_bouts, valid_criterion=timedelta(hours=10)):

    # Generate day-long windows
    start = start_of_day(channel.timestamps[0])
    day_windows = []
    while start < channel.timeframe[1]:
        day_windows.append(Bout(start, start+timedelta(days=1)))
        start += timedelta(days=1)

    valid_windows = []
    invalid_windows = []
    for window in day_windows:
        # how much does all of wear_bouts intersect with window?
        intersections = bout_list_intersection([window], wear_bouts)

        total = total_time(intersections)

        # If the amount of overlap exceeds the valid criterion, it is valid
        if total >= valid_criterion:
            # show valid day windows in orange
            window.draw_properties = {"lw": 0, "facecolor": "#ffa03a", "alpha": 0.25}
            valid_windows.append(window)
        else:
            invalid_windows.append(window)

    return (invalid_windows, valid_windows)


def bouts_exceeding_threshold(channel, threshold, minimum_length=timedelta(minutes=10)):
    """
    Find the bouts, exceeding minimum_length in length, in a channel during which the value of the data is greater than max_value

    :param channel: data channel
    :param threshold: the maximum value criteria
    :param window_size: the minimum bout length, given as a timedelta
    :return: list of bouts
    """
    # maximum int32 value
    max_int = 2147483647

    bouts_list = channel.bouts(low=threshold, high=max_int)

    final_bouts = limit_to_lengths(bouts_list, min_length=minimum_length)

    return final_bouts


def infer_nonwear_for_qc(x, y, z, noise_cutoff_mg, minimum_length=timedelta(hours=1)):
    """
    Loosely infer when monitor was not being worn, based on QC page-level data
    """

    x_intersect_y_intersect_z = infer_still_bouts_triaxial(x, y, z, window_size=timedelta(minutes=2), noise_cutoff_mg=noise_cutoff_mg,
                                                           minimum_length=timedelta(minutes=5))

    # Restrict those bouts to only those with a length that exceeds the minimum length criterion
    x_intersect_y_intersect_z = limit_to_lengths(x_intersect_y_intersect_z, min_length=minimum_length)

    return x_intersect_y_intersect_z


def create_quadrant_bouts(channel):
    """
    :param channel: channel to be examined
    :return: a list of Bout objects, one for each quadrant

    quadrant_0 = 00:00 -> 06:00
    quadrant_1 = 06:00 -> 12:00
    quadrant_2 = 12:00 -> 18:00
    quadrant_3 = 18:00 -> 00:00
    """

    # define quadrant windows
    q_0, q_1, q_2, q_3 = [],[],[],[]

    start = start_of_day(channel.timestamps[0])
    while start < channel.timeframe[1]:
        q_0.append(Bout(start, start + timedelta(hours=6)))
        q_1.append(Bout(start + timedelta(hours=6), start + timedelta(hours=12)))
        q_2.append(Bout(start + timedelta(hours=12), start + timedelta(hours=18)))
        q_3.append(Bout(start + timedelta(hours=18), start + timedelta(days=1)))
        start += timedelta(days=1)

    return (q_0, q_1, q_2, q_3)
