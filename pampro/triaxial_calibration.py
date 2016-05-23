
from pampro import Time_Series, Bout, Channel, channel_inference, time_utilities
from datetime import datetime, date, time, timedelta
import math
import copy
import random
from scipy import stats
import numpy as np
from collections import OrderedDict

def get_cached_calibration(x,y,z):
    """
    If each of the 3 Channels contain the appropriate offset and scale parameters, return them in the correct order, else return None.
    """

    try:
        x_offset = x.offset
        x_scale = x.scale
        y_offset = y.offset
        y_scale = y.scale
        z_offset = z.offset
        z_scale = z.scale

        return (x_offset, x_scale, y_offset, y_scale, z_offset, z_scale)

    except:

        return None

def is_calibrated(channel):
    """
    Return True if the channel appears to have been autocalibrated already.
    """
    return hasattr(channel, "calibrated") and channel.calibrated == True


def nearest_sphere_surface(x_input, y_input, z_input):
    """Given the 3D co-ordinates of a point, return the 3D co-ordinates of the point on the surface of a unit sphere. """

    vm = math.sqrt(sum([x_input**2, y_input**2, z_input**2]))
    return (x_input/vm, y_input/vm, z_input/vm)

def find_calibration_parameters(x_input, y_input, z_input, num_iterations=1000, offset_only=False):
    """Find the offset and scaling factors for each 3D axis. Assumes the input vectors are only still points."""

    # Need to keep a copy of the original input
    x_input_copy = x_input[::]
    y_input_copy = y_input[::]
    z_input_copy = z_input[::]

    # Need 3 blank arrays to populate
    x_matched = np.empty(len(x_input))
    y_matched = np.empty(len(y_input))
    z_matched = np.empty(len(z_input))


    for i in range(num_iterations):

        for i,a,b,c in zip(range(len(x_input)),x_input, y_input, z_input):

            # For each point, find its nearest point on the surface of a sphere
            closest = nearest_sphere_surface(a,b,c)

            # Put the result in the X,Y,Z arrays
            x_matched[i] = closest[0]
            y_matched[i] = closest[1]
            z_matched[i] = closest[2]

        # Now that each X input is matched up against a "perfect" X on a sphere, do linear regression
        x_regression_results = stats.linregress(x_input, x_matched)
        y_regression_results = stats.linregress(y_input, y_matched)
        z_regression_results = stats.linregress(z_input, z_matched)

        if offset_only:
            # Transform the input points using ONLY the offset co-efficient
            x_input = [x_regression_results[1] + x_input_val for x_input_val in x_input]
            y_input = [y_regression_results[1] + y_input_val for y_input_val in y_input]
            z_input = [z_regression_results[1] + z_input_val for z_input_val in z_input]
        else:
            # Transform the input points using the regression co-efficients
            x_input = [x_regression_results[1] + x_input_val*x_regression_results[0] for x_input_val in x_input]
            y_input = [y_regression_results[1] + y_input_val*y_regression_results[0] for y_input_val in y_input]
            z_input = [z_regression_results[1] + z_input_val*z_regression_results[0] for z_input_val in z_input]

    # Regress the backup copy of the original input against the transformed version, calculate how much we offset and scaled
    final_x_regression = stats.linregress(x_input_copy, x_input)[0:2]
    final_y_regression = stats.linregress(y_input_copy, y_input)[0:2]
    final_z_regression = stats.linregress(z_input_copy, z_input)[0:2]

    final_x_offset, final_x_scale = final_x_regression[1],final_x_regression[0]
    final_y_offset, final_y_scale = final_y_regression[1],final_y_regression[0]
    final_z_offset, final_z_scale = final_z_regression[1],final_z_regression[0]

    return final_x_offset, final_x_scale, final_y_offset, final_y_scale, final_z_offset, final_z_scale

def calibrate(x,y,z, allow_overwrite=True, budget=1000, noise_cutoff_mg=13, ignore_cached=False):
    """ Use still bouts in the given triaxial data to calibrate it and return the calibrated channels """

    calibration_diagnostics = OrderedDict()

    # If there are recommended offset and scale values in the Channels, use those rather than do the whole calibration
    # But ignore them if we've specifically said to
    cached_calibration = get_cached_calibration(x, y, z)
    if cached_calibration is not None and ignore_cached is False:

        print("Calibration found")
        # Apply the best calibration factors to the data
        do_calibration(x, y, z, cached_calibration)

        return (x, y, z, calibration_diagnostics)

    else:
        vm = channel_inference.infer_vector_magnitude(x,y,z)

        # Get a list of bouts where standard deviation in each axis is below given threshold ("still")
        still_bouts = channel_inference.infer_still_bouts_triaxial(x,y,z, noise_cutoff_mg=noise_cutoff_mg, minimum_length=timedelta(minutes=1))
        num_still_bouts = len(still_bouts)
        num_still_seconds = Bout.total_time(still_bouts).total_seconds()

        # Summarise VM in 10s intervals
        vm_windows = vm.piecewise_statistics(timedelta(seconds=10), [("generic", ["mean"])], time_period=vm.timeframe)[0]

        # Get a list where VM was between 0.5 and 1.5g ("reasonable")
        reasonable_bouts = vm_windows.bouts(0.5, 1.5)
        num_reasonable_bouts = len(reasonable_bouts)
        num_reasonable_seconds = Bout.total_time(reasonable_bouts).total_seconds()

        # We only want still bouts where the VM level was within 0.5g of 1g
        # Therefore insersect "still" time with "reasonable" time
        still_bouts = Bout.bout_list_intersection(reasonable_bouts, still_bouts)

        # And we only want bouts where it was still and reasonable for 10s or longer
        still_bouts = Bout.limit_to_lengths(still_bouts, min_length = timedelta(seconds=10))
        num_final_bouts = len(still_bouts)
        num_final_seconds = Bout.total_time(still_bouts).total_seconds()

        # Get the average X,Y,Z for each still bout (inside which, by definition, XYZ should not change)
        still_x, num_samples = x.build_statistics_channels(still_bouts, [("generic", ["mean", "n"])])
        still_y = y.build_statistics_channels(still_bouts, [("generic", ["mean"])])[0]
        still_z = z.build_statistics_channels(still_bouts, [("generic", ["mean"])])[0]

        # Get the octant positions of the points to calibrate on
        occupancy = octant_occupancy(still_x.data, still_y.data, still_z.data)

        # Are they fairly distributed?
        comparisons = {"x<0":[0,1,2,3], "x>0":[4,5,6,7], "y<0":[0,1,4,5], "y>0":[2,3,6,7], "z<0":[0,2,4,6], "z>0":[1,3,5,7]}
        for axis in ["x", "y", "z"]:
            mt = sum(occupancy[comparisons[axis + ">0"]])
            lt = sum(occupancy[comparisons[axis + "<0"]])
            calibration_diagnostics[axis + "_inequality"] = abs(mt-lt)/sum(occupancy)

        # Calculate the initial error without doing any calibration
        start_error = evaluate_solution(still_x, still_y, still_z, num_samples, [0,1,0,1,0,1])

        # Do offset and scale calibration by default
        offset_only_calibration = False
        calibration_diagnostics["calibration_method"] = "offset and scale"

        # If we have less than 500 points to calibrate with, or if more than 2 octants are empty
        if len(still_x.data) < 500 or sum(occupancy == 0) > 2:
            offset_only_calibration = True
            calibration_diagnostics["calibration_method"] = "offset only"


        # Search for the correct way to calibrate the data
        calibration_parameters = find_calibration_parameters(still_x.data, still_y.data, still_z.data, offset_only=offset_only_calibration)

        for param,value in zip("x_offset,x_scale,y_offset,y_scale,z_offset,z_scale".split(","), calibration_parameters):
            calibration_diagnostics[param] = value

        for i,occ in enumerate(occupancy):
            calibration_diagnostics["octant_"+str(i)] = occ

        # Calculate the final error after calibration
        end_error = evaluate_solution(still_x, still_y, still_z, num_samples, calibration_parameters)

        calibration_diagnostics["start_error"] = start_error
        calibration_diagnostics["end_error"] = end_error
        calibration_diagnostics["num_final_bouts"] = num_final_bouts
        calibration_diagnostics["num_final_seconds"] = num_final_seconds
        calibration_diagnostics["num_still_bouts"] = num_still_bouts
        calibration_diagnostics["num_still_seconds"] = num_still_seconds
        calibration_diagnostics["num_reasonable_bouts"] = num_reasonable_bouts
        calibration_diagnostics["num_reasonable_seconds"] = num_reasonable_seconds

        if allow_overwrite:
            # If we do not need to preserve the original x,y,z values, we can just calibrate that data

            # Apply the best calibration factors to the data
            do_calibration(x, y, z, calibration_parameters)

            return (x, y, z, calibration_diagnostics)

        else:
            # Else we create an independent copy of the raw data and calibrate that instead
            cal_x = copy.deepcopy(x)
            cal_y = copy.deepcopy(y)
            cal_z = copy.deepcopy(z)

            # Apply the best calibration factors to the data
            do_calibration(cal_x, cal_y, cal_z, calibration_parameters)

            return (cal_x, cal_y, cal_z, calibration_diagnostics)

def do_calibration(x,y,z,values):
    """
    Performs calibration on given channel using given parameters.
    Values should be in this order: [x_offset, x_scale, y_offset, y_scale, z_offset, z_scale]
     """
    x.data = values[0] + (x.data * values[1])
    y.data = values[2] + (y.data * values[3])
    z.data = values[4] + (z.data * values[5])

    x.calibrated = True
    y.calibrated = True
    z.calibrated = True

def undo_calibration(x,y,z,values):
    """
    Reverses calibration on given channel using given parameters
    Values should be in this order: [x_offset, x_scale, y_offset, y_scale, z_offset, z_scale]
    """

    x.data = -values[0] + (x.data / values[1])
    y.data = -values[2] + (y.data / values[3])
    z.data = -values[4] + (z.data / values[5])

    x.calibrated = False
    y.calibrated = False
    z.calibrated = False

def evaluate_solution(still_x, still_y, still_z, still_n, calibration_parameters):
    """ Calculates the RMSE of the input XYZ signal if calibrated according to input calibration parameters"""

    # Temporarily adjust the channels of still data, which has collapsed x,y,z values
    do_calibration(still_x, still_y, still_z, calibration_parameters)

    # Get the VM of the calibrated channel
    vm = channel_inference.infer_vector_magnitude(still_x, still_y, still_z)

    # se = sum error
    se = 0.0

    for vm_val,n in zip(vm.data, still_n.data):
        se += (abs(1.0 - vm_val)**2)*n

    rmse = math.sqrt(se / len(vm.data))

    # Undo the temporary calibration
    undo_calibration(still_x, still_y, still_z, calibration_parameters)

    return rmse

def octant_occupancy(x, y, z):
    """ Counts number of samples lying in each octal region around the origin """

    octants = np.zeros(8, dtype="int")

    for a,b,c in zip(x,y,z):

        if a < 0 and b < 0 and c < 0:
            octants[0] += 1
        elif a < 0 and b < 0 and c > 0:
            octants[1] += 1
        elif a < 0 and b > 0 and c < 0:
            octants[2] += 1
        elif a < 0 and b > 0 and c > 0:
            octants[3] += 1
        elif a > 0 and b < 0 and c < 0:
            octants[4] += 1
        elif a > 0 and b < 0 and c > 0:
            octants[5] += 1
        elif a > 0 and b > 0 and c < 0:
            octants[6] += 1
        elif a > 0 and b > 0 and c > 0:
            octants[7] += 1
        else:
            # Possible because of edge cases, shouldn't come up in calibration
            pass

    return octants
