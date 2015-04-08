
from pampro import Time_Series, Bout, Channel, channel_inference, time_utilities


from datetime import datetime, date, time, timedelta
import math
import copy
import random
from scipy import stats
import numpy as np

def nearest_sphere_surface(x_input, y_input, z_input):

    """Given the 3D co-ordinates of a point, return the 3D co-ordinates of the point on the surface of a unit sphere. """

    vm = math.sqrt(sum([x_input**2, y_input**2, z_input**2]))
    return (x_input/vm, y_input/vm, z_input/vm)



def find_calibration_parameters(x_input, y_input, z_input, num_iterations=1000):

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

def calibrate(x,y,z, allow_overwrite=True, budget=1000, noise_cutoff_mg=13):


    vm = channel_inference.infer_vector_magnitude(x,y,z)

    still_bouts = channel_inference.infer_still_bouts_triaxial(x,y,z, noise_cutoff_mg=noise_cutoff_mg)
    vm_windows = vm.piecewise_statistics( timedelta(seconds=10), ["mean"], time_period=(time_utilities.start_of_hour(x.timeframe[0]), time_utilities.end_of_hour(x.timeframe[1])) )[0]

    reasonable_bouts = vm_windows.bouts(0.5, 1.5)

    num_still_bouts = len(still_bouts)
    num_still_seconds = Bout.total_time(still_bouts).total_seconds()
    num_reasonable_bouts = len(reasonable_bouts)
    num_reasonable_seconds = Bout.total_time(reasonable_bouts).total_seconds()

    still_bouts = Bout.bout_list_intersection(reasonable_bouts, still_bouts)
    Bout.cache_lengths(still_bouts)
    still_bouts = Bout.limit_to_lengths(still_bouts, min_length = timedelta(seconds=10))

    num_final_bouts = len(still_bouts)
    num_final_seconds = Bout.total_time(still_bouts).total_seconds()

    # Get the average X,Y,Z for each still bout (inside which, by definition, XYZ should not change)
    still_x, num_samples = x.build_statistics_channels(still_bouts, ["mean", "n"])
    still_y = y.build_statistics_channels(still_bouts, ["mean"])[0]
    still_z = z.build_statistics_channels(still_bouts, ["mean"])[0]

    still_x.name = "still_x"
    still_y.name = "still_y"
    still_z.name = "still_z"
    num_samples.name = "n"




    # Calculate the initial error without doing any calibration
    start_error = evaluate_solution(still_x, still_y, still_z, num_samples, [0,1,0,1,0,1])

    calibration_parameters = find_calibration_parameters(still_x.data, still_y.data, still_z.data)

    # Calculate the final error after calibration
    end_error = evaluate_solution(still_x, still_y, still_z, num_samples, calibration_parameters)

    if allow_overwrite:
        # If we do not need to preserve the original x,y,z values, we can just calibrate that data

        # Apply the best calibration factors to the data
        do_calibration(x, y, z, calibration_parameters)

        return (x, y, z, calibration_parameters, (start_error, end_error), (num_final_bouts, num_final_seconds,num_still_bouts, num_still_seconds, num_reasonable_bouts, num_reasonable_seconds, still_bouts ))

    else:
        # Else we create an independent copy of the raw data and calibrate that instead
        cal_x = copy.deepcopy(x)
        cal_y = copy.deepcopy(y)
        cal_z = copy.deepcopy(z)

        # Apply the best calibration factors to the data
        do_calibration(cal_x, cal_y, cal_z, calibration_parameters)

        return (cal_x, cal_y, cal_z, calibration_parameters, (start_error, end_error), (len(still_bouts), Bout.total_time(still_bouts).total_seconds() ))





def do_calibration(x,y,z,values):

    x.data = values[0] + (x.data * values[1])
    y.data = values[2] + (y.data * values[3])
    z.data = values[4] + (z.data * values[5])


def undo_calibration(x,y,z,values):

    x.data = -values[0] + (x.data / values[1])
    y.data = -values[2] + (y.data / values[3])
    z.data = -values[4] + (z.data / values[5])


def evaluate_solution(still_x, still_y, still_z, still_n, calibration_parameters):

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
