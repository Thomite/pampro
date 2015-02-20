
from pampro import Time_Series, Bout, Channel, channel_inference


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

        #if i%100 == 0:
        #    print(i)
        
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

def calibrate(x,y,z, allow_overwrite=True, budget=7500, noise_cutoff_mg=13):


    ts_temp = Time_Series.Time_Series("Temporary")
    ts_temp.add_channels([x,y,z])
    stats = {x.name:["mean", "std"], y.name:["mean", "std"], z.name:["mean", "std"]}

    # Get 10 second windowed means and stdevs of X,Y,Z
    result_chans = ts_temp.piecewise_statistics(timedelta(seconds=10), stats, time_period=x.timeframe)
    ts_temp.add_channels(result_chans)

    x_bouts = ts_temp.get_channel(x.name + "_std").bouts(0, float(noise_cutoff_mg)/1000.0, timedelta(seconds=60))
    y_bouts = ts_temp.get_channel(y.name + "_std").bouts(0, float(noise_cutoff_mg)/1000.0, timedelta(seconds=60))
    z_bouts = ts_temp.get_channel(z.name + "_std").bouts(0, float(noise_cutoff_mg)/1000.0, timedelta(seconds=60))

    # Intersect the bouts to get bouts where all axes have low stdev
    x_y_bouts = Bout.bout_list_intersection(x_bouts, y_bouts)
    still_bouts = Bout.bout_list_intersection(x_y_bouts, z_bouts)

    #print("Num still bouts", len(still_bouts))
    #print("Total still time", Bout.total_time(still_bouts).total_seconds())

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

        return (x, y, z, calibration_parameters, (start_error, end_error), (len(still_bouts), Bout.total_time(still_bouts).total_seconds() ))

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

