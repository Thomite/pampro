# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import statsmodels.api as sm
from .channel_inference import *
from .hdf5 import *

def get_calibrate(hdf5_group):
    """
    This hdf5 file object contains a calirbate cache, load it and return the results
    """

    return dictionary_from_attributes(hdf5_group)


def set_calibrate(cache, hdf5_group):
    """
    Create a group for the cached results of calibrate() and write the cache to it
    """

    dictionary_to_attributes(cache, hdf5_group)


def is_calibrated(channel):
    """
    Return True if the channel appears to have been autocalibrated already.
    """

    return hasattr(channel, "calibrated") and channel.calibrated == True


def calibrate_slave(x, y, z, temperature, budget=1000, noise_cutoff_mg=13, calibration_statistics=False):
    """
    Slave to calibrate()
    """

    stillbouts_ts, calibration_diagnostics = calibrate_stepone(x, y, z, temperature, budget=1000, noise_cutoff_mg=13)

    calibration_diagnostics = calibrate_steptwo(stillbouts_ts, calibration_diagnostics, calibration_statistics)

    return calibration_diagnostics


def calibrate(x, y, z, temperature=None, budget=1000, noise_cutoff_mg=13, hdf5_file=None, calibration_statistics=False):
    """ Use still bouts in the given triaxial data to calibrate it and return the calibrated channels """

    args = {"x": x, "y": y, "z": z, "temperature": temperature, "budget": budget, "noise_cutoff_mg": noise_cutoff_mg,
            "calibration_statistics": calibration_statistics}
    params = ["temperature", "budget", "noise_cutoff_mg", "calibration_statistics"]
    calibration_diagnostics = do_if_not_cached("calibrate", calibrate_slave, args, params, get_calibrate, set_calibrate, hdf5_file)

    # Regardless of how we get the results, extract the offset and scales
    calibration_parameters = [calibration_diagnostics[var] for var in ["x_offset", "x_scale", "y_offset", "y_scale", "z_offset", "z_scale"]]

    if temperature is not None:
        calibration_parameters = [calibration_diagnostics[var] for var in ["x_temp_offset", "y_temp_offset", "z_temp_offset"]]

    # Apply the best calibration factors to the data
    do_calibration(x, y, z, temperature, calibration_parameters)

    return x, y, z, calibration_diagnostics


def calibrate_stepone(x, y, z, temperature=None, battery=None, budget=1000, noise_cutoff_mg=13):
    # All diagnostics and results will be saved to this dictionary
    # calibrate() returns this dictionary, and passes it to hdf5.dictionary_to_attributes() for caching
    stillbouts_ts = Time_Series("")
    calibration_diagnostics = OrderedDict()

    # Saving passed parameters for later reference
    calibration_diagnostics["budget"] = budget
    calibration_diagnostics["noise_cutoff_mg"] = noise_cutoff_mg

    vm = infer_vector_magnitude(x,y,z)

    # Get a list of bouts where standard deviation in each axis is below given threshold ("still")
    still_bouts = infer_still_bouts_triaxial(x,y,z, noise_cutoff_mg=noise_cutoff_mg, minimum_length=timedelta(minutes=1))
    num_still_bouts = len(still_bouts)
    num_still_seconds = total_time(still_bouts).total_seconds()

    # Summarise VM in 10s intervals
    vm_windows = vm.piecewise_statistics(timedelta(seconds=10), [("generic", ["mean"])], time_period=vm.timeframe)[0]
    
    # Get a list where VM was between 0.8 and 1.2g ("reasonable") # resonable criteria changed from between 0.5 and 1.5g as considered too broad 25/06/19
    reasonable_bouts = vm_windows.bouts(0.8, 1.2)
    num_reasonable_bouts = len(reasonable_bouts)
    num_reasonable_seconds = total_time(reasonable_bouts).total_seconds()

    # We only want still bouts where the VM level was within 0.2g of 1g (previously within 0.5g of 1g, see above)
    # Therefore intersect "still" time with "reasonable" time
    still_bouts = bout_list_intersection(reasonable_bouts, still_bouts)

    # And we only want bouts where it was still and reasonable for 10s or longer
    still_bouts = limit_to_lengths(still_bouts, min_length = timedelta(seconds=10))
    num_final_bouts = len(still_bouts)
    num_final_seconds = total_time(still_bouts).total_seconds()

    # Get the average X,Y,Z for each still bout (inside which, by definition, XYZ should not change)
    still_x, std_x, num_samples = x.build_statistics_channels(still_bouts, [("generic", ["mean", "std", "n"])])
    still_y, std_y = y.build_statistics_channels(still_bouts, [("generic", ["mean", "std"])])
    still_z, std_z = z.build_statistics_channels(still_bouts, [("generic", ["mean", "std"])])

    channels = [num_samples, still_x, std_x, still_y, std_y, still_z, std_z]
    # Add the statistics channels to the empty still bouts Time Series
    stillbouts_ts.add_channels(channels)

    # if temperature data is required build the statistics channels and add to the still bouts Time Series
    if temperature is not None:
        still_temperature, std_temperature = temperature.build_statistics_channels(still_bouts, [("generic", ["mean", "std"])])
        calibration_diagnostics["mean_temperature"] = np.mean(temperature.data)
        calibration_diagnostics["min_temperature"] = np.min(temperature.data)
        calibration_diagnostics["max_temperature"] = np.max(temperature.data)
        calibration_diagnostics["std_temperature"] = np.std(temperature.data)
        temp_channels = [still_temperature, std_temperature]
        stillbouts_ts.add_channels(temp_channels)

    # if battery data is required build the statistics channel and add to the still bouts Time Series
    if battery is not None:
        still_battery = battery.build_statistics_channels(still_bouts, [("generic", ["mean"])])[0]
        calibration_diagnostics["min_battery"] = np.min(battery.data)
        calibration_diagnostics["max_battery"] = np.max(battery.data)
        stillbouts_ts.add_channel(still_battery)

    # Still bouts information
    calibration_diagnostics["num_final_bouts"] = num_final_bouts
    calibration_diagnostics["num_final_seconds"] = num_final_seconds
    calibration_diagnostics["num_still_bouts"] = num_still_bouts
    calibration_diagnostics["num_still_seconds"] = num_still_seconds
    calibration_diagnostics["num_reasonable_bouts"] = num_reasonable_bouts
    calibration_diagnostics["num_reasonable_seconds"] = num_reasonable_seconds

    return (stillbouts_ts, calibration_diagnostics)


def calibrate_steptwo(stillbouts_ts, calibration_diagnostics, calibration_statistics=False, num_iterations=1000):

    still_x, still_y, still_z, num_samples, still_temperature = still_bouts_from_ts(stillbouts_ts)

    # Get the octant positions of the points to calibrate on
    occupancy = octant_occupancy(still_x.data, still_y.data, still_z.data)

    # if enhanced calibration statistics are required...
    if calibration_statistics:
        ##############################
        # calculate the standard deviation and 25th, 50th and 75th percentiles of observations for each axis:
        percentiles = [("p25",25), ("p50",50), ("p75",75)]
        axes = [("x",still_x.data), ("y",still_y.data), ("z",still_z.data)]
        for axis, data in axes:
            calibration_diagnostics[axis + "_upper_ratio"] = axis_distribution_ratio(data,0.3, upper_or_lower="upper")
            calibration_diagnostics[axis + "_lower_lower"] = axis_distribution_ratio(data, -0.3, upper_or_lower="lower")
            calibration_diagnostics[axis + "_std"] = np.std(data)
            for name,value in percentiles:
                calibration_diagnostics[axis + "_" + name] = np.percentile(data, value)

        # Are the octants fairly distributed?
        comparisons = {"x<0":[0,1,2,3], "x>0":[4,5,6,7], "y<0":[0,1,4,5], "y>0":[2,3,6,7], "z<0":[0,2,4,6], "z>0":[1,3,5,7]}
        for axis in ["x", "y", "z"]:
            mt = sum(occupancy[comparisons[axis + ">0"]])
            lt = sum(occupancy[comparisons[axis + "<0"]])
            calibration_diagnostics[axis + "_inequality"] = abs(mt-lt)/sum(occupancy)

        for i,occ in enumerate(occupancy):
            calibration_diagnostics["octant_"+str(i)] = occ

        ################################

    # Calculate the initial error without doing any calibration
    # i.e. set the parameters to an 'ideal'
    ideal_parameters = {"x_offset": 0,
                        "x_scale": 1,
                        "x_temp_offset": 0,
                        "y_offset": 0,
                        "y_scale": 1,
                        "y_temp_offset": 0,
                        "z_offset": 0,
                        "z_scale": 1,
                        "z_temp_offset": 0}

    start_error = evaluate_solution(still_x, still_y, still_z, num_samples, ideal_parameters)

    # Search for the correct way to calibrate the data:
    #    offset = use offset factors only
    #    offset_scale = use offset and scale factors
    #    offset_temp = use offset and temperature offset
    #    offset_scale_temp = use offset, scale and temperature offset

    # If we have less than 100 points to calibrate with, or if 3 or more octants are empty we will not use scale:
    use_scale = True
    if len(still_x.data) < 100 or sum(occupancy == 0) > 2:
        use_scale = False

    # Assign calibration method according to parameters 'use_scale' and 'still_temperature'
    if not use_scale and still_temperature is None:
        cal_mode = "offset"
        calibration_diagnostics["calibration_method"] = "offset only"

    elif not use_scale and still_temperature is not None:
        cal_mode = "offset_temp"
        calibration_diagnostics["calibration_method"] = "offset only with temperature"

    elif use_scale and still_temperature is None:
        cal_mode = "offset_scale"
        calibration_diagnostics["calibration_method"] = "offset and scale"

    elif use_scale and still_temperature is not None:
        cal_mode = "offset_scale_temp"
        calibration_diagnostics["calibration_method"] = "offset and scale with temperature"

    # Create a DataFrame containing the original x,y,z data and the x,y,z to be transformed data:
    df = pd.DataFrame({"X_orig": still_x.data,
                       "Y_orig": still_y.data,
                       "Z_orig": still_z.data,
                       "X": still_x.data,
                       "Y": still_y.data,
                       "Z": still_z.data})

    # Create an intercept row in order to find the offset factor
    df["intercept"] = 1

    # Find the first "closest points" for the matched arrays
    update_matched(df)

    # find the calibration parameters required to optimise x,y,z to the closest points
    calibration_parameters = find_calibration_parameters(df, still_temperature, cal_mode, calibration_statistics, num_iterations)

    # update the calibration_diagnostics dictionary with the calibration parameters
    calibration_diagnostics.update(calibration_parameters)

    # Calculate the final error after calibration
    end_error = evaluate_solution(still_x, still_y, still_z, num_samples, calibration_parameters, still_temperature)

    calibration_diagnostics["start_error"] = start_error
    calibration_diagnostics["end_error"] = end_error

    return calibration_diagnostics


def find_calibration_parameters(df, temperature, cal_mode, calibration_statistics, num_iterations, optimal_t=25):
    """Find the offset and scaling factors for each 3D axis. Assumes the input vectors are only still points."""

    if "temp" in cal_mode:
        # create a column of T - optimal_T (mean temperature for each still bout minus the optimal temperature)
        # i.e. the deviation in T from the optimal
        df["T_dev"] = temperature.data - optimal_t

    for i in range(num_iterations):
        # do linear regression:
        x_results, y_results, z_results = dataframe_regression(df, cal_mode, do_or_undo="do")

        # results.params() gives the calibration parameters thus:
        # x_results.params() = [x_scale, x_offset, x_temp_offset]   (last item only applies if temperature is used)
        df = dataframe_transformation(df, x_results.params, y_results.params, z_results.params,
                                                           cal_mode)
        # update the "matched" arrays to reflect the new "closest points" after the dataframe transformation
        update_matched(df)

    # Regress the backup copy of the original input against the transformed version,
    # to calculate offset, scale and temperature offset scalar (if temperature used)
    x_results_final, y_results_final, z_results_final = dataframe_regression(df, cal_mode, do_or_undo="undo")

    calibration_parameters = {"x_offset": x_results_final.params[1],
                              "x_scale": x_results_final.params[0],
                              "y_offset": y_results_final.params[1],
                              "y_scale": y_results_final.params[0],
                              "z_offset": z_results_final.params[1],
                              "z_scale": z_results_final.params[0]
                              }

    if "temp" in cal_mode:
        calibration_parameters["x_temp_offset"] = x_results_final.params[2]
        calibration_parameters["y_temp_offset"] = y_results_final.params[2]
        calibration_parameters["z_temp_offset"] = z_results_final.params[2]
    else:
        calibration_parameters["x_temp_offset"] = 0
        calibration_parameters["y_temp_offset"] = 0
        calibration_parameters["z_temp_offset"] = 0

    # if enhanced calibration statistics are required...
    if calibration_statistics:

        ######################

        # extract the error in the final regression fit for each axis
        calibration_parameters["x_rsquared"] = x_results_final.rsquared
        calibration_parameters["y_rsquared"] = y_results_final.rsquared
        calibration_parameters["z_rsquared"] = z_results_final.rsquared

        x_bse = x_results_final.bse
        y_bse = y_results_final.bse
        z_bse = z_results_final.bse

        calibration_parameters["x_scale_se"] = x_bse[0]
        calibration_parameters["y_scale_se"] = y_bse[0]
        calibration_parameters["z_scale_se"] = z_bse[0]

        calibration_parameters["x_offset_se"] = x_bse[1]
        calibration_parameters["y_offset_se"] = y_bse[1]
        calibration_parameters["z_offset_se"] = z_bse[1]

        if "temp" in cal_mode:
            calibration_parameters["x_temp_offset_se"] = x_bse[2]
            calibration_parameters["y_temp_offset_se"] = y_bse[2]
            calibration_parameters["z_temp_offset_se"] = z_bse[2]

        #########################

    return calibration_parameters


def nearest_sphere_surface(x_input, y_input, z_input):
    """Given the 3D co-ordinates of a point,
    return the 3D co-ordinates of the point on the surface of a unit sphere. """

    vm = math.sqrt(sum([x_input**2, y_input**2, z_input**2]))
    return (x_input/vm, y_input/vm, z_input/vm)


def update_matched(df):
    """Takes a df of still bout data and updates the "matched" arrays"""

    # check if "x_matched", "y_matched" and "z_matched" exist already, if not create them as empty arrays:
    if {"X_matched", "Y_matched", "Z_matched"}.issubset(df.columns):
        pass
    else:
        # Need 3 blank arrays to populate
        x_matched = np.empty(len(df.X_orig))
        y_matched = np.empty(len(df.Y_orig))
        z_matched = np.empty(len(df.Z_orig))

        df["X_matched"] = x_matched
        df["Y_matched"] = y_matched
        df["Z_matched"] = z_matched

    for i in range(len(df.X)):

        # For each point, find its nearest point on the surface of a sphere
        closest = nearest_sphere_surface(df.X[i], df.Y[i], df.Z[i])

        # Put the result in the X,Y,Z matched arrays
        df.at[i, "X_matched"] = closest[0]
        df.at[i, "Y_matched"] = closest[1]
        df.at[i, "Z_matched"] = closest[2]

    return df


def dataframe_regression(df, cal_mode, do_or_undo="do"):
    """Given a dataframe(df), perform liner regression on the columns required, according to the cal_mode variable.
       "do_or_undo" variable determines the direction of the regression.

       The required columns in the dataframe:
       df.X_matched, df.Y_matched, df.Z_matched (the values of x,y,z matched to closest sphere surface point)
       df.X_orig, df.Y_orig, df.Z_orig (the original values of x,y,z that are preserved)
       df.X, df.Y, df.Z (the values of x,y,z that update after the regression)
       df.intercept (column of ones to act as intercept)
       df.T_dev (column of deviation in temperature from the optimal - ONLY USED IF cal_mode = "offset_scale_temp" or "offset_temp")
    """

    # perform linear regression to optimise to matched column
    if do_or_undo == "do":

        # if temperature is used in calibration
        if "temp" in cal_mode:

            x_results = sm.OLS(df.X_matched, df[["X", "intercept", "T_dev"]]).fit()
            y_results = sm.OLS(df.Y_matched, df[["Y", "intercept", "T_dev"]]).fit()
            z_results = sm.OLS(df.Z_matched, df[["Z", "intercept", "T_dev"]]).fit()
        # if temperature NOT used in calibration
        else:
            x_results = sm.OLS(df.X_matched, df[["X", "intercept"]]).fit()
            y_results = sm.OLS(df.Y_matched, df[["Y", "intercept"]]).fit()
            z_results = sm.OLS(df.Z_matched, df[["Z", "intercept"]]).fit()

    # perform linear regression to optimise the transformed x,y,z data back to the original x,y,z data
    elif do_or_undo == "undo":

        # if temperature was used in calibration
        if "temp" in cal_mode:
            x_results = sm.OLS(df["X"], df[["X_orig", "intercept", "T_dev"]]).fit()
            y_results = sm.OLS(df["Y"], df[["Y_orig", "intercept", "T_dev"]]).fit()
            z_results = sm.OLS(df["Z"], df[["Z_orig", "intercept", "T_dev"]]).fit()
        # if temperature was NOT used in calibration
        else:
            x_results = sm.OLS(df["X"], df[["X_orig", "intercept"]]).fit()
            y_results = sm.OLS(df["Y"], df[["Y_orig", "intercept"]]).fit()
            z_results = sm.OLS(df["Z"], df[["Z_orig", "intercept"]]).fit()

    return x_results, y_results, z_results


def dataframe_transformation(df, x_params, y_params, z_params, cal_mode):
    """Given the output from the function dataframe_regression() transform the columns df.X, df.Y, df.Z in a given dataframe, depending on the value of the cal_mode variable.

    results.params() gives the calibration parameters thus:
    x_results.params() = [x_scale, x_offset, x_temp_offset]   (last item only applies if temperature is used)"""

    if cal_mode == "offset":
        # Transform the input points using ONLY the offset co-efficient
        # e.g. X(t) = X(t) + x_offset
        df.X = df.X + x_params[1]
        df.Y = df.Y + y_params[1]
        df.Z = df.Z + z_params[1]

    elif cal_mode == "offset_scale":
        # Transform the input points using the regression co-efficients for offset and scale
        # e.g. X(t) = (X(t)* x_scale) + x_offset
        df.X = (df.X * x_params[0]) + x_params[1]
        df.Y = (df.Y * y_params[0]) + y_params[1]
        df.Z = (df.Z * z_params[0]) + z_params[1]

    elif cal_mode == "offset_temp":
        # Transform the input points using the regression co-efficients for offset and the temperature-scaled offset
        # e.g. X(t) = (X(t) + x_offset + (T_dev(t)*temp_offset)
        df.X = df.X + x_params[1] + (df.T_dev * x_params[2])
        df.Y = df.Y + y_params[1] + (df.T_dev * y_params[2])
        df.Z = df.Z + z_params[1] + (df.T_dev * z_params[2])

    elif cal_mode == "offset_scale_temp":
        # Transform the input points using the regression co-efficients for offset and scale and the temperature-scaled offset
        # e.g. X(t) = (X(t)* x_scale) + x_offset + (T_dev(t)*temp_offset)
        df.X = (df.X * x_params[0]) + x_params[1] + (df.T_dev * x_params[2])
        df.Y = (df.Y * y_params[0]) + y_params[1] + (df.T_dev * y_params[2])
        df.Z = (df.Z * z_params[0]) + z_params[1] + (df.T_dev * z_params[2])

    return df


def do_calibration(x,y,z,temperature,cp, optimal_t=25):
    """
    Performs calibration on given channel using a given dictionary of parameters (cp)
     """
    # if temperature is used for calibration:
    if temperature is not None:
        # create an array of T - optimal_T (temperature minus the optimal temperature) i.e. the deviation in T from the optimum
        temp_dev = np.empty(len(temperature.data))
        for i in range(len(temperature.data)):
            temp_dev[i] = temperature.data[i] - optimal_t

        x.data = cp["x_offset"] + (temp_dev * cp["x_temp_offset"]) + (x.data * cp["x_scale"])
        y.data = cp["y_offset"] + (temp_dev * cp["y_temp_offset"]) + (y.data * cp["y_scale"])
        z.data = cp["z_offset"] + (temp_dev * cp["z_temp_offset"]) + (z.data * cp["z_scale"])

        x.temp_offset = cp["x_temp_offset"]
        y.temp_offset = cp["y_temp_offset"]
        z.temp_offset = cp["z_temp_offset"]

    # if temperature is not used for calibration:
    else:
        x.data = cp["x_offset"] + (x.data * cp["x_scale"])
        y.data = cp["y_offset"] + (y.data * cp["y_scale"])
        z.data = cp["z_offset"] + (z.data * cp["z_scale"])

    x.offset = cp["x_offset"]
    x.scale = cp["x_scale"]
    x.calibrated = True

    y.offset = cp["y_offset"]
    y.scale = cp["y_scale"]
    y.calibrated = True

    z.offset = cp["z_offset"]
    z.scale = cp["z_scale"]
    z.calibrated = True


def undo_calibration(x,y,z,temperature,cp, optimal_t = 25):
    """
    Reverses calibration on given channel using a given dictionary of parameters (cp)
    """

    # if temperature is used for calibration:
    if temperature is not None:
        # create an array of T - optimal_T (temperature minus the optimal temperature) i.e. the deviation in T from the optimum
        temp_dev = np.empty(len(temperature.data))
        for i in range(len(temperature.data)):
            temp_dev[i] = temperature.data[i] - optimal_t

        x.data = -cp["x_offset"] - (temp_dev * cp["x_temp_offset"]) + (x.data / cp["x_scale"])
        y.data = -cp["y_offset"] - (temp_dev * cp["y_temp_offset"]) + (y.data / cp["y_scale"])
        z.data = -cp["z_offset"] - (temp_dev * cp["z_temp_offset"]) + (z.data / cp["z_scale"])

    else:
        x.data = -cp["x_offset"] + (x.data / cp["x_scale"])
        y.data = -cp["y_offset"] + (y.data / cp["y_scale"])
        z.data = -cp["z_offset"] + (z.data / cp["z_scale"])

    x.calibrated = False
    y.calibrated = False
    z.calibrated = False


def undo_calibration_using_diagnostics(x,y,z,cd):
    """
    Convenience function that pulls the offset and scale values out of a regular calibration diagnostics dictionary.
    """
    undo_calibration(x, y, z, [cd["x_offset"],cd["x_scale"],cd["y_offset"],cd["y_scale"],cd["z_offset"],cd["z_scale"]])


def evaluate_solution(still_x, still_y, still_z, still_n, calibration_parameters, still_temperature=None):
    """ Calculates the RMSE of the input XYZ signal if calibrated according to input calibration parameters"""

    # Temporarily adjust the channels of still data, which has collapsed x,y,z values
    do_calibration(still_x, still_y, still_z, still_temperature, calibration_parameters)

    # Get the VM of the calibrated channel
    vm = infer_vector_magnitude(still_x, still_y, still_z)

    # se = sum error
    se = 0.0

    for vm_val,n in zip(vm.data, still_n.data):
        se += (abs(1.0 - vm_val)**2)*n

    rmse = math.sqrt(se / len(vm.data))

    # Undo the temporary calibration
    undo_calibration(still_x, still_y, still_z, still_temperature, calibration_parameters)

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


def axis_distribution_ratio(data, cutoff, upper_or_lower="upper"):
    """Returns a ratio of the number of samples lying either above an upper cutoff value or lying below a lower cutoff as a proportion of the total"""

    count = 0
    if upper_or_lower == "upper":
        for a in data:
            if a > cutoff:
                count += 1

    elif upper_or_lower == "lower":
        for a in data:
            if a < cutoff:
                count += 1

    ratio = count/len(data)

    return ratio


def still_bouts_from_ts(stillbouts_ts):
    """Extracts the still bouts data (including temperature if available) and returns channels of data"""

    still_x = stillbouts_ts["X_mean"]
    still_y = stillbouts_ts["Y_mean"]
    still_z = stillbouts_ts["Z_mean"]
    num_samples = stillbouts_ts["X_n"]

    # Ascertain if temperature data is present:
    try:
        still_temperature = stillbouts_ts["Temperature_mean"]
    except:
        still_temperature = None

    return still_x, still_y, still_z, num_samples, still_temperature
