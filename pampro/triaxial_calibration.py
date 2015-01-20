import Time_Series, Bout, Channel, channel_inference
from optipy import optipy
from datetime import datetime, date, time, timedelta
import math
import copy
import random

# We use a global variable to save enormously on redundant copy operations
ts_still = Time_Series.Time_Series("Still only")

def calibrate(x,y,z, allow_overwrite=True, budget=7500, noise_cutoff_mg=13):

	#print("calibrate A")
	global ts_still
	#print("calibrate B")
	ts_temp = Time_Series.Time_Series("Temporary")
	ts_temp.add_channels([x,y,z])
	stats = {x.name:["mean", "std"], y.name:["mean", "std"], z.name:["mean", "std"]}
	#print("calibrate C")
	# Get 10 second windowed means and stdevs of X,Y,Z
	result_chans = ts_temp.piecewise_statistics(timedelta(seconds=10), stats, time_period=x.timeframe)
	ts_temp.add_channels(result_chans)

	x_bouts = ts_temp.get_channel(x.name + "_std").bouts(0, float(noise_cutoff_mg)/1000.0, timedelta(seconds=60))
	y_bouts = ts_temp.get_channel(y.name + "_std").bouts(0, float(noise_cutoff_mg)/1000.0, timedelta(seconds=60))
	z_bouts = ts_temp.get_channel(z.name + "_std").bouts(0, float(noise_cutoff_mg)/1000.0, timedelta(seconds=60))
	#print("calibrate D")
	# Intersect the bouts to get bouts where all axes have low stdev
	x_y_bouts = Bout.bout_list_intersection(x_bouts, y_bouts)
	still_bouts = Bout.bout_list_intersection(x_y_bouts, z_bouts)

	
	print("Num still bouts", len(still_bouts))
	print("Total still time", Bout.total_time(still_bouts).total_seconds())


	# Get the average X,Y,Z for each still bout (inside which, by definition, XYZ should not change)
	still_x, still_x_std, num_samples = x.build_statistics_channels(still_bouts, ["mean", "std", "n"])
	still_y, still_y_std = y.build_statistics_channels(still_bouts, ["mean", "std"])
	still_z, still_z_std = z.build_statistics_channels(still_bouts, ["mean", "std"])

	#print(still_x_std.data)
	#print(still_y_std.data)
	#print(still_z_std.data)
    


	still_x.name = "still_x"
	still_y.name = "still_y"
	still_z.name = "still_z"


	still_x_std.name = "still_x_std"
	still_y_std.name = "still_y_std"
	still_z_std.name = "still_z_std"

	num_samples.name = "n"

	#print("calibrate E")
	ts_still.add_channels([still_x, still_y, still_z,still_x_std, still_y_std, still_z_std, num_samples])


	# Piecewise statistics currently throws -1s when no data matches the query
	# This is a temporary workaround
	bad_x = still_x_std.data != -1.
	still_x.data = still_x.data[bad_x]
	still_y.data = still_y.data[bad_x]
	still_z.data = still_z.data[bad_x]
	still_x_std.data = still_x_std.data[bad_x]
	still_y_std.data = still_y_std.data[bad_x]
	still_z_std.data = still_z_std.data[bad_x]

	num_samples.data = num_samples.data[bad_x]


	bad_y = still_y_std.data != -1.
	still_x.data = still_x.data[bad_y]
	still_y.data = still_y.data[bad_y]
	still_z.data = still_z.data[bad_y]
	still_x_std.data = still_x_std.data[bad_y]
	still_y_std.data = still_y_std.data[bad_y]
	still_z_std.data = still_z_std.data[bad_y]

	num_samples.data = num_samples.data[bad_y]


	bad_z = still_z_std.data != -1.
	still_x.data = still_x.data[bad_z]
	still_y.data = still_y.data[bad_z]
	still_z.data = still_z.data[bad_z]
	still_x_std.data = still_x_std.data[bad_z]
	still_y_std.data = still_y_std.data[bad_z]
	still_z_std.data = still_z_std.data[bad_z]

	num_samples.data = num_samples.data[bad_z]


	#print("calibrate F")
	# These are settings optipy uses to customise the optimisation procedure
	# We currently give optipy a budget of 7500 attempts to find a good calibration value
	generic_optimisation_parameters = {"evaluations":budget, "minimise":True }
	generic_optimisation_functions = {"evaluation_function":evaluate_solution_2, "generator_function":random_solution}

	specific_optimisation_parameters = {"elitism":0, "selection_strategy":"proportional", "population_size":40}
	specific_optimisation_functions = {"mutation_function":mutate_solution}
	#print("calibrate G")
	#---------------------------------------------------------------------
	# First calculate how "uncalibrated" the data currently is
	nothing = optipy.Solution([0.0,1.0,0.0,1.0,0.0,1.0])
	evaluate_solution_2(nothing)
	print(str(nothing.fitness) + " " + str(nothing.values))
	
	# Perform the actual optimisation procedure and return the best solution
	best_solution, diagnostics = optipy.perform_optimisation ("SA", generic_optimisation_functions, generic_optimisation_parameters, specific_optimisation_functions, specific_optimisation_parameters)
	
	print(str(best_solution.fitness) + " " + str(best_solution.values))

	print(diagnostics)

	if allow_overwrite:
		# If we do not need to preserve the original x,y,z values, we can just calibrate that data

		# Apply the best calibration factors to the data
		x.data = (best_solution.values[0]) + (x.data * best_solution.values[1])
		y.data = (best_solution.values[2]) + (y.data * best_solution.values[3])
		z.data = (best_solution.values[4]) + (z.data * best_solution.values[5])

		return (x, y, z, (best_solution.values), (nothing.fitness, best_solution.fitness), (len(still_bouts), Bout.total_time(still_bouts).total_seconds() ))

	else:
		# Else we create an independent copy of the raw data and calibrate that instead
		cal_x = copy.deepcopy(x)
		cal_y = copy.deepcopy(y)
		cal_z = copy.deepcopy(z)

		# Apply the best calibration factors to the data
		cal_x.data = (best_solution.values[0]) + (x.data * best_solution.values[1])
		cal_y.data = (best_solution.values[2]) + (y.data * best_solution.values[3])
		cal_z.data = (best_solution.values[4]) + (z.data * best_solution.values[5])

		return (cal_x, cal_y, cal_z, (best_solution.values), (nothing.fitness, best_solution.fitness), (len(still_bouts), Bout.total_time(still_bouts).total_seconds() ))

	



def do_calibration(x,y,z,values):
    
    x.data = values[0] + (x.data * values[1])
    y.data = values[2] + (y.data * values[3])
    z.data = values[4] + (z.data * values[5])


def undo_calibration(x,y,z,values):
    
    x.data = -values[0] + (x.data / values[1])
    y.data = -values[2] + (y.data / values[3])
    z.data = -values[4] + (z.data / values[5])


def evaluate_solution_2(solution):

    global ts_still
    
	# Temporarily adjust the time series object called ts_still which has collapsed x,y,z values
    do_calibration(ts_still.get_channel("still_x"),ts_still.get_channel("still_y"),ts_still.get_channel("still_z"),solution.values)
    

    x,y,z,num_samples = ts_still.get_channel("still_x"),ts_still.get_channel("still_y"),ts_still.get_channel("still_z"),ts_still.get_channel("n")

    
	# Get the VM of the calibrated channel
    vm = channel_inference.infer_vector_magnitude(x,y,z)     
    
	# se = sum error
    se = 0.0

    for vm_val,n in zip(vm.data, num_samples.data):
        se += (abs(1.0 - vm_val)**2)*n

    
    #mse = se / len(vm.data)
    #rmse = math.sqrt(mse) 
        
    solution.fitness = se
    
	# Undo the temporary calibration
    undo_calibration(ts_still.get_channel("still_x"),ts_still.get_channel("still_y"),ts_still.get_channel("still_z"),solution.values)
    
    return se



def random_solution():
	# We start off with 0 offset and 1 scale for each axis - in other words, no adjustment at all
    return optipy.Solution([0.0,1.0,0.0,1.0,0.0,1.0])


def mutate_solution(solution):

    mutant = copy.deepcopy(solution)

	# Pick a value to change - 90% of the time, change an offset parameter rather than a scale
    if random.random() < 0.9:
        index = random.choice([0,2,4])
    else:
        index = random.choice([1,3,5])

	# The scale values are at indices 1,3 and 5. Since they change the data much more significantly, we adjust these values much less than the offset values.
    if index in [0,2,4]:
        mutation_value = 0.00025
    else:
        mutation_value = 0.00000025

	# Perturb the value at the random index using a random Cauchy value with mutation_value delta.
    mutant.values[index] = mutant.values[index] + (random.gauss(0.0, mutation_value ))
    

    return mutant