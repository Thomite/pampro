#from pampropy import Channel
from datetime import timedelta
import Channel
import Bout
import numpy as np


def produce_binary_channels(bouts, lengths, skeleton_channel):
	
	Bout.cache_lengths(bouts)
	bouts.sort(key=lambda x: x.length, reverse=True)

	channels = []
	for length in lengths:

		# Drop bouts from list if their length is less than x minutes
		bouts = Bout.limit_to_lengths(bouts, min_length=length, sorted=True)

		channel_name = "{}_mt{}".format(skeleton_channel.name,length)

		# Clone the blank channel, set data to 1 where time is inside any of the bouts
		skeleton_copy = copy.deepcopy(skeleton_channel)
		chan = Channel.channel_from_bouts(bouts, False, False, channel_name, skeleton=skeleton_copy)
		channels.append(chan)

	return channels

def infer_sleep_actiheart(actiheart_activity, actiheart_ecg):

	ecg_ma = actiheart_ecg.moving_average(30)
	activity_ma = actiheart_activity.moving_average(30)

	ecg_norm = ecg_ma.clone()
	ecg_norm.normalise()
	activity_norm = activity_ma.clone()
	activity_norm.normalise()

	product = Channel.Channel("Probability of being awake")
	product.set_contents( np.multiply(ecg_norm.data, activity_norm.data), ecg_norm.timestamps)
	product.moving_average(30)

	#print(product.data)
	#print(activity_norm.data)
	#print(ecg_norm.data)
	return product

def infer_vector_magnitude(x,y,z):

	result = Channel.Channel("VM")

	result.set_contents( np.sqrt( np.multiply(x.data,x.data) + np.multiply(y.data,y.data) + np.multiply(z.data,z.data) ), x.timestamps )

	return result

def infer_pitch_roll(x,y,z):

	pitch = Channel.Channel("Pitch")
	roll = Channel.Channel("Roll")

	pitch_degrees = np.arctan(x.data/np.sqrt((y.data*y.data) + (z.data*z.data))) * 180.0/np.pi * -1.0
	roll_degrees = np.arctan(y.data/np.sqrt((x.data*x.data) + (z.data*z.data))) * 180.0/np.pi * -1.0

	pitch.set_contents( pitch_degrees, x.timestamps)
	roll.set_contents( roll_degrees, x.timestamps)

	return [pitch, roll]

def infer_enmo(vm):

	result = Channel.Channel("ENMO")

	result.set_contents( (vm.data - 1.0)*1000.0 , vm.timestamps )


	result.data[np.where(result.data < 0)] = 0

	return result

def infer_nonwear_actigraph(counts, zero_minutes=timedelta(minutes=60)):

	#nonwear = Channel.Channel("Nonwear")

	nonwear_bouts = counts.bouts(0, 0, zero_minutes)
	wear_bouts = Bout.time_period_minus_bouts([counts.timeframe[0], counts.timeframe[1]], nonwear_bouts)
	
	wear = counts.subset_using_bouts(wear_bouts, "Wear_only", substitute_value=-1)
	#wear.delete_windows(nonwear_bouts)

	#print(len(wear.data))
	bad_indices = np.where(wear.data == -1)
	#print(len(bad_indices[0]))
	wear.data = np.delete(wear.data, bad_indices[0], None)
	wear.timestamps = np.delete(wear.timestamps, bad_indices[0], None)
	wear.calculate_timeframe()

	wear_with_missings = counts.subset_using_bouts(wear_bouts, "Wear_minuses", substitute_value=-1)


	return [wear, wear_with_missings, wear_bouts, nonwear_bouts]

def infer_nonwear_triaxial(x,y,z):
	
	''' Use the 3 channels of triaxial acceleration to infer periods of nonwear '''
	x_std = x.moving_std(30)
	y_std = y.moving_std(30)
	z_std = z.moving_std(30)

	# Find bouts where monitor was still for long periods
	x_bouts = x_std.bouts(0, 0.005, timedelta(minutes=15))
	y_bouts = y_std.bouts(0, 0.005, timedelta(minutes=15))
	z_bouts = z_std.bouts(0, 0.005, timedelta(minutes=15))
	
	# Get the times where those bouts overlap
	x_intersect_y = Bout.bout_list_intersection(x_bouts, y_bouts)
	x_intersect_y_intersect_z = Bout.bout_list_intersection(x_intersect_y, z_bouts)
	
	# Create a parallel, binary channel indicating if that time point was in or out of wear
	nonwear_binary = Channel.channel_from_bouts(x_intersect_y_intersect_z, [x.timeframe[0], x.timeframe[1]], approx_epoch, "nonwear")

	# Invert the nonwear bouts, clipping to the time region of the original channels, to get wear bouts
	wear_bouts = Bout.time_period_minus_bouts([x.timeframe[0],x.timeframe[1]], x_intersect_y_intersect_z)

	# Maybe creating a whole new copy without these bouts isn't a great idea - memory footprint
	wear_only_x = x.subset_using_bouts(wear_bouts, "x_wear_only", substitute_value=0)
	wear_only_y = y.subset_using_bouts(wear_bouts, "y_wear_only", substitute_value=0)
	wear_only_z = z.subset_using_bouts(wear_bouts, "z_wear_only", substitute_value=0)

def infer_valid_days(channel, wear_bouts, epoch_length=False, valid_criterion=timedelta(hours=10)):

	#Generate 7 day-long windows
	start = channel.timeframe[0] - timedelta(hours=channel.timeframe[0].hour, minutes=channel.timeframe[0].minute, seconds=channel.timeframe[0].second, microseconds=channel.timeframe[0].microsecond)
	day_windows = []
	while start < channel.timeframe[1]:
		day_windows.append(Bout.Bout(start, start+timedelta(days=1)))
		start += timedelta(days=1)

	valid_windows = []
	for window in day_windows:
		#how much does all of wear_bouts intersect with window?
		intersections = Bout.bout_list_intersection([window],wear_bouts)
		total = Bout.total_time(intersections)
		if total > valid_criterion:
			#window.draw_properties={"lw":0, "facecolor":[1,0,0], "alpha":0.25}
			valid_windows.append(window)
		#print total


	
	valid_zeroes = channel.subset_using_bouts(valid_windows, channel.name + "_valid_zeroes", substitute_value=0)
	valid_missings = channel.subset_using_bouts(valid_windows, channel.name + "_valid_missings", substitute_value=-1)
	# Create a binary channel
	if epoch_length == False:
		epoch_length = channel.timestamps[1] - channel.timestamps[0] 
	
	valid_binary = Channel.channel_from_bouts(valid_windows, [channel.timeframe[0], channel.timeframe[1]], epoch_length, "valid")
	


	return [valid_zeroes, valid_missings, valid_binary, valid_windows]





