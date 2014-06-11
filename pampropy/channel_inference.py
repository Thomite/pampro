#from pampropy import Channel
from datetime import timedelta
import Channel
import Bout
import numpy as np

def infer_sleep_actiheart(actiheart_activity, actiheart_ecg):

	ecg_ma = actiheart_ecg.moving_average(15)
	activity_ma = actiheart_activity.moving_average(15)

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
	
	wear = counts.subset_using_bouts(wear_bouts, "Wear_only")


	return [wear, wear_bouts, nonwear_bouts]

def infer_valid_days_only(channel, wear_bouts, valid_criterion=timedelta(hours=10)):

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

	valid_only = channel.subset_using_bouts(valid_windows, channel.name + "_valid_only")
	return [valid_only, valid_windows]





