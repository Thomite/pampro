#from pampropy import Channel
import Channel
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

	result = Channel.Channel("Vector magnitude")

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