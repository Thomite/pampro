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

	print len(x.data)
	print len(x.timestamps)

	print len(y.data)
	print len(y.timestamps)

	print len(z.data)
	print len(z.timestamps)

	return result