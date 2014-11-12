from datetime import datetime, date, time, timedelta

def get_still_bouts(x,y,z):


	x_sa = x.sliding_statistics(timedelta(minutes=1), statistics=["mean"], time_period=False)
	y_sa = y.sliding_statistics(timedelta(minutes=1), statistics=["mean"], time_period=False)
	z_sa = z.sliding_statistics(timedelta(minutes=1), statistics=["mean"], time_period=False)
	
	return (x_sa, y_sa, z_sa)


def calibrate_signal(x,y,z, x_offset, y_offset, z_offset):

	''' Given the 3 triaxial axes, add the corresponding parameters to them #'''

	new_x = copy.deepcopy(x)
	new_x.data += x_offset

	new_y = copy.deepcopy(y)
	new_y.data += y_offset

	new_z = copy.deepcopy(z)
	new_z.data += z_offset

	return [new_x, new_y, new_z]


def score_calibration(x,y,z):

	enmo = channel_inference.infer_enmo(x,y,z)
	score = math.sqrt(np.sum ( np.square(enmo) ) )
	return score

def do_calibration(channels, x_offset, y_offset, z_offset):

	x,y,z = channels

	return score_calibration(calibrate_signal(x,y,z, x_offset, y_offset, z_offset))

def autocalibrate_signal(x,y,z):

	channels = (x,y,z)

	popt, pcov = curve_fit(do_calibration, channels, np.zeroes(len(x.data)))
	