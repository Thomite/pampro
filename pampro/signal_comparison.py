from scipy import stats
from scipy.optimize import curve_fit


def func(x, a, b, c, d):

	return a + x*b + x**2*c + x**3*d

def channel_linear_regression(channel1, channel2):

	return stats.linregress(channel1.data,channel2.data)

def channel_polynomial_regression(channel1, channel2):

	return curve_fit(func, channel1.data, channel2.data)