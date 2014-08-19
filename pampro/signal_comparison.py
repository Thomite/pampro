from scipy import stats
from scipy.optimize import curve_fit
from sklearn import linear_model

def func(x, a, b, c, d):

	return a + x*b + x**2*c + x**3*d

def channel_linear_regression(channel1, channel2):

	return stats.linregress(channel1.data,channel2.data)

def channel_polynomial_regression(channel1, channel2):

	return curve_fit(func, channel1.data, channel2.data)


def channel_multivariate_regression(channel1, channels):

	yvar = channel1.data
	xvars = [x.data for x in channels]
    
	clf = linear_model.LinearRegression(fit_intercept=True)
	clf.fit(zip(*xvars), yvar)

	return (clf.intercept_, clf.coef_)