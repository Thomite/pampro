
from datetime import timedelta
from pampro import Channel, Bout, signal_comparison
import numpy as np
import copy

def axis_scatter_a_vs_b(ax, channel1, channel2):

	ax.scatter(channel1.data, channel2.data, lw=0, alpha=0.25)

	min_x, max_x = min(channel1.data),max(channel1.data)

	ax.set_xlim(min_x, max_x)
	ax.set_xlabel(channel1.name)
	ax.set_ylabel(channel2.name)



def axis_linear_regression_a_vs_b(ax, channel1, channel2, linear_parameters=False):

	min_x, max_x = min(channel1.data), max(channel1.data)

	if linear_parameters == False:
		slope, intercept, r_value, p_value, std_err = signal_comparison.channel_linear_regression(channel1, channel2)
	else:
		slope, intercept, r_value, p_value, std_err = linear_parameters
	
	ax.plot([min_x, max_x], [intercept+slope*min_x - std_err*min_x,intercept + slope*max_x - std_err*min_x], c=[0.5,0.5,0.5])
	ax.plot([min_x, max_x], [intercept+slope*min_x + std_err*min_x,intercept + slope*max_x + std_err*min_x], c=[0.5,0.5,0.5])
	ax.plot([min_x, max_x], [intercept+slope*min_x,intercept+slope*max_x], c=[1,0,0], lw=2, alpha=0.5)


def axis_polynomial_regression_a_vs_b(ax, channel1, channel2, polynomial_parameters=False):
	# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

	if polynomial_parameters == False:
		popt, pcov = signal_comparison.channel_polynomial_regression(channel1, channel2)
	else:
		popt, pcov = polynomial_parameters

	perr = np.sqrt(np.diag(pcov))

	sorted_x = np.sort(channel1.data)

	ax.plot(sorted_x, [signal_comparison.func(x, popt[0], popt[1], popt[2], popt[3]) for x in sorted_x], c=[1,0,0], lw=2, alpha=0.5)
	ax.plot(sorted_x, [signal_comparison.func(x, popt[0]+perr[0], popt[1]+perr[1], popt[2]+perr[2], popt[3]+perr[3]) for x in sorted_x], c=[0.5,0.5,0.5])
	ax.plot(sorted_x, [signal_comparison.func(x, popt[0]-perr[0], popt[1]-perr[1], popt[2]-perr[2], popt[3]-perr[3]) for x in sorted_x], c=[0.5,0.5,0.5])
