import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy
import math


from pampro import Time_Series, Channel, channel_inference, Bout, time_utilities

execution_start = datetime.now()

ts = Time_Series.Time_Series("Actiheart")


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data\example_actiheart.txt')

chans = Channel.load_channels(filename, "Actiheart")
activity, ecg = chans
ts.add_channels(chans)

# Summarise at hourly intervals
tp = (time_utilities.start_of_hour(activity.timestamps[0]),time_utilities.end_of_hour(activity.timestamps[-1]))
ecg_hourly = ecg.piecewise_statistics(timedelta(hours=1), time_period=tp)
activity_hourly = activity.piecewise_statistics(timedelta(hours=1), time_period=tp)



#ts.add_channels([ecg_hourly[0], activity_hourly[0]])
#ts.draw_separate()

## -- -- -- -- --

# Collapse the channels to simpler forms
ecg_collapsed = ecg_hourly[0].collapse_auto(bins=5)
activity_collapsed = activity_hourly[0].collapse_auto(bins=30)

# Get the transitions from one level to each other level
act_bigrams = activity_collapsed.bigrams()
ecg_bigrams = ecg_collapsed.bigrams()


# Function to take bigram transitions and make a simple visualisation
def ngrams_axis(ax, bigrams):

	unique_values = bigrams.keys()

	connection_info = {}
	values = []

	for val1 in unique_values:

		max_leaving = max(bigrams[val1].values())
		connection_info[val1] = max_leaving

		for val2 in unique_values:
			values.append( float(bigrams[val1][val2]) ) 

	values = np.array(values)

	# --------------------------------

	for val1 in unique_values:
	    for val2 in unique_values:
	    	if val1 != val2:

				strongest_from_val1 = connection_info[val1]
				if strongest_from_val1 > 0:
					connection_strength = float(bigrams[val1][val2])/ connection_info[val1]
				else:
					connection_strength = 0
				ax.plot([0, 1], [val1, val2], c="black", lw=2, alpha=0.025+min(0.975,connection_strength) )

	ax.tick_params(labelleft=True, labelright=True,labeltop=True, labelbottom=True)
	ax.set_yticks(unique_values)
	ax.xaxis.set_ticks([0,0.5,1])
	ax.xaxis.set_ticklabels(["Start","-->","Finish"])
	ax.set_xlim(0,1)
	ax.set_ylim(min(unique_values)-1, max(unique_values)+1)
	return ax

def ngrams_axis_graph(ax, bigrams):

	unique_values = bigrams.keys()

	connection_info = {}
	values = []

	for val1 in unique_values:

		max_leaving = max(bigrams[val1].values())
		connection_info[val1] = max_leaving

		for val2 in unique_values:
			values.append( float(bigrams[val1][val2]) ) 

	values = np.array(values)

	# --------------------------------

	r = 1.0

	positions = {}

	for index, val1 in enumerate(unique_values):

		t = math.pi * 2.0 * (float(index) / float(len(unique_values)))
		x = r*math.cos(t)
		y = r*math.sin(t)

		positions[val1] = (x,y)

		ax.text(1.1*math.cos(t), 1.1*math.sin(t), val1)

	for val1 in unique_values:
	    for val2 in unique_values:
	    	if val1 != val2:

				strongest_from_val1 = connection_info[val1]
				if strongest_from_val1 > 0:
					connection_strength = float(bigrams[val1][val2])/ connection_info[val1]
				else:
					connection_strength = 0

				val1_pos = positions[val1]
				val2_pos = positions[val2]
				ax.plot([val1_pos[0], val2_pos[0]], [val1_pos[1], val2_pos[1]], c="black", lw=2, alpha=0.025+min(0.975,connection_strength) )



	for val1 in unique_values:

		x,y = positions[val1]
		ax.scatter([x],[y], s=100, c=[1,1,1])

	ax.set_xlim(-1.2,1.2)
	ax.set_ylim(-1.2,1.2)
	ax.set_xticks([])
	ax.set_xticklabels([])
	ax.set_yticks([])
	ax.set_yticklabels([])

	# is this the correct height?
	# last minute decision
	ax.set_aspect("equal")

	return ax



# Visualise transitions of activity channel and ECG channel bigrams
fig, (ax1, ax2) = plt.subplots(1,2)

ngrams_axis(ax1, ecg_bigrams)
ngrams_axis_graph(ax2, ecg_bigrams)

#ax1.set_title("ECG")
#ax2.set_title("Activity")
fig.tight_layout()

plt.show()

