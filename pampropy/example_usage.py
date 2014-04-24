import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy

import Time_Series
import Channel
import Annotation
import channel_inference

#from pampropy import Time_Series, Channel, Annotation, channel_inference
execution_start = datetime.now()

ts = Time_Series.Time_Series()


# Create artificial channel 

dt = datetime.strptime("11-Mar-2014 18:00", "%d-%b-%Y %H:%M")
epoch_length = timedelta(minutes=1)

length = 1400
timestamp_list = []
for i in range(0,length):
	timestamp_list.append(dt)
	dt = dt + epoch_length

timestamps_c = np.array(timestamp_list)
channel_c = Channel.Channel("C")
channel_c.set_contents(np.array([120+random.random()*2+np.sin(x*0.025)*10 for x in range(0,length)]), timestamps_c)


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data\ARBOTW.txt')
chans = Channel.load_channels(filename, "Actiheart")
activity = chans[0]
ecg = chans[1]


# Calculate moving averages of the channels
ecg_ma = ecg.moving_average(15)
activity_ma = activity.moving_average(15)
ts.add_channel(ecg_ma)
ts.add_channel(activity_ma)

blah = activity.time_derivative()
blah = blah.moving_average(121)
ts.add_channel(blah)

# Infer sleep from Actiheart channels
awake_probability = channel_inference.infer_sleep_actiheart(activity, ecg)
ts.add_channel(awake_probability)


# Output channel summary statistics
#chan_stat = activity.channel_statistics(statistics=["mean","sum","n"],file_target= os.path.join(os.path.dirname(__file__), '..', 'data/blah.txt'))
#for x in chan_stat:#
	#print x

# Define a period of interest
start = datetime.strptime("17-Mar-2014 00:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=3)

for i in range(1000):
	print i
	# Save some stats about the time series to a file
	stats = ["mean", "sum", "std", "min", "max", "n"]
	ts.piecewise_statistics( timedelta(minutes=10), statistics=stats, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ah_10m_') )

# Infer vector magnitude from three channels
#vm = channel_inference.infer_vector_magnitude(awake_probability, ecg, activity)


# Get a list of bouts where awake probability was >= 0 and <= 0.001 for 240 epochs or more
bouts = awake_probability.bouts(0,0.001,240)
#for bout in bouts:
#	print bout[0].day, bout[0], " -- ", bout[1]


# Subset a channel where the bouts occur
#subset_channel = ecg.subset_using_bouts(bouts, "Low activity")
#ts.add_channel(subset_channel)

# Turn the bouts into annotations and highlight those sections in the signals
annotations = Annotation.annotations_from_bouts(bouts)
ecg.add_annotations(annotations)
activity.add_annotations(annotations)
ecg_ma.add_annotations(annotations)
activity_ma.add_annotations(annotations)
awake_probability.add_annotations(annotations)
blah.add_annotations(annotations)

# Define the appearance of the signals
ecg_ma.draw_properties = {'alpha':1, 'lw':2, 'color':[0.78431,0.196,0.196]}
activity_ma.draw_properties = {'alpha':1, 'lw':2, 'color':[0.196,0.196,0.78431]}
awake_probability.draw_properties = {'alpha':1, 'lw':2, 'color':[0.78431,0.196,0.78431]}
#ts.draw_separate(time_period=[start,end])#, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/blah.png'))


# Save some stats about the time series to a file
stats = ["mean", "sum", "std", "min", "max", "n", [0,499],[500,5000]]
#ts.piecewise_statistics( timedelta(days=1), statistics=stats, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ts_') )

# We can also return the data by not supplying a file_target
#dataset = activity.piecewise_statistics( timedelta(days=1), statistics=stats )
#for row in dataset:
	#print row

execution_end = datetime.now()
print("{} to {} = {}".format( execution_start, execution_end, execution_end - execution_start))