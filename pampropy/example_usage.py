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

ts = Time_Series.Time_Series()


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

# Create a new Time Series for output, add some hourly channels to it and write that to a file
ts_output = Time_Series.Time_Series()
test = activity.piecewise_statistics(timedelta(hours=1), statistics=["mean", "max"])
for t in test:
	ts_output.add_channel(t)
	
test2 = ecg.piecewise_statistics(timedelta(hours=1), statistics=["mean", "max"])
for t in test2:
	ts_output.add_channel(t)

ts_output.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ah_hourly.csv'))


# Define a period of interest
start = datetime.strptime("17-Mar-2014 00:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=3)


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
ts.draw_separate()#, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/blah.png'))


