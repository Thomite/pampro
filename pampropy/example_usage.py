import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random

import Time_Series
import Channel
import Annotation


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
ts.add_channels(chans)
ts.add_channel(channel_c)

activity = ts.get_channel("Actiheart-Activity")
ecg = ts.get_channel("Actiheart-ECG")


# Get a list of bouts where activity was >= 0 and <= 30 for 240 epochs or more
bouts = activity.bouts(0,30,240)

# Turn the bouts into annotations and highlight those sections in the signals
annotations = Annotation.annotations_from_bouts(bouts)
ecg.add_annotations(annotations)
activity.add_annotations(annotations)

#subset_channel = ecg.subset_using_bouts(bouts, "Low activity")
#ts.add_channel(subset_channel)

# Save some stats about a channel to a file
stats = ["mean", "sum", "std", "min", "max", "n", [0,499],[500,5000]]
#activity.piecewise_statistics( timedelta(minutes=15), statistics=stats, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/activity_test.csv') )

# We can also return the data by not supplying a file_target
#dataset = activity.piecewise_statistics( timedelta(days=1), statistics=stats )
#for row in dataset:
	#print row

# Define a period of interest
start = datetime.strptime("17-Mar-2014 00:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=2)

ecg.draw_properties = {'alpha':1, 'color':[0.78431,0.196,0.196]}
activity.draw_properties = {'alpha':1, 'color':[0.196,0.196,0.78431]}
#ts.draw_separate()#time_period=[start,end])

#activity.piecewise_statistics(timedelta(hours=2), statistics=stats,  file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ts_act'))
#channel_c.piecewise_statistics(timedelta(hours=2), statistics=stats,  file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ts_2'))

ts.piecewise_statistics( timedelta(hours=2), statistics=stats, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ts_') )