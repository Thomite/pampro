import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random

import Time_Series
import Channel

ts = Time_Series.Time_Series()


# Create channel A

dt = datetime.strptime("15-Mar-2014 16:32", "%d-%b-%Y %H:%M")
epoch_length = timedelta(minutes=1)

timestamp_list = []
for i in range(0,1000):
	timestamp_list.append(dt)
	dt = dt + epoch_length

timestamps_a = np.array(timestamp_list)

channel_a = Channel.Channel("A")
channel_a.set_contents([350+random.random()*2+np.cos(x*0.02)*30 for x in range(0,1000)], timestamps_a)

# Create channel B

dt = datetime.strptime("15-Mar-2014 21:00", "%d-%b-%Y %H:%M")
epoch_length = timedelta(seconds=30)

timestamp_list = []
for i in range(0,3000):
	timestamp_list.append(dt)
	dt = dt + epoch_length

timestamps_b = np.array(timestamp_list)

channel_b = Channel.Channel("B")
channel_b.set_contents([500+random.random()+np.sin(x*0.025)*10 for x in range(0,3000)], timestamps_b)

# Create channel C

dt = datetime.strptime("15-Mar-2014 18:03", "%d-%b-%Y %H:%M")
epoch_length = timedelta(minutes=2, seconds=30)

timestamp_list = []
for i in range(0,900):
	timestamp_list.append(dt)
	dt = dt + epoch_length

timestamps_c = np.array(timestamp_list)

channel_c = Channel.Channel("C")
channel_c.set_contents([420+random.random()*2+np.sin(x*0.025)*10 for x in range(0,900)], timestamps_c)

#ts.add_channel(channel_a)
#ts.add_channel(channel_b)
#ts.add_channel(channel_c)





here = os.path.dirname(__file__)
filename = os.path.join(here, '..', 'data\ARBOTW.txt')

chans = Channel.load_channels(filename, "Actiheart")
ts.add_channels(chans)


activity = chans[0]
ecg = chans[1]

bouts = activity.bouts(100,99999)
#for bout in bouts:
	#diff = bout[1] - bout[0]
	#total = total + diff
	#print bout[0], " ~ ", bout[1], ": ", diff


#subset_channel = ecg.subset_using_bouts(bouts, "ECG when activity > 100")

#ts.add_channel(subset_channel)
stats = ["mean", "sum", "std", "min", "max", "n"]

dataset = activity.piecewise_statistics( timedelta(days=1, hours=3), statistics=stats )
print stats
for row in dataset:
	print row

ts.draw_separate()
