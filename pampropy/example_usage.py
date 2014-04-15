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


chans = Channel.load_channels("V:/Projects/pampropy/data/ARBOTW.txt", "Actiheart")

for chan in chans:
	chan.normalise(0,100)
	ts.add_channel(chan)


activity = chans[0]


window = timedelta(hours=1)
start = activity.timeframe[0]
cutoff1 = start - timedelta(hours=start.time().hour, minutes=start.time().minute, seconds=start.time().second, microseconds=start.time().microsecond)
cutoff2 = cutoff1 + window

val = 0

while(cutoff1 < activity.timeframe[1]):

	print("Getting indices between {} and {}".format(cutoff1, cutoff2))
	i1 = chan.get_window(cutoff1, cutoff2)

	if (len(i1) > 0):
		mean_val1 = stats.describe(chan.data[i1])
		print(mean_val1)

	cutoff1 = cutoff1 + window
	cutoff2 = cutoff2 + window
	
	val = val+1
	#ts.draw_separate()

print(val)