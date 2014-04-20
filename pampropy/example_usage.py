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

dt = datetime.strptime("15-Mar-2014 18:03", "%d-%b-%Y %H:%M")
epoch_length = timedelta(minutes=5)

timestamp_list = []
for i in range(0,900):
	timestamp_list.append(dt)
	dt = dt + epoch_length

timestamps_c = np.array(timestamp_list)

channel_c = Channel.Channel("C")
channel_c.set_contents([420+random.random()*2+np.sin(x*0.025)*10 for x in range(0,900)], timestamps_c)


# Load sample Actiheart data

filename = os.path.join(os.path.dirname(__file__), '..', 'data\ARBOTW.txt')

chans = Channel.load_channels(filename, "Actiheart")
ts.add_channels(chans)
ts.add_channel(channel_c)

activity = ts.get_channel("Actiheart-Activity")

bouts = activity.bouts(100,99999)
#for bout in bouts:
	#diff = bout[1] - bout[0]
	#total = total + diff
	#print bout[0], " ~ ", bout[1], ": ", diff


#subset_channel = ecg.subset_using_bouts(bouts, "ECG when activity > 100")
#ts.add_channel(subset_channel)

stats = ["mean", "sum", "std", "min", "max", "n", [0,499],[500,5000]]
#activity.piecewise_statistics( timedelta(minutes=15), statistics=stats, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/activity_test.csv') )

# We can also return the data by not supplying a file_target
#dataset = activity.piecewise_statistics( timedelta(days=1), statistics=stats )
#for row in dataset:
	#print row


start = datetime.strptime("16-Mar-2014 00:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=1)

a1 = Annotation.Annotation("whatever", start + timedelta(hours=2, minutes=42), start + timedelta(hours=10, minutes=40))
a1.draw_properties = {'alpha':0.1, 'lw':0, 'facecolor':[0.196,0.196,0.196]}
a2 = Annotation.Annotation("whatever", start + timedelta(hours=16, minutes=0), start + timedelta(hours=17, minutes=30))
a2.draw_properties = {'alpha':0.5, 'lw':0, 'facecolor':[0.196,0.78431,0.196]}

ts.get_channel("Actiheart-ECG").add_annotations([a1,a2])
ts.get_channel("Actiheart-Activity").add_annotation(a1)

ts.get_channel("Actiheart-ECG").draw_properties = {'alpha':0.9, 'color':[0.78431,0.196,0.196]}

ts.draw_separate()

