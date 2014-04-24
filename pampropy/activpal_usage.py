import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy
import time

import Time_Series
import Channel
import Annotation
import channel_inference

#from pampropy import Time_Series, Channel, Annotation, channel_inference

start_time = datetime.now()

ts = Time_Series.Time_Series()




# Load sample activPAL data
#filename = os.path.join(os.path.dirname(__file__), '..', 'data\ARBOTW.txt')
chans = Channel.load_channels("Q:/Data/CSVs for Tom/Parallel files/514538L-AP1336594 12Oct13 10-00am for 8d 0m.csv", "activPAL")
x,y,z = chans[0],chans[1],chans[2]
print("Finished reading in data")

# Calculate moving averages of the channels
#ecg_ma = ecg.moving_average(15)


# Infer vector magnitude from three channels
vm = channel_inference.infer_vector_magnitude(x,y,z)
#pitch, roll = channel_inference.infer_pitch_roll(x,y,z)
ts.add_channel(vm)
print("Inferred VM, pitch and roll")

# Turn the bouts into annotations and highlight those sections in the signals
#annotations = Annotation.annotations_from_bouts(bouts)
#ecg.add_annotations(annotations)

# Save some stats about the time series to a file
stats = ["mean", "sum", "std", "min", "max", "n"]
ts.piecewise_statistics( timedelta(hours=1), statistics=stats, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ap_daily_') )

end_time = start_time = datetime.now()
duration = end_time - start_time
print(duration)

# Define the appearance of the signals
#ecg_ma.draw_properties = {'alpha':1, 'lw':2, 'color':[0.78431,0.196,0.196]}
#activity_ma.draw_properties = {'alpha':1, 'lw':2, 'color':[0.196,0.196,0.78431]}
#awake_probability.draw_properties = {'alpha':1, 'lw':2, 'color':[0.78431,0.196,0.78431]}
ts.draw_separate(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/blah.png'))




# We can also return the data by not supplying a file_target
#dataset = activity.piecewise_statistics( timedelta(days=1), statistics=stats )
#for row in dataset:
	#print row