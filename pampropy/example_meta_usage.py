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



ts = Time_Series.Time_Series("AP")


chans = Channel.load_channels(os.path.join(os.path.dirname(__file__), '..', 'data/ap_data_10m.csv'), "CSV")
ts.add_channels(chans)


stat_dict = {"ap_data_10m.csv - Vector magnitude_std":["mean", "std", [0,0.025]]}
blah = ts.piecewise_statistics( timedelta(minutes=60), stat_dict)
print(len(blah[0].name))

ts_visualisation = Time_Series.Time_Series("Visualisation")

ts_visualisation.add_channel(ts.get_channel("ap_data_10m.csv - Vector magnitude_std"))
ts_visualisation.add_channels(blah)

#bouts = ts.get_channel("ap_data_10m.csv - Vector magnitude_std").bouts(0,0.025,18)
#bouts2 = ts.get_channel("ap_data_10m.csv - Pitch_mean").bouts(-10,10,18)
#bouts3 = ts.get_channel("ap_data_10m.csv - Pitch_std").bouts(0,10,18)
#for bout in bouts:
#	print bout[0].day, bout[0], " -- ", bout[1]


# Turn the bouts into annotations and highlight those sections in the signals
#annotations = Annotation.annotations_from_bouts(bouts)
#annotations2 = Annotation.annotations_from_bouts(bouts2)
#annotations3 = Annotation.annotations_from_bouts(bouts3)
#for chan in ts.channels:
	#chan.add_annotations(annotations)
#ts.get_channel("ap_data_10m.csv - Vector magnitude_std").add_annotations(annotations)
#ts.get_channel("ap_data_10m.csv - Pitch_mean").add_annotations(annotations2)
#ts.get_channel("ap_data_10m.csv - Pitch_std").add_annotations(annotations3)

# Define a period of interest
start = datetime.strptime("12-Oct-2013 09:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=6)


# Define the appearance of the signals



ts_visualisation.draw_separate(time_period=[start,end])

