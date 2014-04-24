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



ts = Time_Series.Time_Series()


chans = Channel.load_channels(os.path.join(os.path.dirname(__file__), '..', 'data/vm.csv'), "CSV")
ts.add_channels(chans)


bouts = ts.get_channel("vm.csv - std").bouts(0,0.025,3)
for bout in bouts:
	print bout[0].day, bout[0], " -- ", bout[1]


# Turn the bouts into annotations and highlight those sections in the signals
annotations = Annotation.annotations_from_bouts(bouts)
bouts = ts.get_channel("vm.csv - std").add_annotations(annotations)

# Define a period of interest
start = datetime.strptime("12-Oct-2013 09:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=7)


# Define the appearance of the signals

ts.draw_separate(time_period=[start,end], channels=["vm.csv - mean", "vm.csv - std", "vm.csv - min", "vm.csv - max"])

