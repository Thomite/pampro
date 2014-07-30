import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy



from pampro import Time_Series, Channel, channel_inference, Bout

execution_start = datetime.now()

ts = Time_Series.Time_Series("Actiheart")


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data\ARBOTW.txt')

chans = Channel.load_channels(filename, "Actiheart")
#ts.add_channels(chans)
activity = chans[0]
ecg = chans[1]


# Calculate moving averages of the channels
ecg_ma = ecg.moving_average(15)
activity_ma = activity.moving_average(15)
ts.add_channel(ecg_ma)
ts.add_channel(activity_ma)

blah = activity.time_derivative()
blah = blah.moving_average(121)
#ts.add_channel(blah)

# Infer sleep from Actiheart channels
awake_probability = channel_inference.infer_sleep_actiheart(activity, ecg)
ts.add_channel(awake_probability)



ts.draw_separate()

execution_end = datetime.now()
print("{} to {} = {}".format( execution_start, execution_end, execution_end - execution_start))