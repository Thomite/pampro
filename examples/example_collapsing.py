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
activity, ecg = chans
ts.add_channel(ecg)

collape_values = [(40,49,1),(50,59,2),(60,69,3),(70,79,4),(80,89,5),(90,99,6),(100,109,7),(110,119,8),(120,129,9),(130,139,10),(140,999,11)]
ecg_collapsed = ecg.collapse(collape_values)
ts.add_channel(ecg_collapsed)

ecg_collapsed2 = ecg.collapse_auto(bins=25)
ts.add_channel(ecg_collapsed2)

ts.draw_separate()

execution_end = datetime.now()
print("{} to {} = {}".format( execution_start, execution_end, execution_end - execution_start))