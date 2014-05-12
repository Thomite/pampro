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

chans2 = Channel.load_channels(os.path.join(os.path.dirname(__file__), '..', 'data/ah_data_10m.csv'), "CSV")
ts.add_channels(chans2)


ts_visualisation = Time_Series.Time_Series("Visualisation")
ts_visualisation.add_channel(ts.get_channel("ap_custom.csv - Pitch_mean"))


angle_levels = [
[-90,-85],[-85,-80],[-80,-75],[-75,-70],[-70,-65],[-65,-60],[-60,-55],[-55,-50],[-50,-45],[-45,-40],[-40,-35],[-35,-30],[-30,-25],[-25,-20],[-20,-15],[-15,-10],[-10,-05],[-05,0],
[0,05],[05,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60],[60,65],[65,70],[70,75],[75,80],[80,85],[85,90]]

chan_stats = ts.get_channel("ap_custom.csv - Pitch_mean").channel_statistics(statistics=["mean"] + angle_levels)
print(chan_stats)


angle_levels = [[0,05],[10,15],[20,25],[30,35],[40,45],[50,55],[60,65],[70,75],[80,85],[85,90]]

for combo in angle_levels:
	name = "ap_custom.csv - Pitch_"+str(combo[0])+"_"+str(combo[1])

	ts_visualisation.add_channel(ts.get_channel(name))



# Define a period of interest
start = datetime.strptime("12-Oct-2013 09:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=6)




draw_channels = ["ap_data_10m.csv - Vector magnitude_mean"]


ts_visualisation.draw_separate(time_period=[start,end])

