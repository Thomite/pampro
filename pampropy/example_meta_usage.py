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


chans = Channel.load_channels(os.path.join(os.path.dirname(__file__), '..', 'data/ap_10m_data.csv'), "CSV")
ts.add_channels(chans)

chans2 = Channel.load_channels(os.path.join(os.path.dirname(__file__), '..', 'data/ap_data.csv'), "A")
ts.add_channels(chans2)


bouts = ts.get_channel("ap_10m_data.csv - Vector magnitude_std").bouts(0,0.025,18)
bouts2 = ts.get_channel("ap_10m_data.csv - Pitch_mean").bouts(-10,10,18)
bouts3 = ts.get_channel("ap_10m_data.csv - Pitch_std").bouts(0,10,18)
#for bout in bouts:
#	print bout[0].day, bout[0], " -- ", bout[1]


# Turn the bouts into annotations and highlight those sections in the signals
annotations = Annotation.annotations_from_bouts(bouts)
annotations2 = Annotation.annotations_from_bouts(bouts2)
annotations3 = Annotation.annotations_from_bouts(bouts3)
#for chan in ts.channels:
	#chan.add_annotations(annotations)
ts.get_channel("ap_10m_data.csv - Vector magnitude_std").add_annotations(annotations)
ts.get_channel("ap_10m_data.csv - Pitch_mean").add_annotations(annotations2)
ts.get_channel("ap_10m_data.csv - Pitch_std").add_annotations(annotations3)

# Define a period of interest
start = datetime.strptime("12-Oct-2013 09:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=7)


# Define the appearance of the signals
ts.get_channel("ap_10m_data.csv - Vector magnitude_mean").draw_properties = {'lw':2, 'color':[0.196,0.196,0.78431]}
ts.get_channel("ap_10m_data.csv - Vector magnitude_std").draw_properties = {'lw':2, 'color':[0.196,0.196,0.78431]}
ts.get_channel("ap_10m_data.csv - Vector magnitude_max").draw_properties = {'lw':2, 'color':[0.196,0.196,0.78431]}

ts.get_channel("ap_10m_data.csv - Roll_mean").draw_properties = {'lw':2, 'color':[0.78431,0.196,0.196]}
ts.get_channel("ap_10m_data.csv - Roll_std").draw_properties = {'lw':2, 'color':[0.78431,0.196,0.196]}
ts.get_channel("ap_10m_data.csv - Roll_max").draw_properties = {'lw':2, 'color':[0.78431,0.196,0.196]}

ts.get_channel("ap_10m_data.csv - Pitch_mean").draw_properties = {'lw':2, 'color':[0.196,0.78431,0.196]}
ts.get_channel("ap_10m_data.csv - Pitch_std").draw_properties = {'lw':2, 'color':[0.196,0.78431,0.196]}
ts.get_channel("ap_10m_data.csv - Pitch_max").draw_properties = {'lw':2, 'color':[0.196,0.78431,0.196]}

draw_channels = ["ap_10m_data.csv - Vector magnitude_mean", "ap_10m_data.csv - Vector magnitude_std","ap_10m_data.csv - Vector magnitude_max", "ap_10m_data.csv - Pitch_mean", "ap_10m_data.csv - Pitch_std", "ap_10m_data.csv - Pitch_max", "ap_10m_data.csv - Roll_mean", "ap_10m_data.csv - Roll_std", "ap_10m_data.csv - Roll_max"]
#draw_channels = ["ap_10m_data.csv - Pitch_mean", "ap_data.csv - Pitch_mean", "ap_10m_data.csv - Pitch_std", "ap_data.csv - Pitch_std",]

ts.draw_separate(time_period=[start,end], channels=draw_channels)

