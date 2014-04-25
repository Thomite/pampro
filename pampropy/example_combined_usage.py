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


awake_probability = Channel.Channel("Awake")

ap_pitch_std_clone = ts.get_channel("ap_data_10m.csv - Pitch_std").moving_average(6)
ap_pitch_std_clone.normalise()

ap_vm_std_clone = ts.get_channel("ap_data_10m.csv - Vector magnitude_std").moving_average(18)
ap_vm_std_clone.normalise()

ah_mean_clone = ts.get_channel("ah_data_10m.csv - AH_Activity_mean").clone()
ah_mean_clone.normalise()

ah_std_clone = ts.get_channel("ah_data_10m.csv - AH_Activity_std").clone()
ah_std_clone.normalise()

ah_ecg_clone = ts.get_channel("ah_data_10m.csv - AH-ECG_mean").clone()
ah_ecg_clone.normalise()




#print(ap_vm_clone.size)
#print(ah_std_clone.size)
awake_probability.set_contents(ah_ecg_clone.data[0:1000] * ah_mean_clone.data[0:1000], ah_std_clone.timestamps[0:1000])

pitch_bouts = ap_pitch_std_clone.bouts(0,10,6)
for b in pitch_bouts:
	awake_probability.data[b[2]:b[3]] = awake_probability.data[b[2]:b[3]] * 0.5

awake_probability.normalise()

ts.add_channel(awake_probability)

bouts = awake_probability.bouts(0,0.02,18)

#bouts = ts.get_channel("ap_data_10m.csv - Vector magnitude_std").bouts(0,0.025,18)

#bouts2 = ts.get_channel("ap_data_10m.csv - Pitch_mean").bouts(-10,10,18)
#bouts3 = ts.get_channel("ap_data_10m.csv - Pitch_std").bouts(0,10,18)
#for bout in bouts:
#	print bout[0].day, bout[0], " -- ", bout[1]





# Turn the bouts into annotations and highlight those sections in the signals
annotations = Annotation.annotations_from_bouts(bouts)
#annotations2 = Annotation.annotations_from_bouts(bouts2)
#annotations3 = Annotation.annotations_from_bouts(bouts3)
for chan in ts.channels:
	chan.add_annotations(annotations)
#ts.get_channel("ap_10m_data.csv - Vector magnitude_std").add_annotations(annotations)
#ts.get_channel("ap_10m_data.csv - Pitch_mean").add_annotations(annotations2)
#ts.get_channel("ap_10m_data.csv - Pitch_std").add_annotations(annotations3)

# Define a period of interest
start = datetime.strptime("12-Oct-2013 09:00", "%d-%b-%Y %H:%M")
end = start + timedelta(days=6)


# Define the appearance of the signals
tom_red = [0.78431,0.196,0.196]
tom_green = [0.196,0.78431,0.196]
tom_blue = [0.196,0.196,0.78431]

ts.get_channel("ap_data_10m.csv - Vector magnitude_max").draw_properties = {'lw':2, 'color':tom_blue}
ts.get_channel("ap_data_10m.csv - Vector magnitude_std").draw_properties = {'lw':2, 'color':tom_blue}

ts.get_channel("ah_data_10m.csv - AH_Activity_mean").draw_properties = {'lw':2, 'color':tom_green}
ts.get_channel("ah_data_10m.csv - AH_Activity_std").draw_properties = {'lw':2, 'color':tom_green}

ts.get_channel("ah_data_10m.csv - AH-ECG_mean").draw_properties = {'lw':2, 'color':tom_red}
ts.get_channel("ah_data_10m.csv - AH-ECG_max").draw_properties = {'lw':2, 'color':tom_red}

ts.get_channel("Awake").draw_properties = {'lw':2, 'color':[0.78431,0.196,0.78431]}

draw_channels = ["ap_data_10m.csv - Pitch_std", "ap_data_10m.csv - Vector magnitude_std", "ah_data_10m.csv - AH_Activity_mean", "ah_data_10m.csv - AH_Activity_std", "Awake"]

ts.draw_separate(time_period=[start,end], channels=draw_channels)

