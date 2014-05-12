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



ts = Time_Series.Time_Series("Axivity")


chans = Channel.load_channels('V:/Projects/Fitness/longitudinal_data.cwa', "Axivity")
ts.add_channels(chans)

vm = channel_inference.infer_vector_magnitude(chans[0], chans[1], chans[2])
ts.add_channel(vm)

chans = channel_inference.infer_pitch_roll(chans[0], chans[1], chans[2])
ts.add_channels(chans)

print ts.earliest
print ts.latest
print ts.latest - ts.earliest


# Define the appearance of the signals
tom_red = [0.78431,0.196,0.196]
tom_green = [0.196,0.78431,0.196]
tom_blue = [0.196,0.196,0.78431]


ts.draw_separate()

