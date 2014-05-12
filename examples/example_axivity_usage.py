import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy
import time

#import Time_Series
#import Channel
#import Annotation
#import channel_inference

from pampropy import Time_Series, Channel, Annotation, channel_inference



ts = Time_Series.Time_Series("Axivity")


chans = Channel.load_channels(os.path.join(os.path.dirname(__file__), '..', 'data/nomovement.cwa'), "Axivity")
ts.add_channels(chans)

vm = channel_inference.infer_vector_magnitude(chans[0], chans[1], chans[2])
ts.add_channel(vm)

chans = channel_inference.infer_pitch_roll(chans[0], chans[1], chans[2])
ts.add_channels(chans)

stats = {"X":["mean", "std"],"Y":["mean", "std"],"Z":["mean", "std"],"Vector magnitude":["mean", "std", "n"]}#,"Pitch":["mean", "std"],"Roll":["mean", "std"]}
chans = ts.summary_statistics(stats)

print len(chans)

for c in chans:
	print c

ts_output = Time_Series.Time_Series("Output")
ts_output = Time_Series.Time_Series("Visualisation")
simplified = ts.piecewise_statistics(timedelta(minutes=1), stats)
ts_output.add_channels(simplified)
ts_output.write_channels_to_file(os.path.join(os.path.dirname(__file__), '..', 'data/axivity_summary.csv'))

print(np.mean(vm.data))
print(np.std(vm.data))

print ts.earliest
print ts.latest
print ts.latest - ts.earliest


# Define the appearance of the signals
tom_red = [0.78431,0.196,0.196]
tom_green = [0.196,0.78431,0.196]
tom_blue = [0.196,0.196,0.78431]

start = datetime.strptime("11-May-2014 13:20", "%d-%b-%Y %H:%M")
end = start + timedelta(seconds=1)

ts.draw_separate()
#ts_visualisation.draw_separate()

#fig = plt.figure(figsize=(18,10))
#ax = fig.add_subplot(1,1,1)
#ax.scatter(simplified[1].data, simplified[2].data, lw=0, alpha=0.7, s=50)
#plt.show()