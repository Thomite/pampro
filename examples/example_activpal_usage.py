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

start_time = datetime.now()

ts = Time_Series.Time_Series("activPAL")




# Load sample activPAL data
#filename = os.path.join(os.path.dirname(__file__), '..', 'data\ARBOTW.txt')
chans = Channel.load_channels("Q:/Data/CSVs for Tom/Parallel files/514538L-AP1336594 12Oct13 10-00am for 8d 0m.csv", "activPAL")
x,y,z = chans[0],chans[1],chans[2]
print("Finished reading in data")

# Calculate moving averages of the channels
#ecg_ma = ecg.moving_average(15)


# Infer vector magnitude from three channels
vm = channel_inference.infer_vector_magnitude(x,y,z)
enmo = channel_inference.infer_enmo(vm)
pitch, roll = channel_inference.infer_pitch_roll(x,y,z)
ts.add_channels([x,y,z,vm,enmo,pitch,roll])
print("Inferred VM, ENMO, pitch and roll")

# Turn the bouts into annotations and highlight those sections in the signals
#annotations = Annotation.annotations_from_bouts(bouts)
#ecg.add_annotations(annotations)

ts_output = Time_Series.Time_Series("activPAL output")

angle_levels = [
[-90,-85],[-85,-80],[-80,-75],[-75,-70],[-70,-65],[-65,-60],[-60,-55],[-55,-50],[-50,-45],[-45,-40],[-40,-35],[-35,-30],[-30,-25],[-25,-20],[-20,-15],[-15,-10],[-10,-05],[-05,0],
[0,05],[05,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60],[60,65],[65,70],[70,75],[75,80],[80,85],[85,90]]

# Save some stats about the time series to a file
very_basic = ["mean", "std"]
basic_stats = ["mean", "sum", "std", "min", "max"]
stat_dict = {"AP_X":very_basic, "AP_Y":very_basic, "AP_Z":very_basic, "VM":basic_stats, "Pitch":angle_levels+basic_stats, "Roll":angle_levels+basic_stats, "ENMO":basic_stats+[[0,40],[40,80],[80,120],[120,160],[160,200],[240,280],[320,360],[360,400],[400,440],[440,480],[480,520],[520,560],[560,600],[600,640],[640,680],[680,720],[720,760],[760,800],[800,840],[840,880],[880,920],[920,960],[960,1000],[1000,1040],[1040,1080],[1080,1120],[1120,1160],[1160,1200],[1240,1280],[1320,1360],[1360,1400],[1400,1440],[1440,1480],[1480,1520],[1520,1560],[1560,1600],[1600,1640],[1640,1680],[1680,1720],[1720,1760],[1760,1800],[1800,1840],[1840,1880],[1880,1920],[1920,1960],[1960,2000]]}
#for channel in [x,y,z,vm,pitch,roll]:
#	derived_channels = channel.piecewise_statistics( timedelta(minutes=10), statistics=stats )
#	ts_output.add_channels(derived_channels)
ts_output.add_channels(ts.piecewise_statistics( timedelta(minutes=10), stat_dict ))

ts_output.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ap_custom.csv'))

end_time = datetime.now()
duration = end_time - start_time
print(duration)



# Define the appearance of the signals
#ts_output.draw_separate()

