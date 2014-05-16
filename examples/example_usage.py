import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy



from pampropy import Time_Series, Channel, channel_inference, Bout

execution_start = datetime.now()

ts = Time_Series.Time_Series("Actiheart")


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data\ARBOTW.txt')

chans = Channel.load_channels(filename, "Actiheart")
ts.add_channels(chans)
activity = chans[0]
ecg = chans[1]


# Calculate moving averages of the channels
ecg_ma = ecg.moving_average(15)
activity_ma = activity.moving_average(15)
ts.add_channel(ecg_ma)
ts.add_channel(activity_ma)

blah = activity.time_derivative()
blah = blah.moving_average(121)
ts.add_channel(blah)

# Infer sleep from Actiheart channels
awake_probability = channel_inference.infer_sleep_actiheart(activity, ecg)
ts.add_channel(awake_probability)


# Define a period of interest
start = datetime.strptime("15-Mar-2014 00:00", "%d-%b-%Y %H:%M")
end = start + timedelta(hours=12)

ts_output = Time_Series.Time_Series("Actiheart output")
# Save some stats about the time series to a file
stats = ["mean", "sum", "std", "min", "max", "n"]
result_chans = activity.piecewise_statistics( timedelta(minutes=10), statistics=stats )
ts_output.add_channels(result_chans)
result_chans = ecg.piecewise_statistics( timedelta(minutes=10), statistics=stats )
ts_output.add_channels(result_chans)
#ts_output.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ah_data_10m.csv'))

result_chans = activity.summary_statistics( statistics=stats )
ts_summary = Time_Series.Time_Series("Actiheart summary")
ts_summary.add_channels(result_chans)
#ts_summary.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ah_data_summary.csv'))

# Infer vector magnitude from three channels
#vm = channel_inference.infer_vector_magnitude(awake_probability, ecg, activity)


# Get a list of bouts where heart rate > 80bpm
bouts = ecg.bouts(80,999,10)
total = Bout.total_time(bouts)
print(total)

bouts2 = ecg.bouts(80,999)
total2 = Bout.total_time(bouts2)
print(total2)

# Create a channel only containing data where those bouts occurred
subset = ecg.subset_using_bouts(bouts, "whatever", substitute_value=-1)

ts.add_channel(subset)

# Summarise where those levels occurred again
just_n = {"whatever":[[80,999]], "AH_ECG":[[80,999]]}
result_chans = ts.piecewise_statistics( timedelta(minutes=10), statistics=just_n )

output = Time_Series.Time_Series("Min bout output")
output.add_channels(result_chans)
output.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/10m_test.csv'))


Bout.write_bouts_to_file(bouts, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/bouts.csv'))
bout_list = Bout.read_bouts(os.path.join(os.path.dirname(__file__), '..', 'data/bouts.csv'))
for bout in bout_list:
	print bout.start_timestamp, bout.end_timestamp

#ecg_low = ecg.bouts(0,80,20)
#intersection = Bout.bout_list_intersection(ecg_low, bouts)


awake_probability.add_annotations(bouts)
#ecg_ma.add_annotations(ecg_low)
activity_ma.add_annotations(bouts)
#ts.get_channel("AH_Activity_td_ma").add_annotations(intersection)

#for bout in bouts:
# print bout[0].day, bout[0], " -- ", bout[1]

#activity_channels = activity.build_statistics_channels(inverted, stats)
#ecg_channels = ecg.build_statistics_channels(inverted, stats)
#ts_test = Time_Series.Time_Series("Bouts")
#ts_test.add_channels(activity_channels)
#ts_test.add_channels(ecg_channels)
#ts_test.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/ah_data_bouts.csv'))

# Define the appearance of the signals
ecg_ma.draw_properties = {'alpha':1, 'lw':2, 'color':[0.78431,0.196,0.196]}
activity_ma.draw_properties = {'alpha':1, 'lw':2, 'color':[0.196,0.196,0.78431]}
awake_probability.draw_properties = {'alpha':1, 'lw':2, 'color':[0.78431,0.196,0.78431]}
#ts.draw_separate(time_period=[start,end])#, file_target=os.path.join(os.path.dirname(__file__), '..', 'data/blah.png'))


# We can also return the data by not supplying a file_target
#dataset = activity.piecewise_statistics( timedelta(days=1), statistics=stats )
#for row in dataset:
#print row

ts.draw_separate()

execution_end = datetime.now()
print("{} to {} = {}".format( execution_start, execution_end, execution_end - execution_start))