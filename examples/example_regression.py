import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy
import math


from pampro import Time_Series, Channel, channel_inference, Bout, time_utilities, pampro_visualisation, signal_comparison

execution_start = datetime.now()

ts = Time_Series.Time_Series("Actiheart")


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data\example_actiheart.txt')

chans = Channel.load_channels(filename, "Actiheart")
activity, ecg = chans
ts.add_channels(chans)

# Summarise at hourly intervals
tp = (time_utilities.start_of_hour(activity.timestamps[0]),time_utilities.end_of_hour(activity.timestamps[-1]))
ecg_hourly = ecg.piecewise_statistics(timedelta(minutes=10), time_period=tp)[0]
activity_hourly = activity.piecewise_statistics(timedelta(minutes=10), time_period=tp)[0]





# Visualise transitions of activity channel and ECG channel bigrams
fig, (ax1,ax2) = plt.subplots(1,2)

pampro_visualisation.axis_scatter_a_vs_b(ax1, ecg, activity)
pampro_visualisation.axis_scatter_a_vs_b(ax2, ecg_hourly, activity_hourly)

linear_parameters = signal_comparison.channel_linear_regression(ecg, activity)
polynomial_parameters = signal_comparison.channel_polynomial_regression(ecg, activity)
print linear_parameters
print polynomial_parameters
pampro_visualisation.axis_linear_regression_a_vs_b(ax1, ecg, activity, linear_parameters=linear_parameters)
pampro_visualisation.axis_polynomial_regression_a_vs_b(ax1, ecg_hourly, activity_hourly, polynomial_parameters=polynomial_parameters)




fig.tight_layout()

plt.show()

