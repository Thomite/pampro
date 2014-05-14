import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy



from pampropy import Time_Series, Channel, Annotation, channel_inference, Bout

execution_start = datetime.now()

ts = Time_Series.Time_Series("Actigraph")


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data/random.DAT')

chans = Channel.load_channels(filename, "Actigraph")
counts = chans[0]

ts.add_channel(counts)


ts.draw_separate()

execution_end = datetime.now()
print("{} to {} = {}".format( execution_start, execution_end, execution_end - execution_start))