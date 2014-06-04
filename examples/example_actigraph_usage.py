import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy
import collections



from pampropy import Time_Series, Channel, channel_inference, Bout


stats = ["

cutpoints = [ [0,49],[0,99],[0,150],[0,199],[0,299],[0,399],[0,499],[0,599],[0,699],[0,799],[0,899],[0,999],[0,1099],[0,1199],[0,1299],[0,1399],[0,1499],[0,1599],[0,1699],[0,1799],[0,1899],[0,1999],[0,2099],[0,2199],[0,2299],[0,2399],[0,2499],[0,2599],[0,2699],[0,2799],[0,2899],[0,2999],[0,3099],[0,3199],[0,3299],[0,3399],[0,3499],[0,3599],[0,3699],[0,3799],[0,3899],[0,3999],[0,4099],[0,4199],[0,4299],[0,4399],[0,4499],[0,4599],[0,4699],[0,4799],[0,4899],[0,4999],
[50,99999],[100,99999],[150,99999],[200,99999],[300,99999],[400,99999],[500,99999],[600,99999],[700,99999],[800,99999],[900,99999],[1000,99999],[1100,99999],[1200,99999],[1300,99999],[1400,99999],[1500,99999],[1600,99999],[1700,99999],[1800,99999],[1900,99999],[2000,99999],[2100,99999],[2200,99999],[2300,99999],[2400,99999],[2500,99999],[2600,99999],[2700,99999],[2800,99999],[2900,99999],[3000,99999],[3100,99999],[3200,99999],[3300,99999],[3400,99999],[3500,99999],[3600,99999],[3700,99999],[3800,99999],[3900,99999],[4000,99999],[4100,99999],[4200,99999],[4300,99999],[4400,99999],[4500,99999],[4600,99999],[4700,99999],[4800,99999],[4900,99999],[5000,99999] ] 
lengths = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100,110,120,150,180]

variable_names = []

for cutpoint in cutpoints:
	
	low, high = cutpoint[0], cutpoint[1]
	for length in lengths:
		name = "_{}_{}_mt{}".format(low,high,length)
		variable_names.append(name)


def write_summary_values(dict, file_target):

	for variable_name in variable_names:
		output.write(str(dict[variable_name].total_seconds()/60) + ",")
	output.write("\n")


def actigraph_analysis(id):

	ts = Time_Series.Time_Series("Actigraph")
	values = collections.OrderedDict()

	# Load Actigraph data
	filename = os.path.join(os.path.dirname(__file__), '..', 'data/random.DAT')

	chans = Channel.load_channels(filename, "Actigraph")
	counts = chans[0]

	ts.add_channel(counts)
	wear_counts, wear_bouts, nonwear_bouts = channel_inference.infer_nonwear_actigraph(counts, zero_minutes=90)
	ts.add_channel(wear_counts)

	valid_only, valid_windows = channel_inference.infer_valid_days_only(wear_counts, wear_bouts, valid_criterion=timedelta(hours=10))
	ts.add_channel(valid_only)

	for cutpoint in cutpoints:
		
		low, high = cutpoint[0], cutpoint[1]

		bouts = valid_only.bouts(low, high)

		for length in lengths:

			time_length = timedelta(minutes=length)

			still_ok = []

			# drop if bouts < length
			for bout in bouts:
				if bout.end_timestamp - bout.start_timestamp >= time_length:
					still_ok.append(bout)

			bouts = still_ok

			name = "_{}_{}_mt{}".format(low,high,length)
			total_time = Bout.total_time(bouts)

			values[name] = total_time

	write_summary_values(values, output)

	counts.add_annotations(wear_bouts)

	chan = Channel.channel_from_bouts(bouts=wear_bouts, time_period=[counts.timeframe[0],counts.timeframe[1]], channel_name="Wear", time_resolution=timedelta(minutes=1))
	ts.add_channel(chan)

	wear_counts.add_annotations(valid_windows)

	ts.draw_separate()



output = file("V:/P5_PhysAct/People/Tom/pampropy/data/summary_data.csv", "w")
for var in variable_names:
	output.write(var + ",")
output.write("\n")

actigraph_analysis("test")



output.close()
