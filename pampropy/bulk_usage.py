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

basic_stats = ["mean", "sum", "std", "min", "max"]
angle_levels = [
[-90,-85],[-85,-80],[-80,-75],[-75,-70],[-70,-65],[-65,-60],[-60,-55],[-55,-50],[-50,-45],[-45,-40],[-40,-35],[-35,-30],[-30,-25],[-25,-20],[-20,-15],[-15,-10],[-10,-05],[-05,0],
[0,05],[05,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60],[60,65],[65,70],[70,75],[75,80],[80,85],[85,90]]
stat_dict = {"AP_X":basic_stats, "AP_Y":basic_stats, "AP_Z":basic_stats, "Vector magnitude":basic_stats, "Pitch":angle_levels+["mean", "std"], "Roll":angle_levels+["mean", "std"]}

file_header = ""
for k,v in stat_dict.items():
	for stat in v:
		if isinstance(stat, list):
			variable_name = str(k) + "_" + str(stat[0]) + "_" + str(stat[1])
		else:
			variable_name = str(k) + "_" + str(stat)
		
		file_header = file_header + "," + variable_name

print file_header
file_header = file_header[1:]
print file_header

def activpal_analysis(source):

	ts = Time_Series.Time_Series("activPAL")

	# Load sample activPAL data
	chans = Channel.load_channels(source, "activPAL")
	x,y,z = chans[0],chans[1],chans[2]
	print("Finished reading in data")

	# Infer vector magnitude from three channels
	vm = channel_inference.infer_vector_magnitude(x,y,z)
	pitch, roll = channel_inference.infer_pitch_roll(x,y,z)
	ts.add_channels([x,y,z,vm,pitch,roll])
	print("Inferred VM, pitch and roll")

	# Save some stats about the whole time series to a file
	
	
	all_results = ts.summary_statistics(stat_dict)
	return all_results
	

files = ["example1", "example2"]

test = []
for f in files:
	results = activpal_analysis(f)
	test.append(results)

output = open("Q:/Data/CSVs for Tom/output.csv", "w")

output.write(file_header + "\n")
for t in test:
	output.write(str(t)[1:-1] + "\n")
output.close()
