import numpy as np
import scipy as sp
from datetime import datetime, date, time, timedelta
from pampro import Bout
import copy

from struct import *
from math import *
import time
from datetime import datetime
from urllib import unquote_plus
import sys
import io
import re
import cProfile, pstats, StringIO
import string
from scipy.io.wavfile import write

percentile_pattern = re.compile("\A([p])([0-9]*)")

class Channel(object):

	def __init__(self, name):

		self.name = name
		self.size = 0
		self.timeframe = 0
		self.data = []
		self.timestamps = []
		self.annotations = []
		self.draw_properties = {}
		self.cached_indices = {}

	def clone(self):

		return copy.deepcopy(self)

	def set_contents(self, data, timestamps):

		self.data = data
		self.timestamps = timestamps

		self.calculate_timeframe()

	def append(self, other_channel):

		print(self.name + " " + str(len(self.data)))
		self.data = np.concatenate((self.data, other_channel.data))
		self.timestamps = np.concatenate((self.timestamps, other_channel.timestamps))

		indices = np.argsort(self.timestamps)


		self.timestamps = np.array(self.timestamps)[indices]
		self.data = np.array(self.data)[indices]
		print(self.name + " " + str(len(self.data)))
		print("")
		self.calculate_timeframe()

	def calculate_timeframe(self):

		self.size = len(self.data)

		if self.size > 0:
			self.timeframe = self.timestamps[0], self.timestamps[-1]
		else:
			self.timeframe = False,False

	def add_annotation(self, annotation):

		self.annotations.append(annotation)

	def add_annotations(self, annotations):

		for a in annotations:
			self.add_annotation(a)

	def normalise(self, floor=0, ceil=1):

		max_value = max(self.data)
		min_value = min(self.data)
		self.data = ((ceil - floor) * (self.data - min_value))/(max_value - min_value) + floor


	def collapse_auto(self, bins=10):

		max_value = max(self.data)
		min_value = min(self.data)
		increment = float(max_value - min_value)/float(bins)

		print min_value, max_value

		ranges = []
		low = min_value
		for i in range(bins):

			if i == bins-1:
				high = max_value
			else:
				high = low+increment

			ranges.append((low, high, i))
			low += increment

		print ranges

		return self.collapse(ranges)

	def collapse(self, ranges):

		# Each range is a tuple: (>= low, <= high, replacement)

		clone = self.clone()

		for low, high, replacement in ranges:

			indices = np.where((self.data >= low) & (self.data <= high))[0]
			clone.data[indices] = replacement

		return clone


	def bigrams(self):

		unique_values = np.unique(self.data)

		# Create a dictionary to count each permutation of unique_value -> unique_value
		pairs = {}
		for val1 in unique_values:
			pairs[val1] = {}
			for val2 in unique_values:
				pairs[val1][val2] = 0

		# Count transitions from each unique value to the next
		for val1, val2 in zip(self.data, self.data[1:]):
			pairs[val1][val2] += 1

		return pairs


	def get_window(self, datetime_start, datetime_end):

		key_value = str(datetime_start) + "|" + str(datetime_end)
		indices = [-1]

		# If we already know what the indices are for this time range:
		#if key_value in self.cached_indices.keys():
		try:
			indices = self.cached_indices[key_value]

		except:
		#else:
			start = np.searchsorted(self.timestamps, datetime_start, 'left')
			end = np.searchsorted(self.timestamps, datetime_end, 'right')
			indices = np.arange(start, end-1)

			# Cache those for next time
			self.cached_indices[key_value] = indices

		return indices

	# This is the old, uncached method
	def get_window_old(self, datetime_start, datetime_end):

		start = np.searchsorted(self.timestamps, datetime_start, 'left')
		end = np.searchsorted(self.timestamps, datetime_end, 'right')
		return np.arange(start, end-1)


	def window_statistics(self, start_dts, end_dts, statistics):


		indices = self.get_window(start_dts, end_dts)

		#pretty_timestamp = start_dts.strftime("%d/%m/%Y %H:%M:%S:%f")
		#output_row = [pretty_timestamp]

		output_row = []
		if (len(indices) > 0):

			for stat in statistics:
				if stat == "mean":
					output_row.append(np.mean(self.data[indices]))
				elif stat == "sum":
					output_row.append(sum(self.data[indices]))
				elif stat == "std":
					output_row.append(np.std(self.data[indices]))
				elif stat == "min":
					output_row.append(np.min(self.data[indices]))
				elif stat == "max":
					output_row.append(np.max(self.data[indices]))
				elif stat == "n":
					output_row.append(len(indices))
				elif isinstance(stat, list):

					indices2 = np.where((self.data[indices] >= stat[0]) & (self.data[indices] <= stat[1]))[0]
					output_row.append(len(indices2))
				elif percentile_pattern.match(stat):
					# for p25, p50, etc
					percentile = int(percentile_pattern.match(stat).groups()[1])
					output_row.append(np.percentile(self.data[indices],percentile))

				elif stat == "bigrams":


					unique_values = np.unique(self.data)

					# 1 channel for each permutation of unique_value -> unique_value
					pairs = {}
					for val1 in unique_values:
						pairs[val1] = {}
						for val2 in unique_values:
							pairs[val1][val2] = 0

					# Count transitions from each unique value to the next
					for val1, val2 in zip(self.data[indices], self.data[indices[1:]]):
						pairs[val1][val2] += 1

					for val1 in unique_values:
						for val2 in unique_values:
							output_row.append(pairs[val1][val2])

				else:
					output_row.append(-1)
		else:

			num_missings = len(statistics)

			# len(statistics) doesn't work for bigrams
			if "bigrams" in statistics:
				unique_values = np.unique(self.data)
				possible_transitions = len(unique_values)**2
				num_missings = num_missings -1 + possible_transitions

			for i in range(num_missings):

				output_row.append(-1)


		return output_row

	def build_statistics_channels(self, windows, statistics):

		channel_list = []
		for var in statistics:
			name = self.name
			if isinstance(var, list):
				name = self.name + "_" + str(var[0]) + "_" + str(var[1])
				channel = Channel(name)
				channel_list.append(channel)

			elif var == "bigrams":
				unique_values = np.unique(self.data)

				# 1 channel for each permutation of unique_value -> unique_value
				for val1 in unique_values:
					for val2 in unique_values:
						name = self.name + "_" + str(val1) + "tr" + str(val2)

						channel = Channel(name)
						channel_list.append(channel)

			else:
				name = self.name + "_" + var
				channel = Channel(name)
				channel_list.append(channel)

		for window in windows:

			results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)
			for i in range(len(results)):
				#print len(results)
				channel_list[i].append_data(window.start_timestamp, results[i])

		for channel in channel_list:
			channel.calculate_timeframe()
			channel.data = np.array(channel.data)
			channel.timestamps = np.array(channel.timestamps)

		return channel_list

	def append_data(self, timestamp, data_row):
		self.timestamps.append(timestamp)
		self.data.append(data_row)

	def sliding_statistics(self, window_size, statistics=["mean"], time_period=False):

		if time_period == False:
			start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
			end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
		else:
			start = time_period[0]
			end = time_period[1]

		#print("Sliding statistics: {}".format(self.name))

		windows = []

		for timestamp in self.timestamps:

			start_dts = timestamp - (window_size/2.0)
			end_dts = timestamp + (window_size/2.0)

			windows.append(Bout.Bout(start_dts, end_dts))

		return self.build_statistics_channels(windows, statistics)


	def piecewise_statistics(self, window_size, statistics=["mean"], time_period=False):

		if time_period == False:
			start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
			end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
		else:
			start = time_period[0]
			end = time_period[1]

		#print("Piecewise statistics: {}".format(self.name))

		windows = []

		start_dts = start
		end_dts = start + window_size

		while start_dts < end:

			window = Bout.Bout(start_dts, end_dts)
			windows.append(window)

			start_dts = start_dts + window_size
			end_dts = end_dts + window_size

		return self.build_statistics_channels(windows, statistics)


	def summary_statistics(self, statistics=["mean"]):

		windows = [Bout.Bout(self.timeframe[0], self.timeframe[1])]
		#results = self.window_statistics(self.timeframe[0], self.timeframe[1], statistics)

		return self.build_statistics_channels(windows, statistics)

	def bouts(self, low, high, minimum_length=timedelta(minutes=0)):

		state = 0
		start_index = 0
		end_index = 1
		bouts = []

		for i, value in enumerate(self.data):

			if state == 0:

				if value >= low and value <= high:

					state = 1
					start_index = i
					end_index = i

			else:

				if value >= low and value <= high:

					end_index = i

				else:

					state = 0

					start_time =  self.timestamps[start_index]
					end_time = self.timestamps[end_index]
					if end_index+1 < self.size:
						end_time = self.timestamps[end_index+1]

					if (end_time - start_time >= minimum_length):
						bouts.append(Bout.Bout(start_time, end_time))


		if state == 1:
			start_time =  self.timestamps[start_index]
			end_time = self.timestamps[end_index]
			if (end_time - start_time >= minimum_length):
				bouts.append(Bout.Bout(start_time, end_time))

		return bouts

	def subset_using_bouts(self, bout_list, name, substitute_value=-1):
		# Given a list of bouts, create a new channel from this taking only the data from inside those bouts
		c = Channel(name)

		filled = np.empty(self.size)
		filled.fill(substitute_value)
		#print(len(filled))

		c.set_contents(filled, self.timestamps)

		for bout in bout_list:
			#print(bout)

			indices = self.get_window(bout.start_timestamp, bout.end_timestamp)

			#c.data[bout[2]:bout[3]] = self.data[bout[2]:bout[3]]
			c.data[indices] = self.data[indices]

		return c

	def delete_windows(self, windows):

		for window in windows:
			indices = self.get_window(window.start_timestamp, window.end_timestamp)

			self.data = np.delete(self.data, indices, None)
			self.timestamps = np.delete(self.timestamps, indices, None)

		self.calculate_timeframe()
			#del self.data[indices[0]:indices[-1]]
			#del self.timestamps[indices[0]:indices[-1]]

	def restrict_timeframe(self, start, end):

		indices = self.get_window(start, end)

		self.set_contents(self.data[indices], self.timestamps[indices])

	def moving_average(self, size):

		averaged = []
		half = (size-1)/2

		for i in range(0,self.size):
			total = 0
			contributors = 0
			for j in range(i-half,i+half):
				if (j >= 0) & (j < self.size):
					contributors+=1
					total += self.data[j]
			averaged.append(total/contributors)

		result = Channel(self.name + "_ma")
		result.set_contents(np.array(averaged), self.timestamps)
		return result

	def moving_std(self, size):

		averaged = []
		half = (size-1)/2

		for i in range(0,self.size):

			low = max(0,i-half)
			high = min(self.size,i+half)

			averaged.append(np.std(self.data[low:high]))

		result = Channel(self.name + "_mstd")
		result.set_contents(np.array(averaged), self.timestamps)
		return result

	def time_derivative(self):

		result = Channel(self.name + "_td")
		result.set_contents(np.diff(self.data), self.timestamps[:-1])
		return result

	def absolute(self):

		result = Channel(self.name + "_abs")
		result.set_contents(np.abs(self.data), self.timestamps)
		return result

	def fill(self, bout, fill_value=0):

		indices = self.get_window(bout.start_timestamp,bout.end_timestamp)

		self.data[indices] = fill_value

	def fft(self):

		return np.fft.fft(self.data)


	def output_as_tone(self, filename, note_duration=0.15, volume=10000):

		rate = 1378.125

		self.normalise(floor=83,ceil=880)
		tone = np.array([], dtype=np.int16)

		for note in self.data:

			t = np.linspace(0,note_duration,note_duration*rate)
			data = np.array(np.sin(2.0*np.pi*note*t)*volume, dtype=np.int16)
			tone = np.append(tone, data)

		write(filename, rate, tone)

	def draw_experimental(self, axis):

		axis.plot(self.timestamps, self.data, label=self.name, **self.draw_properties)
		for a in self.annotations:
			axis.axvspan(xmin=a.start_timestamp, xmax=a.end_timestamp, **a.draw_properties)

def channel_from_coefficients(coefs, timestamps):
    chan = Channel("Recreated")

    recreated = np.fft.ifft(coefs, n=len(timestamps))
    chan.set_contents(recreated, timestamps)

    return chan

def channel_from_bouts(bouts, time_period, time_resolution, channel_name, skeleton=False, in_value=1, out_value=0):

	result = False
	if skeleton==False:
		result = Channel(channel_name)

		#timestamps = []
		#timestamp = time_period[0]

		num_epochs = int(((time_period[1] - time_period[0]).total_seconds()) / time_resolution.total_seconds())

		#while timestamp < time_period[1]:

		#	timestamps.append(timestamp)
		#	timestamp += time_resolution

		timestamps = [time_period[0] + time_resolution*x for x in range(num_epochs)]

		filled = np.empty(len(timestamps))
		filled.fill(out_value)

		print time_period
		print("Length of timestamps:" + str(len(timestamps)))
		print("Length of filled:" + str(len(filled)))

		result.set_contents(filled, timestamps)
	else:
		result = skeleton
		result.name = channel_name
		result.data.fill(out_value)


	for bout in bouts:
		result.fill(bout, in_value)



	return result

# Axivity import code adapted from source provided by Open Movement: https://code.google.com/p/openmovement/. Their license terms are reproduced here in full, and apply only to the Axivity related code:
# Copyright (c) 2009-2014, Newcastle University, UK. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def byte(value):
	return (value + 2 ** 7) % 2 ** 8 - 2 ** 7

def ushort(value):
	return value % 2 ** 16

def short(value):
	return (value + 2 ** 15) % 2 ** 16 - 2 ** 15

def axivity_read_timestamp(stamp):
	stamp = unpack('I', stamp)[0]
	year = ((stamp >> 26) & 0x3f) + 2000
	month = (stamp >> 22) & 0x0f
	day   = (stamp >> 17) & 0x1f
	hours = (stamp >> 12) & 0x1f
	mins  = (stamp >>  6) & 0x3f
	secs  = (stamp >>  0) & 0x3f
	try:
		t = datetime(year, month, day, hours, mins, secs)
	except ValueError:
		t = None
	return t

def axivity_read(fh, bytes):
	data = fh.read(bytes)
	if len(data) == bytes:
		return data
	else:
		raise IOError

def axivity_parse_header(fh):
	blockSize = unpack('H', axivity_read(fh,2))[0]
	performClear = unpack('B', axivity_read(fh,1))[0]
	deviceId = unpack('H', axivity_read(fh,2))[0]
	sessionId = unpack('I', axivity_read(fh,4))[0]
	shippingMinLightLevel = unpack('H', axivity_read(fh,2))[0]
	loggingStartTime = axivity_read(fh,4)
	loggingEndTime = axivity_read(fh,4)
	loggingCapacity = unpack('I', axivity_read(fh,4))[0]
	allowStandby = unpack('B', axivity_read(fh,1))[0]
	debuggingInfo = unpack('B', axivity_read(fh,1))[0]
	batteryMinimumToLog = unpack('H', axivity_read(fh,2))[0]
	batteryWarning = unpack('H', axivity_read(fh,2))[0]
	enableSerial = unpack('B', axivity_read(fh,1))[0]
	lastClearTime = axivity_read(fh,4)
	samplingRate = unpack('B', axivity_read(fh,1))[0]
	lastChangeTime = axivity_read(fh,4)
	firmwareVersion = unpack('B', axivity_read(fh,1))[0]

	reserved = axivity_read(fh,22)

	annotationBlock = axivity_read(fh,448 + 512)

	if len(annotationBlock) < 448 + 512:
		annotationBlock = ""

	annotation = ""
	for x in annotationBlock:
		if ord(x) != 255 and x != ' ':
			if x == '?':
				x = '&'
			annotation += x
	annotation = annotation.strip()

	annotationElements = annotation.split('&')
	annotationNames = {'_c': 'studyCentre', '_s': 'studyCode', '_i': 'investigator', '_x': 'exerciseCode', '_v': 'volunteerNum', '_p': 'bodyLocation', '_so': 'setupOperator', '_n': 'notes', '_b': 'startTime', '_e': 'endTime', '_ro': 'recoveryOperator', '_r': 'retrievalTime',           '_co': 'comments'}
	annotations = dict()
	for element in annotationElements:
		kv = element.split('=', 2)
		if kv[0] in annotationNames:
			annotations[annotationNames[kv[0]]] = unquote_plus(kv[1])

	for x in ('startTime', 'endTime', 'retrievalTime'):
		if x in annotations:
			if '/' in annotations[x]:
				annotations[x] = time.strptime(annotations[x], '%d/%m/%Y')
			else:
				annotations[x] = time.strptime(annotations[x], '%Y-%m-%d %H:%M:%S')

	annotations = annotations
	deviceId = deviceId
	sessionId = sessionId
	lastClearTime = axivity_read_timestamp(lastClearTime)
	lastChangeTime = axivity_read_timestamp(lastChangeTime)
	firmwareVersion = firmwareVersion if firmwareVersion != 255 else 0

def parse_header(header, type, datetime_format):

	header_info = {}

	if type == "Actiheart":

		for i,row in enumerate(header):
			try:
				values = row.split(",")
				header_info[values[0]] = values[1]
			except:
				pass

		time1 = datetime.strptime(header[-2].split(",")[0], "%H:%M:%S")
		time2 = datetime.strptime(header[-1].split(",")[0], "%H:%M:%S")
		header_info["epoch_length"] = time2 - time1

		header_info["start_date"] = datetime.strptime(header_info["Started"], "%d-%b-%Y  %H:%M")

		if "Start trimmed to" in header_info:
			header_info["Start trimmed to"] = datetime.strptime(header_info["Start trimmed to"], "%Y-%m-%d %H:%M")


		for i,row in enumerate(header):

			if row.split(",")[0] == "Time":
				header_info["data_start"] = i+1
				break

	elif type == "Actigraph":

		test = header[2].split(" ")
		timeval = datetime.strptime(test[-1], "%H:%M:%S")
		start_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
		header_info["start_time"] = start_time

		test = header[3].split(" ")
		start_date = string.replace(test[-1], "-", "/")

		try:
			start_date = datetime.strptime(start_date, datetime_format)
		except:
			start_date = datetime.strptime(start_date, "%d/%m/%Y")
		header_info["start_date"] = start_date

		test = header[4].split(" ")
		delta = datetime.strptime(test[-1], "%H:%M:%S")
		epoch_length = timedelta(hours=delta.hour, minutes=delta.minute, seconds=delta.second)
		header_info["epoch_length"] = epoch_length

		start_datetime = start_date + start_time
		header_info["start_datetime"] = start_datetime

		mode = 0
		splitup = header[8].split(" ")
		if "Mode" in splitup:
			index = splitup.index("Mode")
			mode = splitup[index + 2]
		header_info["mode"] = int(mode)

	elif type == "GT3X+_CSV":

		test = header[2].split(" ")
		timeval = datetime.strptime(test[-1], "%H:%M:%S")
		start_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
		header_info["start_time"] = start_time

		test = header[0].split(" ")
		if "Hz" in test:
			index = test.index("Hz")
			hz = int(test[index-1])
			epoch_length = timedelta(seconds=1) / hz
			header_info["epoch_length"] = epoch_length

		if "format" in test:
			index = test.index("format")
			format = test[index+1]
			format = string.replace(format, "dd", "%d")
			format = string.replace(format, "MM", "%m")
			format = string.replace(format, "yyyy", "%Y")

			start_date = datetime.strptime(header[3].split(" ")[2], format)
			header_info["start_date"] = start_date

		start_datetime = start_date + start_time
		header_info["start_datetime"] = start_datetime


	return header_info

def load_channels(source, source_type, datetime_format="%d/%m/%Y %H:%M:%S:%f", datetime_column=0, ignore_columns=False, unique_names=False, average_over=False):

	if (source_type == "Actiheart"):



		first_lines = []
		f = open(source, 'r')
		for i in range(0,30):
			s = f.readline().strip()
			first_lines.append(s)
		f.close()

		header_info = parse_header(first_lines, "Actiheart", "%d-%b-%Y  %H:%M")

		start_date = header_info["start_date"]
		epoch_length = header_info["epoch_length"]
		data_start = header_info["data_start"]
		#timestamp_list = []
		#for i in range(0,len(activity)):
		#	timestamp_list.append(start_date)
		#	start_date = start_date + epoch_length

		activity, ecg  = np.loadtxt(source, delimiter=',', unpack=True, skiprows=data_start, usecols=[1,2])

		timestamp_list = [start_date+i*epoch_length for i in range(len(activity))]
		timestamps = np.array(timestamp_list)


		indices1 = []

		if "Start trimmed to" in header_info:
			indices1 = np.where((ecg > 0) & (timestamps > header_info["Start trimmed to"]))
		else:
			indices1 = np.where(ecg > 0)

		activity = activity[indices1]
		ecg = ecg[indices1]
		timestamps2 = timestamps[indices1]

		actiheart_activity = Channel("AH_Activity")
		actiheart_activity.set_contents(activity, timestamps2)

		actiheart_ecg = Channel("AH_ECG")
		actiheart_ecg.set_contents(ecg, timestamps2)

		return [actiheart_activity, actiheart_ecg]

	elif (source_type == "activPAL"):

		ap_timestamp, ap_x, ap_y, ap_z = np.loadtxt(source, delimiter=',', unpack=True, skiprows=5, dtype={'names':('ap_timestamp','ap_x','ap_y','ap_z'), 'formats':('S16','f8','f8','f8')})
		#print("A")
		dt = datetime.strptime("30-Dec-1899", "%d-%b-%Y")

		ap_timestamps = []
		for val in ap_timestamp:

			test = val.split(".")

			while len(test[1]) < 10:
				test[1] = test[1] + "0"

			finaltest = dt + timedelta(days=int(test[0]), microseconds=int(test[1])*8.64)
			ap_timestamps.append(finaltest)

		ap_timestamps = np.array(ap_timestamps)
		#print("B")
		x = Channel("AP_X")
		y = Channel("AP_Y")
		z = Channel("AP_Z")

		ap_x = (ap_x-128.0)/64.0
		ap_y = (ap_y-128.0)/64.0
		ap_z = (ap_z-128.0)/64.0

		x.set_contents(np.array(ap_x, dtype=np.float64), ap_timestamps)
		y.set_contents(np.array(ap_y, dtype=np.float64), ap_timestamps)
		z.set_contents(np.array(ap_z, dtype=np.float64), ap_timestamps)
		#print("C")
		return [x,y,z]

	elif (source_type == "GeneActiv_CSV"):

		ga_timestamp, ga_x, ga_y, ga_z, ga_lux, ga_event, ga_temperature = np.genfromtxt(source, delimiter=',', unpack=True, skip_header=80, dtype=str)

		ga_x = np.array(ga_x, dtype=np.float64)
		ga_y = np.array(ga_y, dtype=np.float64)
		ga_z = np.array(ga_z, dtype=np.float64)
		ga_lux = np.array(ga_lux, dtype=np.int32)
		ga_event = np.array(ga_event, dtype=np.bool_)
		ga_temperature = np.array(ga_temperature, dtype=np.float32)

		ga_timestamps = []

		for i in range(0, len(ga_timestamp)):
			ts = datetime.strptime(ga_timestamp[i], "%Y-%m-%d %H:%M:%S:%f")
			ga_timestamps.append(ts)
		ga_timestamps = np.array(ga_timestamps)

		x = Channel("GA_X")
		y = Channel("GA_Y")
		z = Channel("GA_Z")
		lux = Channel("GA_Lux")
		event = Channel("GA_Event")
		temperature = Channel("GA_Temperature")

		x.set_contents(ga_x, ga_timestamps)
		y.set_contents(ga_y, ga_timestamps)
		z.set_contents(ga_z, ga_timestamps)
		lux.set_contents(ga_lux, ga_timestamps)
		event.set_contents(ga_event, ga_timestamps)
		temperature.set_contents(ga_temperature, ga_timestamps)

		return [x,y,z,lux,event,temperature]

	elif (source_type == "Actigraph"):

		first_lines = []
		f = open(source, 'r')
		for i in range(0,10):
			s = f.readline().strip()
			first_lines.append(s)


		header_info = parse_header(first_lines, "Actigraph", datetime_format)

		time = header_info["start_datetime"]
		epoch_length = header_info["epoch_length"]
		mode = header_info["mode"]

		count_list = []
		timestamp_list = []

		line = f.readline().strip()
		while (len(line) > 0):

			counts = line.split()
			for index, c in enumerate(counts):
			#	print index, index % 2
				if mode == 0 or mode == 4 or (mode == 1 and index % 2 == 0) or (mode == 3 and index % 2 == 0):
					#print("eh?")
					count_list.append(int(c))
					timestamp_list.append(time)
					time = time + epoch_length
				#else:
					#print("eh?")


			line = f.readline().strip()
		f.close()

		timestamps = np.array(timestamp_list)
		counts = np.array(count_list)

		#print timestamps[0], timestamps[-1]
		#print sum(counts)



		chan = Channel("AG_Counts")
		chan.set_contents(counts, timestamps)

		return [chan, header_info]

	elif (source_type == "GT3X+_CSV"):

		first_lines = []
		f = open(source, 'r')
		for i in range(0,10):
			s = f.readline().strip()
			first_lines.append(s)
		f.close()

		header_info = parse_header(first_lines, "GT3X+_CSV")

		time = header_info["start_datetime"]
		epoch_length = header_info["epoch_length"]

		x, y, z = np.genfromtxt(source, delimiter=',', unpack=True, skip_header=10, dtype=np.float64)

		timestamps = []
		for i in range(len(x)):
			timestamps.append(time)
			time += epoch_length
		timestamps = np.array(timestamps)

		x_chan = Channel("X")
		y_chan = Channel("Y")
		z_chan = Channel("Z")

		x_chan.set_contents(x, timestamps)
		y_chan.set_contents(y, timestamps)
		z_chan.set_contents(z, timestamps)

		return [x_chan,y_chan,z_chan]

	elif (source_type == "CSV"):

		f = open(source, 'r')
		s = f.readline().strip()
		f.close()

		test = s.split(",")

		source_split = source.split("/")


		data = np.loadtxt(source, delimiter=',', skiprows=1, dtype='str')

		print(data.shape)
		#print(data[:,0])
		#print(data[:,1])
		#print(data[:,2])
		#print(data[:,3])

		timestamps = []
		for date_row in data[:,datetime_column]:
			timestamps.append(datetime.strptime(date_row, datetime_format))
		timestamps = np.array(timestamps)

		data_columns = list(range(0,len(test)))
		del data_columns[datetime_column]

		if ignore_columns != False:
			for ic in ignore_columns:
				del data_columns[ic]

		#print data_columns

		channels = []
		for col in data_columns:
			#print col
			if unique_names:
				name = source_split[-1] + " - " + test[col]
			else:
				name = test[col]
			c = Channel(name)
			c.set_contents(np.array(data[:,col], dtype=np.float64), timestamps)
			channels.append(c)

		return channels

	elif (source_type == "Axivity"):

		channel_x = Channel("X")
		channel_y = Channel("Y")
		channel_z = Channel("Z")

		fh = open(source, 'rb')

		n= 0

		axivity_timestamps = []
		axivity_x = []
		axivity_y = []
		axivity_z = []
		last_time = False

		try:
			header = axivity_read(fh,2)

			temp_x = []
			temp_y = []
			temp_z = []
			temp_time = False

			while len(header) == 2:

				if header == 'MD':
					#print 'MD'
					axivity_parse_header(fh)
				elif header == 'UB':
					#print 'UB'
					blockSize = unpack('H', axivity_read(fh,2))[0]
				elif header == 'SI':
					#print 'SI'
					pass
				elif header == 'AX':
					packetLength = unpack('H', axivity_read(fh,2))[0]
					deviceId = unpack('H', axivity_read(fh,2))[0]
					sessionId = unpack('I', axivity_read(fh,4))[0]
					sequenceId = unpack('I', axivity_read(fh,4))[0]
					sampleTime = axivity_read_timestamp(axivity_read(fh,4))
					light = unpack('H', axivity_read(fh,2))[0]
					temperature = unpack('H', axivity_read(fh,2))[0]
					events = axivity_read(fh,1)
					battery = unpack('B', axivity_read(fh,1))[0]
					sampleRate = unpack('B', axivity_read(fh,1))[0]
					numAxesBPS = unpack('B', axivity_read(fh,1))[0]
					timestampOffset = unpack('h', axivity_read(fh,2))[0]
					sampleCount = unpack('H', axivity_read(fh,2))[0]

					sampleData = io.BytesIO(axivity_read(fh,480))
					checksum = unpack('H', axivity_read(fh,2))[0]

					if packetLength != 508:
						continue

					if sampleTime == None:
						continue

					if sampleRate == 0:
						chksum = 0
					else:
						# rewind for checksum calculation
						fh.seek(-packetLength - 4, 1)
						# calculate checksum
						chksum = 0
						for x in range(packetLength / 2 + 2):
							chksum += unpack('H', axivity_read(fh,2))[0]
						chksum %= 2 ** 16

					if chksum != 0:
						continue

					#if sessionId != self.sessionId:
					#	print "x"
					#	continue

					if ((numAxesBPS >> 4) & 15) != 3:
						print '[ERROR: num-axes not expected]'

					if (numAxesBPS & 15) == 2:
						bps = 6
					elif (numAxesBPS & 15) == 0:
						bps = 4

					timestamp = sampleTime
					freq = 3200 / (1 << (15 - sampleRate & 15))
					if freq <= 0:
						freq = 1
					offsetStart = float(-timestampOffset) / float(freq)

					#print freq

					#print offsetStart
					time0 = timestamp + timedelta(milliseconds=offsetStart)

					if average_over != False and last_time == False:
						last_time = time0

					#print time0
					#print "* - {}".format(sampleCount)
					for sample in range(sampleCount):

						x,y,z,t = 0,0,0,0

						if bps == 6:
							x = unpack('h', sampleData.read(2))[0] / 256.0
							y = unpack('h', sampleData.read(2))[0] / 256.0
							z = unpack('h', sampleData.read(2))[0] / 256.0
						elif bps == 4:
							temp = unpack('I', sampleData.read(4))[0]
							temp2 = (6 - byte(temp >> 30))
							x = short(short((ushort(65472) & ushort(temp << 6))) >> temp2) / 256.0
							y = short(short((ushort(65472) & ushort(temp >> 4))) >> temp2) / 256.0
							z = short(short((ushort(65472) & ushort(temp >> 14))) >> temp2) / 256.0

						#t = timedelta(milliseconds=(float(sample) / float(freq))*8.64) + time0
						t = sample*(timedelta(seconds=1) / 120) + time0
						#print sample, "--", t


						if average_over != False:

							temp_x.append(x)
							temp_y.append(y)
							temp_z.append(z)

							if t - last_time >= average_over:

								mean_x = np.mean(temp_x)
								mean_y = np.mean(temp_y)
								mean_z = np.mean(temp_z)
								temp_x = []
								temp_y = []
								temp_z = []

								#print last_time
								axivity_timestamps.append(last_time)
								axivity_x.append(mean_x)
								axivity_y.append(mean_y)
								axivity_z.append(mean_z)
								last_time = t


						else:
							axivity_timestamps.append(t)
							axivity_x.append(x)
							axivity_y.append(y)
							axivity_z.append(z)


				header = axivity_read(fh,2)

				n=n+1
		except IOError:
			pass

		print n

		axivity_x = np.array(axivity_x)
		axivity_y = np.array(axivity_y)
		axivity_z = np.array(axivity_z)
		axivity_timestamps = np.array(axivity_timestamps)

		print(len(axivity_x))

		channel_x.set_contents(axivity_x, axivity_timestamps)
		channel_y.set_contents(axivity_y, axivity_timestamps)
		channel_z.set_contents(axivity_z, axivity_timestamps)

		return [channel_x,channel_y,channel_z]
