import numpy as np
from datetime import datetime, date, time, timedelta
from pampropy import Bout
import copy

from struct import *
from math import *
import time
from datetime import datetime
from urllib import unquote_plus
import sys
import io



class Channel(object):

	def __init__(self, name):
		
		self.name = name
		self.size = 0
		self.timeframe = 0
		self.data = []
		self.timestamps = []
		self.annotations = []
		self.draw_properties = {}

	def clone(self):

		return copy.deepcopy(self)

	def set_contents(self, data, timestamps):
		
		self.data = data
		self.timestamps = timestamps

		self.calculate_timeframe()

	def calculate_timeframe(self):

		self.size = len(self.data)
		self.timeframe = self.timestamps[0], self.timestamps[self.size-1], (self.timestamps[self.size-1]-self.timestamps[0]), self.size

	def add_annotation(self, annotation):

		self.annotations.append(annotation)

	def add_annotations(self, annotations):

		for a in annotations:
			self.add_annotation(a)

	def normalise(self, floor=0, ceil=1):

		max_value = max(self.data)
		min_value = min(self.data)
		self.data = ((ceil - floor) * (self.data - min_value))/(max_value - min_value) + floor

	def get_window(self, datetime_start, datetime_end):

		start = np.searchsorted(self.timestamps, datetime_start, 'left')
		end = np.searchsorted(self.timestamps, datetime_end, 'right')
		return np.arange(start, end-1)

	def window_statistics(self, start_dts, end_dts, statistics):

		indices = self.get_window(start_dts, end_dts)

		pretty_timestamp = start_dts.strftime("%d/%m/%Y %H:%M:%S:%f")

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

					indices2 = np.where((self.data[indices] >= stat[0]) & (self.data[indices] < stat[1]))[0]
					output_row.append(len(indices2))

				else:
					output_row.append(-1)
		else:
			for i in range(len(statistics)):
				output_row.append(-1)	

		return output_row

	def build_statistics_channels(self, windows, statistics):

		channel_list = []
		for var in statistics:
			name = self.name
			if isinstance(var, list):
				name = self.name + "_" + str(var[0]) + "_" + str(var[1])
			else:
				name = self.name + "_" + var
			channel = Channel(name)
			channel_list.append(channel)

		for window in windows:

			results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)
			for i in range(len(results)):
				
				channel_list[i].append_data(window.start_timestamp, results[i])

		for channel in channel_list:
			channel.calculate_timeframe()
			channel.data = np.array(channel.data)
			channel.timestamps = np.array(channel.timestamps)

		return channel_list

	def append_data(self, timestamp, data_row):
		self.timestamps.append(timestamp)
		self.data.append(data_row)

	def piecewise_statistics(self, window_size, statistics=["mean"], time_period=False):

		if time_period == False:
			start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
			end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
		else:
			start = time_period[0]
			end = time_period[1]

		print start, end

		windows = []

		start_dts = start
		end_dts = start + window_size

		while start_dts < end:
			
			window = Bout.Bout(start_dts, end_dts)
			windows.append(window)

			start_dts = start_dts + window_size
			end_dts = end_dts + window_size
			
		return self.build_statistics_channels(windows, statistics)

	# def piecewise_statistics(self, window_size, statistics=["mean"], time_period=False):

		# if time_period == False:
			# start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
			# end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
		# else:
			# start = time_period[0]
			# end = time_period[1]
		# # ------------------------------

		# #print start , "---", end

		# channel_list = []
		# for var in statistics:
			# name = self.name
			# if isinstance(var, list):
				# name = self.name + "_" + str(var[0]) + "_" + str(var[1])
			# else:
				# name = self.name + "_" + var
			# channel = Channel(name)
			# channel_list.append(channel)

		# window = window_size
		# start_dts = start
		# end_dts = start + window

		# while start_dts < end:
			
			# results = self.window_statistics(start_dts, end_dts, statistics)
			# for i in range(len(results)):
				
				# channel_list[i].append_data(start_dts, results[i])

			# start_dts = start_dts + window
			# end_dts = end_dts + window

		# for channel in channel_list:
			# channel.calculate_timeframe()
			# channel.data = np.array(channel.data)
			# channel.timestamps = np.array(channel.timestamps)

		# return channel_list


	def summary_statistics(self, statistics=["mean"]):

		windows = [Bout.Bout(self.timeframe[0], self.timeframe[1])]
		#results = self.window_statistics(self.timeframe[0], self.timeframe[1], statistics)

		return self.build_statistics_channels(windows, statistics)

	def bouts(self, low, high, minimum_length=0):

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
					if (end_index - start_index + 1 >= minimum_length):

						bouts.append(Bout.Bout(self.timestamps[start_index], self.timestamps[end_index]))	
						
	
		return bouts

	def subset_using_bouts(self, bout_list, name):
		# Given a list of bouts, create a new channel from this taking only the data from inside those bouts
		c = Channel(name)

		c.set_contents(np.zeros(self.size), self.timestamps)

		#print(self.data[2345])

		for bout in bout_list:
			#print(bout)

			indices = self.get_window(bout[0], bout[1])

			#c.data[bout[2]:bout[3]] = self.data[bout[2]:bout[3]]
			c.data[indices] = self.data[indices]

		return c

	def restrict_timeframe(self, start, end):

		indices = self.get_window(start,end)

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

def load_channels(source, source_type, datetime_format="%d/%m/%Y %H:%M:%S:%f", datetime_column=0, use_columns=False):

	if (source_type == "Actiheart"):

		activity, ecg  = np.loadtxt(source, delimiter=',', unpack=True, skiprows=15, usecols=[1,2])

		first_lines = []
		f = open(source, 'r')
		for i in range(0,30):
			s = f.readline().strip()
			first_lines.append(s)
		f.close()

		time1 = datetime.strptime(first_lines[20].split(",")[0], "%H:%M:%S")
		time2 = datetime.strptime(first_lines[21].split(",")[0], "%H:%M:%S")

		line8 = first_lines[8]
		test = line8.split(",")
		dt = datetime.strptime(test[1], "%d-%b-%Y  %H:%M")
		delta = time2 - time1

		timestamp_list = []
		for i in range(0,len(activity)):
			timestamp_list.append(dt)
			dt = dt + delta

		timestamps = np.array(timestamp_list)

		indices1 = np.where(ecg > 1)
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


		line2 = first_lines[2]
		test = line2.split(" ")
		timeval = datetime.strptime(test[-1], "%H:%M:%S")
		timeadd = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
	
		line3 = first_lines[3]
		test = line3.split(" ")
		time = timeadd + datetime.strptime(test[-1], "%m-%d-%Y")

		line4 = first_lines[4]
		test = line4.split(" ")
		delta = datetime.strptime(test[-1], "%H:%M:%S")
		timeadd = timedelta(hours=delta.hour, minutes=delta.minute, seconds=delta.second)
		
		count_list = []
		timestamp_list = []

		line = f.readline().strip()
		while (len(line) > 0):
	
			counts = line.split()
			for c in counts:
				count_list.append(int(c))
				timestamp_list.append(time)
				time = time + timeadd

			line = f.readline().strip()
		f.close()

		timestamps = np.array(timestamp_list)
		counts = np.array(count_list)
		
		print timestamps[0], timestamps[-1]
		print sum(counts)

		chan = Channel("AG_Counts")
		chan.set_contents(counts, timestamps)

		return [chan]

	elif (source_type == "CSV"):

		f = open(source, 'r')
		s = f.readline().strip()
		f.close()

		test = s.split(",")

		source_split = source.split("/")
		
		data = np.loadtxt(source, delimiter=',', skiprows=1, dtype='str')
		
		#print(data.shape)
		#print(data[:,0])

		timestamps = []
		for date_row in data[:,datetime_column]:
			timestamps.append(datetime.strptime(date_row, datetime_format))
		timestamps = np.array(timestamps)

		data_columns = list(range(0,len(test)))
		del data_columns[datetime_column]
		print data_columns
		channels = []
		for col in data_columns:
			print col
			name = source_split[-1] + " - " + test[col]
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

		try:
			header = axivity_read(fh,2)
			while len(header) == 2:
				
				if header == 'MD':
					print 'MD'
					axivity_parse_header(fh)
				elif header == 'UB':
					print 'UB'
					blockSize = unpack('H', axivity_read(fh,2))[0]
				elif header == 'SI':
					print 'SI'
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
