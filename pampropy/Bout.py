import copy
from datetime import datetime, date, time, timedelta
import numpy as np

class Bout(object):

	def __init__(self, start_timestamp, end_timestamp, label=""):
		
		self.label = label
		self.start_timestamp = start_timestamp
		self.end_timestamp = end_timestamp
		self.draw_properties = {'lw':0, 'alpha':0.8, 'facecolor':[0.78431,0.78431,0.78431]}

	def contains(self, timepoint):

		return timepoint >= self.start_timestamp and timepoint <= self.end_timestamp

	def overlaps(self, other):
		
		return self.contains(other.start_timestamp) or self.contains(other.end_timestamp) or other.contains(self.start_timestamp) or other.contains(self.end_timestamp)

	def intersection(self, other):

		return Bout(max(self.start_timestamp, other.start_timestamp), min(self.end_timestamp, other.end_timestamp))



def total_time(bouts):

	total = timedelta()
	for bout in bouts:
		total += bout.end_timestamp - bout.start_timestamp

	return total

def bout_list_intersection(bouts_a, bouts_b):

	intersection = []

	for bout_a in bouts_a:
		for bout_b in bouts_b:

			if bout_a.overlaps(bout_b):

				bout_c = bout_a.intersection(bout_b)
				#print bout_a.start_timestamp, bout_a.end_timestamp, "overlaps", bout_b.start_timestamp, bout_b.end_timestamp, "=", bout_c.start_timestamp, bout_c.end_timestamp
				intersection.append(bout_c)
		
	#print "Bouts a", len(bouts_a)
	#print "Bouts b", len(bouts_b)
	#print "Intersections", len(intersection)

	#for i in intersection:
	#	i.draw_properties = {"lw":1, "facecolor":[1.0,0.2,0.2], "alpha":0.38}

	return intersection

def bout_list_union(bouts_a, bouts_b):

	# TODO: Return the union of two bout lists

	return -1

def time_period_minus_bouts(time_period, bouts):

	# Currently assumes bouts are sorted in chronological order
	# Add sort in here for safety?

	results = []

	start = time_period[0]

	for bout in bouts:

		results.append(Bout(start, bout.start_timestamp))
		start = bout.end_timestamp

	results.append(Bout(start, time_period[1]))

	#for r in results:
	#	r.draw_properties = {'lw':0, "alpha":0.75, "facecolor":[0.95,0.1,0.1]}

	return results

def bout_confusion_matrix(a1_bouts, b1_bouts, time_period):

	#Calculate time spent in and out of bouts_a and bouts_b with respect to each other, within time_period


	a0_bouts = time_period_minus_bouts(time_period, a1_bouts)
	b0_bouts = time_period_minus_bouts(time_period, b1_bouts)
	
	a0_b0_bouts = bout_list_intersection(a0_bouts, b0_bouts)
	a1_b0_bouts = bout_list_intersection(a1_bouts, b0_bouts)
	a0_b1_bouts = bout_list_intersection(a0_bouts, b1_bouts)
	a1_b1_bouts = bout_list_intersection(a1_bouts, b1_bouts)

	a0_b0_time = total_time(a0_b0_bouts)
	a1_b0_time = total_time(a1_b0_bouts)
	a0_b1_time = total_time(a0_b1_bouts)
	a1_b1_time = total_time(a1_b1_bouts)

	results = {}
	results["a0_b0_time"] = a0_b0_time
	results["a1_b0_time"] = a1_b0_time
	results["a0_b1_time"] = a0_b1_time
	results["a1_b1_time"] = a1_b1_time
	results["a0_b0_bouts"] = a0_b0_bouts
	results["a1_b0_bouts"] = a1_b0_bouts
	results["a0_b1_bouts"] = a0_b1_bouts
	results["a1_b1_bouts"] = a1_b1_bouts
	
	return results

def limit_to_lengths(bouts, min_length=False, max_length=False, sorted=False):
	
	within_length = []
	for bout in bouts:
		#bout_length = bout.end_timestamp - bout.start_timestamp
		if (min_length==False or bout.length >= min_length) and (max_length==False or bout.length <= max_length):
			within_length.append(bout)

		else:
			if sorted:
				break
	
	return within_length

def cache_lengths(bouts):

	for bout in bouts:
		bout.length = bout.end_timestamp - bout.start_timestamp


def write_bouts_to_file(bouts, file_target):

	file_output = open(file_target,"w")

	for bout in bouts:

		pretty_start = str(bout.start_timestamp.strftime("%d/%m/%Y %H:%M:%S:%f"))
		pretty_end = str(bout.end_timestamp.strftime("%d/%m/%Y %H:%M:%S:%f"))
		file_output.write(pretty_start + "," + pretty_end + "\n")

	file_output.close()

def read_bouts(file_source, date_format="%d/%m/%Y %H:%M:%S:%f"):


	data = np.loadtxt(file_source, delimiter=',', dtype='str')

	bouts = []
	for start,end in zip(data[:,0],data[:,1]):
		bouts.append(Bout(datetime.strptime(start, date_format), datetime.strptime(end, date_format)))
	
	return bouts
