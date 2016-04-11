import copy
from datetime import datetime, date, time, timedelta
import numpy as np

class Bout(object):

	def __init__(self, start_timestamp, end_timestamp):

		#self.label = label
		self.start_timestamp = start_timestamp
		self.end_timestamp = end_timestamp
		self.length = self.end_timestamp - self.start_timestamp
		self.draw_properties = {'lw':0, 'alpha':0.8, 'facecolor':[0.78431,0.78431,0.78431]}

	def contains(self, timepoint):
		""" Returns true if a time value is contained by the start and end of this bout. """

		return timepoint >= self.start_timestamp and timepoint <= self.end_timestamp

	def overlaps(self, other):
		""" Returns true if there is some overlapping time between this Bout and the other. """

		inter = self.intersection(other)

		overlap_type = str(type(inter.length))

		if "int" in overlap_type:
			return inter.length > 0
		else:
			return inter.length > timedelta(microseconds=0)

	def intersection(self, other):
		""" Create a Bout object that represents the overlap of this Bout object and another. """

		return Bout(max(self.start_timestamp, other.start_timestamp), min(self.end_timestamp, other.end_timestamp))

	def approximate_timestamps(self, channel):
		"""If start_timestamp and end_timestamp are indices, convert them to timestamps. """

		self.start_timestamp = channel.infer_timestamp(self.start_timestamp)
		self.end_timestamp = channel.infer_timestamp(self.end_timestamp)

	def __str__(self):
		""" Printing a Bout yields a string of the form: start -> end (duration) """
		return str(self.start_timestamp) + " -> " + str(self.end_timestamp) + " (" + str(self.length) + ")"

def approximate_timestamps(bouts, channel):

	for b in bouts:
		b.approximate_timestamps(channel)

def total_time(bouts):
	""" Calculate the total time contained by the given list of Bouts. Returns a datetime.timedelta object. """

	total = timedelta(minutes=0)
	for bout in bouts:
		total += bout.end_timestamp - bout.start_timestamp

	return total

def bout_list_intersection(bouts_a, bouts_b):

	intersection = []

	if len(bouts_a) > 0 and len(bouts_b) > 0:
		for bout_a in bouts_a:
			for bout_b in bouts_b:

				if bout_a.overlaps(bout_b):

					bout_c = bout_a.intersection(bout_b)
					intersection.append(bout_c)

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

	""" Calculate time spent in and out of bouts_a and bouts_b with respect to each other, within time_period """


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
	""" Given a list of Bouts, return those whose length is >= min_length and <= max_length. """

	within_length = []
	for bout in bouts:

		if (min_length==False or bout.length >= min_length) and (max_length==False or bout.length <= max_length):
			within_length.append(bout)

		else:
			if sorted:
				break

	return within_length

def cache_lengths(bouts):
	pass



def write_bouts_to_file(bouts, file_target, date_format="%d/%m/%Y %H:%M:%S:%f"):
	""" Write the given list of Bouts to a file, 1 bout per row. """

	file_output = open(file_target, "w")

	for bout in bouts:
		# Format the timestamps as requested
		pretty_start = str(bout.start_timestamp.strftime(date_format))
		pretty_end = str(bout.end_timestamp.strftime(date_format))
		file_output.write(pretty_start + "," + pretty_end + "\n")

	file_output.close()

def read_bouts(file_source, date_format="%d/%m/%Y %H:%M:%S:%f"):
	""" Read a list of Bouts from a file, 1 bout per row. """

	data = np.loadtxt(file_source, delimiter=',', dtype='S').astype("U")

	bouts = []
	for start,end in zip(data[:,0],data[:,1]):
		bouts.append(Bout(datetime.strptime(start, date_format), datetime.strptime(end, date_format)))

	return bouts
