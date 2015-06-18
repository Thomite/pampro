import numpy as np
import scipy as sp
from datetime import datetime, date, time, timedelta
from pampro import Bout, pampro_utilities
import copy
from struct import *
from math import *
import time
from datetime import datetime
import sys
import io
import re
import string
from scipy.io.wavfile import write
import zipfile



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
        self.sparsely_timestamped = False

    def clone(self):

        return copy.deepcopy(self)

    def set_contents(self, data, timestamps):

        self.data = data
        self.timestamps = timestamps

        self.calculate_timeframe()

    def append(self, other_channel):

        #print(self.name + " " + str(len(self.data)))
        self.data = np.concatenate((self.data, other_channel.data))
        self.timestamps = np.concatenate((self.timestamps, other_channel.timestamps))

        indices = np.argsort(self.timestamps)


        self.timestamps = np.array(self.timestamps)[indices]
        self.data = np.array(self.data)[indices]
        #print(self.name + " " + str(len(self.data)))
        #print("")
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

        print(min_value)
        print(max_value)

        ranges = []
        low = min_value
        for i in range(bins):

            if i == bins-1:
                high = max_value
            else:
                high = low+increment

            ranges.append((low, high, i))
            low += increment

        print(str(ranges))

        return self.collapse(ranges)

    def collapse(self, ranges):

        # Each range is a tuple: (>= low, <= high, replacement)

        clone = self.clone()

        for low, high, replacement in ranges:

            indices = np.where((self.data >= low) & (self.data <= high))[0]
            clone.data[indices] = replacement

        return clone

    def get_window(self, datetime_start, datetime_end):

        key_value = str(datetime_start) + "|" + str(datetime_end)
        indices = [-1]

        # If we already know what the indices are for this time range:

        try:
            indices = self.cached_indices[key_value]

        except:

            if not self.sparsely_timestamped:

                indices = self.get_data_indices(datetime_start, datetime_end)

            else:

                indices = self.get_sparse_data_indices(datetime_start, datetime_end)


            # Cache those for next time
            self.cached_indices[key_value] = indices

        return indices


    def get_data_indices(self, datetime_start, datetime_end):
        """ Returns the indices of the data array to use if every observation is timestamped """

        start = np.searchsorted(self.timestamps, datetime_start, 'left')
        end = np.searchsorted(self.timestamps, datetime_end, 'right')
        return (start, end-1)

    def get_data_index(self, timestamp):

        if timestamp < self.timestamps[0]:
            return -1
        elif timestamp > self.timestamps[-1]:
            return -1
        else:

            start = np.searchsorted(self.timestamps, timestamp, 'left')
            if self.timestamps[start] != timestamp:

                previous_timestamp = self.timestamps[start-1]
                previous_difference = self.timestamps[start]-previous_timestamp
                sample_difference = self.indices[start]-self.indices[start-1]
                per_sample_difference = previous_difference / sample_difference
                num_samples_back = int(previous_difference / per_sample_difference)

                a = max(0, self.indices[start] - num_samples_back)

                self.timestamps = np.insert(self.timestamps, start, timestamp)
                self.indices = np.insert(self.indices, start, a)

            else:
                a = self.indices[start]

            return a

    def get_sparse_data_indices(self, datetime_start, datetime_end):
        """ Returns the indices of the data array to use if it is sparsely timestamped """

        a = self.get_data_index(datetime_start)
        b = self.get_data_index(datetime_end)

        if a == -1 and b != -1:
            a = 0

        if a != -1 and b == -1:
            b = len(self.timestamps)-1

        return (a, b)

    def window_statistics(self, start_dts, end_dts, statistics):

        start_index,end_index = self.get_window(start_dts, end_dts)

        window_data = self.data[start_index:end_index]

        output_row = []
        if (end_index-start_index > 0):


            for stat in statistics:
                if stat[0] == "top_frequencies" or stat[0] == "frequency_ranges":
                    spectrum = np.fft.fft(window_data)
                    spectrum = np.array([abs(e) for e in spectrum[:int((end_index-start_index)/2)]])
                    sum_spec = sum(spectrum)
                    spectrum /= sum_spec
                    frequencies = np.fft.fftfreq(int(end_index-start_index), d=1.0/self.frequency)[:int((end_index-start_index)/2)]
                    break


            for stat in statistics:
                # Every stat is a tuple of the form ("type", [details])

                if stat[0] == "generic":
                # Example: ("generic", ["mean", "min", "sum"])

                    for val in stat[1]:
                        if val == "mean":
                            output_row.append(np.mean(window_data))
                        elif val == "sum" or stat[1] == "total":
                            output_row.append(sum(window_data))
                        elif val == "std" or stat[1] == "stdev":
                            output_row.append(np.std(window_data))
                        elif val == "min" or stat[1] == "minimum":
                            output_row.append(np.min(window_data))
                        elif val == "max" or stat[1] == "maximum":
                            output_row.append(np.max(window_data))
                        elif val == "n":
                            output_row.append(end_index-start_index)

                elif stat[0] == "cutpoints":
                # Example: ("cutpoints", [[0,10],[10,20],[20,30]])

                    sorted_vals = np.sort(window_data)
                    for low,high in stat[1]:
                        start = np.searchsorted(sorted_vals, low, 'left')
                        end = np.searchsorted(sorted_vals, high, 'right')

                        output_row.append(end-start)
                        """
                        This is the old way - much slower
                        for low,high in stat[1]:
                            indices2 = np.where((self.data[indices] >= low) & (self.data[indices] <= high))[0]
                            output_row.append(len(indices2))
                        """

                elif stat[0] == "bigrams":
                # Example: ("bigrams", [0,1,2])

                    unique_values = stat[1]
                    # 1 channel for each permutation of unique_value -> unique_value
                    pairs = {}
                    for val1 in unique_values:
                        pairs[val1] = {}
                        for val2 in unique_values:
                            pairs[val1][val2] = 0

                    # Count transitions from each unique value to the next
                    for val1, val2 in zip(window_data, window_data[1:]):
                        if val1 in unique_values and val2 in unique_values:
                            pairs[val1][val2] += 1

                    for val1 in unique_values:
                        for val2 in unique_values:
                            output_row.append(pairs[val1][val2])

                elif stat[0] == "frequency_ranges":
                # Example: ("frequency_ranges", [[0,1],[1,2],[2,3]])

                    for low,high in stat[1]:

                        start = np.searchsorted(frequencies, low, 'left')
                        end = np.searchsorted(frequencies, high, 'right')
                        index_range = np.arange(start, end-1)
                        sum_range = sum(spectrum[index_range])

                        output_row.append(sum_range)

                elif stat[0] == "top_frequencies":
                # Example: ("top_frequencies", 5)

                    sorted_spectrum = np.sort(spectrum)[::-1]
                    dom_magnitudes = sorted_spectrum[:stat[1]]

                    dom_indices = [np.where(spectrum==top)[0] for top in dom_magnitudes]
                    dom_frequencies = [frequencies[index] for index in dom_indices]

                    for freq,mag in zip(dom_frequencies,dom_magnitudes):
                        output_row.append(freq[0])
                        output_row.append(mag)

                elif stat[0] == "percentiles":
                # Example: ("percentiles", [10,20,30,40,50,60,70,80,90])

                    values = np.percentile(window_data, stat[1])
                    for v in values:
                        output_row.append(v)

                else:
                    output_row.append("Unknown statistic")
        else:

            # There was no data for the time period
            # Output -1 for each missing variable
            for i in range(self.expected_results(statistics)):
                output_row.append(-1)


        return output_row


    def expected_results(self, statistics):
        """ Calculate the number of expected results for this statistics request """

        expected = 0
        for stat in statistics:
            if stat[0] == "generic":
                expected += len(stat[1])

            elif stat[0] == "cutpoints":
                expected += len(stat[1])

            elif stat[0] == "bigrams":
                expected += len(stat[1])**2

            elif stat[0] == "frequency_ranges":
                expected += len(stat[1])

            elif stat[0] == "top_frequencies":
                expected += int(stat[1])*2

            elif stat[0] == "percentiles":
                expected += len(stat[1])

        return expected


    def build_statistics_channels(self, windows, statistics):

        channel_list = []

        for stat in statistics:
            #print(stat)
            channel_names = pampro_utilities.design_variable_names(self.name, stat)
            #print(channel_names)
            for cn in channel_names:
                channel_list.append(Channel(cn))

        num_expected_results = len(channel_list)

        for window in windows:

            results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)
            if len(results) != num_expected_results:
                raise Exception("Incorrect number of statistics yielded. {} expected, {} given. Channel: {}. Statistics: {}.".format(num_expected_results, len(results), self.name, statistics))

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

    def sliding_statistics(self, window_size, statistics=[("generic", "mean")], time_period=False):

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


    def piecewise_statistics(self, window_size, statistics=[("generic", ["mean"])], time_period=False):

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

    def bouts(self, low, high):

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

                    bouts.append(Bout.Bout(start_time, end_time))


        if state == 1:
            start_time =  self.timestamps[start_index]
            end_time = self.timestamps[end_index]
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

            start_index,end_index = self.get_window(bout.start_timestamp, bout.end_timestamp)

            #c.data[bout[2]:bout[3]] = self.data[bout[2]:bout[3]]
            c.data[indices] = self.data[start_index:end_index]

        return c

    def delete_windows(self, windows):

        for window in windows:
            start_index,end_index = self.get_window(window.start_timestamp, window.end_timestamp)

            self.data = np.delete(self.data, range(start_index, end_index), None)

            if not self.sparsely_timestamped:
                self.timestamps = np.delete(self.timestamps, range(start_index, end_index), None)

            else:
                start = np.searchsorted(self.timestamps, window.start_timestamp, 'left')
                end = np.searchsorted(self.timestamps, window.end_timestamp, 'right')
                self.timestamps = np.delete(self.timestamps, range(start,end), None)
                self.indices[start:] -= end_index-start_index
                self.indices = np.delete(self.indices, range(start,end), None)


        self.calculate_timeframe()

        self.cached_indices = {}

    def restrict_timeframe(self, start, end):

        start_index,end_index = self.get_window(start, end)

        self.set_contents(self.data[start_index:end_index], self.timestamps[start_index:end_index])


    def time_derivative(self):

        result = Channel(self.name + "_td")
        result.set_contents(np.diff(self.data), self.timestamps[:-1])
        return result

    def absolute(self):

        result = Channel(self.name + "_abs")
        result.set_contents(np.abs(self.data), self.timestamps)
        return result

    def fill(self, bout, fill_value=0):

        start_index,end_index = self.get_window(bout.start_timestamp,bout.end_timestamp)

        self.data[start_index:end_index] = fill_value

    def fill_windows(self, bouts, fill_value=0):

        for b in bouts:
            self.fill(b, fill_value = fill_value)

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

    def draw(self, axis):

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

        #    timestamps.append(timestamp)
        #    timestamp += time_resolution

        timestamps = [time_period[0] + time_resolution*x for x in range(num_epochs)]

        filled = np.empty(len(timestamps))
        filled.fill(out_value)

        print(time_period)
        print("Length of timestamps:" + str(len(timestamps)))
        print("Length of filled:" + str(len(filled)))

        result.set_contents(filled, timestamps)
    else:
        result = skeleton.clone()
        result.name = channel_name
        result.data.fill(out_value)


    for bout in bouts:
        result.fill(bout, in_value)



    return result
