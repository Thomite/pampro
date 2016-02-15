import numpy as np
import scipy as sp
from datetime import datetime, date, time, timedelta
from pampro import Time_Series, Bout, pampro_utilities
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
        self.missing_value = False

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

    def inherit_time_properties(self, channel):

        self.timestamps = channel.timestamps
        self.missing_value = channel.missing_value

        if channel.sparsely_timestamped:
            self.sparsely_timestamped = True
            self.indices = channel.indices
            self.cached_indices = channel.cached_indices

        try:
            self.frequency = channel.frequency
        except:
            pass

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

        #print(min_value)
        #print(max_value)

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

        key_a = str(datetime_start)
        key_b = str(datetime_end)
        indices = [-1]

        # If we already know what the indices are for these timestamps:
        try:
            index_a = self.cached_indices[key_a]
            index_b = self.cached_indices[key_b]
            indices = (index_a, index_b)

        except:

            if not self.sparsely_timestamped:

                indices = self.get_data_indices(datetime_start, datetime_end)

            else:
                for timestamp in [datetime_start, datetime_end]:
                    self.ensure_timestamped_at(timestamp)

                indices = self.get_data_indices(datetime_start, datetime_end)


            # Cache those for next time
            self.cached_indices[key_a] = indices[0]
            self.cached_indices[key_b] = indices[1]

        return indices

    def get_data_indices(self, datetime_start, datetime_end):
        """ Returns the indices of the data array to use if every observation is timestamped """

        if datetime_start > self.timestamps[-1] or datetime_end < self.timestamps[0]:
            start = -1
            end = -1
        else:

            if self.sparsely_timestamped:

                if datetime_start < self.timestamps[0]:
                    start = -1
                else:

                    start = np.searchsorted(self.timestamps, datetime_start, 'left')
                    try:
                        start = self.indices[max(0,start)]
                    except:
                        print("!!!!!!!! EXCEPTION !!!!!!!!!!")
                        print(start, end)
                        print(datetime_start, datetime_end)
                        print("!!!!!!!! EXCEPTION !!!!!!!!!!")

                if datetime_end < self.timestamps[0]:
                    end = -1
                else:
                    end = np.searchsorted(self.timestamps, datetime_end, 'left')
                    end = self.indices[min(len(self.indices)-1,end)]

            else:

                if datetime_start < self.timestamps[0]:
                    start = -1
                else:
                    start = np.searchsorted(self.timestamps, datetime_start, 'left')

                if datetime_end < self.timestamps[0]:
                    end = -1
                else:
                    end = np.searchsorted(self.timestamps, datetime_end, 'left')


        if start == -1 and end != -1:
            start = 0

        if start != -1 and end == -1:
            end = len(self.timestamps)

        return (start, end)

    def inject_timestamp_index(self, timestamp, index):

        i = np.searchsorted(self.indices, index, "left")
        if self.indices[i] != index:

            self.timestamps = np.insert(self.timestamps, i, timestamp)
            self.indices = np.insert(self.indices, i, index)




    def ensure_timestamped_at(self, timestamp):
        """ Guarantees a timestamp will be in the timestamps array """
        # Is this check necessary?
        if timestamp >= self.timestamps[0] and timestamp < self.timestamps[-1]:

            start = np.searchsorted(self.timestamps, timestamp, 'left')

            # If this timestamp didn't exactly match an existing timestamp in the array
            if self.timestamps[start] != timestamp:

                # self.timestamps[start] > desired timestamp - but by how much?
                overshoot = (self.timestamps[start]-timestamp).total_seconds()
                # seconds difference * num samples per second = sample difference
                num_samples_back = overshoot*self.frequency
                a = int(self.indices[start] - num_samples_back)
                b = a + 1

                a_timestamp = self.infer_timestamp(a)
                b_timestamp = self.infer_timestamp(b)

                self.inject_timestamp_index(a_timestamp, a)
                self.inject_timestamp_index(b_timestamp, b)
                #self.timestamps = np.insert(self.timestamps, start, [a_timestamp, b_timestamp])
                #self.indices = np.insert(self.indices, start, [a,b])



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

        # Allow direct indexing numerically, or by timestamp
        index_type = str(type(start_dts))
        if "int" in index_type:
            start_index,end_index = start_dts, end_dts
        else:
            start_index,end_index = self.get_window(start_dts, end_dts)

        window_data = self.data[start_index:end_index]
        #print(start_dts,end_dts,window_data)
        initial_n = len(window_data)
        missing_n = 0

        if self.missing_value is not False:
            window_data = window_data[window_data != self.missing_value]
            missing_n = initial_n - len(window_data)

        output_row = []
        data_found = len(window_data) > 0
        #if (len(window_data) > 0):

        # Cache the frequency spectrum, at least 1 statistic needs it
        for stat in statistics:
            if data_found and (stat[0] == "top_frequencies" or stat[0] == "frequency_ranges"):
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

                        if data_found:
                            output_row.append(np.mean(window_data))
                        else:
                            output_row.append(-1)

                    elif val == "sum" or stat[1] == "total":

                        if data_found:
                            output_row.append(sum(window_data))
                        else:
                            output_row.append(-1)

                    elif val == "std" or stat[1] == "stdev":

                        if data_found:
                            output_row.append(np.std(window_data))
                        else:
                            output_row.append(-1)

                    elif val == "min" or stat[1] == "minimum":

                        if data_found:
                            output_row.append(np.min(window_data))
                        else:
                            output_row.append(-1)

                    elif val == "max" or stat[1] == "maximum":

                        if data_found:
                            output_row.append(np.max(window_data))
                        else:
                            output_row.append(-1)

                    elif val == "n":

                        output_row.append(len(window_data))

                    elif val == "missing":

                        output_row.append(missing_n)


            elif stat[0] == "cutpoints":
            # Example: ("cutpoints", [[0,10],[10,20],[20,30]])

                if data_found:
                    sorted_vals = np.sort(window_data)
                    for low,high in stat[1]:
                        start = np.searchsorted(sorted_vals, low, 'left')
                        end = np.searchsorted(sorted_vals, high, 'right')

                        output_row.append(end-start)
                else:
                    for i in range(len(stat[1])):
                        output_row.append(0)

            elif stat[0] == "bigrams":
            # Example: ("bigrams", [0,1,2])

                unique_values = stat[1]

                if data_found:

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

                else:

                    for i in range(len(unique_values)**2):
                        output_row.append(0)

            elif stat[0] == "frequency_ranges":
            # Example: ("frequency_ranges", [[0,1],[1,2],[2,3]])


                for low,high in stat[1]:

                    if data_found:
                        start = np.searchsorted(frequencies, low, 'left')
                        end = np.searchsorted(frequencies, high, 'right')
                        index_range = np.arange(start, end-1)
                        sum_range = sum(spectrum[index_range])

                        output_row.append(sum_range)

                    else:
                        output_row.append(-1)

            elif stat[0] == "top_frequencies":
            # Example: ("top_frequencies", 5)

                if data_found:
                    sorted_spectrum = np.sort(spectrum)[::-1]
                    dom_magnitudes = sorted_spectrum[:stat[1]]

                    dom_indices = [np.where(spectrum==top)[0] for top in dom_magnitudes]
                    dom_frequencies = [frequencies[index] for index in dom_indices]

                    for freq,mag in zip(dom_frequencies,dom_magnitudes):
                        output_row.append(freq[0])
                        output_row.append(mag)
                else:
                    for i in range(stat[1]*2):
                        output_row.append(-1)

            elif stat[0] == "percentiles":
            # Example: ("percentiles", [10,20,30,40,50,60,70,80,90])

                if data_found:
                    values = np.percentile(window_data, stat[1])
                    for v in values:
                        output_row.append(v)
                else:
                    for i in range(len(stat[1])):
                        output_row.append(-1)

            else:
                output_row.append("Unknown statistic")

        """
        else:

        # There was no data for the time period
        # Output -1 for each missing variable
        for i in range(self.expected_results(statistics)):
            output_row.append(-1)
        """

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


    def build_statistics_channels(self, windows, statistics, name=""):
        """ Describe the contents of this channel in the given time windows using the given statistics  """

        using_indices = True
        if str(type(windows[0])) == "<class 'pampro.Bout.Bout'>":
            using_indices = False

        channel_list = []

        for stat in statistics:

            channel_names = pampro_utilities.design_variable_names(self.name, stat)

            for cn in channel_names:
                channel_list.append(Channel(cn))

        num_expected_results = len(channel_list)

        for window in windows:

            if using_indices:
                results = self.window_statistics(window[0], window[1], statistics)

            else:

                results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)

            if len(results) != num_expected_results:

                raise Exception("Incorrect number of statistics yielded. {} expected, {} given. Channel: {}. Statistics: {}.".format(num_expected_results, len(results), self.name, statistics))

            if using_indices:
                for i in range(len(results)):
                    channel_list[i].append_data(window[0], results[i])

            else:

                for i in range(len(results)):
                    channel_list[i].append_data(window.start_timestamp, results[i])

        for channel in channel_list:
            channel.missing_value = -1
            channel.data = np.array(channel.data)
            channel.timestamps = np.array(channel.timestamps)
            channel.calculate_timeframe()

        ts = Time_Series.Time_Series(name)
        ts.add_channels(channel_list)
        return ts

    def append_data(self, timestamp, data_row):
        """Append a single observation to the end of the timestamp and data arrays. """

        self.timestamps.append(timestamp)
        self.data.append(data_row)

    def infer_timestamp(self, index):
        """ Given an index of the data array, approximate its timestamp using the sparse timestamps around it """

        start = np.searchsorted(self.indices, index, 'left')
        #print("infer_timestamp | start:", start)
        if self.indices[start] == index:

            return self.timestamps[start]

        elif start == len(self.indices):
            return self.timestamps[-1]

        else:
            # it's before "start" & after "start"-1
            index_difference = self.indices[start]-index
            time_difference = index_difference * timedelta(seconds=1)/self.frequency
            #print("infer_timestamp | index_difference:", index_difference)
            #print("infer_timestamp | time_difference:", time_difference)
            return self.timestamps[start] - time_difference

    def sliding_statistics(self, window_size, statistics=[("generic", ["mean"])], time_period=False, name=""):

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

        channels = self.build_statistics_channels(windows, statistics, name=name)

        for c in channels:
            c.timestamps = self.timestamps
            c.calculate_timeframe()

        return channels


    def piecewise_statistics(self, window_size, statistics=[("generic", ["mean"])], time_period=False, name=""):

        if time_period == False:
            start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
            end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
        else:
            start = time_period[0]
            end = time_period[1]

        #print("Piecewise statistics: {}".format(self.name))
        windows = []

        # If we passed a timedelta object as our window size
        if str(type(window_size)) == "<class 'datetime.timedelta'>":

            start_dts = start
            end_dts = start + window_size

            while start_dts < end:

                window = Bout.Bout(start_dts, end_dts)
                windows.append(window)

                start_dts = start_dts + window_size
                end_dts = end_dts + window_size

        # Else if we passed an integer as our window size
        elif str(type(window_size)) == "<class 'int'>":

            windows = [[i,i+window_size] for i in range(0,len(self.data),window_size)]



        return self.build_statistics_channels(windows, statistics, name=name)


    def summary_statistics(self, statistics=[("generic", ["mean"])], time_period=False, name=""):

        if time_period == False:
            windows = [Bout.Bout(self.timeframe[0], self.timeframe[1]+timedelta(days=1111))]
        else:
            windows = [Bout.Bout(time_period[0],time_period[1])]

        return self.build_statistics_channels(windows, statistics, name=name)

    def bouts(self, low, high):

        # 0 indicates not currently in a bout, 1 indicates in
        state = 0

        # start_index will be the variable that tracks the start of bouts
        start_index = 0

        # end_index will track the end
        end_index = 1
        bouts = []

        for i, value in enumerate(self.data):

            # If we're currently not in a bout
            if state == 0:

                # And if this value is in the range we want
                if value >= low and value <= high:

                    # Start a bout
                    state = 1
                    start_index = i
                    end_index = i

            # Else, we're currently in a bout
            else:

                # And this value is in the range we want
                if value >= low and value <= high:

                    # So the bout expands to include this value
                    end_index = i

                # But this value is out of our range
                else:

                    # So we end the bout at the previous value
                    state = 0

                    start_time =  self.timestamps[start_index]
                    end_time = self.timestamps[end_index]
                    if end_index+1 < self.size:
                        end_time = self.timestamps[end_index+1]

                    bouts.append(Bout.Bout(start_time, end_time))


        # Bout finishes at end of file
        if state == 1:
            start_time =  self.timestamps[start_index]
            end_time = self.timestamps[end_index]

            if not self.sparsely_timestamped:
                end_time += self.timestamps[-1]-self.timestamps[-2]

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

            start_index,end_index = self.bout.start_timestamp, bout.end_timestamp

            #c.data[bout[2]:bout[3]] = self.data[bout[2]:bout[3]]
            c.data[indices] = self.data[start_index:end_index]

        return c

    def delete_windows(self, windows, missing_value = -111):

        # New approach - don't delete the data, mask it with a set value
        # Then when we summarise, check if data has been masked (missing_value is not False)
        # Then analyse only unmasked data
        self.fill_windows(windows, fill_value=missing_value)
        self.missing_value = missing_value

    def restrict_timeframe(self, start, end):

        # Don't delete the data anymore, just mask the data outside of the range
        bout1 = Bout.Bout(self.timestamps[0]-timedelta(days=1), start-timedelta(microseconds=1))
        bout2 = Bout.Bout(end+timedelta(microseconds=1), self.timestamps[-1]+timedelta(days=1))

        self.delete_windows([bout1, bout2])
        """
        start_index,end_index = self.get_window(start, end)

        if not self.sparsely_timestamped:
            self.set_contents(self.data[start_index:end_index], self.timestamps[start_index:end_index])
        else:
            print("WARNING - can't restrict timeframe on sparsely timestamped data yet")
            pass
        """

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
        #print(start_index, end_index)
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
