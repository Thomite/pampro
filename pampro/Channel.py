import numpy as np
from datetime import datetime, date, time, timedelta
import copy
from struct import *
from math import *
import sys
import re
from scipy.io.wavfile import write
import zipfile
from collections import OrderedDict
from scipy.interpolate import interp1d

from pampro import Time_Series, Bout, pampro_utilities, time_utilities

from bisect import bisect_left, bisect_right


class Channel(object):

    def __init__(self, name):

        self.name = name
        self.size = 0
        self.timeframe = (0,0)
        self.time_period = (0,0)
        self.data = []
        self.timestamps = []
        self.indices = []
        self.annotations = []
        self.draw_properties = {}
        self.cached_indices = {}
        self.timestamp_policy = "normal" # sparse, offset
        self.missing_value = False


    def clone(self):
        """ Return an independent copy of this Channel. """

        return copy.deepcopy(self)

    def set_contents(self, data, timestamps, timestamp_policy="normal"):
        """ Override current contents of data and timestamp arrays, and update timeframe accordingly. """

        self.data = data
        self.timestamps = timestamps

        self.timestamp_policy = timestamp_policy
        self.calculate_timeframe()

        self.determine_appropriate_methods()

    def determine_appropriate_methods(self):
        """
        Create some interface shortcuts to certain methods to optimise speed.
        """

        # This allows us to call self.get_index_appropriately() rather than be slowed down by various if statements.
        get_index_methods = {"normal":self.get_data_index, "sparse":self.get_sparse_data_index, "offset":self.get_offset_data_index}
        self.get_index_appropriately = get_index_methods[self.timestamp_policy]

    def resample(self, frequency):
        """
        Resample channel data to a given frequency
        """

        # This code will only produce sensible results for offset data right now!
        if self.timestamp_policy == "offset":

            # Yields a function that can be called with a new timestamp value
            func = interp1d(self.timestamps, self.data)

            # Every offset value from 0 to the highest offset seen, in increments of delta
            # Where delta is in milliseconds (eg 100 Hz = 10 milliseconds)
            delta = int((timedelta(seconds=1)/frequency).microseconds/1000)

            new_timestamps = np.arange(0, max(self.timestamps), delta)

            # func is a function, so we just give it new hypothetical offsets
            new_data = func(new_timestamps)

            self.set_contents(new_data, new_timestamps, timestamp_policy=self.timestamp_policy)
            self.frequency = frequency
        else:
            print("NOPE.")

    def append(self, other_channel):
        """ Take the data and timestamps from another Channel and incorporate them into this one. """

        self.data = np.concatenate((self.data, other_channel.data))
        self.timestamps = np.concatenate((self.timestamps, other_channel.timestamps))

        #indices = np.argsort(self.timestamps)

        #self.timestamps = np.array(self.timestamps)[indices]
        #self.data = np.array(self.data)[indices]

        #self.calculate_timeframe()

    def calculate_timeframe(self):
        """ Update timeframe and time_period variables to reflect start and end of timestamps. """

        self.size = len(self.data)

        if self.timestamp_policy == "normal":

            self.timeframe = self.timestamps[0], self.timestamps[-1]

        elif self.timestamp_policy == "sparse":

            self.timeframe = self.timestamps[0], self.timestamps[-1]

        elif self.timestamp_policy == "offset":

            self.timeframe = self.start + self.timestamps[0]*timedelta(microseconds=1000), self.start + self.timestamps[-1]*timedelta(microseconds=1000)

        self.time_period = self.timeframe # Sick of getting these the wrong way around!

    def inherit_time_properties(self, channel):
        """ Make this Channel inherit all the time properties of the given Channel. """

        self.timestamps = channel.timestamps
        self.missing_value = channel.missing_value
        self.timestamp_policy = channel.timestamp_policy
        self.indices = channel.indices
        self.cached_indices = channel.cached_indices
        self.timeframe = channel.timeframe
        self.time_period = channel.time_period

        try:
            self.frequency = channel.frequency
        except:
            pass

        if channel.timestamp_policy == "offset":
            self.start = channel.start

        # If we inherit timestamps, we need to inherit the appropriate methods to interpret them!
        self.determine_appropriate_methods()

    def normalise(self, floor=0, ceil=1):

        max_value = max(self.data)
        min_value = min(self.data)
        self.data = ((ceil - floor) * (self.data - min_value))/(max_value - min_value) + floor

    def collapse_auto(self, bins=10):

        max_value = max(self.data)
        min_value = min(self.data)
        increment = float(max_value - min_value)/float(bins)

        ranges = []
        low = min_value
        for i in range(bins):

            if i == bins-1:
                high = max_value
            else:
                high = low+increment

            ranges.append((low, high, i))
            low += increment

        #print(str(ranges))

        return self.collapse(ranges)

    def collapse(self, ranges):
        """ Replace a range of data with a static value. """

        # Each range is a tuple: (>= low, <= high, replacement)

        clone = self.clone()

        for low, high, replacement in ranges:

            indices = np.where((self.data >= low) & (self.data <= high))[0]
            clone.data[indices] = replacement

        return clone

    def get_index(self, datetimestamp):
        """
        Return the data index of the given datetimestamp.
        If it has already been cached, return the cached value.
        Otherwise, call the appropriate method based on timestamp_policy, cache the result, and return it.
        """

        if datetimestamp < self.time_period[0] or datetimestamp > self.time_period[1]:
            return -1
        else:

            try:
                i = self.cached_indices[datetimestamp]
            except:
                i = self.get_index_appropriately(datetimestamp)
                self.cached_indices[datetimestamp] = i

            return i

    def get_window(self, datetime_start, datetime_end):
        """
        Return the indices of the data array that contain the given timestamps
        """

        # Making this more straightforward with the 6 scenarios that can actually occur
        # a
        if datetime_start <= self.time_period[0] and datetime_end > self.time_period[0] and datetime_end <= self.time_period[1]:
            start = 0
            end = self.get_index(datetime_end)
        # b
        elif datetime_start >= self.time_period[0] and datetime_start <= self.time_period[1] and datetime_end >= self.time_period[0] and datetime_end <= self.time_period[1]:
            start = self.get_index(datetime_start)
            end = self.get_index(datetime_end)
        # c
        elif datetime_start >= self.time_period[0] and datetime_start < self.time_period[1] and datetime_end >= self.time_period[1]:
            start = self.get_index(datetime_start)
            end = len(self.data)
        # d
        elif datetime_start <= self.time_period[0] and datetime_end >= self.time_period[1]:
            start = 0
            end = len(self.data)
        # e and f
        elif datetime_end <= self.time_period[0] or datetime_start >= self.time_period[1]:
            start = -1
            end = -1
        else:
            raise Exception("Corner case in get_window() of {}.\nChannel time period: {} to {}.\nQuery: {} to {}.".format(self.name, self.time_period[0], self.time_period[1], datetime_start, datetime_end))

        return (start,end)



    def get_data_index(self, datetimestamp):
        """
        Returns the indices of the data array to use if every observation is timestamped
        """

        index = bisect_left(self.timestamps, datetimestamp)

        return index

    def get_sparse_data_index(self, datetimestamp):
        """ Returns the indices of the data array to use if it is sparsely timestamped """

        self.ensure_timestamped_at(datetimestamp)

        search = bisect_left(self.timestamps, datetimestamp)
        index = self.indices[max(0,search)]

        return index

    def get_offset_data_index(self, datetimestamp):

        start_index = (datetimestamp - self.time_period[0])/timedelta(microseconds=1000)

        index = bisect_left(self.timestamps, start_index)

        return index

    def inject_timestamp_index(self, timestamp, index):
        """ Add a new timestamp pointing at the given index in the data array. Used to be more specific with timestamps when data is sparsely timestamped. """

        i = bisect_left(self.indices, index)
        if self.indices[i] != index:

            self.timestamps = np.insert(self.timestamps, i, timestamp)
            self.indices = np.insert(self.indices, i, index)

    def ensure_timestamped_at(self, timestamp):
        """ Guarantees a timestamp will be in the timestamps array """


        # Is this check necessary?
        if timestamp >= self.timestamps[0] and timestamp < self.timestamps[-1]:

            start = bisect_left(self.timestamps, timestamp)

            # If this timestamp didn't exactly match an existing timestamp in the array
            # And the index is a useable range in the timestamp array
            if self.timestamps[start] != timestamp and (start > 0) and (start < len(self.timestamps)):

                try:
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

                except:
                    pass

    def window_statistics(self, start_dts, end_dts, statistics):
        """ Summarise the data between these timestamps using the statistics listed. """

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
                frequencies = np.fft.fftfreq(len(window_data), d=1.0/self.frequency)

                frequencies = frequencies[np.where(frequencies >= 0)]
                magnitudes = np.abs(spectrum[np.where(frequencies >= 0)])
                magnitudes[0] = 0

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
                        start = bisect_left(sorted_vals, low)
                        end = bisect_right(sorted_vals, high)

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
                        start = bisect_left(frequencies, low)
                        end = bisect_left(frequencies, high)
                        index_range = np.arange(start, end-1)
                        sum_range = sum(magnitudes[index_range])

                        output_row.append(sum_range)

                    else:
                        output_row.append(-1)

            elif stat[0] == "top_frequencies":
            # Example: ("top_frequencies", 5)

                if data_found:
                    sorted_spectrum = np.sort(magnitudes)[::-1]
                    dom_magnitudes = sorted_spectrum[:stat[1]]

                    dom_indices = [np.where(magnitudes==top)[0] for top in dom_magnitudes]
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

            elif stat[0] == "bouts":
            # Example: ("bouts", [(0,99),(100,200)])
            # Example 2: ("bouts", [(0,99,5),(100,200,5)])

                for b in stat[1]:

                    # Earlier in build_statistics_channels(), all bouts for this were extracted across the channel
                    # They were indexed in a dictionary using this key format
                    key = "{}_{}".format(b[0], b[1])

                    if len(b) == 3:
                        key += "_{}".format(b[2])

                    # Bout object to represent window currently being summarised
                    bout_window = Bout.Bout(start_dts, end_dts)

                    # Get pre-computed list of bouts for the whole channel
                    all_bouts = self.bouts_cache[key]

                    # Prune the list to only those that overlap this window
                    relevant_bouts = [b for b in all_bouts if b.overlaps(bout_window)]

                    # Get the exact intersection of the relevant bouts with this window
                    intersection = Bout.bout_list_intersection([bout_window], relevant_bouts)

                    # Two variables - total time overlapping the window, and number of bouts it contains
                    sum_seconds = Bout.total_time(intersection).total_seconds()
                    num_bouts = len(intersection)

                    output_row.append(sum_seconds)
                    output_row.append(num_bouts)

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

            elif stat[0] == "bouts":
                expected += len(stat[1])*2
                print("SADDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
            else:
                print(stat)

        return expected

    def build_statistics_channels(self, windows, statistics, name=""):
        """ Describe the contents of this channel in the given time windows using the given statistics  """

        channel_list = []

        for stat in statistics:
            # For each statistic, decide what should it be called
            channel_names = pampro_utilities.design_variable_names(self.name, stat)

            # Create a Channel for each output
            for cn in channel_names:
                channel_list.append(Channel(cn))

            # If any pre-processing needs to happen for this statistic, now is the time to do it

            # For bouts, we want to get an exhaustive list of bouts for the whole channel
            # Then when we call window_statistics, we narrow that list of bouts down to what overlaps the window
            if stat[0] == "bouts":

                # To be indexed as "X_Y" or "X_Y_Z"
                self.bouts_cache = {}

                for b in stat[1]: # examples (0,99) or (0,99,5)

                    low, high = b[0], b[1]

                    # Extract the bouts >= X and <= Y and save them
                    bouts = self.bouts(low, high)
                    self.bouts_cache["{}_{}".format(low, high)] = bouts

                    # Also, if they specified a minimum duration of the bouts
                    if len(b) == 3:

                        bouts_restricted = Bout.limit_to_lengths(bouts, min_length=timedelta(seconds=b[2]))
                        self.bouts_cache["{}_{}_{}".format(low, high, b[2])] = bouts_restricted

        num_expected_results = len(channel_list)

        for window in windows:

            results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)

            if len(results) != num_expected_results:

                raise Exception("Incorrect number of statistics yielded. {} expected, {} given. Channel: {}. Statistics: {}.".format(num_expected_results, len(results), self.name, statistics))

            for i in range(len(results)):
                channel_list[i].append_data(window.start_timestamp, results[i])

        for channel in channel_list:
            channel.missing_value = -1
            channel.data = np.array(channel.data)
            channel.timestamps = np.array(channel.timestamps)
            channel.calculate_timeframe()
            channel.determine_appropriate_methods()

        ts = Time_Series.Time_Series(name)
        ts.add_channels(channel_list)
        return ts

    def append_data(self, timestamp, data_row):
        """ Append a single observation to the end of the timestamp and data arrays. """

        self.timestamps.append(timestamp)
        self.data.append(data_row)

    def infer_timestamp(self, index):
        """ Given an index of the data array, approximate its timestamp using the sparse timestamps around it """

        start = bisect_left(self.indices, index)
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

    def infer_timestamp_delta(self):

        deltas = np.diff(self.timestamps)
        self.mean_timedelta = np.mean(deltas)
        self.max_timedelta = np.max(deltas)
        self.min_timedelta = np.min(deltas)

    def generate_sliding_windows(self, window_size):

        for timestamp in self.timestamps:

            start_dts = timestamp - (window_size/2.0)
            end_dts = timestamp + (window_size/2.0)

            yield Bout.Bout(start_dts, end_dts)

    def sliding_statistics(self, window_size, statistics=[("generic", ["mean"])], time_period=False, name=""):

        windows = self.generate_sliding_windows(window_size)

        channels = self.build_statistics_channels(windows, statistics, name=name)

        for c in channels:
            c.timestamps = self.timestamps
            c.calculate_timeframe()

        return channels

    def generate_piecewise_windows(self, start, end, window_size):

        start_dts = start
        end_dts = start + window_size

        while start_dts < end:

            yield Bout.Bout(start_dts, end_dts)

            start_dts = start_dts + window_size
            end_dts = end_dts + window_size

    def piecewise_statistics(self, window_size, statistics=[("generic", ["mean"])], time_period=False, name=""):

        if time_period == False:
            start = time_utilities.start_of_day(self.timeframe[0])
            end = time_utilities.end_of_day(self.timeframe[1])
        else:
            start = time_period[0]
            end = time_period[1]

        #print("Piecewise statistics: {}".format(self.name))
        windows = self.generate_piecewise_windows(start,end, window_size)

        # Else if we passed an integer as our window size
        #elif str(type(window_size)) == "<class 'int'>":

        #    windows = [[i,i+window_size] for i in range(0,len(self.data),window_size)]

        return self.build_statistics_channels(windows, statistics, name=name)

    def summary_statistics(self, statistics=[("generic", ["mean"])], time_period=False, name=""):

        if time_period == False:
            windows = [Bout.Bout(self.timeframe[0], self.timeframe[1]+timedelta(days=1111))]
        else:
            windows = [Bout.Bout(time_period[0],time_period[1])]

        return self.build_statistics_channels(windows, statistics, name=name)

    def bouts(self, low, high):
        """ Return a list of Bout objects where data is >= low and <= high. """

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
            start_time = self.timestamps[start_index]
            end_time = self.timestamps[end_index]

            #if not self.sparsely_timestamped:
            end_time += self.timestamps[-1]-self.timestamps[-2]

            bouts.append(Bout.Bout(start_time, end_time))

        if self.timestamp_policy == "offset":

            for b in bouts:

                b.start_timestamp = self.time_period[0]+timedelta(microseconds=1000)*b.start_timestamp
                b.end_timestamp = self.time_period[0]+timedelta(microseconds=1000)*b.end_timestamp
                b.length = b.end_timestamp - b.start_timestamp

        return bouts

    def delete_windows(self, windows, missing_value = -111):
        """ Given a list of Bouts, replace any data inside those time windows with the given missing_value. This masks the data when being summarised by any statistic methods. """

        # New approach - don't delete the data, mask it with a set value
        # Then when we summarise, check if data has been masked (missing_value is not False)
        # Then analyse only unmasked data
        self.fill_windows(windows, fill_value=missing_value)
        self.missing_value = missing_value

    def restrict_timeframe(self, start, end):
        """ Mask all the data outside of the given time range, by calling delete_windows. """

        # Don't delete the data anymore, just mask the data outside of the range

        # First bout represents all time up to "start"
        bout1 = Bout.Bout(self.timestamps[0]-timedelta(days=1), start-timedelta(microseconds=1))
        # Second bout represents all time after "end"
        bout2 = Bout.Bout(end+timedelta(microseconds=1), self.timestamps[-1]+timedelta(days=1))

        self.delete_windows([bout1, bout2])

    def fill(self, bout, fill_value=0):
        """ Given a Bout representing a window of time, replace all the data values of this Channel within the time window with a given fill_value. """

        start_index,end_index = self.get_window(bout.start_timestamp,bout.end_timestamp)
        #print(start_index, end_index)
        self.data[start_index:end_index] = fill_value

    def fill_windows(self, bouts, fill_value=0):
        """ Given a list of Bouts, iteratively call self.fill() with each Bout."""

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

    def add_annotation(self, annotation):

        self.annotations.append(annotation)

    def add_annotations(self, annotations):

        for a in annotations:
            self.add_annotation(a)

    def draw(self, axis, time_period=False):

        #if not self.sparsely_timestamped:
        start_index,end_index = self.get_window(time_period[0], time_period[1])
        window_data = self.data[start_index:end_index]
        window_timestamps = self.timestamps[start_index:end_index]

        if self.timestamp_policy == "offset":

            window_timestamps = time_period[0] + timedelta(microseconds=1000)*window_timestamps

        axis.plot(window_timestamps, window_data, label=self.name, **self.draw_properties)

        for a in self.annotations:
            axis.axvspan(xmin=a.start_timestamp, xmax=a.end_timestamp, **a.draw_properties)

    def __str__(self):

        description = OrderedDict()
        description["Channel name"] = self.name
        description["Start"] = self.timeframe[0]
        description["End"] = self.timeframe[1]
        description["Duration"] = self.timeframe[1] - self.timeframe[0]
        description["Data count"] = len(self.data)
        description["Timestamp count"] = len(self.timestamps)
        description["Timestamp policy"] = self.timestamp_policy

        if not hasattr(self, "mean_timedelta"):
            self.infer_timestamp_delta()

        description["mean_timedelta"] = self.mean_timedelta
        description["max_timedelta"] = self.max_timedelta
        description["min_timedelta"] = self.min_timedelta

        output = ""
        for k,v in description.items():
            output += str(k) + ": " + str(v) + "\n"
        output = output[:-1]

        return output

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
