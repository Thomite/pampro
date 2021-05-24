# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import re
import collections
from collections import OrderedDict
import pandas as pd
import os,shutil
import json
from datetime import timedelta


def design_variable_names(signal_name, stat):

    name = signal_name
    varnames = []

    if (stat[0] == "generic"):

        for val1 in stat[1]:
            varnames.append(signal_name + "_" + val1)

    if (stat[0] == "binary"):

        for val1 in stat[1]:
            varnames.append(signal_name + "_" + val1)
    
    elif (stat[0] == "cutpoints"):

        for low,high in stat[1]:
            varnames.append(signal_name + "_" + str(low) + "_" + str(high))

    elif (stat[0] == "bigrams"):

        for val1 in stat[1]:
            for val2 in stat[1]:

                varnames.append(signal_name + "_" + str(val1) + "tr" + str(val2))

    elif (stat[0] == "frequency_ranges"):

        for low,high in stat[1]:
            varnames.append(signal_name + "_" + str(low) + "hz_" + str(high) + "hz")

    elif (stat[0] == "top_frequencies"):

        for i in range(stat[1]):

            varnames.append(signal_name + "_topfreq_" + str(i))
            varnames.append(signal_name + "_topmag_" + str(i))

    elif (stat[0] == "percentiles"):

        for i in stat[1]:

            varnames.append(signal_name + "_p" + str(i))

    elif (stat[0] == "sdx"):

        for i in stat[1]:

            varnames.append(signal_name + "_sd" + str(i))

    elif (stat[0] == "bouts"):

        for i in stat[1]:
            basic = "{}_{}_{}_bouts".format(signal_name, i[0], i[1])
            if len(i) == 3:
                basic += "_mt{}".format(i[2])

            varnames.append(basic+"_sum")
            varnames.append(basic+"_num")

    return varnames

def design_file_header(statistics):

    file_header = "id,timestamp"
    for k,v in statistics.items():
        for stat in v:

            variable_names = design_variable_names(k,stat)
            for vn in variable_names:

                file_header = file_header + "," + vn

    #print(file_header)
    return file_header

def design_variable_descriptions(signal_name, stat):

    variable_descriptions = []

    if stat[0] == "generic":

        for stat1 in stat[1]:
            if stat1 == "mean":
                variable_descriptions.append("Mean average {}".format(signal_name))
            elif stat1 == "sum":
                variable_descriptions.append("Sum {}".format(signal_name))
            elif stat1 == "std":
                variable_descriptions.append("Standard deviation {}".format(signal_name))
            elif stat1 == "min":
                variable_descriptions.append("Lowest {}".format(signal_name))
            elif stat1 == "max":
                variable_descriptions.append("Highest {}".format(signal_name))
            elif stat1 == "n":
                variable_descriptions.append("Number of observations in {}".format(signal_name))
            elif stat1 == "missing":
                variable_descriptions.append("Number of missing observations in {}".format(signal_name))

    elif stat[0] == "cutpoints":
        for low,high in stat[1]:

            variable_descriptions.append("Number of values in {} at >= {} & <= {}".format(signal_name, low, high))

    elif stat[0] == "bigrams":

        for val1 in stat[1]:
            for val2 in stat[1]:
                variable_descriptions.append("Number of transitions from {} to {}".format(val1, val2))

    elif stat[0] == "percentiles":

        for percentile in stat[1]:

            suffix = "th"
            if str(percentile)[-1] == "3":
                suffix = "rd"
            elif str(percentile)[-1] == "1":
                suffix = "st"
            elif str(percentile)[-1] == "2":
                suffix = "nd"
            variable_descriptions.append("{}{} percentile of {}".format(percentile, suffix, signal_name))

    return variable_descriptions

def design_data_dictionary(statistics_dictionary):

    data_dictionary = collections.OrderedDict()

    for channel, statistics in statistics_dictionary.items():

        for stat in statistics:

            variable_names = design_variable_names(channel, stat)
            variable_descriptions = design_variable_descriptions(channel, stat)

            for variable_name, variable_description in zip(variable_names, variable_descriptions):
                data_dictionary[variable_name] = variable_description

    return data_dictionary


def csv_line(vals):
    """ CSV formatted output of a list """
    strval = str(vals[0])
    for val in vals[1:]:
        strval += "," + str(val)
    strval += "\n"
    return strval


def dict_write(file_location, id, dictionary, other_index=False):
    """
    Append a dictionary as a row to a CSV, or create one if necessary. Indexed by the given ID.
    other_index = False by default, which sets index as "id".  If any other index is used, give as string e.g. "device"
    """
    if other_index == False:
        index_name = "id"
    else:
        index_name = other_index

    dictionary = dictionary.copy()

    # Wrap each value in a list
    for k,v in dictionary.items():
        dictionary[k] = [v]
    dictionary[index_name] = [id]

    # Create a 1 row dataset using the passed dictionary
    df = pd.DataFrame.from_dict(dictionary).set_index(index_name)

    try:
        # Append the dataset from the file location
        loaded = pd.read_csv(file_location).set_index(index_name)
        df = df.append(loaded)
    except:
        pass

    # Save everything to given location
    df.to_csv(file_location, na_rep="-1")


def json_epochs_to_dict(json_string):
    """ Converts a json string from a settings file to a dcitionary of epochs,
    where the name is the key and the timedelta is the value"""

    epochs = json.loads(json_string)
    epoch_dict = dict()

    for e in epochs:
        inc = e["increment"]
        if e["unit"] == "hour(s)":
            name = str(inc) + "h"
            epoch = timedelta(hours=inc)
        elif e["unit"] == "minute(s)":
            name = str(inc) + "m"
            epoch = timedelta(minutes=inc)
        elif e["unit"] == "second(s)":
            name = str(inc) + "s"
            epoch= timedelta(seconds=inc)
        else:
            name = epoch = ""
        epoch_dict[name] = epoch

    return epoch_dict


def json_cutpoints_to_list(json_string):
    """ Converts a json string from a settings file to a list of cut points.
       Each cut point is itself a list comprising of start and end points"""
    cutpoints = json.loads(json_string)

    cutpoints_list = []
    for c in cutpoints:
        start = c["start"]
        end = c["end"]
        list1 = [start, end]

        cutpoints_list.append(list1)

    return cutpoints_list


def define_statistics(stats_list, intensities_list, angles_list):
    """Produces the analysis statistics from a list of statistics,
    a list of intensity cut points and a list of angle cut points"""

    stats = OrderedDict()
    if 'enmo' in stats_list:
        stats["ENMO"] = [("generic", ["mean", "n", "missing", "sum"]), ("cutpoints", intensities_list)]
    if 'hpfvm' in stats_list:
        stats["HPFVM"] = [("generic", ["mean", "n", "missing", "sum"]), ("cutpoints", intensities_list)]
    if 'pitch' in stats_list:
        stats["PITCH"] = [("generic", ["mean", "std", "min", "max"]), ("cutpoints", angles_list)]
    if 'roll' in stats_list:
        stats["ROLL"] = [("generic", ["mean", "std", "min", "max"]), ("cutpoints", angles_list)]
    if 'zangle' in stats_list:
        stats["ZANGLE"] = [("generic", ["mean", "std", "min", "max"]), ("cutpoints", angles_list)]
    if 'temperature' in stats_list:
        stats["Temperature"] = [("generic", ["mean"])]
    if 'battery' in stats_list:
        stats["Battery"] = [("generic", ["mean"])]
    if 'integrity' in stats_list:
        stats["Integrity"] = [("generic", ["sum"])]

    return stats
