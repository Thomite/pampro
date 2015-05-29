import re
import collections

def design_variable_names(signal_name, stat):

    name = signal_name
    varnames = []

    if (stat[0] == "generic"):

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
