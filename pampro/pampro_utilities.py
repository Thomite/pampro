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

def design_data_dictionary(statistics_dictionary):

    data_dictionary = collections.OrderedDict()

    for channel, statistics in statistics_dictionary.items():


        for stat in statistics:

            variable_names = design_variable_names(channel, stat)
            for variable_name, stat1 in zip(variable_names, stat[1]):
                if stat[0] == "generic":
                    if stat1 == "mean":
                        data_dictionary[variable_name] = "Mean average of {}".format(channel)
                    elif stat1 == "sum":
                        data_dictionary[variable_name] = "Sum of values in {}".format(channel)
                    elif stat1 == "std":
                        data_dictionary[variable_name] = "Standard deviation of {}".format(channel)
                    elif stat1 == "min":
                        data_dictionary[variable_name] = "Lowest value in {}".format(channel)
                    elif stat1 == "max":
                        data_dictionary[variable_name] = "Highest value in {}".format(channel)
                    elif stat1 == "n":
                        data_dictionary[variable_name] = "Number of observations in {}".format(channel)

                elif stat[0] == "cutpoints":

                    for low,high in stat1:
                        data_dictionary[variable_name] = "Number of values in {} at >= {} & <= {}".format(channel, low, high)

                elif stat[0] == "bigrams":

                    for val1 in stat1:
                        for val2 in stat1:
                            data_dictionary[variable_name] = "Number of transitions from {} to {}".format(val1, val2)

                elif stat[0] == "percentiles":

                    for percentile in stat1:

                        suffix = "th"
                        if str(percentile)[-1] == "3":
                            suffix = "rd"
                        elif str(percentile)[-1] == "1":
                            suffix = "st"
                        elif str(percentile)[-1] == "2":
                            suffix = "nd"
                        data_dictionary[variable_name] = "{}{} percentile of {}".format(percentile, suffix, channel)

    return data_dictionary
