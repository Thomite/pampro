import re
import collections

def design_variable_name(signal, statistic):

	variable_name = "error"
	if isinstance(statistic, list):
		variable_name = str(signal) + "_" + str(statistic[0]) + "_" + str(statistic[1])
	else:
		variable_name = str(signal) + "_" + str(statistic)

	return variable_name

def design_file_header(statistics):
	
	file_header = "id,timestamp"
	for k,v in statistics.items():
		for stat in v:

			variable_name = design_variable_name(k,stat)
			
			file_header = file_header + "," + variable_name

	print file_header
	return file_header

def design_data_dictionary(statistics_dictionary):
    
    percentile_pattern = re.compile("\A([p])([0-9]*)")
    data_dictionary = collections.OrderedDict()
    
    for channel, statistics in statistics_dictionary.items():


        for stat in statistics:
            
            variable_name = design_variable_name(channel, stat)
            if stat == "mean":
                sans_mean = variable_name[:-5]
                data_dictionary[variable_name] = "Mean average of {}".format(sans_mean)
            elif stat == "sum":
                sans_sum = variable_name[:-4]
                data_dictionary[variable_name] = "Sum of values in {}".format(sans_sum)
            elif stat == "std":
                sans_std = variable_name[:-4]
                data_dictionary[variable_name] = "Standard deviation of {}".format(sans_std)
            elif stat == "min":
                sans_min = variable_name[:-4]
                data_dictionary[variable_name] = "Lowest value in {}".format(sans_min)
            elif stat == "max":
                sans_max = variable_name[:-4]
                data_dictionary[variable_name] = "Highest value in {}".format(sans_max)
            elif stat == "n":
                sans_n = variable_name[:-2]
                data_dictionary[variable_name] = "Number of observations in {}".format(sans_n)
            elif isinstance(stat, list):
                sans_limits = variable_name.replace("_{}_{}".format(stat[0], stat[1]), "")
                data_dictionary[variable_name] = "Number of values in {} at >= {} & <= {}".format(sans_limits, stat[0], stat[1])
            elif percentile_pattern.match(stat):
                match_value = percentile_pattern.match(stat).groups()[1]
                percentile = int(match_value)
                sans_px = variable_name.replace("_p" + match_value, "")
                suffix = "th"
                if match_value[-1] == "3":
                    suffix = "rd"
                elif match_value[-1] == "1":
                    suffix = "st"
                elif match_value[-1] == "2":
                    suffix = "nd"
                data_dictionary[variable_name] = "{}{} percentile of {}".format(percentile, suffix, sans_px)
            
    return data_dictionary