import collections
import math
import numpy as np
import sys
from datetime import datetime

def job_indices(n, num_jobs, job_list_size):

	n = n-1

	job_size = int(math.ceil(job_list_size/num_jobs))
	remaining = job_list_size % job_size
	start_index = 0
	for i in range(num_jobs):

		end_index = min( job_list_size, start_index + job_size )

		if remaining > 0:
			end_index += 1
			remaining -= 1

		if i == n:
			return (start_index,end_index)
		start_index = end_index

def load_job_details(job_file):
    
    data = np.genfromtxt(job_file, delimiter=',', dtype='str', skiprows=0)
    
    master_dictionary = collections.OrderedDict()
    
    for row in data[1:]:
        master_dictionary[row[0]] = {}
        for index,col in enumerate(row):
            master_dictionary[row[0]][data[0,index]] = col
    
    return master_dictionary


def batch_process(analysis_function, job_file, job_num, num_jobs):

	batch_start_time = datetime.now()

	job_details = load_job_details(job_file)

	job_section = job_indices(job_num, num_jobs, len(job_details))

	my_jobs = job_details.keys()[job_section[0]:job_section[1]]

	output_log = open(job_file + "_log_{}.csv".format(job_num), "w")

	for n, job in enumerate(my_jobs):

		print("Job {}/{}: {}\n".format(n+1, len(my_jobs), job))
		job_start_time = datetime.now()
		
		try:
			analysis_function( job_details[job] )
		except:
			print "Exception:", sys.exc_info()[0]
			output_log.write("Exception:" + sys.exc_info()[0] + "\n")

		job_end_time = datetime.now()
		job_duration = job_end_time - job_start_time
		print("\nJob run time: " + str(job_duration))

		batch_duration = job_end_time - batch_start_time
		batch_remaining = (len(my_jobs)-n)*job_duration
		print("Batch run time: " + str(batch_duration))
		print("Time remaining: " + str(batch_remaining) + "\n")

	batch_end_time = datetime.now()
	batch_duration = batch_end_time - batch_start_time
	print("Batch run time: " + str(batch_duration))

	output_log.close()
