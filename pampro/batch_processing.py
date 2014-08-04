import collections
import math
import numpy as np

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

	job_details = load_job_details(job_file)

	job_section = job_indices(job_num, num_jobs, len(job_details))

	my_jobs = job_details.keys()[job_section[0]:job_section[1]]

	for job in my_jobs:
		analysis_function( job_details[job] )
	

def generic_analysis(id):
	for k,v in id.items():
		print k,v

batch_process(generic_analysis, "/pa/data/STVS/_documents/sources.csv", 1, 10)
	