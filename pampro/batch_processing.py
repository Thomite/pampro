import collections
import math
import numpy as np
import sys
from datetime import datetime
import json
import traceback

def job_indices(n, num_jobs, job_list_size):

    n = n-1

    job_size = math.floor(job_list_size/num_jobs)
    remaining = job_list_size - (num_jobs*job_size)

    start_index = 0
    for i in range(num_jobs):

        end_index = min( job_list_size, start_index + job_size )

        if remaining > 0:
            end_index += 1
            remaining -= 1

        if i == n:
            return (start_index,end_index)
        start_index = end_index

def get_feedback(filename):

    file = open(filename, "r")
    feedback = json.loads(file.read())
    file.close()
    return feedback

def write_feedback(feedback, filename):

    file = open(filename, "w")
    file.write(json.dumps(feedback))
    file.flush()
    file.close()

def load_job_details(job_file):

    data = np.genfromtxt(job_file, delimiter=',', dtype='str', skiprows=0)

    master_dictionary = collections.OrderedDict()

    for row in data[1:]:
        master_dictionary[row[0]] = {}
        for index,col in enumerate(row):
            master_dictionary[row[0]][data[0,index]] = col

    return master_dictionary

def batch_process(analysis_function, job_file, job_num, num_jobs, live_feedback=False):

    batch_start_time = datetime.now()

    # Load the document listing all the files to be processed
    job_details = load_job_details(job_file)

    # Using job_num and num_jobs, calculate which files this process should handle
    job_section = job_indices(job_num, num_jobs, len(job_details))
    my_jobs = list(job_details.keys())[job_section[0]:job_section[1]]

    output_log = False

    if live_feedback:
        # Create a JSON file to store progress information in
        feedback_filename = job_file + "_{}_status.json".format(job_num)
        write_feedback({"job":job_num, "num_jobs":len(job_details), "progress":1, "complete":0}, feedback_filename)

    for n, job in enumerate(my_jobs):

        print("Job {}/{}: {}\n".format(n+1, len(my_jobs), job))
        job_start_time = datetime.now()

        try:
            if live_feedback:
                analysis_function( job_details[job], feedback_filename )
            else:
                analysis_function( job_details[job] )

        except:

            tb = traceback.format_exc()

            # Create the output file only if an error has occurred
            if output_log is False:
                output_log = open("error_log_{}.csv".format(job_num), "w")

            print("Exception:" + str(sys.exc_info()))
            print(tb)

            output_log.write( str(job_details[job]) + "\n" )
            output_log.write("Exception:" + str(sys.exc_info()) + "\n")
            output_log.write(tb + "\n\n")

        job_end_time = datetime.now()
        job_duration = job_end_time - job_start_time
        print("\nJob run time: " + str(job_duration))

        batch_duration = job_end_time - batch_start_time
        batch_remaining = (len(my_jobs)-n)*job_duration
        print("Batch run time: " + str(batch_duration))
        print("Time remaining: " + str(batch_remaining))
        print("Predicted completion time:" + str((batch_remaining + datetime.now())) + "\n")

        if live_feedback:
            feedback = get_feedback(feedback_filename)
            feedback["progress"] += 1
            write_feedback(feedback, feedback_filename)

    batch_end_time = datetime.now()
    batch_duration = batch_end_time - batch_start_time
    print("Batch run time: " + str(batch_duration))

    if live_feedback:
        feedback = get_feedback(feedback_filename)
        feedback["complete"] = 1
        write_feedback(feedback, feedback_filename)

    # If everything went smoothly, output_log is False because it was never a file object
    if output_log is not False:
        output_log.close()
