
from pampro import Channel, Time_Series
import numpy as np
from datetime import datetime, timedelta
import copy

start = datetime.now()
data = np.zeros(1000)
full_timestamps = np.array([start+(timedelta(seconds=1)/100)*i for i in range(len(data))])
c = Channel.Channel("")
indices = list(range(0,len(data), 100)) + [999]
sparse_timestamps = full_timestamps[indices]

def setup_func():

    global c
    global data
    global full_timestamps
    global indices
    global sparse_timestamps

    c.set_contents(data, sparse_timestamps)

    c.frequency = 100
    c.sparsely_timestamped = True
    c.indices = indices

def teardown_func():
    pass


def test_insertion():

    # Channel starts off with 11 timestamps & corresponding indices
    assert(len(c.timestamps) == 11)
    assert(len(c.indices) == 11)

    # full_timestamps[50] is inbetween the timestamps channel c has
    first_call = c.get_data_index(full_timestamps[50])

    # This should cause an insert to both indices and timestamps
    assert(len(c.timestamps) == 12)
    assert(len(c.indices) == 12)

    # full_timestamps[50] now corresponds to an entry
    second_call = c.get_data_index(full_timestamps[50])

    # So there should be no new entries to indices and timestamps
    assert(len(c.timestamps) == 12)
    assert(len(c.indices) == 12)

    # And it should return the same index
    assert(first_call == second_call)

def test_piecewise_secondly():

    # Channel starts off with 11 timestamps & corresponding indices
    assert(len(c.timestamps) == 11)
    assert(len(c.indices) == 11)

    # Snapshot the timestamps
    before_copy = copy.deepcopy(c.timestamps)

    # Since the timestamps are at second intervals, there should be no change to the timestamps
    test = c.piecewise_statistics(timedelta(seconds=1), time_period=(c.timestamps[0],c.timestamps[-1]))

    # Assert no change in length
    assert(len(c.timestamps) == 11)
    assert(len(c.indices) == 11)

    # Snapshot again
    after_copy = copy.deepcopy(c.timestamps)

    # Iterate and compare each timestamp entry, should be identical
    for before,after in zip(before_copy, after_copy):
        assert(before == after)


    

test_insertion.setup = setup_func
test_insertion.teardown = teardown_func
test_piecewise_secondly.setup = setup_func
test_piecewise_secondly.teardown = teardown_func
