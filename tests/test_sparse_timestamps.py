
from pampro import Channel, Time_Series
import numpy as np
from datetime import datetime, timedelta
import copy

start = datetime.strptime("01/01/2000 00:00", "%d/%m/%Y %H:%M")
data = np.array([0 for i in range(500)] + [1 for i in range(500)])
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

    print(len(data))
    print(indices)

def teardown_func():
    pass


def test_ensure_timestamped_at():

    # Channel starts off with 11 timestamps & corresponding indices
    assert(len(c.timestamps) == 11)
    assert(len(c.indices) == 11)

    # full_timestamps[50] is inbetween the timestamps channel c has
    c.ensure_timestamped_at(full_timestamps[50])

    # This should cause 2 inserts to both indices and timestamps
    assert(len(c.timestamps) == 13)
    assert(len(c.indices) == 13)

    # full_timestamps[50] now corresponds to an entry
    c.ensure_timestamped_at(full_timestamps[50])

    # So there should be no new entries to indices and timestamps
    assert(len(c.timestamps) == 13)
    assert(len(c.indices) == 13)

    # The inserts should be at 50 and 51
    assert(c.indices[1] == 50)
    assert(c.indices[2] == 51)

    # Now, ensuring the channel is timestamped between 50 & 51 should also cause no change
    middle_timestamp = full_timestamps[50] + timedelta(microseconds=198)
    c.ensure_timestamped_at(middle_timestamp)
    assert(len(c.timestamps) == 13)
    assert(len(c.indices) == 13)
    assert(c.indices[1] == 50)
    assert(c.indices[2] == 51)


def test_piecewise_secondly():

    # Channel starts off with 11 timestamps & corresponding indices
    assert(len(c.timestamps) == 11)
    assert(len(c.indices) == 11)

    # Snapshot the timestamps
    before_copy = copy.deepcopy(c.timestamps)

    # Since the timestamps are at second intervals, there should be no change to the timestamps
    test = c.piecewise_statistics(timedelta(seconds=1), statistics=[("generic", ["mean", "n", "sum"])], time_period=(c.timestamps[0], c.timestamps[-1]))
    for channel in test:
        print(channel.name, channel.data)

    # Assert no change in length
    assert(len(c.timestamps) == 11)
    assert(len(c.indices) == 11)

    # Snapshot again
    after_copy = copy.deepcopy(c.timestamps)

    # Iterate and compare each timestamp entry, should be identical
    for before,after in zip(before_copy, after_copy):
        assert(before == after)

def test_timestamp_inference():

    # Test the
    for i,ts in enumerate(full_timestamps):

        assert(c.infer_timestamp(i) == ts)


test_ensure_timestamped_at.setup = setup_func
test_ensure_timestamped_at.teardown = teardown_func
test_piecewise_secondly.setup = setup_func
test_piecewise_secondly.teardown = teardown_func
test_timestamp_inference.setup = setup_func
test_timestamp_inference.teardown = teardown_func
