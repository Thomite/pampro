
from pampro import data_loading, Time_Series, Channel, Bout
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile18.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass



def test_restrict_timeframe():

    # File contains 24 hours of 1s, then 15 hours of 0s, then 9 hours of 1s, then 24 hours of 1s

    start = counts.timestamps[0]
    end = counts.timestamps[-1]

    # Summarise the data before restricting
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"])]))

    # Number of 1s = 24 hours then 9 hours then 24 hours
    assert(summary_before.get_channel("AG_Counts_sum").data[0] == (24+9+24)*60)

    # Trim an hour off the start
    counts.restrict_timeframe(start+timedelta(hours=1), end)
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"])]))

    # Should be 1 hour less
    assert(summary_after.get_channel("AG_Counts_sum").data[0] == int((23+9+24)*60))

    # Repeating exactly the same thing should change nothing
    counts.restrict_timeframe(start+timedelta(hours=1), end)
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"])]))
    assert(summary_after.get_channel("AG_Counts_sum").data[0] == int((23+9+24)*60))

    # Now trim an hour off the end
    counts.restrict_timeframe(start+timedelta(hours=1), end-timedelta(hours=1))
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"])]))

    # Should now be an hour less again
    assert(summary_after.get_channel("AG_Counts_sum").data[0] == int((23+9+23)*60))

    # Now trim to a single hour
    counts.restrict_timeframe(start+timedelta(hours=12), start+timedelta(hours=12, minutes=59))
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"])]))

    # Should now be an hour of 1s
    assert(summary_after.get_channel("AG_Counts_sum").data[0] == 60)


test_restrict_timeframe.setup = setup_func
test_restrict_timeframe.teardown = teardown_func
