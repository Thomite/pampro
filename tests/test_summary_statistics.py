
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile16.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    counts = False


def test_summary_statistics():

    # There should be 16x30 1s
    assert sum(counts.data == 1) == 16*30

    # There should be 10x30 0s
    assert sum(counts.data == 0) == 10*30

    ts_results = counts.summary_statistics(statistics=[("generic", ["sum", "min", "max"]), ("cutpoints", [[0,0],[1,1],[0,1]])])

    # Sum is == number of 1s == 16*30
    assert(ts_results.get_channel("AG_Counts_sum").data[0] == 16*30)

    # Lowest value is 0
    assert(ts_results.get_channel("AG_Counts_min").data[0] == 0)

    # Highest value is 1
    assert(ts_results.get_channel("AG_Counts_max").data[0] == 1)

    # Number of 0s is 10*30
    assert(ts_results.get_channel("AG_Counts_0_0").data[0] == 10*30)

    # Number of 0s and 1s is 26*30
    assert(ts_results.get_channel("AG_Counts_0_1").data[0] == 26*30)

    # Number of 1s is 16*30
    assert(ts_results.get_channel("AG_Counts_1_1").data[0] == 16*30)

def test_summary_statistics_with_tp():

    # Same again, but defining a time period (which extends beyond the time period of the signal, so there should be no difference)

    start = counts.timeframe[0] - timedelta(minutes=2)
    end = counts.timeframe[1] + timedelta(minutes=2)
    ts_results2 = counts.summary_statistics(statistics=[("generic", ["sum", "min", "max", "n"]), ("cutpoints", [[0,0],[1,1],[0,1]])], time_period=(start,end))

    # Sum is == number of 1s == 16*30
    assert(ts_results2.get_channel("AG_Counts_sum").data[0] == 16*30)

    # Lowest value is 0
    assert(ts_results2.get_channel("AG_Counts_min").data[0] == 0)

    # Highest value is 1
    assert(ts_results2.get_channel("AG_Counts_max").data[0] == 1)

    # Number of 0s is 10*30
    assert(ts_results2.get_channel("AG_Counts_0_0").data[0] == 10*30)

    # Number of 0s and 1s is 26*30
    assert(ts_results2.get_channel("AG_Counts_0_1").data[0] == 26*30)

    # Number of 1s is 16*30
    assert(ts_results2.get_channel("AG_Counts_1_1").data[0] == 16*30)

test_summary_statistics.setup = setup_func
test_summary_statistics.teardown = teardown_func
test_summary_statistics_with_tp.setup = setup_func
test_summary_statistics_with_tp.teardown = teardown_func
