
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection
from datetime import datetime, timedelta


ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load("_data/testfile16.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass


def test_summary_statistics():

    # There should be 16x30 1s
    assert sum(counts.data == 1) == 16*30

    # There should be 10x30 0s
    assert sum(counts.data == 0) == 10*30

    ts_results = Time_Series.Time_Series("")
    ts_results.add_channels( counts.summary_statistics(statistics=[("generic", ["sum", "min", "max"]), ("cutpoints", [[0,0],[1,1],[0,1]])]))

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

test_summary_statistics.setup = setup_func
test_summary_statistics.teardown = teardown_func
