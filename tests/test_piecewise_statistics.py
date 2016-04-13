
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile17.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

def teardown_func():
    pass


def test_hourly():

    # File has 60 0s, then 60 1s, 60 2s, 60 3s
    hourly_results = counts.piecewise_statistics(timedelta(hours=1), statistics=[("generic", ["mean", "sum", "n", "missing"])], time_period=counts.timeframe, name="Blah")

    assert(hourly_results.name == "Blah")

    hourly_mean = hourly_results[0]
    assert(hourly_mean.data[0] == 0)
    assert(hourly_mean.data[1] == 1)
    assert(hourly_mean.data[2] == 2)
    assert(hourly_mean.data[3] == 3)

    hourly_sum = hourly_results[1]
    assert(hourly_sum.data[0] == 0)
    assert(hourly_sum.data[1] == 1*60)
    assert(hourly_sum.data[2] == 2*60)
    assert(hourly_sum.data[3] == 3*60)

    # Should be 60 observations per hour
    hourly_n = hourly_results[2]
    for d in hourly_n.data:
        assert(d == 60)

    # Nothing should be missing
    hourly_missing = hourly_results[3]
    assert(sum(hourly_missing.data) == 0)

def test_half_hourly():

    # File has 60 0s, then 60 1s, 60 2s, 60 3s
    half_hourly_results = counts.piecewise_statistics(timedelta(minutes=30), statistics=[("generic", ["mean", "sum", "n", "missing"])], time_period=counts.timeframe)

    half_hourly_mean = half_hourly_results[0]
    assert(half_hourly_mean.data[0] == 0)
    assert(half_hourly_mean.data[1] == 0)
    assert(half_hourly_mean.data[2] == 1)
    assert(half_hourly_mean.data[3] == 1)
    assert(half_hourly_mean.data[4] == 2)
    assert(half_hourly_mean.data[5] == 2)
    assert(half_hourly_mean.data[6] == 3)
    assert(half_hourly_mean.data[7] == 3)

    half_hourly_sum = half_hourly_results[1]
    assert(half_hourly_sum.data[0] == 0)
    assert(half_hourly_sum.data[1] == 0)
    assert(half_hourly_sum.data[2] == 30)
    assert(half_hourly_sum.data[3] == 30)
    assert(half_hourly_sum.data[4] == 60)
    assert(half_hourly_sum.data[5] == 60)
    assert(half_hourly_sum.data[6] == 90)
    assert(half_hourly_sum.data[7] == 90)

    # Should be 30 observations per half hour
    half_hourly_n = half_hourly_results[2]
    for d in half_hourly_n.data:
        assert(d == 30)

    # Nothing should be missing
    half_hourly_missing = half_hourly_results[3]
    assert(sum(half_hourly_missing.data) == 0)

test_hourly.setup = setup_func
test_hourly.teardown = teardown_func

def test_custom_time_period():

    # Offset start by 30 minutes, do hourly piecewise sums of counts
    tf = (counts.timestamps[0]+timedelta(minutes=30), counts.timestamps[-1])
    hourly_results = counts.piecewise_statistics(timedelta(hours=1), statistics=[("generic", ["sum", "n"])], time_period=tf)

    hourly_sum = hourly_results[0]

    # 30 minutes of 0s, 30 minutes of 1s
    assert(hourly_sum.data[0] == 30)

    # 30 minutes of 1s, 30 minutes of 2s
    assert(hourly_sum.data[1] == 90)

    # 30 minutes of 2s, 30 minutes of 3s
    assert(hourly_sum.data[2] == 150)

    # 30 minutes of 3s, 30 minutes of 4s
    assert(hourly_sum.data[3] == 90)

test_custom_time_period.setup = setup_func
test_custom_time_period.teardown = teardown_func

def test_piecewise_sequence_a():

    # Test that doing piecewise_statistics twice gets identical results
    results1 = counts.piecewise_statistics(timedelta(minutes=30), statistics=[("generic", ["mean"])])
    results2 = counts.piecewise_statistics(timedelta(minutes=30), statistics=[("generic", ["mean"])])

    for a,b in zip(results1.channels[0].data, results2.channels[0].data):
        assert(a == b)

test_piecewise_sequence_a.setup = setup_func
test_piecewise_sequence_a.teardown = teardown_func

def test_piecewise_sequence_b():

    # Test that doing piecewise statistics at a shorter level doesn't affect the results at longer level
    results1 = counts.piecewise_statistics(timedelta(minutes=2), statistics=[("generic", ["mean"])])
    test_hourly()

test_piecewise_sequence_b.setup = setup_func
test_piecewise_sequence_b.teardown = teardown_func

def test_piecewise_sequence_c():

    # Test that doing piecewise statistics at a longer level doesn't affect the results at shorter level
    results1 = counts.piecewise_statistics(timedelta(hours=3), statistics=[("generic", ["mean"])])
    test_hourly()

test_piecewise_sequence_c.setup = setup_func
test_piecewise_sequence_c.teardown = teardown_func

def test_piecewise_sequence_d():

    # Test that doing piecewise statistics at various levels doesn't affect the results at hourly or half hourly level
    for i in [1,5,15]:
        print(i)
        results1 = counts.piecewise_statistics(timedelta(minutes=i), statistics=[("generic", ["mean"])])
        test_half_hourly()
        test_hourly()

test_piecewise_sequence_d.setup = setup_func
test_piecewise_sequence_d.teardown = teardown_func
