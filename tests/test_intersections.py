
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection, channel_inference
from datetime import datetime, timedelta


ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load("_data/testfile22.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass



def test_extracted_bouts():

    one_bouts = counts.bouts(1,1)
    zero_bouts = counts.bouts(0,0)

    # Bouts where counts == 0 and counts == 1 should be mutually excluse
    # So there should be no intersections between them
    intersections = Bout.bout_list_intersection(one_bouts, zero_bouts)

    assert(len(intersections) == 0)

    # A bout that spans the whole time period should completely intersect with bouts where counts == 1
    one_big_bout = Bout.Bout(counts.timestamps[0]-timedelta(days=1), counts.timestamps[-1]+timedelta(days=1))

    one_intersections = Bout.bout_list_intersection(one_bouts, [one_big_bout])
    assert(Bout.total_time(one_intersections) == Bout.total_time(one_bouts))

    # Same for zeros
    zero_intersections = Bout.bout_list_intersection(zero_bouts, [one_big_bout])
    assert(Bout.total_time(zero_intersections) == Bout.total_time(zero_bouts))

    # Filling in the bout gaps of one bouts should recreate the zero bouts
    inverse_of_one_bouts = Bout.time_period_minus_bouts((counts.timeframe[0], counts.timeframe[1]+timedelta(minutes=1)), one_bouts)

    # They should have the same n
    assert(len(inverse_of_one_bouts) == len(zero_bouts))

    # Same total amount of time
    assert(Bout.total_time(inverse_of_one_bouts) == Bout.total_time(zero_bouts))

test_extracted_bouts.setup = setup_func
test_extracted_bouts.teardown = teardown_func


def test_artificial_bouts():

    start_a = datetime.strptime("01/01/2000", "%d/%m/%Y")
    end_a = start_a + timedelta(hours=1)
    bout_a = Bout.Bout(start_a, end_a)

    # Hour long bout
    assert(bout_a.length == timedelta(hours=1))

    start_b = datetime.strptime("01/01/2000", "%d/%m/%Y")
    end_b = start_a + timedelta(minutes=15)
    bout_b = Bout.Bout(start_b, end_b)

    # They share common time
    assert(bout_a.overlaps(bout_b))

    # 15 minutes, to be precise
    intersection = bout_a.intersection(bout_b)
    assert(intersection.length == timedelta(minutes=15))

    start_c =  datetime.strptime("01/02/2000", "%d/%m/%Y")
    end_c = start_c + timedelta(days=1)
    bout_c = Bout.Bout(start_c, end_c)

    # No overlap of those bouts
    assert(not bout_a.overlaps(bout_c))

    # bout_a ends exactly as bout_d starts
    # there should be no overlap (0 common time)
    start_d = end_a
    end_d = start_d + timedelta(minutes=1)
    bout_d = Bout.Bout(start_d, end_d)
    assert(not bout_a.overlaps(bout_d))
