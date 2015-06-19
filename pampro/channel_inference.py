
from datetime import timedelta
from pampro import Channel, Bout, Time_Series, time_utilities, pampro_fourier
import numpy as np
import copy

def produce_binary_channels(bouts, lengths, skeleton_channel):

    Bout.cache_lengths(bouts)
    bouts.sort(key=lambda x: x.length, reverse=True)

    channels = []
    for length in lengths:

        # Drop bouts from list if their length is less than x minutes
        bouts = Bout.limit_to_lengths(bouts, min_length=length, sorted=True)

        channel_name = "{}_mt{}".format(skeleton_channel.name,length)

        # Clone the blank channel, set data to 1 where time is inside any of the bouts
        skeleton_copy = copy.deepcopy(skeleton_channel)
        chan = Channel.channel_from_bouts(bouts, False, False, channel_name, skeleton=skeleton_copy)
        channels.append(chan)

    return channels


def infer_sleep_triaxial_wrist(vm,enmo,pitch,roll):



    print("A")
    result = Channel.Channel("Sleep")
    enmo_prob = enmo.piecewise_statistics(timedelta(minutes=10), statistics=["sum"], time_period=enmo.timeframe)[0]
    pitch_prob = pitch.piecewise_statistics(timedelta(minutes=10), statistics=["std"], time_period=pitch.timeframe)[0]
    roll_prob = roll.piecewise_statistics(timedelta(minutes=10), statistics=["std"], time_period=roll.timeframe)[0]
    print("C")
    pitch_prob.normalise()
    roll_prob.normalise()
    enmo_prob.normalise()
    print("D")
    product = np.multiply(np.multiply(pitch_prob.data, roll_prob.data), enmo_prob.data)
    print("E")
    result.set_contents(product, pitch_prob.timestamps)
    print("F")

    return result



def activpal_classification(pitch):

    transition_sit_stand = 32 # Sit -> Stand if angle >= 32
    transition_stand_sit = 22 # Stand -> Sit if angle <= 22
    ten_seconds_in_samples = 200 #20Hz

    # This should be faster than pointlessly copying the input array
    classification = Channel.Channel("activPAL_class")
    classification.set_contents(np.zeros(len(pitch.data)), pitch.timestamps)

    current_classification = 0 # 0 = sitting, 1 = standing, 2 = stepping
    bout_length = 0
    bout_index = 0
    for index,value in enumerate(pitch.data):

        classification.data[index] = current_classification

        if current_classification == 0:

            if value >= transition_sit_stand:

                if bout_length == 0:
                    bout_length = 1
                    bout_index = index

                else:
                    bout_length += 1
                    if bout_length >= ten_seconds_in_samples:
                        current_classification = 1
                        classification.data[bout_index:index] = 1

            else:
                bout_length = 0
                bout_index = index

        elif current_classification == 1:
            if value <= transition_stand_sit:

                if bout_length == 0:
                    bout_length = 1
                    bout_index = index

                else:
                    bout_length += 1
                    if bout_length >= ten_seconds_in_samples:
                        current_classification = 0
                        classification.data[bout_index:index] = 0

            else:
                bout_length = 0
                bout_index = index

    return classification

def infer_sleep_actiheart(actiheart_activity, actiheart_ecg):

    ecg_ma = actiheart_ecg.moving_average(30)
    activity_ma = actiheart_activity.moving_average(30)

    ecg_norm = ecg_ma.clone()
    ecg_norm.normalise()
    activity_norm = activity_ma.clone()
    activity_norm.normalise()

    product = Channel.Channel("Probability of being awake")
    product.set_contents( np.multiply(ecg_norm.data, activity_norm.data), ecg_norm.timestamps)
    product.moving_average(30)

    #print(product.data)
    #print(activity_norm.data)
    #print(ecg_norm.data)
    return product

def infer_magic(x,y,z):

    xdiff = np.abs(np.diff(x.data))
    ydiff = np.abs(np.diff(y.data))
    zdiff = np.abs(np.diff(z.data))

    xdiffchan = Channel.Channel("Xdiff")
    ydiffchan = Channel.Channel("Ydiff")
    zdiffchan = Channel.Channel("Zdiff")

    xdiffchan.set_contents(xdiff, x.timestamps)
    ydiffchan.set_contents(ydiff, y.timestamps)
    zdiffchan.set_contents(zdiff, z.timestamps)

    vmdiff = infer_vector_magnitude(xdiffchan, ydiffchan, zdiffchan)
    vmdiff.name = "Magic"

    return vmdiff

def infer_vector_magnitude(x,y,z):

    result = Channel.Channel("VM")

    result.set_contents( np.sqrt( np.multiply(x.data,x.data) + np.multiply(y.data,y.data) + np.multiply(z.data,z.data) ), x.timestamps )

    if x.sparsely_timestamped:
        result.indices = x.indices
        result.sparsely_timestamped = True

    return result

def infer_pitch_roll(x,y,z):

    pitch = Channel.Channel("Pitch")
    roll = Channel.Channel("Roll")

    pitch_degrees = np.arctan(x.data/np.sqrt((y.data*y.data) + (z.data*z.data))) * 180.0/np.pi
    roll_degrees = np.arctan(y.data/np.sqrt((x.data*x.data) + (z.data*z.data))) * 180.0/np.pi

    pitch.set_contents( pitch_degrees, x.timestamps)
    roll.set_contents( roll_degrees, x.timestamps)

    if x.sparsely_timestamped:
        pitch.indices = x.indices
        roll.indices = x.indices
        pitch.sparsely_timestamped = True
        roll.sparsely_timestamped = True

    return [pitch, roll]

def infer_enmo(vm):

    result = Channel.Channel("ENMO")

    result.set_contents( (vm.data - 1.0)*1000.0 , vm.timestamps )

    result.data[np.where(result.data < 0)] = 0

    if vm.sparsely_timestamped:
        result.indices = vm.indices
        result.sparsely_timestamped = True

    return result

def infer_enmo_a(vm):

    result = Channel.Channel("ENMOa")

    result.set_contents( np.absolute((vm.data - 1.0)*1000.0) , vm.timestamps )

    if vm.sparsely_timestamped:
        result.indices = vm.indices
        result.sparsely_timestamped = True
    return result

def infer_vm_hpf(vm):
    """Apply high pass filter to VM at 0.2 hertz. Absolute resulting data. Returned in mg. """

    vm_hpf = pampro_fourier.high_pass_filter(vm, 0.2, frequency=vm.frequency, order=4)
    vm_hpf.name = "VM_HPF"
    vm_hpf.data = np.multiply(1000.0, abs(vm_hpf.data))

    if vm.sparsely_timestamped:
        vm_hpf.indices = vm.indices
        vm_hpf.sparsely_timestamped = True

    return vm_hpf

def infer_nonwear_actigraph(counts, zero_minutes=timedelta(minutes=60)):

    #nonwear = Channel.Channel("Nonwear")

    nonwear_bouts = counts.bouts(0, 0)
    Bout.cache_lengths(nonwear_bouts)
    nonwear_bouts = Bout.limit_to_lengths(nonwear_bouts, min_length=zero_minutes)

    #print("Num nonwear bouts: ", len(nonwear_bouts))
    wear_bouts = Bout.time_period_minus_bouts([counts.timeframe[0], counts.timeframe[1]], nonwear_bouts)

    """
    wear = counts.subset_using_bouts(wear_bouts, "Wear_only", substitute_value=-1)
    #wear.delete_windows(nonwear_bouts)


    print("wear.data length: " + str(len(wear.data)))
    bad_indices = np.where(wear.data == -1)
    print("Bad indices length:")
    print(len(bad_indices[0]))
    wear.data = np.delete(wear.data, bad_indices[0], None)
    wear.timestamps = np.delete(wear.timestamps, bad_indices[0], None)
    print("Here")
    wear.calculate_timeframe()

    print("Calculated timeframe")
    wear_with_missings = counts.subset_using_bouts(wear_bouts, "Wear_minuses", substitute_value=-1)
    print("Done infer_nonwear")

    return [wear, wear_with_missings, wear_bouts, nonwear_bouts]
    """
    return nonwear_bouts, wear_bouts



def infer_still_bouts_triaxial(x, y, z, window_size=timedelta(seconds=10), noise_cutoff_mg=13, approximate_timestamps=False):

    results = Time_Series.Time_Series("Results")
    timeframe = (time_utilities.start_of_hour(x.timeframe[0]), time_utilities.end_of_hour(x.timeframe[1]))

    if x.sparsely_timestamped:
        results.add_channels(x.piecewise_statistics(window_size.seconds*int(x.frequency), statistics=[("generic",["std"])], time_period=timeframe))
        results.add_channels(y.piecewise_statistics(window_size.seconds*int(x.frequency), statistics=[("generic",["std"])], time_period=timeframe))
        results.add_channels(z.piecewise_statistics(window_size.seconds*int(x.frequency), statistics=[("generic",["std"])], time_period=timeframe))

    else:
        results.add_channels(x.piecewise_statistics(window_size, statistics=[("generic",["std"])], time_period=timeframe))
        results.add_channels(y.piecewise_statistics(window_size, statistics=[("generic",["std"])], time_period=timeframe))
        results.add_channels(z.piecewise_statistics(window_size, statistics=[("generic",["std"])], time_period=timeframe))

    x_std = results.get_channel(x.name + "_std")
    y_std = results.get_channel(y.name + "_std")
    z_std = results.get_channel(z.name + "_std")

    # Find bouts where monitor was still for long periods
    x_bouts = x_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    y_bouts = y_std.bouts(0, float(noise_cutoff_mg)/1000.0)
    z_bouts = z_std.bouts(0, float(noise_cutoff_mg)/1000.0)

    # Get the times where those bouts overlap
    x_intersect_y = Bout.bout_list_intersection(x_bouts, y_bouts)
    x_intersect_y_intersect_z = Bout.bout_list_intersection(x_intersect_y, z_bouts)

    if approximate_timestamps:
        Bout.approximate_timestamps(x_intersect_y_intersect_z, x)

    return x_intersect_y_intersect_z


def infer_nonwear_triaxial(x, y, z, minimum_length=timedelta(hours=1), noise_cutoff_mg=13, return_nonwear_binary=False):

    ''' Use the 3 channels of triaxial acceleration to infer periods of nonwear '''

    x_intersect_y_intersect_z = infer_still_bouts_triaxial(x,y,z, approximate_timestamps=True)

    Bout.cache_lengths(x_intersect_y_intersect_z)


    x_intersect_y_intersect_z = Bout.limit_to_lengths(x_intersect_y_intersect_z, min_length=minimum_length)

    if return_nonwear_binary:
        # Create a parallel, binary channel indicating if that time point was in or out of wear
        nonwear_binary = Channel.channel_from_bouts(x_intersect_y_intersect_z, x.timeframe, False, "nonwear", skeleton=x)

        return (x_intersect_y_intersect_z, nonwear_binary)
    else:
        return x_intersect_y_intersect_z

def infer_valid_days(channel, wear_bouts, valid_criterion=timedelta(hours=10)):

    #Generate day-long windows
    start = time_utilities.start_of_day(channel.timestamps[0])
    day_windows = []
    while start < channel.timeframe[1]:
        day_windows.append(Bout.Bout(start, start+timedelta(days=1)))
        start += timedelta(days=1)

    valid_windows = []
    for window in day_windows:
        #how much does all of wear_bouts intersect with window?
        intersections = Bout.bout_list_intersection([window], wear_bouts)

        total = Bout.total_time(intersections)

        if total > valid_criterion:
            #window.draw_properties={"lw":0, "facecolor":[1,0,0], "alpha":0.25}
            valid_windows.append(window)
        #print total


    invalid_windows = Bout.time_period_minus_bouts(channel.timeframe, valid_windows)

    return(invalid_windows, valid_windows)
