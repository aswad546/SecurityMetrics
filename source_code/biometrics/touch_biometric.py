"""
MIT License

Copyright (c) 2021, Sohail Habib

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

------------------------------------------------------------------------------------------------------------------------

Touch Biometric
===============
Touch Biometric class implementation. 
An Implementation of:
    @ARTICLE{touchalytics,
    author    = {Mario Frank and Ralf Biedert and Eugene Ma and Ivan Martinovic and Dawn Song},
    journal={Information Forensics and Security, IEEE Transactions on},
     title={Touchalytics: On the Applicability of Touchscreen Input as a Behavioral Biometric for
      Continuous Authentication},
    year={2013},
    month={1 },
    volume={8},
    number={1},
    pages={136 -148},
    doi={http://dx.doi.org/10.1109/TIFS.2012.2225048},
    ISSN={1556-6013},}
"""

import math
import cmath
from source_code.biometrics.biometric import Biometric


class TouchBiometric(Biometric):
    """Touch biometric which deals with the swiping behaviour. 
        Does not deal with multi-stroke gestures.
    """

    def get_feature_header(self):
        """Returns a list containing name of all features."""
        return ['user_id', 'session_id', 'activity', 'swipe_id', 'direction_flag', \
                'interstroke_time', 'stroke_duration', 'start_x', 'start_y', \
                'stop_x', 'stop_y', 'direct_end_to_end_distance', \
                'mean_resultant_length', 'direction_of_end_to_end_line', \
                'pairwise_velocity_20p_perc', 'pairwise_velocity_50p_perc', \
                'pairwise_velocity_80p_perc', 'pairwise_acc_20p_perc', \
                'pairwise_acc_50p_perc', 'pairwise_acc_80p_perc', \
                'median_velocity_last_3_pts', 'largest_dev_end_to_end_line', \
                'dev_end_to_end_line_20p_perc', 'dev_end_to_end_line_50p_perc', \
                'dev_end_to_end_line_80p_perc', 'avg_direction', \
                'length_of_trajectory', 'ratio_end_to_end_dist_length_of_traj', \
                'avg_velocity', 'median_acc_first_5_pts', 'mid_stroke_pressure', \
                'mid_stroke_area', 'phone_orientation']

    def get_required_raw_data(self):
        """Returns a list raw data required to calculate features."""
        return 'tuple consisting of a list of TouchPoint objects of current' + \
               'swipe and timestamp for last swipe'

    def raw_to_feature_vector(self, raw_data):
        """
        Returns a feature vector generated from raw_data, 
        which is a tuple (list of TouchPoint, last swipe time).
        """
        tps = raw_data[0]
        last_swipe_time = raw_data[1]
        fv = [0.0] * len(self.get_feature_header())

        fv[0] = tps[0].user_id
        fv[1] = tps[0].session_id
        fv[2] = tps[0].activity
        fv[3] = tps[0].swipe_id

        fv[5] = tps[0].tstamp - last_swipe_time  # interstroke_time
        fv[6] = tps[-1].tstamp - tps[0].tstamp  # stroke_duration
        fv[7] = tps[0].x
        fv[8] = tps[0].y
        fv[9] = tps[-1].x
        fv[10] = tps[-1].y

        # direct end-to-end distance
        fv[11] = math.sqrt(math.pow(fv[9] - fv[7], 2) + math.pow(fv[10] - fv[6], 2))

        x_disp, y_disp, t_disp = list(), list(), list()
        for i in range(1, len(tps)):
            x_disp.append(tps[i].x - tps[i - 1].x)
            y_disp.append(tps[i].y - tps[i - 1].y)
            t_disp.append(tps[i].tstamp - tps[i - 1].tstamp)

        pairw_angle = []
        for i in range(0, len(x_disp)):
            pairw_angle.append(math.atan2(y_disp[i], x_disp[i]))

        fv[12] = circ_r(pairw_angle)  # 8 Mean Resultant Length

        # Direction Flag (up, down, left, right are 0,1,2,3)
        fv[4] = 'down'  # down is default
        x_diff = fv[9] - fv[7]
        y_diff = fv[10] - fv[8]
        if math.fabs(x_diff) > math.fabs(y_diff):
            if x_diff < 0:
                fv[4] = 'left'
            else:
                fv[4] = 'right'
        else:
            if y_diff < 0:
                fv[4] = 'up'

        fv[13] = math.atan2(fv[10] - fv[8], fv[9] - fv[7])  # direction of end-to-end line

        pairw_dist = []
        for i in range(0, len(x_disp)):
            pairw_dist.append(math.sqrt(math.pow(x_disp[i], 2) + math.pow(y_disp[i], 2)))

        pairw_v = []
        for i in range(0, len(pairw_dist)):
            if t_disp[i] == 0:
                pairw_v.append(0)
            else:
                pairw_v.append(pairw_dist[i] / t_disp[i])
        max_v = max(pairw_v)  # replace 0 v with max(v) as that is more appropriate
        for i in range(0, len(pairw_v)):
            if pairw_v[i] == 0:
                pairw_v[i] = max_v
        pairw_a = []
        for i in range(1, len(pairw_v)):
            pairw_a.append(pairw_v[i] - pairw_v[i - 1])
        for i in range(0, len(pairw_a)):
            if t_disp[i] == 0:
                pairw_a[i] = 0  # replace with max acceleration-done below
            else:
                pairw_a[i] = pairw_a[i] / t_disp[i]

        max_a = max(pairw_a)
        for i in range(0, len(pairw_a)):
            if pairw_a[i] == 0:
                pairw_a[i] = max_a

        pairw_v3 = pairw_v[-4:]
        pairw_a6 = pairw_a[0:6]
        pairw_v.sort()
        pairw_a.sort()
        pairw_v3.sort()
        pairw_a6.sort()

        fv[14] = percentile(pairw_v, 0.20)  # 20% percentile of velocity
        fv[15] = percentile(pairw_v, 0.50)  # 50% percentile of velocity
        fv[16] = percentile(pairw_v, 0.80)  # 80% percentile of velocity
        fv[17] = percentile(pairw_a, 0.20)  # 20% percentile of acceleration
        fv[18] = percentile(pairw_a, 0.50)  # 50% percentile of acceleration
        fv[19] = percentile(pairw_a, 0.80)  # 80% percentile of acceleration

        fv[20] = percentile(pairw_v3, 0.50)  # median velocity at last 3 points

        # 26 Largest deviation from end-end line
        xvek, yvek = list(), list()
        for i in range(0, len(tps)):
            xvek.append(tps[i].x - fv[7])
            yvek.append(tps[i].y - fv[8])

        pervek = [yvek[-1], xvek[-1] * -1, 0]
        temp = math.sqrt(pervek[0] * pervek[0] + pervek[1] * pervek[1])
        if temp == 0:
            for i in range(0, len(pervek)):
                pervek[i] = 0
        else:
            for i in range(0, len(pervek)):
                pervek[i] = pervek[i] / temp

        proj_perp_straight = []
        abs_proj = []
        for i in range(0, len(xvek)):
            proj_perp_straight.append(xvek[i] * pervek[0] + yvek[i] * pervek[1])
            abs_proj.append(math.fabs(proj_perp_straight[i]))
        fv[21] = max(abs_proj)
        fv[22] = percentile(abs_proj, 0.20)  # 20% deviation from end-end line
        fv[23] = percentile(abs_proj, 0.50)  # 50% deviation from end-end line
        fv[24] = percentile(abs_proj, 0.80)  # 80% deviation from end-end line

        fv[25] = circ_mean(pairw_angle)  # average direction of ensemble pairs

        fv[26] = 0  # length of trajectory
        for pd in pairw_dist:
            fv[26] += pd

        if fv[26] == 0:
            fv[27] = 0  # Ratio of direct distance and trajectory length
        else:
            fv[27] = fv[11] / fv[26]

        if fv[6] == 0:  # fv[6] is stroke duration; fv[26] length of traj.
            fv[28] = 0  # Average Velocity
        else:
            fv[28] = fv[26] / fv[6]

        fv[29] = percentile(pairw_a6, 0.50)  # Median acceleration at first 5 points

        fv[30] = tps[int(len(tps) / 2)].pressure  # pressure in the middle of stroke
        fv[31] = tps[int(len(tps) / 2)].area  # area in the middle of stroke

        fv[32] = tps[0].orientation

        return fv


class TouchPoint:
    """Data for a single touch point on the screen"""

    def __init__(self, user_id, session_id, swipe_id, tstamp, x, y, pressure,
                 area, orientation, activity):
        self.user_id = user_id
        self.session_id = session_id
        self.swipe_id = swipe_id
        self.tstamp = tstamp
        self.x = x
        self.y = y
        self.pressure = pressure
        self.area = area
        self.orientation = orientation
        self.activity = activity

    def __str__(self):
        return "user: " + self.user_id + ", session_id: " + str(self.session_id) + \
               ", swipe_id: " + str(self.swipe_id) + \
               ", time: " + str(self.tstamp) + ", x: " + str(self.x) + ", y: " + str(self.y) + \
               ", pressure: " + str(self.pressure) + ", area: " + str(self.area) + \
               ", orientation: " + str(self.orientation) + ", activity: " + str(self.activity)


# Translated from www.kyb.mpg.de/~berens/circStat.html
def circ_r(x):
    r = cmath.exp(1j * x[0])
    for i in range(1, len(x)):
        r += cmath.exp(1j * x[i])
    return abs(r) / len(x)


# Translated from www.kyb.mpg.de/~berens/circStat.html
def circ_mean(x):
    r = cmath.exp(1j * x[0])
    for i in range(1, len(x)):
        r += cmath.exp(1j * x[i])
    return math.atan2(r.imag, r.real)


def percentile(N, percent, key=lambda x: x):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c - k)
    d1 = key(N[int(c)]) * (k - f)
    return d0 + d1
