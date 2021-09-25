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

Mouse Biometric
===============
Mouse Biometric class implementation.

"""

import math
import numpy as np
import numpy.linalg as la
from statistics import stdev
from statistics import mean
from source_code.biometrics.biometric import Biometric


class MouseBiometric(Biometric):

    def get_feature_header(self):
        """Returns a list containing name of all features."""
        return ['user_id', 'std_dev_direction_angle', 'max_direction_angle', 'min_direction_angle',
                'mean_direction_angle',
                'std_dev_speed', 'max_speed', 'min_speed', 'mean_speed',
                'std_dev_acc', 'max_acc', 'min_acc', 'mean_acc',
                'click_duration',
                'std_dev_angle_of_curvature', 'max_angle_of_curvature', 'min_angle_of_curvature',
                'mean_angle_of_curvature',
                'std_dev_curvature_distance', 'max_curvature_distance', 'min_curvature_distance',
                'mean_curvature_distance']

    def calculate_angle_3points(self, centrepoint, point2,
                                point3):  # I am uncertain about what to do here leaving it in

        ba = point2 - centrepoint
        bc = point3 - centrepoint
        if (np.linalg.norm(ba) * np.linalg.norm(bc)) != 0:
            cosine_angle = np.arccos(np.round(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), 5))

        else:
            cosine_angle = 0  # set to 1 as cos_inverse of 1 is 0 as a denominator of 0 implies that the set of points in on top of itself

        return cosine_angle

    def extract_features(self, segment, user):
        """
        Features List:
        0     - User ID
        1-4   - std_dev_direction_angle, max_direction_angle, min_direction_angle, mean_direction_angle
        5-8   - std_dev_speed, max_speed, min_speed, mean_speed
        9-12   - std_dev_acc, max_acc, min_acc, mean_acc
        13    - click duration,
        14-17 - std_dev_angle_of_curvature, max_angle_of_curvature,  min_angle_of_curvature, mean_angle_of_curvature
        18-21 - std_dev_curvature_distance, max_curvature_distance,  min_curvature_distance, mean_curvature_distance
        """
        if len(segment) < 6:
            return
        fv = []
        fv.append(user)  # FV[0]
        '''print("new segment")'''
        downclick_index = 0
        seg_distance = 0
        pairw_speed = []
        pairw_distance = []
        angle = []
        for x in range(len(segment) - 1):  # calculations requiring 2 consecutive points

            if segment[x][1] == "mouseleftdown":
                downclick_index = x

            point1 = np.array([int(segment[x][3]), int(segment[x][4])])  # anchor point
            point2 = np.array([int(segment[x + 1][3]), int(segment[x + 1][4])])
            horz_point = np.array([int(segment[x + 1][3]), int(segment[x][4])])

            '''DIRECTION CALCULATION(angle)'''

            angle.append(self.calculate_angle_3points(point1, point2, horz_point))  # in cos_theta form
            if point2[1] < point1[1]:
                angle[x] = angle[x] * -1  # for angles below the horizontal line

            distance = math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))
            pairw_distance.append(distance)
            time = abs(int(segment[x][0]) - int(segment[x + 1][0]))
            seg_distance += distance

            '''PAIRWISE SPEED CALCULATION'''
            if time != 0:
                pairw_speed.append(distance / time)
            else:
                pairw_speed.append(0)

        # DIRECTION ANGLE BASED FEATURES
        fv.append(stdev(angle))  # FV[1]
        fv.append(max(angle))  # FV[2]
        fv.append(min(angle))  # FV[3]
        fv.append(mean(angle))  # FV[4]

        # SPEED BASED FEATURES
        fv.append(stdev(pairw_speed))  # FV[5]
        fv.append(max(pairw_speed))  # FV[6]
        fv.append(min(pairw_speed))  # FV[7]
        fv.append(mean(pairw_speed))  # FV[8]

        '''PAIRWISE ACC CALCULATION'''
        pairw_acc = []
        for s in range(len(pairw_speed) - 1):
            pairw_acc.append(pairw_speed[s + 1] - pairw_speed[s])
            if pairw_distance[s] == 0:
                pairw_acc.append(0)
            else:
                pairw_acc.append(pairw_acc[s] / pairw_distance[s])

        # ACC BASED FEATURES
        fv.append(stdev(pairw_acc))  # FV[9]
        fv.append(max(pairw_acc))  # FV[10]
        fv.append(min(pairw_acc))  # FV[11]
        fv.append(mean(pairw_acc))  # FV[12]

        ''' TOTAL SPEED CALCULATION'''
        seg_time = abs(int(segment[0][0]) - int(segment[len(segment) - 1][0]))
        seg_speed = seg_distance / seg_time

        '''CLICK DURATION'''
        click_dura = abs(int(segment[downclick_index][0]) - int(segment[len(segment) - 1][0]))
        # CLICK DURATION BASED FEATURE
        fv.append(click_dura)  # FV[13]

        angle_of_curvature = []
        distance_of_curvature = []
        for x in range(len(segment) - 2):  # 3 consecutive points

            point1 = np.array([int(segment[x][3]), int(segment[x][4])])
            point2 = np.array([int(segment[x + 1][3]), int(segment[x + 1][4])])  # anchor point, POINT B
            point3 = np.array([int(segment[x + 2][3]), int(segment[x + 2][4])])

            '''ANGLE OF CURVATURE CALCULATION'''

            angle_of_curvature.append(self.calculate_angle_3points(point2, point1, point3))  # in cos_theta form

            '''DISTANCE OF CURVATURE CALCULATION 
            ratio of the length of −→AC to the perpendicular distance from
            point B to the line −→AC'''

            lengthAC = math.sqrt(
                ((point1[0] - point3[0]) ** 2) + ((point1[1] - point3[1]) ** 2))  # Distance between point 1 and 3

            if la.norm(point3 - point1) != 0:
                perpend_dist = np.absolute(np.cross(point3 - point1, point1 - point2)) / la.norm(point3 - point1)
            else:  # point AC are on top of eachother so perpendicular distance is the distance from point A to point B
                perpend_dist = math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

            if perpend_dist != 0:
                distance_of_curvature.append(lengthAC / perpend_dist)
            else:
                distance_of_curvature.append(0)

        # ACC BASED FEATURES
        fv.append(stdev(angle_of_curvature))  # FV[14]
        fv.append(max(angle_of_curvature))  # FV[15]
        fv.append(min(angle_of_curvature))  # FV[16]
        fv.append(mean(angle_of_curvature))  # FV[17]

        # ACC BASED FEATURES
        fv.append(stdev(distance_of_curvature))  # FV[18]
        fv.append(max(distance_of_curvature))  # FV[19]
        fv.append(min(distance_of_curvature))  # FV[20]
        fv.append(mean(distance_of_curvature))  # FV[21]

        return fv
