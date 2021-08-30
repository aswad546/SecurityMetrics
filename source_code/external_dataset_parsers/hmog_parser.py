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

HMOG Parser
===========
A class that parses HMOG data set. Current implementation only pulls Touch 
data belonging to scroll events in the dataset
More details on dataset: http://www.cs.wm.edu/~qyang/hmog.html
"""

import collections
import os
from io import StringIO

import pandas as pd

from biometrics import touch_biometric
from external_dataset_parsers.external_parser import ExternalParser
from utilities import io_utilities as io


class HMOGParser(ExternalParser):
    """A class that contains parsers for hmog dataset"""

    def __init__(self, dataset_path=None, user_limit=None):
        """Initializer for hmogParser.
         Takes optional user_limit and only uses user_limit users' data"""
        self.dataset_path = dataset_path
        # dirs = io.get_directory_list(self.dataset_path)
        # self.users = dirs[:min(len(dirs), user_limit)]
        # print(str(len(self.len(dirs))), 'Users found, using data for', self.users, "users")
        return

    def raw_to_feature_vectors(self, raw_data_path, output_path=None, limit=None):
        """ 
        Takes root path of the dataset where the individual user folders are located
         Current implementation only supports dumping Touch input data
        from HMOG"""

        self.dataset_path = raw_data_path
        dirs = io.get_directory_list(self.dataset_path)

        if limit is None:
            users = dirs
        else:
            users = dirs[:min(len(dirs), limit)]

        print("Users found: " + str(len(users)))
        # if not os.path.isdir(output_path):
        #    os.makedirs(output_path)

        tb = touch_biometric.TouchBiometric()
        # create a string that can be loaded as csv to dataframe
        csv_str = StringIO()
        csv_str.write('user,')
        csv_str.write('%s' % ','.join(map(str, tb.get_feature_header()[5:-1])) + '\n')

        for u in users:
            print("Generating Features for user",u,"Please Wait")
            tps = self.get_user_raw_touch_data(u)
            if len(tps) < 6:
                continue
            # fout = open(os.path.join(output_path, u + "_touch_fv"), 'w')
            # fout.write('%s' % ', '.join(map(str, tb.get_feature_header())) + '\n')

            i = 0
            swipe_id = tps[0].swipe_id
            while i < len(tps):
                start_idx = i
                while swipe_id == tps[i].swipe_id:
                    i += 1
                    if i == len(tps):
                        break

                end_idx = i - 1
                if i == len(tps):
                    swipe_id = tps[i - 1].swipe_id
                else:
                    swipe_id = tps[i].swipe_id

                last_swipe_end_time = tps[0].tstamp  # if last swipe DNE, set to current rather than 0
                if (start_idx != 0):
                    last_swipe_end_time = tps[start_idx - 1].tstamp
                if end_idx - start_idx > 5:
                    # Cleanse data
                    fv = tb.raw_to_feature_vector((tps[start_idx:end_idx], last_swipe_end_time))
                    if fv[5] <= 600000 and fv[5] > 0:  # interstroke_time must be above 0 and less than 10 minutes
                        if fv[6] <= 60000 and fv[6] > 0:  # stroke_duration must be above 0 and less than 1 minute
                            if fv[3] > -1:  # swipe_id should be greater than -1
                                csv_str.write(u + ',' + str(fv[5:-1])[1:-1] + '\n')
            # fout.close()
        # print(csv_str.getvalue())

        csv_str.seek(0)
        df = pd.read_csv(csv_str, header=0, index_col=False)
        if output_path is not None:
            df.to_csv(output_path, index=False, mode='w+')
            print("Feature Generation Complete files are available at path", output_path)
        else:
            print("Feature Generation Complete")
        return df

    def get_feature_vectors(self, input_path, limit=None):
        df = pd.read_csv(input_path, header=0, index_col=False)
        if limit is not None:
            ctr = collections.Counter(df['user'])
            top_users_ctr = ctr.most_common(min(len(ctr), limit))
            users = list()
            for user in top_users_ctr:
                users.append(user[0])
        df = df[df['user'].isin(users)]
        return df

    @property
    def get_user_list(self):
        """Returns the list of users"""
        return self.users

    def get_activityID_to_task_desc(self, u):
        """Maps activityID to activity description"""
        user_data_path = os.path.join(self.dataset_path, u)
        dirs = io.get_directory_list(user_data_path)
        activity = dict()
        for d in dirs:
            if not d.startswith(u + "_session_"):
                continue
            touch_data_path = os.path.join(self.dataset_path, u, d, "Activity.csv")
            with open(touch_data_path) as f:
                lines = f.readlines()
            for l in lines:
                tokens = [int(x) for x in l.split(',')]
                activity[tokens[0]] = self.get_task_desc(tokens[-2])
        return activity

    def get_task_desc(self, task_id):
        if task_id in [1, 7, 13, 19]:
            return "Reading + Sitting"
        elif task_id in [2, 8, 14, 20]:
            return "Reading + Walking"
        elif task_id in [3, 9, 15, 21]:
            return "Writing + Sitting"
        elif task_id in [4, 10, 16, 22]:
            return "Writing + Walking"
        elif task_id in [5, 11, 17, 23]:
            return "Map + Sitting"
        elif task_id in [6, 12, 18, 24]:
            return "Map + Walking"
        else:
            raise ValueError("Invalid Activity ID received: " + str(task_id))

    def get_user_raw_touch_data(self, u):
        """Returns raw touch data of user 'u' """
        touchpoints = list()
        task_desc = self.get_activityID_to_task_desc(u)
        user_data_path = os.path.join(self.dataset_path, u)
        dirs = io.get_directory_list(user_data_path)
        for d in dirs:
            if not d.startswith(u + "_session_"):
                continue
            session = d.split('_')[-1]
            touch_data_path = os.path.join(self.dataset_path, u, d, "ScrollEvent.csv")
            with open(touch_data_path) as f:
                lines = f.readlines()

            for l in lines:
                l = [float(x) for x in l.split(',')]
                touchpoints.append(touch_biometric.TouchPoint(u, session, int(l[4]), l[0], l[11], l[12],
                                                              l[13], l[14], l[-1],
                                                              task_desc[int(l[3])]))
        return touchpoints

    def get_all_raw_touch_data(self):
        """Returns raw touch data of users"""
        touchpoints = list()
        for u in self.users:
            task_desc = self.get_activityID_to_task_desc(u)
            swipe_id = -1
            dirs = os.listdir(os.path.join(self.dataset_path, u))
            for d in dirs:
                if not d.startswith(u + "_session_"):
                    continue
                session = d.split('_')[-1]
                touch_data_path = os.path.join(self.dataset_path, u, d, "ScrollEvent.csv")
                with open(touch_data_path) as f:
                    lines = f.readlines()

                for l in lines:
                    l = [float(x) for x in l.split(',')]
                    touchpoints.append(touch_biometric.TouchPoint(u, session, int(l[4]), l[0], l[11], l[12],
                                                                  l[13], l[14], l[-1],
                                                                  task_desc[int(l[3])]))
        return touchpoints
