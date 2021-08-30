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

Hypervolume User Recommendation
=====================
This class gives a sorted list of users with most overlap in a given hyper space.
"""

import pandas as pd
from analytics.analytics import Analytics


class HypVolUserRecommend(Analytics):

    def __init__(self, hyp_vol_df=None, hyp_vol_df_path=None, sort_type=0):
        """
        Initializes the class object
        @param hyp_vol_df: Data frame containing hypervolume overlaps
        @param hyp_vol_df: path to csv file containing hypervolume overlaps, expects file is standard
        hypervolume overlap format
        @param sort_type: 0 = sort by ["user_overlap_others_mean", "user_overlap_by_others_mean"]
                                 1 = sort by ["user_overlap_by_others_mean", "user_overlap_others_mean"]

        """

        if hyp_vol_df_path is None:
            self.hyp_vol_df = hyp_vol_df
        else:
            self.hyp_vol_df = pd.read_csv(hyp_vol_df_path)
        self.users = self.hyp_vol_df.V1.unique()
        self.hyp_vol_data_mean = self.hyp_vol_df[['V1', 'V2', 'port_mean']]
        self.sort_type = sort_type
        self.means = pd.DataFrame()
        return

    def get_analytics(self):
        user_ol_by_others = dict()
        user_ol_others = dict()
        for user in self.users:
            user_ol_by_others[user] = self.hyp_vol_data_mean[self.hyp_vol_data_mean['V1'] == user]
            user_ol_others[user] = self.hyp_vol_data_mean[self.hyp_vol_data_mean['V2'] == user]

        means_ol_others = [user_ol_others[x].port_mean.mean() for x in self.users]
        means_ol_by_others = [user_ol_by_others[x].port_mean.mean() for x in self.users]

        df_cols = ['user', 'user_overlap_others_mean', 'user_overlap_by_others_mean']
        self.means = pd.DataFrame(zip(self.users, means_ol_others, means_ol_by_others), columns=df_cols)

        if self.sort_type == 0:
            return self.means.sort_values(by=[df_cols[1], df_cols[2]], ascending=[False, False])
        else:
            return self.means.sort_values(by=[df_cols[2], df_cols[1]], ascending=[False, False])
