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

Data Overlap interval
=====================
This class extracts data from overlapping regions
"""

import sys

import pandas as pd

from source_code.analytics.analytics import Analytics
from source_code.dataset.outlier_removal import OutLierRemoval


def over_lap_range(ser1, ser2):
    search_min = max(ser1.min(), ser2.min())
    search_max = min(ser1.max(), ser2.max())

    return search_min, search_max


class OverLapInt(Analytics):

    def __init__(self, df1, df2, std_dev=None):
        """
        Initializes the class object



        """
        self.ol_flag = False
        self.df1 = df1.copy()
        self.df1_user_name = None
        self.df2_user_name = None
        if 'user' in self.df1:
            self.df1_user_name = self.df1.iloc[0, 0]
            self.df1 = self.df1.drop('user', axis=1)

        self.df2 = df2.copy()
        if 'user' in self.df2:
            self.df2_user_name = self.df2.iloc[0, 0]
            self.df2 = self.df2.drop('user', axis=1)

        self.std_dev = std_dev
        true_cond = (len(self.df1.columns.values) == len(self.df2.columns.values)) \
                    & (self.df1.columns.values == self.df1.columns.values).all()

        if self.std_dev is not None:
            self.df1 = OutLierRemoval().operate(data=self.df1, z_score_threshold=self.std_dev)
            self.df2 = OutLierRemoval().operate(data=self.df2, z_score_threshold=self.std_dev)
            if len(self.df1) == 0 | len(self.df2) == 0:
                self.ol_flag = True

        if not true_cond:
            print("both data frames need to have same columns")
            sys.exit(1)

        return

    def get_analytics(self):
        df_overlaps = pd.DataFrame()
        if self.ol_flag is False:
            if len(self.df1) == 0 | len(self.df2) == 0:
                print("Error Empty dataframe passed")
                return sys.exit(1)
        else:
            return self.df1, self.df2
        for col_name, content in self.df1.iteritems():
            ser_a = self.df1[col_name]
            ser_b = self.df2[col_name]
            overlap_min_max = over_lap_range(ser_a, ser_b)
            df_overlaps[col_name] = [i for i in overlap_min_max]

        df_overlaps = df_overlaps.rename(index={0: 'min', 1: 'max'})
        df1_filter = pd.DataFrame()
        df2_filter = pd.DataFrame()
        for col_name, content in self.df1.iteritems():
            df1_filter[col_name] = self.df1[col_name].between(left=df_overlaps.loc['min', col_name],
                                                              right=df_overlaps.loc['max', col_name], inclusive=True)

            df2_filter[col_name] = self.df2[col_name].between(left=df_overlaps.loc['min', col_name],
                                                              right=df_overlaps.loc['max', col_name], inclusive=True)

        df1_overlap = self.df1[df1_filter]
        df2_overlap = self.df2[df2_filter]
        df1_overlap = df1_overlap.dropna()
        df2_overlap = df2_overlap.dropna()
        df1_overlap.reset_index(drop=True)
        df2_overlap.reset_index(drop=True)

        if self.df1_user_name is not None:
            df1_overlap.insert(0, 'user', self.df1_user_name)

        if self.df2_user_name is not None:
            df2_overlap.insert(0, 'user', self.df2_user_name)

        return df1_overlap, df2_overlap

    pass
