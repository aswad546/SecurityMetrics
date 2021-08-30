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
Gait Data Parser
=====================
This class implements Gait data parser from accelerometer
"""
import sys

import numpy as np
import pandas as pd

from biometrics.gait_acl_biometric import GaitAclBiometric
from external_dataset_parsers.external_parser import ExternalParser


class GaitParser(ExternalParser):

    def __init__(self):
        """Initializer for gait parser.
         """
        self.data_set_path = None
        self.data = None
        self.users = None
        self.win_size = None
        self.win_step = None
        self.col_names = GaitAclBiometric().get_feature_header()
        self.freq_col_names = ['users']
        freq_col_n = [f"{num}_fft_freq" for num in range(40)]
        self.freq_col_names.extend(freq_col_n)
        self.feat_df = pd.DataFrame()
        self.freq_df = pd.DataFrame()
        return

    def raw_to_feature_vectors(self, data=None, raw_data_path=None, output_path=None, limit=None, win_size=8, win_step=4
                               ):
        """

        @param win_step:
        @param win_size:
        @param data:
        @param raw_data_path: Path for raw data
        @param output_path: Path to save extracted features
        @param limit: User limit for number of users
        @return:
        """
        if data is None:
            if raw_data_path is None:
                print("Please pass data or enter raw data path")
                sys.exit(1)
            else:
                self.data = pd.read_csv(raw_data_path, header=False)

        self.data = data
        self.users = self.data.users.unique().tolist()
        self.win_size = int(win_size)
        self.win_step = int(win_step)

        for user in self.users:
            user_df = self.data[self.data.users == user]
            win_start_sample = 0
            win_end_sample = win_start_sample + self.win_size
            final_sample = len(user_df)
            feat_df = pd.DataFrame()
            freq_df = pd.DataFrame()
            feat_ext_sts = False

            while feat_ext_sts is False:
                win_data = user_df.drop("users", axis=1).iloc[win_start_sample:win_end_sample, :].values.flatten()
                fft_coef, fft_freq = GaitAclBiometric().raw_to_feature_vector(win_data)
                win_start_sample += self.win_step
                win_end_sample += self.win_step
                fft_coef = np.insert(fft_coef, 0, user)
                fft_freq = np.insert(fft_freq, 0, user)
                feat_df = pd.concat([feat_df, pd.DataFrame(fft_coef).T])
                freq_df = pd.concat([freq_df, pd.DataFrame(fft_freq).T])

                if win_end_sample >= final_sample:
                    feat_ext_sts = True

            self.feat_df = pd.concat([self.feat_df, feat_df]).reset_index(drop=True)
            self.freq_df = pd.concat([self.freq_df, freq_df]).reset_index(drop=True)
        self.feat_df.columns = self.col_names
        self.freq_df.columns = self.freq_col_names

        return self.feat_df
