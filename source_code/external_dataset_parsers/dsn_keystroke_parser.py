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
DSN keystroke parser
=====================
This class implements parser for DSN keystroke data
@inproceedings{killourhy2009comparing,
  title={Comparing anomaly-detection algorithms for keystroke dynamics},
  author={Killourhy, Kevin S and Maxion, Roy A},
  booktitle={2009 IEEE/IFIP International Conference on Dependable Systems \& Networks},
  pages={125--134},
  year={2009},
  organization={IEEE}
}
"""
import collections
import os
import sys

import pandas as pd

from external_dataset_parsers.external_parser import ExternalParser

"""
    Parses the DSN keystroke data
    More details:
    Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics,"
    In Proceedings of the 39th Annual International Conference on Dependable Systems and Networks (DSN-2009)
"""


class DSNParser(ExternalParser):
    def __init__(self):
        return

    def raw_to_feature_vectors(self, raw_data_path, output_path=None, limit=None):
        ''' We ignore limit since data is already a csv
        
        Parameters
        ----------
        raw_data_path : TYPE
            DESCRIPTION.
        output_path : TYPE, optional
            DESCRIPTION. The default is None.
        limit : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''

        if not os.path.exists(raw_data_path):
            print("Path does not exist " + raw_data_path)
            sys.exit(1)

        with open(raw_data_path) as f:
            lines = f.readlines()

        df = pd.read_csv(raw_data_path, header=0, index_col=False)
        del df['sessionIndex']
        del df['rep']
        df.rename(columns={'subject': 'user'}, inplace=True)

        if output_path is not None:
            df.to_csv(output_path, index=False)
        return df

    def get_feature_vectors(self, input_path, limit=None):
        df = pd.read_csv(input_path, header=0, index_col=False)

        if limit is not None:
            ctr = collections.Counter(df['user'])
            top_users_ctr = ctr.most_common(min(len(ctr), limit))
            users = list()
            for user in top_users_ctr:
                users.append(user[0])
        else:
            users = df.user.unique()
        df = df[df['user'].isin(users)]
        return df
