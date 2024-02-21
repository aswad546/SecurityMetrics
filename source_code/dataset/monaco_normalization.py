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

Normalization Technique by John Monaco
=====================
This class implements normalization Technique by John Monaco

@article{monaco2016robust,
  title={Robust keystroke biometric anomaly detection},
  author={Monaco, John V},
  journal={arXiv preprint arXiv:1606.09075},
  year={2016}

  https://arxiv.org/pdf/1606.09075.pdf
}

pn_i = max(0, min(1, (p_i - p_lower)/ p_upper - p_lower ))

p_neg = floor(mean - (H*std))
p_pos = floor(mean + (H*std))

"""
from dataset.dataset_operation import DatasetOperation
import pandas as pd
import numpy as np


class MonacoNormalize(DatasetOperation):
    """
    Initialize class object
    """

    def __init__(self):
        self.pos_user = None
        self.pos_user_mean_vec = None
        self.pos_user_std_vec = None
        self.bounds = pd.DataFrame(columns=['lower_bound', 'upper_bound'])

        return

    def operate(self, pos_user, data, h=1, output_path=None):
        """

        @param pos_user: Positive user Identification
        @param data: Pandas Data frame containing data to be normalized
        @param h: normalizing paramater for features to  normalize within h SD of the mean.
        @param output_path: Output path for normalized data
        @return monaco normalized dataframe
        """
        df = data
        # print(df.head(10).to_string())
        if 'labels' in df.columns:
            labels = df['labels']
            df = df.drop('labels', axis=1)
            ins_labels = True
        else:
            ins_labels = False
        #careful here
        users = df.user
        pos_user_fil = df['user'] == pos_user

        self.pos_user_mean_vec = df[pos_user_fil].drop('user', axis=1).to_numpy().mean(axis=0).tolist()
        self.pos_user_std_vec = df[pos_user_fil].drop('user', axis=1).to_numpy().std(axis=0).tolist()

        res_df = pd.DataFrame()
        self.pos_user = pos_user
        col_name = df.drop("user", axis=1).columns
        # print(col_name)
        res_df['user'] = users

        for col, mean, std in zip(col_name, self.pos_user_mean_vec, self.pos_user_std_vec):
            fd = df[col].to_numpy()
            lb = np.floor(mean - (1 * std))
            ub = np.ceil(mean + (1 * std))
            self.bounds.loc[col, 'lower_bound'] = lb
            self.bounds.loc[col, 'upper_bound'] = ub
            # print('Here')
            res_df[col] = self.monormalize(feature=fd, lower_bound=lb, upper_bound=ub)

        if output_path is not None:
            res_df.to_csv(output_path, index=False, mode='w+')

        if ins_labels is True:
            res_df['labels'] = labels

        # print(df.to_string())
        #644
        # print()
        # print(res_df.to_string())
        return res_df

    def monormalize(self, feature, lower_bound, upper_bound):
        """
        This function does monaco normalization on a feature vector
        @param feature: numpy feature vector
        @param lower_bound:
        @param upper_bound:
        @return: Monaco Normalized Vector
        """
        lower_bound = lower_bound
        upper_bound = upper_bound
        diff = upper_bound -lower_bound
        if (upper_bound-lower_bound == 0):
            # print('Touching here')
            diff = 1
        n_feat = np.maximum(0, np.minimum(1, (feature - lower_bound) / (diff)))

        return n_feat
