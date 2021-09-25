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

K-means++ attack
=====================
This class implements K-means++ attack from paper K-means++ vs. Behavioral Biometrics: One Loop to
Rule Them All

@inproceedings{negi2018k,
  title={K-means++ vs. Behavioral Biometrics: One Loop to Rule Them All.},
  author={Negi, Parimarjan and Sharma, Prafull and Jain, Vivek and Bahmani, Bahman},
  booktitle={NDSS},
  year={2018}
}

"""
from source_code.adversaries.adversarial_attacks import Attacks
import pandas as pd
import numpy as np


class KppAttack(Attacks):

    def __init__(self, data, required_attack_samples):
        """

        @param required_attack_samples: Expects an integer for number of attack samples to generate
        @param data: Expects a Pandas dataframe
        """
        self.attack_df = data
        self.attack_samples = required_attack_samples
        self.attack_df_kpp = None

    def generate_attack(self):
        if 'user' in self.attack_df.columns:
            centroid = self.attack_df.drop('user', axis=1).mean().values.reshape(1, -1)
            # Using numpy arrays for more efficient usage
            k_mean_ar = self.attack_df.drop('user', axis=1).to_numpy()
            feat_list = self.attack_df.columns.drop('user').to_list()
        else:
            centroid = self.attack_df.mean()
            # Using numpy arrays for more efficient usage
            k_mean_ar = self.attack_df.to_numpy()
            feat_list = self.attack_df.columns.drop('user').to_list()
        # Generating attack set, first point is the mean of the attack data
        init_point = centroid
        self.attack_df_kpp = pd.DataFrame(init_point, columns=feat_list)
        num_samples = self.attack_samples - 1

        for num in range(num_samples):
            # Calculating euclidean distance from mean and find index of the farthest point
            d_x = np.linalg.norm((k_mean_ar - init_point), axis=1) ** 2
            prob_d = d_x / d_x.sum()
            max_val, max_val_idx = prob_d.max(), prob_d.argmax()
            # set initial point to farthest point
            init_point = k_mean_ar[max_val_idx, :]
            self.attack_df_kpp.loc[num + 1] = k_mean_ar[max_val_idx, :]
            # drop farthest point from attack data
            k_mean_ar = np.delete(k_mean_ar, max_val_idx, axis=0)

        return self.attack_df_kpp
