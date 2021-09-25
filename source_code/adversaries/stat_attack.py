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

Vanilla Stat attack
=====================
This class implements Vanilla Stat attack which is inspired by these three papers listed below

@inproceedings{serwadda2013when,
  title={When kids' toys breach mobile phone security},
  author={Serwadda, Abdul and Phoha, Vir V},
  booktitle={ACM SIGSAC Conference on Computer \& Communications Security},
  year={2013},
  organization={ACM}
}

@article{serwadda2013examining,
  title={Examining a large keystroke biometrics dataset for statistical-attack openings},
  author={Serwadda, Abdul and Phoha, Vir V},
  journal={ACM Transactions on Information and System Security},
  volume={16},
  number={2},
  pages={8},
  year={2013},
  publisher={ACM}
}

@article{serwadda2016toward,
  title={Toward robotic robbery on the touch screen},
  author={Serwadda, Abdul and Phoha, Vir V and Wang, Zibo and Kumar, Rajesh and Shukla, Diksha},
  journal={ACM Transactions on Information and System Security (TISSEC)},
  volume={18},
  number={4},
  pages={1--25},
  year={2016},
  publisher={ACM New York, NY, USA}
}
"""
from source_code.adversaries.adversarial_attacks import Attacks
from source_code.synth_data_gen.gauss_blob_generator import GaussBlob
import pandas as pd
import numpy as np
import os


class StatAttack(Attacks):

    def __init__(self, data, required_attack_samples, bootstrap_data_path, run_bootstrap=True,
                 bootstrap_iter=10000, random_state=42):
        """

        @param required_attack_samples: Expects an integer for number of attack samples to generate
        @param data: Expects a Pandas dataframe
        """
        self.attack_df = data
        self.attack_samples = required_attack_samples
        self.attack_df_stat = None
        self.boot_strap_st_at = run_bootstrap
        self.bootstrap_iterations = bootstrap_iter
        self.bs_data_path = bootstrap_data_path
        self.rand_state = random_state

    def generate_attack(self):
        if 'user' in self.attack_df.columns:
            # Using numpy arrays for more efficient usage
            feat_list = self.attack_df.columns.drop('user').to_list()
        else:
            # Using numpy arrays for more efficient usage
            feat_list = self.attack_df.columns.drop('user').to_list()

        # Generating attack set
        bs_results = dict()
        attack_feat_stats = pd.DataFrame(columns=["mean", "std"], index=feat_list)

        if self.boot_strap_st_at is True:

            # calculating mean ans std by bootstrap

            for feat in feat_list:
                bs_results[feat] = pd.DataFrame(columns=['mean', "std"])

            bs_iter = self.bootstrap_iterations
            print("starting bootstrap experiment")
            for feat in feat_list:
                print(f"starting bootstrap for {feat}")
                for itera in range(bs_iter):
                    ar = np.random.choice(self.attack_df.drop('user', axis=1)[feat].to_numpy(), replace=True,
                                          size=len(self.attack_df))
                    bs_results[feat].loc[itera, 'mean'] = ar.mean()
                    bs_results[feat].loc[itera, 'std'] = ar.std()
                bs_results[feat].to_csv(os.path.join(self.bs_data_path, f"{feat}_bs.csv"), index=False, mode='w+')
                stats = bs_results[feat].to_numpy().mean(axis=0)
                attack_feat_stats.loc[feat, "mean"] = stats[0]
                attack_feat_stats.loc[feat, "std"] = stats[1]
            print("starting bootstrap experiment done")
        else:
            print(f"Reading bootstrap data from disk")

            for feat in feat_list:
                bs_results[feat] = pd.read_csv(os.path.join(self.bs_data_path, f"{feat}_bs.csv"))
                stats = bs_results[feat].to_numpy().mean(axis=0)
                attack_feat_stats.loc[feat, "mean"] = stats[0]
                attack_feat_stats.loc[feat, "std"] = stats[1]
            print(f"Reading bootstrap data from disk")

        print(f"Generating Stats attack data")
        self.attack_df_stat = pd.DataFrame(columns=feat_list)
        num_samples = self.attack_samples
        for feat in feat_list:
            X, y = GaussBlob().generate_data(n_classes=1, n_features=1, n_samples=num_samples,
                                             centers=[attack_feat_stats.loc[feat, 'mean']],
                                             random_state=self.rand_state,
                                             cluster_std=[attack_feat_stats.loc[feat, 'std']])
            self.attack_df_stat[feat] = X.Dim_00

        return self.attack_df_stat
