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

Hypervolume attack
=====================
This class implements our Hypervolume attack

Note: if calculating cluster then you have to calculate hypervolumes using the R script (hyper_vol_usage.R)
TODO Impliment Hypervolume calculation form this class
"""
from source_code.adversaries.adversarial_attacks import Attacks
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import sys
from analytics.dataoverlap_interval import OverLapInt
from scipy.spatial import distance
import math


class HypVolAttack(Attacks):

    def __init__(self, data, equal_user_data=False, random_state=42, calc_clusters=False, clusters_path=None,
                 gr_num=None, cluster_count=None, ol_path=None, ol_cut_off=0.01, std_dev_at_gr=2, cluster_bins=10,
                 attack_samples=1000, cls_use_index=3):
        self.random_state = random_state
        self.equal_user_data = equal_user_data
        self.data = data
        self.user_dict = dict()
        self.red_user_dict = dict()
        self.red_data = pd.DataFrame()
        self.cls_dict = dict()
        self.users = None
        self.df = pd.DataFrame()
        self.calc_clusters = calc_clusters
        self.clusters_path = clusters_path
        self.gr_num = gr_num
        self.cluster_count = cluster_count
        self.ol_path = ol_path
        self.ol_data = dict()
        self.per_dim_ol_data = dict()
        self.ol_cut_off = ol_cut_off
        self.std_dev_at_gr = std_dev_at_gr
        self.cls_extracted = dict()
        self.cls_mean_ol = dict()
        self.attack_df_hyp = pd.DataFrame()
        self.attack_samples = attack_samples
        self.cluster_bins = cluster_bins
        self.cls_rank_df = None
        self.cls_use_index = cls_use_index
        return

    def generate_attack(self):
        self.users = self.data.user.unique()
        self.user_dict = {u: self.data[self.data.user == u] for u in self.users}

        if self.equal_user_data is True:
            user_data_len = [(user, len(self.user_dict[user])) for user in self.users]
            user_data_len = pd.DataFrame(user_data_len, columns=["user", "data_length"])
            data_min = user_data_len.data_length.min()
            self.red_user_dict = \
                {u: (self.user_dict[u].sample(n=data_min, replace=False,
                                              random_state=self.random_state)) for u in self.users}
            for user in self.users:
                self.red_data = pd.concat([self.red_data, self.red_user_dict[user]])
            self.df = self.red_data
        else:
            self.df = self.data

        if self.calc_clusters is True:
            self.generate_clusters()
        else:
            for cls in range(self.cluster_count):
                self.cls_dict[cls] = pd.read_csv(os.path.join(self.clusters_path, f"cls_group_{self.gr_num}_{cls}.csv"))
                if "cls_labels" in self.cls_dict[cls]:
                    self.cls_dict[cls] = self.cls_dict[cls].drop("cls_labels", axis=1)

        if self.ol_path is None:
            print(f"provide overlap data path")
            sys.exit(1)
        else:
            self.get_attack_data()

        return self.attack_df_hyp

    def generate_clusters(self):

        """
        K-means clustering
        """
        print("Performing clustering")
        k_range = range(2, len(self.users))
        sum_sq_distance = np.zeros(0)

        for k in k_range:
            df_kmeans = \
                KMeans(n_clusters=k, random_state=self.random_state, verbose=False).fit(self.df.drop('user', axis=1))
            sum_sq_distance = np.append(sum_sq_distance, df_kmeans.inertia_)

        # Calculating gradients
        sum_sq_distance_av = pd.DataFrame(sum_sq_distance / len(df_kmeans.labels_))
        sum_sq_distance_gr = pd.DataFrame(np.gradient(sum_sq_distance_av, axis=0))
        sum_sq_distance_gr_diff = sum_sq_distance_gr.diff().dropna()
        print("Performing clustering done")

        sns.set_theme(context="poster")
        sns.set_style("whitegrid")
        df_km = pd.DataFrame()
        df_km["k_range"] = k_range
        df_km["sq_err"] = sum_sq_distance_av
        df_km["grd_sq_er"] = sum_sq_distance_gr

        fig_km = plt.figure(figsize=(19.2, 10.8))
        ax_sq_er = fig_km.add_subplot(2, 2, 1)
        sns.lineplot(data=df_km, x="k_range", y="sq_err", ax=ax_sq_er)

        ax_gr = fig_km.add_subplot(2, 2, 2)
        sns.lineplot(data=df_km, x="k_range", y="grd_sq_er", ax=ax_gr)

        knee = KneeLocator(df_km["k_range"], df_km["grd_sq_er"], S=1,
                           curve='concave', direction='increasing')
        fig_km.tight_layout()
        ax_gr_kn = fig_km.add_subplot(2, 2, 4)
        knee.plot_knee()
        plt.title(f"knee point at {knee.knee}")
        plt.xlabel("Points")
        plt.ylabel("Distance")
        opt_cls = int(knee.knee)
        self.cluster_count = opt_cls

        df_kmeans = \
            KMeans(n_clusters=opt_cls, random_state=self.random_state, verbose=False).fit(self.df.drop('user', axis=1))
        self.df["cls_labels"] = df_kmeans.labels_
        self.cls_dict = {cls: self.df[self.df.cls_labels == cls].drop("cls_labels", axis=1) for cls in range(opt_cls)}

        return self.cls_dict

    def get_attack_data(self):
        self.ol_data = \
            {cls: pd.read_csv(os.path.join(self.ol_path, f"cls_group_{self.gr_num}_{cls}_hyper_vol.csv"))
                .drop("Unnamed: 0", axis=1)
             for cls in range(self.cluster_count)}

        self.per_dim_ol_data = \
            {cls: pd.read_csv(
                os.path.join(self.ol_path, f"cls_group_{self.gr_num}_{cls}_hyper_vol_size_overlap_per_dim.csv"))
                .drop("Unnamed: 0", axis=1)
             for cls in range(self.cluster_count)}

        for cls in range(self.cluster_count):
            cls_users = self.cls_dict[cls].user.unique()
            cls_users.sort()
            ol_cls_user = np.unique(np.concatenate((self.ol_data[cls].V1.unique(), self.ol_data[cls].V2.unique())))
            ol_cls_user.sort()
            per_dim_ol_cls_user = \
                np.unique(
                    np.concatenate((self.per_dim_ol_data[cls].V1.unique(), self.per_dim_ol_data[cls].V2.unique())))
            per_dim_ol_cls_user.sort()
            if (np.array_equal(cls_users, ol_cls_user) and np.array_equal(ol_cls_user, per_dim_ol_cls_user)) is True:
                pass
            else:
                print("Overlap and cluster data don't match")
                sys.exit(1)

        # Extracting Overlap Data
        for cls in range(self.cluster_count):
            cls_users = self.cls_dict[cls].user.unique()
            ol_mins = self.per_dim_ol_data[cls].min()
            ol_mins = ol_mins[ol_mins < self.ol_cut_off]
            query_list = []
            for col in ol_mins.index:
                query_list.append(f"{col} >= {self.ol_cut_off}")
            query = ' & '.join(query_list)
            if len(query) != 0:
                fil_cls_overlap = self.per_dim_ol_data[cls].query(query)
            else:
                fil_cls_overlap = self.per_dim_ol_data[cls]
            seed_ol_user_dict = dict()

            for seed_user in cls_users:
                fil_seed_user_pd = fil_cls_overlap[
                    (fil_cls_overlap['V1'] == seed_user) | (fil_cls_overlap['V1'] == seed_user)]
                seed_arr = fil_seed_user_pd.V1.unique().tolist()
                seed_arr.extend(fil_seed_user_pd.V2.unique().tolist())
                seed_arr = np.array(seed_arr)
                seed_arr = np.unique(seed_arr).tolist()
                seed_ol_user_dict[seed_user] = seed_arr

            overlap_data_dict = dict()
            overlap_data_gr_2_seed_user_ = pd.DataFrame()
            for s_user in cls_users:
                seed_user = s_user

                overlap_data_gr_2_seed_user = self.cls_dict[cls][self.cls_dict[cls]['user'] == seed_user]
                if len(overlap_data_gr_2_seed_user) > 50:

                    for usr in seed_ol_user_dict[seed_user]:
                        df2 = self.cls_dict[cls][self.cls_dict[cls].user == usr].copy()
                        if len(df2) > 50:
                            overlap_data_gr_2_seed_user_r, df2 = \
                                OverLapInt(overlap_data_gr_2_seed_user, df2,
                                           std_dev=self.std_dev_at_gr).get_analytics()
                            overlap_data_gr_2_seed_user_ = pd.concat(
                                [overlap_data_gr_2_seed_user_, overlap_data_gr_2_seed_user_r, df2])
                        else:
                            overlap_data_gr_2_seed_user_ = pd.concat(
                                [overlap_data_gr_2_seed_user_, df2])
                    overlap_data_gr_2_seed_user_ = overlap_data_gr_2_seed_user_.drop_duplicates()
                    overlap_data_dict[s_user] = overlap_data_gr_2_seed_user_
                else:
                    overlap_data_dict[s_user] = overlap_data_gr_2_seed_user

            overlap_data_extracted = pd.DataFrame()
            for user_key in overlap_data_dict.keys():
                overlap_data_extracted = overlap_data_extracted.append(overlap_data_dict[user_key])
            overlap_data_extracted = overlap_data_extracted.drop_duplicates()
            self.cls_extracted[cls] = overlap_data_extracted
            self.cls_extracted[cls] = self.cls_extracted[cls].reset_index(drop=True)

        # Checking for empty extracted data
        valid_cluster_list = list()
        for cls in range(self.cluster_count):
            cls_dat_len = len(self.cls_extracted[cls])
            if cls_dat_len != 0:
                valid_cluster_list.extend([cls])
            else:
                pass

        # Calculating Mean Overlaps for clusters
        self.cls_mean_ol = \
            [(cls, self.ol_data[cls].loc[
                self.per_dim_ol_data[cls].V1 != self.per_dim_ol_data[cls].V2].port_mean.values.mean())
             for cls in valid_cluster_list]

        cls_rank_df = pd.DataFrame(self.cls_mean_ol, columns=["cluster_number", "mean_ol"])
        cls_rank_df["mean_ol"] = np.around(cls_rank_df["mean_ol"].values, 2)

        cls_rank_df["cls_user"] = [self.cls_dict[cls].user.nunique() for cls in valid_cluster_list]
        cls_rank_df["cls_user"] = np.around(cls_rank_df["cls_user"].values / len(self.users), 2)

        cls_rank_df["cls_samples"] = [len(self.cls_dict[cls]) for cls in valid_cluster_list]
        cls_rank_df["cls_samples"] = np.around((cls_rank_df["cls_samples"].values / len(self.data)), 3)

        data_centroid = self.data.drop("user", axis=1).mean()
        cls_rank_df["cls_pop_dis"] = [distance.euclidean(self.cls_extracted[c].drop("user", axis=1).values.mean(axis=0),
                                                         data_centroid) for c in valid_cluster_list]
        # Weights for scoring function
        ol_w = 0.9
        dis_w = 0
        num_user_w = 0.05
        sample_w = 0.05

        cls_rank_df["cls_score"] = np.round(np.average(cls_rank_df.drop("cluster_number", axis=1).values,
                                                       weights=[ol_w, num_user_w, sample_w, dis_w], axis=1), 2)
        cls_rank_df = cls_rank_df.sort_values(by=['cls_score', 'cls_pop_dis'], ascending=[False, True])
        cls_rank_df = cls_rank_df.reset_index(drop=True)
        # cls_rank_df = cls_rank_df.sort_values(by="cls_score", ascending=False).reset_index(drop=True)
        cls_ranked_list = cls_rank_df["cluster_number"].to_list()
        cls_ranked_list_ = cls_rank_df["cluster_number"].to_list()

        for cls in cls_ranked_list_:
            if len(self.cls_extracted[cls]) < 5:
                cls_ranked_list.remove(cls)

        lc = np.empty(0)
        for cls in cls_ranked_list:
            lc = np.append(lc, len(self.cls_extracted[cls]))
        lc_c_sum = np.cumsum(lc)

        # cls_use_index = self.cls_use_index
        # if lc_c_sum[cls_use_index] < self.attack_samples:
        #     cls_use_index = np.where(lc_c_sum > self.attack_samples)
        top_3_clusters = cls_ranked_list

        cls_centroids = {c: self.cls_extracted[c].drop("user", axis=1).values.mean(axis=0) for c in top_3_clusters}

        # Attack
        self.attack_df_hyp = pd.DataFrame.from_dict(cls_centroids).T
        self.attack_df_hyp.columns = self.data.columns.drop("user")
        self.attack_df_hyp = self.attack_df_hyp.reset_index(drop=True)
        self.attack_df_hyp = pd.concat([self.attack_df_hyp, cls_rank_df["cluster_number"]], axis=1)
        rem_samples = self.attack_samples - len(self.attack_df_hyp)
        cls_kmean = {c: KMeans(n_clusters=min(self.attack_samples * 1, len(self.cls_extracted[c])),
                               random_state=self.random_state, verbose=False).fit(
            self.cls_extracted[c].drop("user", axis=1).values) for c in top_3_clusters}
        cls_kmean_data = {c: cls_kmean[c].cluster_centers_ for c in top_3_clusters}
        iter_req = math.ceil(rem_samples / len(top_3_clusters))
        row_start_num = len(self.attack_df_hyp)
        for row in range(iter_req):
            r = 0
            for cls in top_3_clusters:
                at_dat = cls_kmean_data[cls][row, :].reshape(1, -1)
                self.attack_df_hyp.loc[(row + r + row_start_num), :] = np.append(at_dat, cls)
                r += 1
            row_start_num = len(self.attack_df_hyp)

        self.attack_df_hyp = self.attack_df_hyp.head(self.attack_samples)
        self.attack_df_hyp = self.attack_df_hyp.drop_duplicates()
        self.attack_df_hyp = self.attack_df_hyp.reset_index(drop=True)
        self.attack_df_hyp = self.attack_df_hyp.head(self.attack_samples)
        self.cls_rank_df = cls_rank_df


        return self.attack_df_hyp
