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

Experiment using SVM classifier
=====================
This class implements our experiment using Support Vector Machine (SVM) classifier, if different classifier is required
then refer to the comments in the classifier section for instructions, only few changes needed for updating the experiment

Note: This experiment would need data from hypervolume calculation which can be done by using the R script (hyper_vol_usage.R)
       This experiment uses optional sklearnex package to provoide optimization for sklearn library running on intel processors
"""
import collections
import sys
from operator import itemgetter
from sklearn.cluster import KMeans
from source_code.adversaries.kpp_attack import KppAttack
from source_code.adversaries.mk_attack import MkAttack
from source_code.adversaries.stat_attack import StatAttack
from source_code.adversaries.hyp_attack import HypVolAttack
from source_code.dataset.biometric_dataset import BioDataSet
import numpy as np
import pandas as pd
import os
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt
from source_code.metrics.confusion_matrix import ConfusionMatrix
from source_code.metrics.fcs import FCS
from source_code.metrics.roc_curve import RocCurve
from source_code.synth_data_gen.gauss_blob_generator import GaussBlob
from source_code.analytics.dataoverlap_interval import OverLapInt
import traceback


class HypExp:
    def __init__(self, pop_df, attack_df, pop_classifier_path, pos_user_per_dim_ol_path, active_gr,
                 results_save_path=None,
                 attack_samples=1000, boot_strap_st_at=False, bs_data_path=None, bs_mul=1,
                 hv_cut_off=0.04, gr2_per_dim_ol_path=None, std_dev_at_gr=5, clf_type=None,
                 hyp_at_u_data=None, rand_state=None, train_data_size=0.6,
                 train_classifiers=False, cluster_data_path=None, hyp_vol_data_path=None,
                 num_cls=None, cv=10, random_grid_search_iter=25):

        self.pop_df = pop_df.copy()
        self.attack_df = attack_df.copy()
        self.active_gr = active_gr

        self.classifier_training = train_classifiers
        self.results_save_path = results_save_path

        self.clf_type = clf_type
        self.rand_state = rand_state

        self.attack_samples = attack_samples
        self.boot_strap_st_at = boot_strap_st_at
        self.train_data_size = train_data_size
        self.num_cls = num_cls
        self.cv = cv
        self.n_iter = random_grid_search_iter

        self.bs_mul = bs_mul
        self.hv_cut_off = hv_cut_off
        self.feat_list = None

        self.clf_path = pop_classifier_path
        self.pos_user_per_dim_ol_path = pos_user_per_dim_ol_path
        self.bs_data_path = bs_data_path
        self.gr2_per_dim_ol_path = gr2_per_dim_ol_path
        self.std_dev_at_gr = std_dev_at_gr
        self.cluster_data_path = cluster_data_path
        self.hyp_vol_data_path = hyp_vol_data_path

        #  creating dictionaries for data gathering
        self.test_prd_dict = dict()
        self.test_prd_prob_dict = dict()
        self.test_cm_dict = dict()
        self.test_precision = dict()
        self.roc_dict = dict()
        self.fcs_dict = dict()
        self.fcs_plt = dict()
        self.att_prd_mk = dict()
        self.att_prd_prob_mk = dict()
        self.att_prd_kpp = dict()
        self.att_prd_prob_kpp = dict()
        self.att_prd_stat = dict()
        self.att_prd_prob_stat = dict()
        self.att_prd_hyp = dict()
        self.att_prd_prob_hyp = dict()

        # Attack Data
        self.attack_df_kpp = None
        self.attack_df_mk = None
        self.attack_df_stat = None
        if hyp_at_u_data is not None:
            self.attack_df_hyp = hyp_at_u_data
        else:
            self.attack_df_hyp = None
            # Result Data
            self.acc_res_full_df = None
            self.acc_res_df = None
            self.acc_per_df = None
            self.acc_eer_df = None
            self.stack_res_df = None

            return

    def run_exp(self):
        data_group_1 = dict()

        clf_dict = dict()

        gr2_means = self.attack_df.mean()
        gr2_means_fv = gr2_means.drop('user', axis=0).to_numpy().reshape(1, -1)
        gr2_std = self.attack_df.std()
        gr2_std_fv = gr2_std.drop('user', axis=0).to_numpy().reshape(1, -1)

        tb_data_group_1 = BioDataSet(feature_data_frame=self.pop_df, random_state=self.rand_state)
        tb_data_group_2 = BioDataSet(feature_data_frame=self.attack_df, random_state=self.rand_state)

        # Extracting users in both groups
        users_group_1 = tb_data_group_1.user_list
        users_group_2 = tb_data_group_2.user_list

        self.feat_list = self.pop_df.columns.drop('user').to_list()

        """
        Generating user data
        """
        user_g1_df_dict = dict()
        for user in users_group_1:
            data_group_1[user] = tb_data_group_1.get_data_set(user, neg_sample_sources=None, neg_test_limit=True)
            user_g1_df_dict[user] = self.pop_df[self.pop_df['user'] == user]

        if self.classifier_training is True:
            scoring_metric = 'precision'
            self.cv = 10 # specify a number for cv fold cross validation
            self.n_iter = 25 #number of iterations for random grid search
            precision_tup = list()
            eer_tup = list()
            print(f"training classifiers")
            if self.clf_type == 'svm':
                # Commnet out two lines below if not using an intel processor or sklearnex is not installed
                from sklearnex import patch_sklearn
                patch_sklearn()
                from classifiers.svm_classifier import SvmClassifier
                # Classifier training grid params, update with classifer specic hyper parameters
                c_range = np.unique(np.logspace(start=0.1, stop=4, num=100 + 20, dtype=int))
                grid_svm = {'estimator__C': c_range,
                            'estimator__gamma': ['auto', 'scale']}
                # Update classifier on line below for using a different classifer
                clf_dict = {usr: SvmClassifier(pos_user=data_group_1[usr], random_state=self.rand_state)
                                for usr in users_group_1}
                for usr in users_group_1:
                    print(f'training for user {usr}')
                    clf_name_string = f"clf_{usr}_{self.clf_type}_{self.rand_state}.joblib"

                    clf_dict[usr].split_data(data_frame=data_group_1[usr], training_data_size=0.6)
                    clf_dict[usr].random_train_tune_parameters(pram_dist=grid_svm, cv=self.cv, scoring_metric=scoring_metric,
                                                                   n_itr=self.n_iter)
                    dump(clf_dict[usr], os.path.join(self.clf_path, clf_name_string))

            elif self.clf_type == 'knn':
                from classifiers.knn_classifier import KnnClassifier
                leaf_size = list(range(1, 70))
                n_neighbors = list(range(1, 50))
                p = [1, 2]
                grid_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
                clf_dict = {usr: KnnClassifier(pos_user=data_group_1[usr], random_state=self.rand_state,
                                                   n_jobs=-1) for usr in users_group_1}
                for usr in users_group_1:
                    print(f'training for user {usr}')
                    clf_name_string = f"clf_{usr}_{self.clf_type}_{self.rand_state}.joblib"
                    clf_dict[usr].split_data(data_frame=data_group_1[usr], training_data_size=0.6)
                    clf_dict[usr].random_train_tune_parameters(pram_dist=grid_knn, cv=self.cv,
                                                                   scoring_metric=scoring_metric,
                                                                   n_itr=self.n_iter)
                    dump(clf_dict[usr], os.path.join(self.clf_path, clf_name_string))

            elif self.clf_type == 'rf':
                # Commnet out two lines below if not using an intel processor or sklearnex is not installed
                from classifiers.random_forest_classifier import RandomForestClassifier
                from sklearnex import patch_sklearn
                patch_sklearn()

                # Number of trees in random forest
                n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
                # Number of features to consider at every split
                max_features = ['auto', 'sqrt']
                # Maximum number of levels in tree
                max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
                max_depth.append(None)
                # Minimum number of samples required to split a node
                min_samples_split = [2, 5, 10]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 4]
                # Method of selecting samples for training each tree
                bootstrap = [True, False]
                grid_rf = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
                clf_dict = {usr: RandomForestClassifier(pos_user=data_group_1[usr], random_state=self.rand_state,
                                                       n_jobs=-1) for usr in users_group_1}
                for usr in users_group_1:
                    print(f'training for user {usr}')
                    clf_name_string = f"clf_{usr}_{self.clf_type}_{self.rand_state}.joblib"
                    clf_dict[usr].split_data(data_frame=data_group_1[usr], training_data_size=0.6)
                    clf_dict[usr].random_train_tune_parameters(pram_dist=grid_rf, cv=self.cv,
                                                               scoring_metric=scoring_metric,
                                                               n_itr=self.n_iter)
                    dump(clf_dict[usr], os.path.join(self.clf_path, clf_name_string))



            else:
                print('classifier not implimented')
                sys.exit(1)


            print(f"training classifiers complete")
        else:
            """
            Loading classifiers from disk
            """
            print(f"Loading classifiers")
            try:
                clf_dict = {usr: load(os.path.join(self.clf_path, f"clf_{usr}_{self.clf_type}_{self.rand_state}.joblib"))
                                for usr in users_group_1}
            except Exception():
                traceback.print_exc()

            print(f"Loading classifiers complete")

        """
        Calculating mean overlaps on feature level
        """
        print(f"Calculating mean overlaps on feature level started")
        overlap_other_per_user_means_df = pd.DataFrame()
        overlap_by_other_per_user_means_df = pd.DataFrame()
        for pos_user in users_group_1:
            pos_user_per_dim_ol_path = self.pos_user_per_dim_ol_path
            pos_user_per_dim_ol = pd.read_csv(pos_user_per_dim_ol_path)
            pos_user_per_dim_ol = pos_user_per_dim_ol.drop('Unnamed: 0', axis=1)

            pos_user_pd_ol_others = pos_user_per_dim_ol[(pos_user_per_dim_ol['V2'] == pos_user)]
            pos_user_pd_ol_others_mean = pos_user_pd_ol_others.drop(['V1', 'V2'], axis=1).mean()
            overlap_other_per_user_means_df[pos_user] = pos_user_pd_ol_others_mean

            pos_user_pd_ol_by_others = pos_user_per_dim_ol[(pos_user_per_dim_ol['V1'] == pos_user)]
            pos_user_pd_ol_by_others_mean = \
                pos_user_pd_ol_by_others.drop(['V1', 'V2'], axis=1).mean().sort_values()
            overlap_by_other_per_user_means_df[pos_user] = pos_user_pd_ol_by_others_mean
        print(f"Calculating mean overlaps on feature level complete")

        """
        Calculating mean statistics for overlaps over entire population
        """

        overlap_other_means = overlap_other_per_user_means_df.mean(axis=1)
        overlap_other_means = overlap_other_means.sort_values(ascending=True)
        overlap_other_range = overlap_other_per_user_means_df.max(axis=1) - overlap_other_per_user_means_df.min(axis=1)
        overlap_other_range = overlap_other_range.sort_values(ascending=True)

        overlap_by_other_means = overlap_by_other_per_user_means_df.mean(axis=1)
        overlap_by_other_means = overlap_by_other_means.sort_values(ascending=True)
        overlap_by_other_range = overlap_by_other_per_user_means_df.max(
            axis=1) - overlap_by_other_per_user_means_df.min(axis=1)
        overlap_by_other_range = overlap_by_other_range.sort_values(ascending=True)

        '''  
        Model Classification
        '''
        print(f"Starting model classification")
        self.test_prd_dict = {usr: clf_dict[usr].classify() for usr in users_group_1}
        self.test_prd_prob_dict = {usr: clf_dict[usr].predictions_prob for usr in users_group_1}
        print(f"Model classification complete")

        """
           Test set and labels extraction
        """
        test_set = {usr: clf_dict[usr].test_data_frame.drop('labels', axis=1) for usr in users_group_1}
        test_labels = {usr: clf_dict[usr].test_data_frame.labels.values for usr in users_group_1}

        """
            Confusion Matrix 
        """
        self.test_cm_dict = {usr: ConfusionMatrix() for usr in users_group_1}
        matrix_svm = {usr: self.test_cm_dict[usr].get_metric(true_labels=test_labels[usr],
                                                             predicted_labels=self.test_prd_dict[usr])
                      for usr in users_group_1}
        self.test_precision = {usr: self.test_cm_dict[usr].tp / (self.test_cm_dict[usr].tp + self.test_cm_dict[usr].fp)
                               for usr in users_group_1}
        """
            ROC Curves
        """
        self.roc_dict = {usr: RocCurve() for usr in users_group_1}
        roc_svm = {usr: self.roc_dict[usr].get_metric(test_set_features=test_set[usr].values,
                                                      test_set_labels=test_labels[usr],
                                                      classifier=clf_dict[usr].classifier, ax=None)
                   for usr in users_group_1}

        """
            FCS
        """

        self.fcs_dict = {usr: FCS(classifier_name='SVM') for usr in users_group_1}
        self.fcs_plt = {usr: self.fcs_dict[usr].get_metric(
            true_labels=test_labels[usr],
            predicted_probs=clf_dict[usr].predictions_prob,
            pred_labels=clf_dict[usr].predictions)
            for usr in users_group_1}
        plt.close('all')

        """
        Master Key Attack
        """
        # Generating attack set
        mk_adv = MkAttack(data=self.attack_df, required_attack_samples=self.attack_samples)
        self.attack_df_mk = mk_adv.generate_attack()

        # Performing attack
        self.att_prd_mk = {usr: clf_dict[usr].classifier.predict(self.attack_df_mk.values)
                           for usr in users_group_1}
        att_prd_prob_mk = {usr: clf_dict[usr].classifier.predict_proba(self.attack_df_mk.values)
                           for usr in users_group_1}
        self.att_prd_prob_mk = {usr: att_prd_prob_mk[usr][:, 1]
                                for usr in users_group_1}

        """
        Targeted K-means++ Attack
        """
        # Generating attack set, first point is the mean of the attack data
        kpp_adv = KppAttack(data=self.attack_df, required_attack_samples=self.attack_samples)
        self.attack_df_kpp = kpp_adv.generate_attack()

        # Performing attack
        self.att_prd_kpp = {usr: clf_dict[usr].classifier.predict(self.attack_df_kpp.values)
                            for usr in users_group_1}
        att_prd_prob_kpp = {usr: clf_dict[usr].classifier.predict_proba(self.attack_df_kpp.values)
                            for usr in users_group_1}
        self.att_prd_prob_kpp = {usr: att_prd_prob_kpp[usr][:, 1]
                                 for usr in users_group_1}

        """
        Stats Attack
        """
        stat_adv = StatAttack(data=self.attack_df, required_attack_samples=self.attack_samples,
                              bootstrap_data_path=self.bs_data_path,
                              run_bootstrap=self.boot_strap_st_at, bootstrap_iter=self.bs_mul * 1000)
        self.attack_df_stat = stat_adv.generate_attack()

        # Performing attack
        self.att_prd_stat = {usr: clf_dict[usr].classifier.predict(self.attack_df_stat.values)
                             for usr in users_group_1}
        att_prd_prob_stat = {usr: clf_dict[usr].classifier.predict_proba(self.attack_df_stat.values)
                             for usr in users_group_1}
        self.att_prd_prob_stat = {usr: att_prd_prob_stat[usr][:, 1]
                                  for usr in users_group_1}

        """
        Hypervolume Attack
        """
        if self.attack_df_hyp is None:
            hyp_adv = HypVolAttack(data=self.attack_df, equal_user_data=False, random_state=self.rand_state,
                                   calc_clusters=False,
                                   clusters_path=self.cluster_data_path, gr_num=1, cluster_count=self.num_cls,
                                   ol_path=self.hyp_vol_data_path, attack_samples=self.attack_samples,
                                   ol_cut_off=self.hv_cut_off, std_dev_at_gr=None)
            self.attack_df_hyp = hyp_adv.generate_attack()
        else:
            self.attack_df_hyp = self.attack_df_hyp

        # Performing attack
        self.att_prd_hyp = {usr: clf_dict[usr].classifier.predict(
            self.attack_df_hyp.drop(["cluster_number"], axis=1).values)
                            for usr in users_group_1}
        att_prd_prob_hyp = {usr: clf_dict[usr].classifier.predict_proba(
            self.attack_df_hyp.drop(["cluster_number"], axis=1).values)
                            for usr in users_group_1}
        self.att_prd_prob_hyp = {usr: att_prd_prob_hyp[usr][:, 1]
                                 for usr in users_group_1}

        df_hyp = pd.DataFrame.from_dict(self.att_prd_hyp)
        df_stat = pd.DataFrame.from_dict(self.att_prd_stat)
        df_kpp = pd.DataFrame.from_dict(self.att_prd_kpp)
        df_mk = pd.DataFrame.from_dict(self.att_prd_mk)

        df_prob_hyp = pd.DataFrame.from_dict(self.att_prd_prob_hyp)
        df_prob_stat = pd.DataFrame.from_dict(self.att_prd_prob_stat)
        df_prob_kpp = pd.DataFrame.from_dict(self.att_prd_prob_kpp)
        df_prob_mk = pd.DataFrame.from_dict(self.att_prd_prob_mk)

        df_hyp = pd.concat([df_hyp, self.attack_df_hyp['cluster_number']], axis=1)
        df_hyp.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_hyp_at_prd_{self.clf_type}.csv"), index=False,
                      mode='w+')
        df_stat.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_stat_at_prd_{self.clf_type}.csv"), index=False,
                       mode='w+')
        df_kpp.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_kpp_at_prd_{self.clf_type}.csv"), index=False,
                      mode='w+')
        df_mk.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_mk_at_prd_{self.clf_type}.csv"),
                     index=False, mode='w+')

        df_prob_hyp = pd.concat([df_hyp, self.attack_df_hyp['cluster_number']], axis=1)

        df_prob_hyp.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_hyp_at_prd_prob_{self.clf_type}.csv"),
                           index=False,
                           mode='w+')
        df_prob_stat.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_stat_at_prd_prob_{self.clf_type}.csv"),
                            index=False,
                            mode='w+')
        df_prob_kpp.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_kpp_at_prd_prob_{self.clf_type}.csv"),
                           index=False,
                           mode='w+')
        df_prob_mk.to_csv(os.path.join(self.results_save_path, f"{self.active_gr}_mk_at_prd_prob_{self.clf_type}.csv"), index=False,
                          mode='w+')

        df_hyp = df_hyp.drop("cluster_number", axis=1)
        df_prob_hyp = df_prob_hyp.drop("cluster_number", axis=1)

        user_crk_hyp = pd.DataFrame(columns=["try_num", "attack_type", "users_cracked_per"])
        user_crk_stat = pd.DataFrame(columns=["try_num", "attack_type", "users_cracked_per"])
        user_crk_kpp = pd.DataFrame(columns=["try_num", "attack_type", "users_cracked_per"])
        user_crk_mk = pd.DataFrame(columns=["try_num", "attack_type", "users_cracked_per"])
        a0_hyp = pd.Series(dtype="float64")
        a0_stat = pd.Series(dtype="float64")
        a0_kpp = pd.Series(dtype="float64")
        a0_mk = pd.Series(dtype="float64")

        for row in range(self.attack_samples):
            a_hyp = df_hyp.loc[row, :][df_hyp.loc[row, :] == 1]
            a0_hyp = pd.concat([a0_hyp, a_hyp])
            user_crk_hyp.loc[row, "try_num"] = row + 1
            user_crk_hyp.loc[row, "users_cracked_per"] = len(a0_hyp.index.unique())
            user_crk_hyp.loc[row, "attack_type"] = "hyp"

            a_stat = df_stat.loc[row, :][df_stat.loc[row, :] == 1]
            a0_stat = pd.concat([a0_stat, a_stat])
            user_crk_stat.loc[row, "try_num"] = row + 1
            user_crk_stat.loc[row, "users_cracked_per"] = len(a0_stat.index.unique())
            user_crk_stat.loc[row, "attack_type"] = "stat"

            a_kpp = df_kpp.loc[row, :][df_kpp.loc[row, :] == 1]
            a0_kpp = pd.concat([a0_kpp, a_kpp])
            user_crk_kpp.loc[row, "try_num"] = row + 1
            user_crk_kpp.loc[row, "users_cracked_per"] = len(a0_kpp.index.unique())
            user_crk_kpp.loc[row, "attack_type"] = "kpp"

            a_mk = df_mk.loc[row, :][df_mk.loc[row, :] == 1]
            a0_mk = pd.concat([a0_mk, a_mk])
            user_crk_mk.loc[row, "try_num"] = row + 1
            user_crk_mk.loc[row, "users_cracked_per"] = len(a0_mk.index.unique())
            user_crk_mk.loc[row, "attack_type"] = "mk"

        df = pd.concat([user_crk_hyp, user_crk_stat, user_crk_kpp, user_crk_mk])
        df.try_num = df.try_num.astype("float64")
        df.users_cracked_per = df.users_cracked_per.astype("float64")
        df.attack_type = df.attack_type.astype("string")
        df.users_cracked_per = df.users_cracked_per / len(self.pop_df.user.unique())

        user_crk_hyp_sm = user_crk_hyp.head(100).copy()
        user_crk_stat_sm = user_crk_stat.head(100).copy()
        user_crk_kpp_sm = user_crk_kpp.head(100).copy()
        user_crk_mk_sm = user_crk_mk.head(100).copy()

        user_crk_hyp_sm1 = user_crk_hyp.head(10).copy()
        user_crk_stat_sm1 = user_crk_stat.head(10).copy()
        user_crk_kpp_sm1 = user_crk_kpp.head(10).copy()
        user_crk_mk_sm1 = user_crk_mk.head(10).copy()

        df_sm = pd.concat([user_crk_hyp_sm, user_crk_stat_sm, user_crk_kpp_sm, user_crk_mk_sm])
        df_sm.users_cracked_per = df_sm.users_cracked_per / len(self.pop_df.user.unique())

        df_sm1 = pd.concat([user_crk_hyp_sm1, user_crk_stat_sm1, user_crk_kpp_sm1, user_crk_mk_sm1])
        df_sm1.users_cracked_per = df_sm1.users_cracked_per / len(self.pop_df.user.unique())

        df_sm.try_num = df_sm.try_num.astype("float64")
        df_sm.users_cracked_per = df_sm.users_cracked_per.astype("float64")
        df_sm.attack_type = df_sm.attack_type.astype("string")

        df_sm1.try_num = df_sm1.try_num.astype("float64")
        df_sm1.users_cracked_per = df_sm1.users_cracked_per.astype("float64")
        df_sm1.attack_type = df_sm1.attack_type.astype("string")

        attack_per_plt_sm1 = plt.figure(figsize=(19.2, 10.8))
        attack_per_ax_sm1 = attack_per_plt_sm1.add_subplot(1, 1, 1)
        sns.lineplot(data=df_sm1, x="try_num", y="users_cracked_per", hue="attack_type", ax=attack_per_ax_sm1,
                     alpha=0.7)
        attack_per_ax_sm1.set_title(f"Attack performance comparison group {self.active_gr[-1]} ")
        attack_per_plt_sm1.tight_layout()
        attack_per_plt_sm1.savefig(
            os.path.join(self.results_save_path, f"{self.active_gr}_{self.clf_type}_attack_speed_comp_10.png"))
        plt.close('all')

        attack_per_plt = plt.figure(figsize=(19.2, 10.8))
        attack_per_ax = attack_per_plt.add_subplot(1, 1, 1)
        sns.lineplot(data=df, x="try_num", y="users_cracked_per", hue="attack_type", ax=attack_per_ax, alpha=0.7)
        attack_per_ax.set_title(f"Attack performance comparison group {self.active_gr[-1]} ")
        attack_per_plt.tight_layout()
        attack_per_plt.savefig(os.path.join(self.results_save_path, f"{self.active_gr}_{self.clf_type}_attack_speed_comp_{self.attack_samples}.png"))
        plt.close('all')

        return
