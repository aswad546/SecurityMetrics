import time
from operator import itemgetter
from analytics.user_recom_hv import HypVolUserRecommend
from external_dataset_parsers import hmog_parser
from dataset.biometric_dataset import BioDataSet
from dataset.min_max_scaling_operation import MinMaxScaling
from classifiers.svm_classifier import SvmClassifier
import numpy as np
from metrics.confusion_matrix import ConfusionMatrix
from metrics.gini_coefficient import GiniCoef
from metrics.fcs import FCS
from metrics.roc_curve import RocCurve
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dataset.low_variance_feature_removal import LowVarFeatRemoval
from analytics.dataoverlap_interval import OverLapInt
import pandas as pd
from sklearn.utils import shuffle
from analytics.fp_acceptance_percentage import FpAcceptance

root_path = Path(__file__).parent.parent.parent.parent
gr1_feature_path = os.path.join(root_path, 'experiment_results\\HMOG_touch_overlap_test_svm\\df_scaled_group_1.csv')
gr2_feature_path = os.path.join(root_path, 'experiment_results\\HMOG_touch_overlap_test_svm\\df_scaled_group_2.csv')
data_metric_save_path = os.path.join(root_path, 'experiment_results\\HMOG_overlap_C_SVM\\')
gr2_per_dim_overlap_path = os.path.join(root_path,
                                        'experiment_results\\HMOG_touch_overlap_test'
                                        '\\gr2_hyper_vol_size_overlap_per_dim'
                                        '.csv')

random_state_seed = 42
random_state_range = list(range(random_state_seed, random_state_seed * 201, random_state_seed))
neg_sample_sources_base = 23
# neg_sample_sources_range = list(range(neg_sample_sources_base, neg_sample_sources_base * 2, neg_sample_sources_base))
neg_sample_sources_range = [23]
column_names = ["random_state", "neg_sample_sources", "seed_user", "C_value", "test_tn", "test_fp", "test_fn",
                "test_tp",
                "overlap_tn", "overlap_fp", "overlap_fn", "overlap_tp",
                "test_auc", "overlap_auc", "percent_accepted_overlap", "percent_accepted_test"]
results = pd.DataFrame(columns=column_names)
results_int = pd.DataFrame(columns=column_names)
cut_off = 0.001
std_dev_gr2 = 5

"""
    Reading data from disk
"""
df_group_1 = pd.read_csv(gr1_feature_path)
df_group_2 = pd.read_csv(gr2_feature_path)

"""
    Reading ovrelap data from disk and finding best seed user 
"""
gr2_per_dim_overlap = pd.read_csv(gr2_per_dim_overlap_path)
gr2_mins = gr2_per_dim_overlap.min()
gr2_mins = gr2_mins[gr2_mins < cut_off]

query_list = []
for col in gr2_mins.index:
    query_list.append(f"{col} >= {cut_off}")
query = ' & '.join(query_list)

if len(query) != 0:
    fil_gr2_overlap = gr2_per_dim_overlap.query(query)
else:
    fil_gr2_overlap = gr2_per_dim_overlap
seed_ol_user_dict = dict()
users_group_2 = df_group_2.pos_user.unique()
for seed_user in users_group_2:
    fil_seed_user_pd = fil_gr2_overlap[(fil_gr2_overlap['V1'] == seed_user) | (fil_gr2_overlap['V1'] == seed_user)]
    seed_arr = fil_seed_user_pd.V1.unique().tolist()
    seed_arr.extend(fil_seed_user_pd.V2.unique().tolist())
    seed_arr = np.array(seed_arr)
    seed_arr = np.unique(seed_arr).tolist()
    seed_arr.remove(seed_user)
    seed_ol_user_dict[seed_user] = seed_arr

gr2_len_tup = [(key, len(seed_ol_user_dict[key])) for key in seed_ol_user_dict.keys()]
gr2_len_tup = sorted(gr2_len_tup, key=itemgetter(1), reverse=True)
best_seed_user = gr2_len_tup[0][0]
# best_seed_user = 395129


for neg_sample_sources in neg_sample_sources_range:
    print(f'generating results for negative sample sources = {neg_sample_sources}')
    for rand_state in random_state_range:
        print(f'generating results for random state = {rand_state} and negative sample sources = {neg_sample_sources}')

        """
        reading biometrics features from CSV file
        
        """
        tb_data_group_1 = BioDataSet(feature_data_frame=df_group_1, random_state=rand_state)
        tb_data_group_2 = BioDataSet(feature_data_frame=df_group_2, random_state=rand_state)

        """
        Removing features with low variance
        """
        df_group_1 = LowVarFeatRemoval().operate(data=tb_data_group_1.feature_df)
        df_group_2 = LowVarFeatRemoval().operate(data=tb_data_group_2.feature_df)

        ''' 
           get the user list from the dataset class object
        '''

        users_group_1 = tb_data_group_1.user_list
        users_group_2 = tb_data_group_2.user_list

        ''' 
           generate tagged data set for each user
        '''

        data_group_1 = dict()
        data_group_2 = dict()

        for user in users_group_1:
            data_group_1[user] = tb_data_group_1.get_data_set(user, neg_sample_sources=neg_sample_sources,
                                                              neg_test_limit=True)

        ''' 
           Extracting data for individual users from group 2 for overlap analysis
        '''
        for user in users_group_2:
            data_group_2[user] = df_group_2[df_group_2['user'] == user]

        """
            Extracting overlapping features from group 2
        """
        ol_start_time = time.time()
        print(f"Extracting overlapping features from group 2")

        overlap_data_dict = dict()
        for s_user in users_group_2:
            seed_user_gr2 = s_user

            overlap_data_gr_2_seed_user = data_group_2[seed_user_gr2]
            for user in seed_ol_user_dict[seed_user_gr2]:
                overlap_data_gr_2_seed_user, df_2 = \
                    OverLapInt(overlap_data_gr_2_seed_user, data_group_2[user], std_dev=std_dev_gr2).get_analytics()
            overlap_data_dict[s_user] = overlap_data_gr_2_seed_user
        overlap_data_gr_2_extracted = pd.DataFrame()
        for user_key in overlap_data_dict.keys():
            overlap_data_gr_2_extracted = overlap_data_gr_2_extracted.append(overlap_data_dict[user_key])
        overlap_data_gr_2_extracted = overlap_data_gr_2_extracted.drop_duplicates()
        overlap_data_gr_2_extracted['labels'] = np.zeros(len(overlap_data_gr_2_extracted))
        ol_end_time = time.time()
        print(f"Overlap set creation time = {ol_end_time - ol_start_time} seconds")

        ''' 
           Extracting positive samples for the overlap dataset
        '''
        hyp_vol_data_path = os.path.join(root_path,
                                         'experiment_results/HMOG_touch_overlap_test/gr1_hyper_vol.csv')
        fil_p_df_0 = HypVolUserRecommend(hyp_vol_df_path=hyp_vol_data_path, sort_type=0).get_analytics()
        pos_user = fil_p_df_0.iloc[0, 0]

        # pos_user = 366286
        data_frame = data_group_1[pos_user]

        pos_val_for_overlap_data = data_frame[data_frame['user'] == pos_user].iloc[0:25, :]
        overlap_data_gr_2 = overlap_data_gr_2_extracted.append(pos_val_for_overlap_data)
        overlap_data_gr_2 = shuffle(overlap_data_gr_2)
        overlap_data_gr_2 = overlap_data_gr_2.reset_index(drop=True)
        # overlap_data_gr_2_path = os.path.join(root_path,
        #                                       'experiment_results/HMOG_touch_overlap_test/df_overlap_gr_2.csv')
        # overlap_data_gr_2.to_csv(overlap_data_gr_2_path, index=False, mode='w+')

        # Deleting extracted samples from the data frame
        data_frame = pd.concat([data_frame, pos_val_for_overlap_data]).drop_duplicates(keep=False)
        data_frame = data_frame.reset_index(drop=True)

        ''' 
           Initialize classifier object and split the data into training, testing and ovrlaping data
        '''

        clf_svm = SvmClassifier(pos_user=pos_user, random_state=rand_state, cache_size=5000)
        clf_svm.split_data(data_frame=data_frame, training_data_size=0.6)

        # c_range = [x * 0.001 for x in range(1, 150, 50)]
        # c_range = [round(x, 3) for x in c_range]
        # c_med = list(range(1, 100, 25))
        # c_range.extend(c_med)
        # c_high = [x * 1 for x in range(100, 1500, 500)]
        # c_range.extend(c_high)
        # c_range = [1000, 5000]
        # c_opt = [10 ** 4]
        # c_range.extend(c_opt)
        # c_range.sort()
        c_range = [10 ** 4]
        col_range = range(0, len(c_range))

        train_data = clf_svm.training_data_frame.drop('labels', axis=1)
        train_labels = clf_svm.training_data_frame['labels']

        test_data_values = clf_svm.test_data_frame.drop('labels', axis=1)
        test_labels = clf_svm.test_data_frame['labels']

        overlap_data_gr_2_p = overlap_data_gr_2.drop('user', axis=1)
        overlap_set_values = overlap_data_gr_2_p.drop('labels', axis=1)
        overlap_labels = overlap_data_gr_2_p.labels.values

        for c_val, row_num in zip(c_range, col_range):
            print(f"calculating results for SVM regularization  parameter C = {c_val}, random state = {rand_state} and "
                  f"negative sample sources = {neg_sample_sources} ")
            print(f"Training SVM model")
            clf_svm.classifier.C = c_val
            clf_svm.classifier.fit(train_data, train_labels)
            print(f"SVM model training complete")
            '''
              Model Classification using test dataset
            '''
            test_dataset_predictions = clf_svm.classify()

            '''
               Model Classification using overlap dataset
            '''

            overlap_dataset_predictions = clf_svm.classify(df=overlap_data_gr_2_p)

            """
                False positive acceptance percentage example
            """
            fp_pers_svm = FpAcceptance(df=overlap_data_gr_2_p, prediction=clf_svm.predictions_ext_df)
            percent_overlap_sample_accept = round(fp_pers_svm.get_analytics(), 2)

            """
                    Confusion Matrix Test Set
            """

            test_cm = ConfusionMatrix()
            test_c_matrix = test_cm.get_metric(true_labels=test_labels, predicted_labels=test_dataset_predictions)

            """
                    Confusion Overlap Set, FpAcceptance calculates confusion matrix internally
            """

            overlap_cm = fp_pers_svm.cm

            """
                ROC Curves
            """
            ax_roc = plt.gca()
            test_roc = RocCurve().get_metric(test_set_features=test_data_values.values, test_set_labels=test_labels
                                             , classifier=clf_svm.classifier, ax=ax_roc)
            overlap_roc = RocCurve().get_metric(test_set_features=overlap_set_values.values,
                                                test_set_labels=overlap_labels
                                                , classifier=clf_svm.classifier, ax=ax_roc)
            test_auc = round(test_roc.roc_auc, 3)
            overlap_auc = round(overlap_roc.roc_auc, 3)

            plt.close()

            """
            Results Dataframe
            
            """
            results_int.loc[row_num, "random_state"] = rand_state
            results_int.loc[row_num, "neg_sample_sources"] = neg_sample_sources
            results_int.loc[row_num, "seed_user"] = best_seed_user

            results_int.loc[row_num, "C_value"] = c_val
            results_int.loc[row_num, "test_tn"] = test_cm.tn
            results_int.loc[row_num, "test_fp"] = test_cm.fp
            results_int.loc[row_num, "test_fn"] = test_cm.fn
            results_int.loc[row_num, "test_tp"] = test_cm.tp
            results_int.loc[row_num, "test_auc"] = test_auc

            results_int.loc[row_num, "overlap_tn"] = overlap_cm.tn
            results_int.loc[row_num, "overlap_fp"] = overlap_cm.fp
            results_int.loc[row_num, "overlap_fn"] = overlap_cm.fn
            results_int.loc[row_num, "overlap_tp"] = overlap_cm.tp
            results_int.loc[row_num, "overlap_auc"] = overlap_auc
            results_int.loc[row_num, "percent_accepted_overlap"] = percent_overlap_sample_accept
            results_int.loc[row_num, "percent_accepted_test"] = \
                round((100 * results_int.test_fp[row_num] / (
                            results_int.test_fp[row_num] + results_int.test_tn[row_num])), 2)

            print(f"calculations for SVM regularization parameter C = {c_val}, random state = {rand_state} and "
                  f"negative sample sources = {neg_sample_sources} complete")
            plt.close()
            print('%.2f' % percent_overlap_sample_accept, "percent of false users accepted for SVM classifier")

        results = results.append(results_int)
results = results.reset_index(drop=True)
results.to_csv(os.path.join(data_metric_save_path
                            , f'{pos_user}_results_1_5_10K_{random_state_seed}_{len(random_state_range)}RS_cval.csv'), index=False, mode='w+')
a = 1
