import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from classifiers.one_class_svm import OneClassSvmClassifier

from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from synth_data_gen.gauss_blob_generator import GaussBlob

# TODO Impliment defece as a class
def ext_res_usr(ml_dict, n_tries, l1_keys, l2_keys, l3_keys):
    """
    This function extracts common user from dictionary of numpy arrays
    @param ml_dict: 3 level dictionary of 1-D numpy arrays with attack types in first layer and groups in second layer
    @param n_tries: Number of attacker attempts
    @param l1_keys: list or iterable with attacks keys for dictionary
    @param l2_keys:list or iterable with groups keys for dictionary
    @param l3_keys: list or iterable with classifier keys for dictionary
    @return: Dictionary with unique users in all arrays
    """
    res_user_dict = \
        {at: {gr: {clf: ml_dict[at][gr][clf].columns[ml_dict[at][gr][clf].head(n_tries).sum() == 0].to_numpy()
                   for clf in l3_keys}
              for gr in l2_keys}
         for at in l1_keys}
    act_gr_res_lst = {clf: {gr: [res_user_dict[at][gr][clf] for at in l1_keys] for gr in l2_keys} for clf in l3_keys}
    act_gr_res_ary = {clf: {gr: np.concatenate(act_gr_res_lst[clf][gr]) for gr in l2_keys} for clf in l3_keys}
    res_unq = {clf: {gr: np.unique(act_gr_res_ary[clf][gr], return_counts=True) for gr in l2_keys} for clf in l3_keys}
    res_users = {clf: {gr: res_unq[clf][gr][0][np.where(res_unq[clf][gr][1] == len(l1_keys))] for gr in l2_keys}
                 for clf in l3_keys}
    return res_users


def dist_calc(x, y, dis_type="euclidean"):
    """
    This function acts as a wrapper to scipy.spatial library's distance module
    @param x: n dim numpy vector
    @param y:n dim numpy vector
    @param dis_type: Type of distance to calculate
    @return: Calculated distance
    """
    if dis_type == "euclidean":
        dist = distance.euclidean(x, y)
    elif dis_type == "cosine":
        dist = distance.cosine(x, y)
    elif dis_type == "manhattan":
        dist = distance.cityblock(x, y)
    else:
        print(f"Enter one of the valid distance type")
        dist = np.nan

    return dist


def get_distances(scr_data_dict, distance_keys, data_centroid=None):
    """
    @param scr_data_dict:
    @param distance_keys:
    @param data_centroid: Do not provide if calculating distances for establishing s thresholding
    @return:
    """
    dataset_keys = scr_data_dict.keys()
    # Extracting dataset point of interests
    if data_centroid is None:
        data_centroid = {ds: {clf: {gr: {usr: scr_data_dict[ds][clf][gr][usr].mean()
                                         for usr in scr_data_dict[ds][clf][gr].keys()}
                                    for gr in scr_data_dict[ds][clf].keys()}
                              for clf in scr_data_dict[ds].keys()}
                         for ds in dataset_keys}
    else:
        data_centroid = data_centroid

    sample_dis_fr_cent = {
        ds: {
            clf: {gr: {
                usr: {dis: scr_data_dict[ds][clf][gr][usr].apply(lambda x: dist_calc(x, data_centroid[ds][clf][gr][usr],
                                                                                     dis_type=dis), axis=1).values
                      for dis in distance_keys}
                for usr in scr_data_dict[ds][clf][gr].keys()}
                for gr in scr_data_dict[ds][clf].keys()}
            for clf in scr_data_dict[ds].keys()}
        for ds in dataset_keys}

    p1p2_dist = {
        ds: {clf: {gr: {
            usr: {dis: np.array(
                [dist_calc(scr_data_dict[ds][clf][gr][usr].iloc[i + 1, :], scr_data_dict[ds][clf][gr][usr].iloc[i, :],
                           dis_type=dis) for i in np.arange(len(scr_data_dict[ds][clf][gr][usr]) - 1)])
                for dis in distance_keys}
            for usr in scr_data_dict[ds][clf][gr].keys()}
            for gr in scr_data_dict[ds][clf].keys()}
            for clf in scr_data_dict[ds].keys()}
        for ds in dataset_keys}

    return {'data_centroid': data_centroid, 'sample_centroid_dis': sample_dis_fr_cent, 'p1p2_dis': p1p2_dist}


def get_attack_distances(attack_data, distance_, victim_data_centroid):
    data_centroid = victim_data_centroid
    dis = distance_
    sample_dis_fr_cent = attack_data.apply(lambda x: dist_calc(x, data_centroid, dis_type=dis), axis=1).values

    p1p2_dist = [dist_calc(attack_data.iloc[i + 1, :], attack_data.iloc[i, :], dis_type=dis)
                 for i in np.arange(len(attack_data) - 1)]
    return {'data_centroid': data_centroid, 'sample_centroid_dis': sample_dis_fr_cent, 'p1p2_dis': p1p2_dist}


def get_scores(distances_dict, classifier_predictions_dict):
    dataset_keys = classifier_predictions_dict.keys()
    res_usr_scr = {
        ds: {clf: {gr: {
            usr: {dis: pd.DataFrame(
                [(distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i],
                  distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i + 1],
                  distances_dict['p1p2_dis'][ds][clf][gr][usr][dis][i],
                  ((distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i] * scr_w['w1']) +
                   (distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i + 1] * scr_w['w2']) +
                   (distances_dict['p1p2_dis'][ds][clf][gr][usr][dis][i] * scr_w['w3'])) / scr_w_sum)
                 for i in np.arange(len(distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis]) - 1)],
                columns=['p1_cent_dis', 'p2_cent_dis', "p1_p2_dis", "score"])
                for dis in distances_dict['sample_centroid_dis'][ds][clf][gr][usr].keys()}
            for usr in classifier_predictions_dict[ds][clf][gr].keys()}
            for gr in classifier_predictions_dict[ds][clf].keys()}
            for clf in classifier_predictions_dict[ds].keys()}
        for ds in dataset_keys}
    return res_usr_scr


def get_attack_scores(at_distance_dict):
    attack_scoring = {
        at: {ds: {clf: {gr: {usr: {dis: pd.DataFrame(
            [(at_distance_dict[at][ds][clf][gr][usr][dis]['sample_centroid_dis'][i],
              at_distance_dict[at][ds][clf][gr][usr][dis]['sample_centroid_dis'][i+1],
              at_distance_dict[at][ds][clf][gr][usr][dis]['p1p2_dis'][i],
              ((at_distance_dict[at][ds][clf][gr][usr][dis]['sample_centroid_dis'][i] * scr_w['w1']) +
              (at_distance_dict[at][ds][clf][gr][usr][dis]['sample_centroid_dis'][i + 1] * scr_w['w2']) +
              (at_distance_dict[at][ds][clf][gr][usr][dis]['p1p2_dis'][i] * scr_w['w3'])) / scr_w_sum)
             for i in np.arange(len(at_distance_dict[at][ds][clf][gr][usr][dis]['sample_centroid_dis']) - 1)],
            columns=['p1_cent_dis', 'p2_cent_dis', "p1_p2_dis", "score"])
            for dis in at_distance_dict[at][ds][clf][gr][usr].keys()}
            for usr in at_distance_dict[at][ds][clf][gr].keys()}
            for gr in at_distance_dict[at][ds][clf].keys()}
            for clf in at_distance_dict[at][ds].keys()}
            for ds in at_distance_dict[at].keys()}
        for at in at_distance_dict.keys()}
    return attack_scoring


# def get_scores(distances_dict, classifier_predictions_dict):
#     dataset_keys = classifier_predictions_dict.keys()
#     res_usr_scr = {
#         ds: {clf: {gr: {
#             usr: {dis: pd.DataFrame(
#                 [(distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i],
#                   distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i + 1],
#                   distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i + 2],
#                   distances_dict['p1p2_dis'][ds][clf][gr][usr][dis][i],
#                   distances_dict['p1p3_dis'][ds][clf][gr][usr][dis][i],
#                   distances_dict['p2p3_dis'][ds][clf][gr][usr][dis][i],
#                   classifier_predictions_dict[ds][clf][gr][usr][i],
#                   classifier_predictions_dict[ds][clf][gr][usr][i + 1],
#                   classifier_predictions_dict[ds][clf][gr][usr][i + 2],
#                   ((distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i] * scr_w['w1']) +
#                    (distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i + 1] * scr_w['w2']) +
#                    (distances_dict['sample_centroid_dis'][ds][clf][gr][usr][dis][i + 2] * scr_w['w3']) +
#                    (distances_dict['p1p2_dis'][ds][clf][gr][usr][dis][i] * scr_w['w4']) +
#                    (distances_dict['p1p3_dis'][ds][clf][gr][usr][dis][i] * scr_w['w5']) +
#                    (distances_dict['p2p3_dis'][ds][clf][gr][usr][dis][i] * scr_w['w6']) +
#                    ((3 - classifier_predictions_dict[ds][clf][gr][usr][i] -
#                      classifier_predictions_dict[ds][clf][gr][usr][i + 1]
#                      - classifier_predictions_dict[ds][clf][gr][usr][i + 2]) * scr_w['w7'])) / scr_w_sum)
#                  for i in np.arange(len(classifier_predictions_dict[ds][clf][gr][usr]) - 2)],
#                 columns=['p1_cent_dis', 'p2_cent_dis', 'p3_cent_dis', "p1_p2_dis", "p1_p3_dis", "p2_p3_dis",
#                          "p1_clf_prd", "p2_clf_prd", "p3_clf_prd", "score"])
#                 for dis in distances_dict['sample_centroid_dis'][ds][clf][gr][usr]}
#             for usr in classifier_predictions_dict[ds][clf][gr].keys()}
#             for gr in classifier_predictions_dict[ds][clf].keys()}
#             for clf in classifier_predictions_dict[ds].keys()}
#         for ds in dataset_keys}
#     return res_usr_scr


root_path = Path(__file__).parent.parent.parent.parent
data_metric_save_path = os.path.join(root_path, 'experiment_results\\dis_def_exp\\')

# at_types = ['hyp', 'kpp']
at_types = ['hyp', 'kpp', 'stat', 'mk']
# dist_type = ['cosine']
dist_type =  ['cosine', 'euclidean']
data_sets = ['touch', 'keystroke', 'mouse', 'gait', 'voice']
clf_types = ['svm', 'knn', 'rf']

random_state = 42
y_scl_dict = {"euclidean": {"dis_y_max": 1.2, "dis_y_min": -0.1,
                            "sl_y_max": 0.2, "sl_y_min": -0.2},

              "cosine": {"dis_y_max": 0.2, "dis_y_min": -0.1,
                         "sl_y_max": 0.2, "sl_y_min": -0.2}
              }

gr_params = {"touch": {"gr1": 14,
                       "gr2": 12},

             "keystroke": {"gr1": 6,
                           "gr2": 7},

             "mouse": {"gr1": 6,
                       "gr2": 6},

             "gait": {"gr1": 5,
                      "gr2": 5},

             "voice": {"gr1": 8,
                       "gr2": 7},

             }

gr_list = ["gr1", "gr2"]
at_tries = 2
u_samples = at_tries * 100
adv_usr_num = 5
usr_f_det_thr = 0.98
lower_bound_mul = 2

# scr_w = {"w1": 0.1, "w2": 0.1, "w3": 0.1, "w4": 0.3, "w5": 0.3, "w6": 0.3, "w7": 1.2}
# scr_w = {"w1": 0.06, "w2": 0.07, "w3": 0.07, "w4": 0.1, "w5": 0.1, "w6": 0.1, "w7": 0.5}
scr_w = {"w1": 0.05, "w2": 0.15, "w3": 0.80}
scr_w_sum = pd.DataFrame.from_dict(scr_w.values()).sum().values[0]

data_paths = {data_sets[0]: os.path.join(root_path, f'experiment_results\\adv_attack_hmog\\'),
              data_sets[1]: os.path.join(root_path, f'experiment_results\\adv_attack_dsn_cv\\'),
              data_sets[2]: os.path.join(root_path, f'experiment_results\\adv_attack_{data_sets[2]}\\'),
              data_sets[3]: os.path.join(root_path, f'experiment_results\\adv_attack_{data_sets[3]}\\'),
              data_sets[4]: os.path.join(root_path, f'experiment_results\\adv_attack_{data_sets[4]}\\')
              }

cluster_paths = {ds: os.path.join(data_paths[ds], 'cluster_data\\') for ds in data_sets}

clf_paths = {ds: os.path.join(data_paths[ds], 'trained_classifiers_svm\\') for ds in data_sets}

plot_paths = {ds: os.path.join(data_metric_save_path, f'{ds}\\') for ds in data_sets}

sns.set_theme(context="poster", style="whitegrid")

# Generating random data
num_feat = 20

# Loading data
gr_data_paths = {"touch":
                     {gr: os.path.join(data_paths["touch"], f'df_group_{gr[-1]}_50S.csv') for gr in gr_list},
                 "keystroke":
                     {gr: os.path.join(data_paths["keystroke"], f'df_group_{gr[-1]}_gr_scl.csv') for gr in gr_list},
                 "mouse":
                     {gr: os.path.join(data_paths["mouse"], f'df_group_{gr[-1]}_scaled.csv') for gr in gr_list},
                 "gait":
                     {gr: os.path.join(data_paths["gait"], f'dataset_1_group{gr[-1]}_scl.csv') for gr in gr_list},
                 "voice":
                     {gr: os.path.join(data_paths["voice"], f'{gr}_scl_df.csv') for gr in gr_list}
                 }

cluster_data_path = {ds: {gr: {cls: os.path.join(cluster_paths[ds], f"cls_group_{gr[-1]}_{cls}.csv")
                               for cls in range(gr_params[ds][gr])}
                          for gr in gr_list}
                     for ds in data_sets}

data = {ds: {gr: pd.read_csv(gr_data_paths[ds][gr]) for gr in gr_list} for ds in data_sets}
users = {ds: {gr: data[ds][gr].user.unique() for gr in gr_list} for ds in data_sets}

cls_data = {ds: {gr: {cls: pd.read_csv(cluster_data_path[ds][gr][cls])
                      for cls in range(gr_params[ds][gr])}
                 for gr in gr_list}
            for ds in data_sets}

at_data = {at: {ds: {gr: pd.read_csv(os.path.join(data_paths[ds], f"{gr}_{at}_at_data.csv"))
                     for gr in gr_list}
                for ds in data_sets}
           for at in at_types}

for ds in data_sets:
    for gr in gr_list:
        at_data['hyp'][ds][gr] = at_data['hyp'][ds][gr].drop('cluster_number', axis=1)

at_prd = {ds: {at: {gr: {clf: pd.read_csv(os.path.join(data_paths[ds], f"{gr}_{at}_at_prd_{clf}.csv"))
                         for clf in clf_types}
                    for gr in gr_list}
               for at in at_types}
          for ds in data_sets}

# Extracting users not cracked in n tries
res_usr = {ds: ext_res_usr(ml_dict=at_prd[ds], n_tries=at_tries, l1_keys=at_types, l2_keys=gr_list, l3_keys=clf_types)
           for ds in data_sets}

res_usr_dat = {ds: {clf: {gr: {usr: data[ds][gr][data[ds][gr].user == int(float(usr))].drop('user', axis=1)
                               for usr in res_usr[ds][clf][gr]}
                          for gr in gr_list}
                    for clf in clf_types}
               for ds in data_sets}

res_usr_dat_split = \
    {ds: {clf: {gr: {usr: train_test_split(res_usr_dat[ds][clf][gr][usr], test_size=0.5, random_state=42)
                     for usr in res_usr[ds][clf][gr]}
                for gr in gr_list}
          for clf in clf_types}
     for ds in data_sets}

res_usr_scr_dat = {ds: {clf: {gr: {usr: res_usr_dat_split[ds][clf][gr][usr][0]
                                   for usr in res_usr[ds][clf][gr]}
                              for gr in gr_list}
                        for clf in clf_types}
                   for ds in data_sets}

res_usr_test_dat = {ds: {clf: {gr: {usr: res_usr_dat_split[ds][clf][gr][usr][1].reset_index(drop=True)
                                    for usr in res_usr[ds][clf][gr]}
                               for gr in gr_list}
                         for clf in clf_types}
                    for ds in data_sets}

# Loading classifiers for resilient users
res_usr_clf_path = {ds: {clf: {gr: {u: glob.glob(os.path.join(clf_paths[ds], f"*{u}_{clf}*.joblib"))
                                    for u in res_usr[ds][clf][gr]}
                               for gr in gr_list}
                         for clf in clf_types}
                    for ds in data_sets}

res_usr_clf = {ds: {clf: {gr: {u: load(res_usr_clf_path[ds][clf][gr][u][0])
                               for u in res_usr[ds][clf][gr]}
                          for gr in gr_list}
                    for clf in clf_types}
               for ds in data_sets}

# Resilient user scoring and test data predictions
res_usr_prd_scr = {ds: {clf: {gr: {u: res_usr_clf[ds][clf][gr][u].classifier.predict(res_usr_scr_dat[ds][clf][gr][u])
                                   for u in res_usr[ds][clf][gr]}
                              for gr in gr_list}
                        for clf in clf_types}
                   for ds in data_sets}

res_usr_prd_test = {
    ds: {clf: {gr: {u: res_usr_clf[ds][clf][gr][u].classifier.predict(res_usr_test_dat[ds][clf][gr][u])
                    for u in res_usr[ds][clf][gr]}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

res_usr_test_dat_fp = {
    ds: {clf: {gr: {u: res_usr_test_dat[ds][clf][gr][u].loc[np.where(res_usr_prd_test[ds][clf][gr][u] == 0)[0]]
                    for u in res_usr[ds][clf][gr]}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

# Finding users with false positives
fp_usr_list = {ds: {clf: {gr: dict() for gr in gr_list} for clf in clf_types} for ds in data_sets}
for ds in data_sets:
    for clf in clf_types:
        for gr in gr_list:
            usr_list = []
            for u in res_usr[ds][clf][gr]:
                fp_len = len(res_usr_test_dat_fp[ds][clf][gr][u])
                if fp_len >= 3:
                    usr_list.append(u)
                else:
                    pass
            if len(usr_list) != 0:
                fp_usr_list[ds][clf][gr] = usr_list
            else:
                del fp_usr_list[ds][clf][gr]
        if len(fp_usr_list[ds][clf]) == 0:
            del fp_usr_list[ds][clf]
        else:
            pass

res_usr_test_dat_fp_clean = {
    ds: {clf: {gr: {u: res_usr_test_dat_fp[ds][clf][gr][u]
                    for u in fp_usr_list[ds][clf][gr]}
               for gr in fp_usr_list[ds][clf].keys()}
         for clf in fp_usr_list[ds].keys()}
    for ds in fp_usr_list.keys()}

res_usr_test_dat_fp_clean_prd = {
    ds: {clf: {gr: {u: res_usr_clf[ds][clf][gr][u].classifier.predict(res_usr_test_dat_fp_clean[ds][clf][gr][u])
                    for u in fp_usr_list[ds][clf][gr]}
               for gr in fp_usr_list[ds][clf].keys()}
         for clf in fp_usr_list[ds].keys()}
    for ds in fp_usr_list.keys()}

# Extracting distances for generating user scores

res_usr_scr_distances = get_distances(scr_data_dict=res_usr_scr_dat, distance_keys=dist_type)

# Calculating scores on scoring data and finding user threshold
res_usr_scr_scores = get_scores(distances_dict=res_usr_scr_distances, classifier_predictions_dict=res_usr_prd_scr)
res_usr_scr_th_ub = {
    ds: {clf: {gr: {u: {dis: res_usr_scr_scores[ds][clf][gr][u][dis].quantile(usr_f_det_thr).score
                        for dis in dist_type}
                    for u in res_usr[ds][clf][gr]}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

res_usr_scr_th_lb = {
    ds: {clf: {gr: {u: {
        dis: res_usr_scr_scores[ds][clf][gr][u][dis].quantile(lower_bound_mul * (1 - usr_f_det_thr)).score
                        for dis in dist_type}
                    for u in res_usr[ds][clf][gr]}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

# Extracting distances for generating testing user scores
res_usr_fp_distances = get_distances(scr_data_dict=res_usr_test_dat_fp_clean, distance_keys=dist_type,
                                     data_centroid=res_usr_scr_distances['data_centroid'])

# Generating scores for resilient user test cases
res_usr_fp_scores = get_scores(distances_dict=res_usr_fp_distances,
                               classifier_predictions_dict=res_usr_test_dat_fp_clean_prd)
a = 1

res_usr_fp_det_results = {
    ds: {clf: {gr: {u: {dis: (res_usr_fp_scores[ds][clf][gr][u][dis].score.values >
                              res_usr_scr_th_ub[ds][clf][gr][u][dis]) |
                             (res_usr_fp_scores[ds][clf][gr][u][dis].score.values <
                              res_usr_scr_th_lb[ds][clf][gr][u][dis])
                        for dis in dist_type}
                    for u in fp_usr_list[ds][clf][gr]}
               for gr in fp_usr_list[ds][clf].keys()}
         for clf in fp_usr_list[ds].keys()}
    for ds in fp_usr_list.keys()}

test_cat_ar = np.empty(0)
for ds in fp_usr_list.keys():
    for clf in fp_usr_list[ds].keys():
        for gr in fp_usr_list[ds][clf].keys():
            for u in fp_usr_list[ds][clf][gr]:
                for dis in dist_type:
                    test_cat_ar = np.concatenate((test_cat_ar, res_usr_fp_det_results[ds][clf][gr][u][dis]))

TU_, CU_ = np.unique(test_cat_ar, return_counts=True)
print(f"user false det {CU_[1] / CU_.sum() * 100}%")

# Extracting distances for attack samples for detection
at_samples_distances = {at: {ds: {clf: {gr: {u: {dis: dict()
                                                 for dis in dist_type}
                                             for u in fp_usr_list[ds][clf][gr]}
                                        for gr in fp_usr_list[ds][clf].keys()}
                                  for clf in fp_usr_list[ds].keys()}
                             for ds in fp_usr_list.keys()}
                        for at in at_types}

for at in at_types:
    for ds in fp_usr_list.keys():
        for clf in fp_usr_list[ds].keys():
            for gr in fp_usr_list[ds][clf].keys():
                for u in fp_usr_list[ds][clf][gr]:
                    for dis in dist_type:
                        at_samples_distances[at][ds][clf][gr][u][dis] = \
                            get_attack_distances(attack_data=at_data[at][ds][gr].head(at_tries),
                                                 distance_=dis,
                                                 victim_data_centroid=
                                                 res_usr_scr_distances['data_centroid'][ds][clf][gr][u])

a = 1

at_samples_scores = get_attack_scores(at_distance_dict=at_samples_distances)

at_det_results = {
    at: {
        ds: {clf: {gr: {u: {
            dis: (at_samples_scores[at][ds][clf][gr][u][dis].score.values > res_usr_scr_th_ub[ds][clf][gr][u][dis])
                 | (at_samples_scores[at][ds][clf][gr][u][dis].score.values < res_usr_scr_th_lb[ds][clf][gr][u][dis])

            for dis in dist_type}
            for u in fp_usr_list[ds][clf][gr]}
            for gr in fp_usr_list[ds][clf].keys()}
            for clf in fp_usr_list[ds].keys()}
        for ds in fp_usr_list.keys()}
    for at in at_types}


at_hyp_ar = np.empty(0)
at_kpp_ar = np.empty(0)
at_stat_ar = np.empty(0)
at_mk_ar = np.empty(0)

for at in at_types:
    for ds in fp_usr_list.keys():
        for clf in fp_usr_list[ds].keys():
            for gr in fp_usr_list[ds][clf].keys():
                for u in fp_usr_list[ds][clf][gr]:
                    for dis in dist_type:
                        if at == 'hyp':
                            at_hyp_ar = np.concatenate((at_hyp_ar, at_det_results[at][ds][clf][gr][u][dis]))
                        elif at == 'kpp':
                            at_kpp_ar = np.concatenate((at_kpp_ar, at_det_results[at][ds][clf][gr][u][dis]))
                        elif at == 'stat':
                            at_stat_ar = np.concatenate((at_stat_ar, at_det_results[at][ds][clf][gr][u][dis]))
                        elif at == 'mk':
                            at_mk_ar = np.concatenate((at_mk_ar, at_det_results[at][ds][clf][gr][u][dis]))


TA_hyp, CA_hyp = np.unique(at_hyp_ar, return_counts=True)
TA_kpp, CA_kpp = np.unique(at_kpp_ar, return_counts=True)
TA_stat, CA_stat = np.unique(at_stat_ar, return_counts=True)
# TA_mk, CA_mk = np.unique(at_mk_ar, return_counts=True)
print(f"Hyp Attack detected {CA_hyp[1] / CA_hyp.sum() * 100}%")
print(f"kpp Attack detected {CA_kpp[1] / CA_kpp.sum() * 100}%")
print(f"Stat Attack detected {CA_stat[1] / CA_stat.sum() * 100}%")
# print(f"mk Attack detected {CA_mk[1] / CA_mk.sum() * 100}%")

a = 1

at_res_df_comp = pd.DataFrame(columns=['dataset', 'attack', 'group', 'user', 'classifier', 'attack_type',
                                       'distance_type', 'user_threshold_upper', 'user_threshold_lower',
                                       'at_sample', 'adv_score', 'decision'])

row_num = 0
for at in at_types:
    for ds in fp_usr_list.keys():
        for clf in fp_usr_list[ds].keys():
            for gr in fp_usr_list[ds][clf].keys():
                for usr in fp_usr_list[ds][clf][gr]:
                    for dis in dist_type:
                        for ap in range(len(at_det_results[at][ds][clf][gr][usr][dis])):
                            # print(f"{ds} {at} {gr} {usr} {dis}\n", at_det_results[at][ds][clf][gr][usr][dis][ap])
                            at_res_df_comp.loc[row_num, 'dataset'] = ds
                            at_res_df_comp.loc[row_num, 'attack'] = at
                            at_res_df_comp.loc[row_num, 'group'] = gr
                            at_res_df_comp.loc[row_num, 'user'] = usr
                            at_res_df_comp.loc[row_num, 'classifier'] = clf
                            at_res_df_comp.loc[row_num, 'attack_type'] = at
                            at_res_df_comp.loc[row_num, 'distance_type'] = dis
                            at_res_df_comp.loc[row_num, 'at_sample'] = ap
                            at_res_df_comp.loc[row_num, 'user_threshold_upper'] = \
                                res_usr_scr_th_ub[ds][clf][gr][usr][dis]
                            at_res_df_comp.loc[row_num, 'user_threshold_lower'] = \
                                res_usr_scr_th_lb[ds][clf][gr][usr][dis]
                            at_res_df_comp.loc[row_num, 'adv_score'] =\
                                at_samples_scores[at][ds][clf][gr][usr][dis].score[ap]
                            at_res_df_comp.loc[row_num, 'decision'] = at_det_results[at][ds][clf][gr][usr][dis][ap]
                            row_num += 1
print(f"hyp \n {at_res_df_comp[at_res_df_comp['attack_type'] == 'hyp'].decision.value_counts(normalize=True)}")
print(f"kpp \n {at_res_df_comp[at_res_df_comp['attack_type'] == 'kpp'].decision.value_counts(normalize=True)}")
print(f"stat \n {at_res_df_comp[at_res_df_comp['attack_type'] == 'stat'].decision.value_counts(normalize=True)}")
print(f"mk \n {at_res_df_comp[at_res_df_comp['attack_type'] == 'mk'].decision.value_counts(normalize=True)}")
at_res_df_comp.to_csv(os.path.join(data_metric_save_path, f"def_res_2pt_v3.csv"),  index=False, mode='w+')

a = 1