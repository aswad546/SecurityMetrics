import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import load

from metrics.roc_curve import RocCurve

root_path = Path(__file__).parent.parent.parent.parent

data_sets = ['touch', 'keystroke', 'mouse', 'gait']
touch_dir = "adv_attack_hmog"
keystroke_dir = "adv_attack_dsn_cv"
mouse_dir = "adv_attack_mouse"
gait_dir = "adv_attack_gait"

clf_dirs = [touch_dir, keystroke_dir, mouse_dir, gait_dir]

clf_paths = [os.path.join(root_path, f'experiment_results\\{dir}\\trained_classifiers_svm\\')
             for dir in clf_dirs]
trained_clf_files = {ds: glob.glob(os.path.join(pt, f"clf_*_svm_42.joblib")) for ds, pt in zip(data_sets, clf_paths)}
trained_clf_files_svm = {ds: [i for i in trained_clf_files[ds] if i.split('\\')[-1].split("_")[-3] != 'oc']
                         for ds in data_sets}
trained_clf_files_knn = {ds: glob.glob(os.path.join(pt, f"clf_*_knn_42.joblib")) for ds, pt in
                         zip(data_sets, clf_paths)}

clf_types = ['svm', 'knn']
trained_clf_files = {'svm': trained_clf_files_svm,
                     'knn': trained_clf_files_knn}
tr_clf = {ds: {clf: {i.split('\\')[-1].split("_")[-3]: load(i)
                     for i in trained_clf_files[clf][ds]}
               for clf in clf_types}
          for ds in data_sets}

roc_dict = {ds: {clf: {u: RocCurve()
                       for u in tr_clf[ds][clf].keys()}
                 for clf in clf_types}
            for ds in data_sets}

roc_obj = {ds: {clf: {u: roc_dict[ds][clf][u].get_metric(
    test_set_features=tr_clf[ds][clf][u].test_data_frame.drop('labels', axis=1).values,
    test_set_labels=tr_clf[ds][clf][u].test_data_frame.labels.values
    , classifier=tr_clf[ds][clf][u].classifier, ax=None)
    for u in tr_clf[ds][clf].keys()}

    for clf in clf_types}
    for ds in data_sets}

results = pd.DataFrame(columns=['data_set', 'user', 'clf', 'auc', "eer", 'eer_threshold', 'fpr_tpr_len'])
row_num = 0
for ds in data_sets:
    for clf in clf_types:
        for u in tr_clf[ds][clf].keys():
            results.loc[row_num, 'data_set'] = ds
            results.loc[row_num, 'user'] = u
            results.loc[row_num, 'clf'] = clf
            results.loc[row_num, 'auc'] = roc_dict[ds][clf][u].auc_roc
            results.loc[row_num, 'eer'] = roc_dict[ds][clf][u].eer
            results.loc[row_num, 'eer_threshold'] = roc_dict[ds][clf][u].eer_threshold
            results.loc[row_num, 'fpr_tpr_len'] = len(roc_dict[ds][clf][u].fpr)

            row_num += 1

results_means = {
    ds: {cl: results[(results.data_set == ds) & (results.clf == cl)].drop(['data_set', 'user', 'clf', 'fpr_tpr_len']
                                                                          , axis=1).mean(axis=0)
         for cl in clf_types}
    for ds in data_sets}

results_std = {ds: {cl: results[(results.data_set == ds) & (results.clf == cl)].drop(['data_set', 'user', 'clf']
                                                                                     , axis=1).std(axis=0)
                    for cl in clf_types}
               for ds in data_sets}

results_comp_df = pd.DataFrame(columns=['dataset', 'classifier', "auc_mean", "auc_std", "eer_mean", "eer_std"])
row_num = 0
r_des = 2
for ds in data_sets:
    for clf in clf_types:
        results_comp_df.loc[row_num, "dataset"] = ds
        results_comp_df.loc[row_num, "classifier"] = clf
        results_comp_df.loc[row_num, "auc_mean"] = round(results_means[ds][clf]["auc"], r_des)
        results_comp_df.loc[row_num, "auc_std"] = round(results_std[ds][clf]["auc"], r_des)
        results_comp_df.loc[row_num, "eer_mean"] = round(results_means[ds][clf]["eer"], r_des)
        results_comp_df.loc[row_num, "eer_std"] = round(results_std[ds][clf]["eer"], r_des)
        # results_comp_df.loc[row_num, "eer_threshold_mean"] = round(results_means[ds][clf]["eer"], r_des)
        # results_comp_df.loc[row_num, "eer_threshold_std"] = round(results_std[ds][clf]["eer"], r_des)

        row_num += 1


samples = results[(results.data_set == 'touch') & (results.clf == 'svm')].fpr_tpr_len.min()
samples_dict = {ds: {clf:  (results[(results.data_set == ds) & (results.clf == clf)].fpr_tpr_len.min() - 2)
                     for clf in clf_types}
                for ds in data_sets}

sum_fig_tpr_df = {ds: {clf: pd.DataFrame() for clf in clf_types} for ds in data_sets}
sum_fig_fpr_df = {ds: {clf: pd.DataFrame() for clf in clf_types} for ds in data_sets}

one_s = pd.DataFrame([1, 1]).T
one_s.columns = ["fpr", "tpr"]
zero_s = pd.DataFrame([0, 0]).T
zero_s.columns= ["fpr", "tpr"]

for ds in data_sets:
    for clf in clf_types:
        for u in tr_clf[ds][clf].keys():

            fpr_tpr_df = pd.DataFrame([roc_dict[ds][clf][u].fpr , roc_dict[ds][clf][u].tpr]).T
            fpr_tpr_df.columns = ['fpr', 'tpr']
            sample_fpr_tpr = fpr_tpr_df[1: -1].sample(samples_dict[ds][clf], random_state=42)
            con_df = pd.concat([zero_s, sample_fpr_tpr, one_s])
            con_df = con_df.sort_values(by='fpr')
            con_df = con_df.reset_index(drop=True)
            row_fpr_ser = con_df.fpr
            row_tpr_ser = con_df.tpr

            sum_fig_fpr_df[ds][clf] = sum_fig_fpr_df[ds][clf].append(row_fpr_ser).reset_index(drop=True)
            sum_fig_tpr_df[ds][clf] = sum_fig_tpr_df[ds][clf].append(row_tpr_ser).reset_index(drop=True)

a = 1
fig_fpr = {ds: {clf: sum_fig_fpr_df[ds][clf].mean() for clf in clf_types} for ds in data_sets}
fig_tpr = {ds: {clf: sum_fig_tpr_df[ds][clf].mean() for clf in clf_types} for ds in data_sets}

fig_dict = {ds: {clf: plt.figure(figsize=(19.2, 10.8)) for clf in clf_types} for ds in data_sets}
ax_dict = {ds: {clf: fig_dict[ds][clf].add_subplot(111) for clf in clf_types} for ds in data_sets}

sns.set_theme(context='poster', style="whitegrid")

for ds in data_sets:
    for clf in clf_types:
        sns.lineplot(x=fig_fpr[ds][clf], y=fig_tpr[ds][clf], ax=ax_dict[ds][clf])
        ax_dict[ds][clf].set_xlabel('fpr')
        ax_dict[ds][clf].set_ylabel('tpr')
        ax_dict[ds][clf].set_title(f"{ds} dataset, {clf} classifier")
        fig_dict[ds][clf].tight_layout()

