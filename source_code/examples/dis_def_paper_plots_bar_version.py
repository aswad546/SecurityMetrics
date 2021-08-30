import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

root_path = Path(__file__).parent.parent.parent.parent
root_path = Path(__file__).parent.parent.parent.parent
data_metric_save_path = os.path.join(root_path, 'experiment_results\\dis_def_exp\\')

res_files_path = glob.glob(os.path.join(data_metric_save_path, f"def_res*_v2.csv"))
usr_test_files_path = glob.glob((os.path.join(data_metric_save_path, f"usr_test*_v2.csv")))
at_types = ['hyp', 'kpp', 'stat', 'mk']
clf_types = ['SVM', 'KNN', 'RF']
# clf_types = ['SVM']
data_sets = ['Touch', 'Keystroke', 'Mouse', 'Gait', 'Voice']
# data_sets = ['Mouse']
dist_type = ['cosine', 'euclidean']
res_files = dict()
for f in res_files_path:
    pts = f.split('.')[0][-6]
    res_files[f'{pts}_pt'] = pd.read_csv(f)
    res_files[f'{pts}_pt']['des_points'] = int(pts)
    res_files[f'{pts}_pt']['classifier'] =  res_files[f'{pts}_pt']['classifier'].str.upper()
    res_files[f'{pts}_pt']['dataset'] =  res_files[f'{pts}_pt']['dataset'].str.capitalize()

user_test_files = dict()
for f in usr_test_files_path:
    pts = f.split('.')[0][-6]
    user_test_files[f'{pts}_pt'] = pd.read_csv(f)
    user_test_files[f'{pts}_pt']['des_points'] = int(pts)
    user_test_files[f'{pts}_pt']['classifier'] = user_test_files[f'{pts}_pt']['classifier'].str.upper()
    user_test_files[f'{pts}_pt']['dataset'] = user_test_files[f'{pts}_pt']['dataset'].str.capitalize()
#

plot_data_2pt = pd.DataFrame(columns=['Biometric', 'Classifier', 'attack', 'attack_detection_rate',
                                      'user_false_det_rate'])

plot_data_3pt = pd.DataFrame(columns=['Biometric', 'Classifier', 'attack', 'attack_detection_rate',
                                      'user_false_det_rate'])

row = 0
for ds in data_sets:
    for clf in clf_types:
        for at in at_types:
            plot_data_2pt.loc[row, 'Biometric'] = ds
            plot_data_2pt.loc[row, 'Classifier'] = clf
            plot_data_2pt.loc[row, 'attack'] = at
            plot_data_2pt.loc[row, 'attack_detection_rate'] = at
            attack_detection_results = res_files['2_pt'][(res_files['2_pt'].dataset == ds)
                                                         & (res_files['2_pt'].attack == at)
                                                         & (res_files['2_pt'].classifier == clf)]\
                .decision.value_counts(normalize=True)
            attack_det_results_len = len(attack_detection_results)
            if attack_det_results_len == 2:
                plot_data_2pt.loc[row, 'attack_detection_rate'] = float(attack_detection_results[True])
            elif attack_det_results_len == 0:
                pass
            else:
                if attack_detection_results.index.isin([True]):
                    plot_data_2pt.loc[row, 'attack_detection_rate'] = float(attack_detection_results[True])
                else:
                    plot_data_2pt.loc[row, 'attack_detection_rate'] = float(1 - attack_detection_results[False])

            user_detection_results = user_test_files['2_pt'][(user_test_files['2_pt'].dataset == ds)
                                                         & (user_test_files['2_pt'].classifier == clf)]\
                .decision.value_counts(normalize=True)
            user_det_results_len = len(user_detection_results)
            if user_det_results_len == 2:
                plot_data_2pt.loc[row, 'user_false_det_rate'] = float(user_detection_results[True])
            elif user_det_results_len == 0:
                pass
            else:
                if user_detection_results.index.isin([True]):
                    plot_data_2pt.loc[row, 'user_false_det_rate'] = float(user_detection_results[True])
                else:
                    plot_data_2pt.loc[row, 'user_false_det_rate'] = float(1 - user_detection_results[False])
            row += 1

row = 0
for ds in data_sets:
    for clf in clf_types:
        for at in at_types:
            plot_data_3pt.loc[row, 'Biometric'] = ds
            plot_data_3pt.loc[row, 'Classifier'] = clf
            plot_data_3pt.loc[row, 'attack'] = at
            plot_data_3pt.loc[row, 'attack_detection_rate'] = at
            attack_detection_results = res_files['3_pt'][(res_files['3_pt'].dataset == ds)
                                                         & (res_files['3_pt'].attack == at)
                                                         & (res_files['3_pt'].classifier == clf)]\
                .decision.value_counts(normalize=True)
            attack_det_results_len = len(attack_detection_results)
            if attack_det_results_len == 2:
                plot_data_3pt.loc[row, 'attack_detection_rate'] = float(attack_detection_results[True])
            elif attack_det_results_len == 0:
                pass
            else:
                if attack_detection_results.index.isin([True]):
                    plot_data_3pt.loc[row, 'attack_detection_rate'] = float(attack_detection_results[True])
                else:
                    plot_data_3pt.loc[row, 'attack_detection_rate'] = float(1 - attack_detection_results[False])

            user_detection_results = user_test_files['3_pt'][(user_test_files['3_pt'].dataset == ds)
                                                         & (user_test_files['3_pt'].classifier == clf)]\
                .decision.value_counts(normalize=True)
            user_det_results_len = len(user_detection_results)
            if user_det_results_len == 2:
                plot_data_3pt.loc[row, 'user_false_det_rate'] = float(user_detection_results[True])
            elif user_det_results_len == 0:
                pass
            else:
                if user_detection_results.index.isin([True]):
                    plot_data_3pt.loc[row, 'user_false_det_rate'] = float(user_detection_results[True])
                else:
                    plot_data_3pt.loc[row, 'user_false_det_rate'] = float(1 - user_detection_results[False])
            row += 1

plot_data_2pt = plot_data_2pt.dropna()
plot_data_2pt['attack_detection_rate'] = plot_data_2pt['attack_detection_rate'].astype('float64')
plot_data_2pt['user_false_det_rate'] = plot_data_2pt['user_false_det_rate'].astype('float64')

plot_data_3pt = plot_data_3pt.dropna()
plot_data_3pt['attack_detection_rate'] = plot_data_3pt['attack_detection_rate'].astype('float64')
plot_data_3pt['user_false_det_rate'] = plot_data_3pt['user_false_det_rate'].astype('float64')


attack_data_sep_2pt = {at: plot_data_2pt[plot_data_2pt.attack == at] for at in at_types}
attack_data_sep_3pt = {at: plot_data_3pt[plot_data_2pt.attack == at] for at in at_types}

sns.set_theme(context='poster', style="white", font_scale=1.5)

at_l_plt_fig_2pt = {at: plt.figure(figsize=(19.2, 10.8)) for at in at_types}
at_l_plt_ax_2pt = {at: at_l_plt_fig_2pt[at].add_subplot(111) for at in at_types}
at_l_plt_ax_2_2pt = {at: at_l_plt_ax_2pt[at].twinx() for at in at_types}

at_l_plt_fig_3pt = {at: plt.figure(figsize=(19.2, 10.8)) for at in at_types}
at_l_plt_ax_3pt = {at: at_l_plt_fig_3pt[at].add_subplot(111) for at in at_types}
at_l_plt_ax_2_3pt = {at: at_l_plt_ax_3pt[at].twinx() for at in at_types}

for at in at_types:
    # 2 points
    sns.barplot(data=attack_data_sep_2pt[at],
                x='Biometric', y='attack_detection_rate', hue='Classifier', ci=None, ax=at_l_plt_ax_2pt[at], alpha=0.75)

    sns.barplot(data=attack_data_sep_2pt[at],
                x='Biometric', y='user_false_det_rate', hue='Classifier', ci=None, ax=at_l_plt_ax_2pt[at], alpha=1.0,
                hatch='/')
    handle_2pt, label_2pt = at_l_plt_ax_2pt[at].get_legend_handles_labels()
    label_2pt[0] = 'ADR SVM'
    label_2pt[1] = 'ADR KNN'
    label_2pt[2] = 'ADR RF'
    label_2pt[3] = 'FDR SVM'
    label_2pt[4] = 'FDR KNN'
    label_2pt[5] = 'FDR RF'

    at_l_plt_ax_2pt[at].legend(handle_2pt, label_2pt, loc=(0.2, 1.01), ncol=3, columnspacing=1.0, handletextpad=0.4,
                               handlelength=1.0, frameon=False)

    at_l_plt_ax_2pt[at].set_ylabel('Attack Detection Rate (ADR)')
    at_l_plt_ax_2_2pt[at].set_ylabel('False Detection Rate (FDR)')
    at_l_plt_fig_2pt[at].tight_layout()
    at_l_plt_fig_2pt[at].savefig(os.path.join(data_metric_save_path, f"{at}_bar_plot_2pt_v2.pdf"), bbox_inches="tight")

    # 3 points
    sns.barplot(data=attack_data_sep_3pt[at],
                x='Biometric', y='attack_detection_rate', hue='Classifier', ci=None, ax=at_l_plt_ax_3pt[at], alpha=0.75)

    sns.barplot(data=attack_data_sep_3pt[at],
                x='Biometric', y='user_false_det_rate', hue='Classifier', ci=None, ax=at_l_plt_ax_3pt[at], alpha=1.0,
                hatch='/')

    handle_3pt, label_3pt = at_l_plt_ax_3pt[at].get_legend_handles_labels()
    label_3pt[0] = 'ADR SVM'
    label_3pt[1] = 'ADR KNN'
    label_3pt[2] = 'ADR RF'
    label_3pt[3] = 'FDR SVM'
    label_3pt[4] = 'FDR KNN'
    label_3pt[5] = 'FDR RF'

    at_l_plt_ax_3pt[at].legend(handle_3pt, label_3pt, loc=(0.2, 1.01), ncol=3, columnspacing=1.0, handletextpad=0.4,
                               handlelength=1.0, frameon=False)

    at_l_plt_ax_3pt[at].set_ylabel('Attack Detection Rate (ADR)')
    at_l_plt_ax_2_3pt[at].set_ylabel('False Detection Rate (FDR)')
    at_l_plt_fig_3pt[at].tight_layout()
    at_l_plt_fig_3pt[at].savefig(os.path.join(data_metric_save_path, f"{at}_bar_plot_3pt_v2.pdf"),
                                 bbox_inches="tight")


