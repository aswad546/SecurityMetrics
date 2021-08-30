import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt

root_path = Path(__file__).parent.parent.parent.parent
data_metric_save_path = os.path.join(root_path, 'experiment_results\\HMOG_full_pop_ol_mi_comp\\')
box_plt_path = os.path.join(data_metric_save_path, 'box_plots')

results = \
    pd.read_csv(os.path.join(data_metric_save_path, "results_overlap_mi_exp_rs_42_100_sample.csv"))

bf_dr_sub_set_cols_list = \
    ["positive user", "pos_user_ol_others_mean", "pos_user_ol_by_others_mean", "C_value",

     "full_feat_tpr_test_set",
     "ol_others_bf_dr_tpr_test_set",
     "ol_by_others_bf_dr_tpr_test_set",
     "mi_bf_dr_tpr_test_set",

     "full_feat_percent_accepted_test_set",
     "ol_others_bf_dr_percent_accepted_test_set",
     "ol_by_others_bf_dr_percent_accepted_test_set",
     "mi_bf_dr_percent_accepted_test_set",

     "full_feat_percent_accepted_overlap_set",
     "ol_others_bf_dr_percent_accepted_overlap_set",
     "ol_by_others_bf_dr_percent_accepted_overlap_set",
     "mi_bf_dr_percent_accepted_overlap_set",
     ]

wf_dr_sub_set_cols_list = \
    ["positive user", "pos_user_ol_others_mean", "pos_user_ol_by_others_mean", "C_value",

     "full_feat_tpr_test_set",
     "ol_others_wf_dr_tpr_test_set",
     "ol_by_others_wf_dr_tpr_test_set",
     "mi_wf_dr_tpr_test_set",

     "full_feat_percent_accepted_test_set",
     "ol_others_wf_dr_percent_accepted_test_set",
     "ol_by_others_wf_dr_percent_accepted_test_set",
     "mi_wf_dr_percent_accepted_test_set",

     "full_feat_percent_accepted_overlap_set",
     "ol_others_wf_dr_percent_accepted_overlap_set",
     "ol_by_others_wf_dr_percent_accepted_overlap_set",
     "mi_wf_dr_percent_accepted_overlap_set"
     ]

common_cols = ["positive user", "pos_user_ol_others_mean", "pos_user_ol_by_others_mean", "C_value"]

full_feat_results_col_list, ol_other_bf_dr_results_col_list, ol_other_wf_dr_results_col_list, \
ol_by_other_bf_dr_results_col_list, ol_by_other_wf_dr_results_col_list, mi_bf_dr_results_col_list, \
mi_wf_dr_results_col_list = (common_cols.copy() for _ in range(7))

full_feat_results_col_list.extend(["full_feat_tpr_test_set",
                                   "full_feat_percent_accepted_test_set",
                                   "full_feat_percent_accepted_overlap_set"
                                   ])
ol_other_bf_dr_results_col_list.extend(["ol_others_bf_dr_tpr_test_set",
                                        "ol_others_bf_dr_percent_accepted_test_set",
                                        "ol_others_bf_dr_percent_accepted_overlap_set"
                                        ])

ol_other_wf_dr_results_col_list.extend(["ol_others_wf_dr_tpr_test_set",
                                        "ol_others_wf_dr_percent_accepted_test_set",
                                        "ol_others_wf_dr_percent_accepted_overlap_set"
                                        ])

ol_by_other_bf_dr_results_col_list.extend(["ol_by_others_bf_dr_tpr_test_set",
                                           "ol_by_others_bf_dr_percent_accepted_test_set",
                                           "ol_by_others_bf_dr_percent_accepted_overlap_set"
                                           ])

ol_by_other_wf_dr_results_col_list.extend(["ol_by_others_wf_dr_tpr_test_set",
                                           "ol_by_others_wf_dr_percent_accepted_test_set",
                                           "ol_by_others_wf_dr_percent_accepted_overlap_set"
                                           ])

mi_bf_dr_results_col_list.extend(["mi_bf_dr_tpr_test_set",
                                  "mi_bf_dr_percent_accepted_test_set",
                                  "mi_bf_dr_percent_accepted_overlap_set"
                                  ])

mi_wf_dr_results_col_list.extend(["mi_wf_dr_tpr_test_set",
                                  "mi_wf_dr_percent_accepted_test_set",
                                  "mi_wf_dr_percent_accepted_overlap_set"
                                  ])

results_bf_dr = results[bf_dr_sub_set_cols_list]
results_wf_dr = results[wf_dr_sub_set_cols_list]

fil_thresh = 0.01
tpr_bf_dr_fil_cond_1 = \
    abs(results_bf_dr['ol_others_bf_dr_tpr_test_set'] - results_bf_dr['ol_by_others_bf_dr_tpr_test_set']) <= fil_thresh
tpr_bf_dr_fil_cond_2 = \
    abs(results_bf_dr['ol_others_bf_dr_tpr_test_set'] - results_bf_dr['mi_bf_dr_tpr_test_set']) <= fil_thresh
tpr_bf_dr_filter = (tpr_bf_dr_fil_cond_1 & tpr_bf_dr_fil_cond_2)
results_fil_bf_dr = results_bf_dr[tpr_bf_dr_filter]

tpr_wf_dr_fil_cond_1 = \
    abs(results_wf_dr['ol_others_wf_dr_tpr_test_set'] - results_wf_dr['ol_by_others_wf_dr_tpr_test_set']) <= fil_thresh
tpr_wf_dr_fil_cond_2 = \
    abs(results_wf_dr['ol_others_wf_dr_tpr_test_set'] - results_wf_dr['mi_wf_dr_tpr_test_set']) <= fil_thresh
tpr_wf_dr_filter = (tpr_bf_dr_fil_cond_1 & tpr_bf_dr_fil_cond_2)
results_fil_wf_dr = results_wf_dr[tpr_bf_dr_filter]

bf_dr_full_feat_results = results_fil_bf_dr.loc[:, full_feat_results_col_list]
wf_dr_full_feat_results = results_fil_wf_dr.loc[:, full_feat_results_col_list]

ol_other_bf_dr_results = results_fil_bf_dr.loc[:, ol_other_bf_dr_results_col_list]
ol_by_other_bf_dr_results = results_fil_bf_dr.loc[:, ol_by_other_bf_dr_results_col_list]
mi_bf_dr_results = results_fil_bf_dr.loc[:, mi_bf_dr_results_col_list]

ol_other_wf_dr_results = results_fil_wf_dr.loc[:, ol_other_wf_dr_results_col_list]
ol_by_other_wf_dr_results = results_fil_wf_dr.loc[:, ol_by_other_wf_dr_results_col_list]
mi_wf_dr_results = results_fil_wf_dr.loc[:, mi_wf_dr_results_col_list]

bf_dr_full_feat_results['condition'] = "full_feat"
wf_dr_full_feat_results['condition'] = "full_feat"

ol_other_bf_dr_results['condition'] = "ol_other_bf_dr"
ol_by_other_bf_dr_results['condition'] = "ol_by_other_bf_dr"
mi_bf_dr_results['condition'] = "mi_bf_dr"

ol_other_wf_dr_results['condition'] = "ol_other_wf_dr"
ol_by_other_wf_dr_results['condition'] = "ol_by_other_wf_dr"
mi_wf_dr_results['condition'] = "mi_wf_dr"

bf_dr_full_feat_results = bf_dr_full_feat_results.rename(
    columns={"full_feat_tpr_test_set": "tpr_test_set",
             "full_feat_percent_accepted_test_set": "percent_accepted_test_set",
             "full_feat_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

wf_dr_full_feat_results = wf_dr_full_feat_results.rename(
    columns={"full_feat_tpr_test_set": "tpr_test_set",
             "full_feat_percent_accepted_test_set": "percent_accepted_test_set",
             "full_feat_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)
ol_other_bf_dr_results = ol_other_bf_dr_results.rename(
    columns={"ol_others_bf_dr_tpr_test_set": "tpr_test_set",
             "ol_others_bf_dr_percent_accepted_test_set": "percent_accepted_test_set",
             "ol_others_bf_dr_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

ol_other_wf_dr_results = ol_other_wf_dr_results.rename(
    columns={"ol_others_wf_dr_tpr_test_set": "tpr_test_set",
             "ol_others_wf_dr_percent_accepted_test_set": "percent_accepted_test_set",
             "ol_others_wf_dr_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

ol_by_other_bf_dr_results = ol_by_other_bf_dr_results.rename(
    columns={"ol_by_others_bf_dr_tpr_test_set": "tpr_test_set",
             "ol_by_others_bf_dr_percent_accepted_test_set": "percent_accepted_test_set",
             "ol_by_others_bf_dr_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

ol_by_other_wf_dr_results = ol_by_other_wf_dr_results.rename(
    columns={"ol_by_others_wf_dr_tpr_test_set": "tpr_test_set",
             "ol_by_others_wf_dr_percent_accepted_test_set": "percent_accepted_test_set",
             "ol_by_others_wf_dr_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

mi_bf_dr_results = mi_bf_dr_results.rename(
    columns={"mi_bf_dr_tpr_test_set": "tpr_test_set",
             "mi_bf_dr_percent_accepted_test_set": "percent_accepted_test_set",
             "mi_bf_dr_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

mi_wf_dr_results = mi_wf_dr_results.rename(
    columns={"mi_wf_dr_tpr_test_set": "tpr_test_set",
             "mi_wf_dr_percent_accepted_test_set": "percent_accepted_test_set",
             "mi_wf_dr_percent_accepted_overlap_set": "percent_accepted_overlap_set"}
)

filtered_results_stacked_bf_dr = \
    pd.concat([bf_dr_full_feat_results,
               ol_other_bf_dr_results,
               ol_by_other_bf_dr_results,
               mi_bf_dr_results, ], axis=0)

filtered_results_stacked_wf_dr = \
    pd.concat([wf_dr_full_feat_results,
               ol_other_wf_dr_results,
               ol_by_other_wf_dr_results,
               mi_wf_dr_results], axis=0)

fig_1a = plt.figure()
fig_1a.suptitle('Percent negative users accepted as positive on overlap set for best feature drop box plot',
                fontsize=20)
fig_1a.set_figheight(9.5)
fig_1a.set_figwidth(19)
sns.boxplot(x='positive user', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr, hue='condition',
            showmeans=True,
            meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
plt.savefig(os.path.join(box_plt_path, f"bf_dr_pos_user_percent_accepted_overlap_set_box_plt.png"))

fig_1b = plt.figure()
fig_1b.suptitle('Percent negative users accepted as positive on overlap set for best feature drop strip plot',
                fontsize=20)
fig_1b.set_figheight(9.5)
fig_1b.set_figwidth(19)
sns.stripplot(x='positive user', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr, hue='condition')
plt.savefig(os.path.join(box_plt_path, f"bf_dr_pos_user_percent_accepted_overlap_set_strip_plot.png"))

fig_2a = plt.figure()
fig_2a.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop box plot',
                fontsize=20)
fig_2a.set_figheight(9.5)
fig_2a.set_figwidth(19)
sns.boxplot(x='positive user', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr, hue='condition',
            showmeans=True,
            meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
plt.savefig(os.path.join(box_plt_path, f"wf_dr_pos_user_percent_accepted_overlap_set_box_plt.png"))

fig_2b = plt.figure()
fig_2b.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop strip plot',
                fontsize=20)
fig_2b.set_figheight(9.5)
fig_2b.set_figwidth(19)
sns.stripplot(x='positive user', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr, hue='condition')
plt.savefig(os.path.join(box_plt_path, f"wf_dr_pos_user_percent_accepted_overlap_set_strip_plot.png"))

fig_3a = plt.figure()
fig_3a.suptitle('Percent negative users accepted as positive on overlap set for best feature drop box plot',
                fontsize=20)
fig_3a.set_figheight(9.5)
fig_3a.set_figwidth(19)
sns.boxplot(x='pos_user_ol_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr,
            hue='condition', showmeans=True,
            meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
plt.savefig(os.path.join(box_plt_path, f"bf_dr_pos_user_ol_others_mean_overlap_set_box_plt.png"))

fig_3b = plt.figure()
fig_3b.suptitle('Percent negative users accepted as positive on overlap set for best feature drop strip plot',
                fontsize=20)
fig_3b.set_figheight(9.5)
fig_3b.set_figwidth(19)
sns.stripplot(x='pos_user_ol_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr,
              hue='condition')
plt.savefig(os.path.join(box_plt_path, f"bf_dr_pos_user_ol_others_mean_overlap_set_strip_plt.png"))

fig_4a = plt.figure()
fig_4a.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop box plot',
                fontsize=20)
fig_4a.set_figheight(9.5)
fig_4a.set_figwidth(19)
sns.boxplot(x='pos_user_ol_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr,
            hue='condition', showmeans=True,
            meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
plt.savefig(os.path.join(box_plt_path, f"wf_dr_pos_user_ol_others_mean_overlap_set_box_plt.png"))

fig_4b = plt.figure()
fig_4b.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop strip plot',
                fontsize=20)
fig_4b.set_figheight(9.5)
fig_4b.set_figwidth(19)
sns.stripplot(x='pos_user_ol_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr,
              hue='condition')
plt.savefig(os.path.join(box_plt_path, f"wf_dr_pos_user_ol_others_mean_overlap_set_strip_plt.png"))

fig_5a = plt.figure()
fig_5a.suptitle('Percent negative users accepted as positive on overlap set for best feature drop box plot',
                fontsize=20)
fig_5a.set_figheight(9.5)
fig_5a.set_figwidth(19)
sns.boxplot(x='pos_user_ol_by_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr,
            hue='condition', showmeans=True,
            meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
plt.savefig(os.path.join(box_plt_path, f"bf_dr_pos_user_ol_by_others_mean_overlap_set_box_plt.png"))

fig_5b = plt.figure()
fig_5b.suptitle('Percent negative users accepted as positive on overlap set for best feature drop strip plot',
                fontsize=20)
fig_5b.set_figheight(9.5)
fig_5b.set_figwidth(19)
sns.stripplot(x='pos_user_ol_by_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr,
              hue='condition')
plt.savefig(os.path.join(box_plt_path, f"bf_dr_pos_user_ol_by_others_mean_overlap_set_strip_plt.png"))

fig_6a = plt.figure()
fig_6a.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop box plot',
                fontsize=20)
fig_6a.set_figheight(9.5)
fig_6a.set_figwidth(19)
sns.boxplot(x='pos_user_ol_by_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr,
            hue='condition', showmeans=True,
            meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
plt.savefig(os.path.join(box_plt_path, f"wf_dr_pos_user_ol_by_others_mean_overlap_set_box_plt.png"))

fig_6b = plt.figure()
fig_6b.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop strip plot',
                fontsize=20)
fig_6b.set_figheight(9.5)
fig_6b.set_figwidth(19)
sns.stripplot(x='pos_user_ol_by_others_mean', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr,
              hue='condition')
plt.savefig(os.path.join(box_plt_path, f"wf_dr_pos_user_ol_by_others_mean_overlap_set_strip_plt.png"))

# fig_7a = plt.figure()
# fig_7a.suptitle('Percent negative users accepted as positive on overlap set for best feature drop box plot', fontsize=20)
# fig_7a.set_figheight(9.5)
# fig_7a.set_figwidth(19)
# sns.boxplot(x='tpr_test_set', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr, hue='condition', showmeans=True,
#             meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
#
# fig_7b = plt.figure()
# fig_7b.suptitle('Percent negative users accepted as positive on overlap set for best feature drop strip plot', fontsize=20)
# fig_7b.set_figheight(9.5)
# fig_7b.set_figwidth(19)
# sns.stripplot(x='tpr_test_set', y='percent_accepted_overlap_set', data=filtered_results_stacked_bf_dr, hue='condition')
#
# fig_8a = plt.figure()
# fig_8a.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop box plot', fontsize=20)
# fig_8a.set_figheight(9.5)
# fig_8a.set_figwidth(19)
# sns.boxplot(x='tpr_test_set', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr, hue='condition', showmeans=True,
#             meanprops={"markerfacecolor": "firebrick", 'markeredgecolor': 'black'}, fliersize=5)
#
# fig_8b = plt.figure()
# fig_8b.suptitle('Percent negative users accepted as positive on overlap set for worst feature drop strip plot', fontsize=20)
# fig_8b.set_figheight(9.5)
# fig_8b.set_figwidth(19)
# sns.stripplot(x='tpr_test_set', y='percent_accepted_overlap_set', data=filtered_results_stacked_wf_dr, hue='condition')
"""
Calculating stats for tpr_test_set, percent_accepted_test_set, percent_accepted_overlap_set


"""
results_stats = pd.DataFrame(columns=["full_feat_tpr_test_set",
                                      "ol_other_bf_dr_tpr_test_set",
                                      "ol_other_wf_dr_tpr_test_set",
                                      "ol_by_other_bf_dr_tpr_test_set",
                                      "ol_by_other_wf_dr_tpr_test_set",
                                      "mi_bf_dr_tpr_test_set",
                                      "mi_wf_dr_tpr_test_set",

                                      "full_feat_percent_accepted_test_set",
                                      "ol_other_bf_dr_percent_accepted_test_set",
                                      "ol_other_wf_dr_percent_accepted_test_set",
                                      "ol_by_other_bf_dr_percent_accepted_test_set",
                                      "ol_by_other_wf_dr_percent_accepted_test_set",
                                      "mi_bf_dr_percent_accepted_test_set",
                                      "mi_wf_dr_percent_accepted_test_set",

                                      "full_feat_percent_accepted_overlap_set",
                                      "ol_other_bf_dr_percent_accepted_overlap_set",
                                      "ol_other_wf_dr_percent_accepted_overlap_set",
                                      "ol_by_other_bf_dr_percent_accepted_overlap_set",
                                      "ol_by_other_wf_dr_percent_accepted_overlap_set",
                                      "mi_bf_dr_percent_accepted_overlap_set",
                                      "mi_wf_dr_percent_accepted_overlap_set"
                                      ],
                             index=["mean", "median", "standard_deviation", "variance"])

results_stats.loc["mean", "full_feat_tpr_test_set"] = \
    bf_dr_full_feat_results.tpr_test_set.mean()
results_stats.loc["median", "full_feat_tpr_test_set"] =\
    bf_dr_full_feat_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "full_feat_tpr_test_set"] =\
    bf_dr_full_feat_results.tpr_test_set.std()
results_stats.loc["variance", "full_feat_tpr_test_set"] =\
    bf_dr_full_feat_results.tpr_test_set.var()

results_stats.loc["mean", "ol_other_bf_dr_tpr_test_set"] = \
    ol_other_bf_dr_results.tpr_test_set.mean()
results_stats.loc["median", "ol_other_bf_dr_tpr_test_set"] =\
    ol_other_bf_dr_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "ol_other_bf_dr_tpr_test_set"] =\
    ol_other_bf_dr_results.tpr_test_set.std()
results_stats.loc["variance", "ol_other_bf_dr_tpr_test_set"] =\
    ol_other_bf_dr_results.tpr_test_set.var()

results_stats.loc["mean", "ol_other_wf_dr_tpr_test_set"] = \
    ol_other_wf_dr_results.tpr_test_set.mean()
results_stats.loc["median", "ol_other_wf_dr_tpr_test_set"] =\
    ol_other_wf_dr_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "ol_other_wf_dr_tpr_test_set"] =\
    ol_other_wf_dr_results.tpr_test_set.std()
results_stats.loc["variance", "ol_other_wf_dr_tpr_test_set"] =\
    ol_other_wf_dr_results.tpr_test_set.var()

#
results_stats.loc["mean", "ol_by_other_bf_dr_tpr_test_set"] = \
    ol_by_other_bf_dr_results.tpr_test_set.mean()
results_stats.loc["median", "ol_by_other_bf_dr_tpr_test_set"] =\
    ol_by_other_bf_dr_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "ol_by_other_bf_dr_tpr_test_set"] =\
    ol_by_other_bf_dr_results.tpr_test_set.std()
results_stats.loc["variance", "ol_by_other_bf_dr_tpr_test_set"] =\
    ol_by_other_bf_dr_results.tpr_test_set.var()

results_stats.loc["mean", "ol_by_other_wf_dr_tpr_test_set"] = \
    ol_by_other_wf_dr_results.tpr_test_set.mean()
results_stats.loc["median", "ol_by_other_wf_dr_tpr_test_set"] =\
    ol_by_other_wf_dr_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "ol_by_other_wf_dr_tpr_test_set"] =\
    ol_by_other_wf_dr_results.tpr_test_set.std()
results_stats.loc["variance", "ol_by_other_wf_dr_tpr_test_set"] =\
    ol_by_other_wf_dr_results.tpr_test_set.var()
#
results_stats.loc["mean", "mi_bf_dr_tpr_test_set"] = \
    mi_bf_dr_results.tpr_test_set.mean()
results_stats.loc["median", "mi_bf_dr_tpr_test_set"] =\
    mi_bf_dr_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "mi_bf_dr_tpr_test_set"] =\
    mi_bf_dr_results.tpr_test_set.std()
results_stats.loc["variance", "mi_bf_dr_tpr_test_set"] =\
    mi_bf_dr_results.tpr_test_set.var()

results_stats.loc["mean", "mi_wf_dr_tpr_test_set"] = \
    mi_wf_dr_results.tpr_test_set.mean()
results_stats.loc["median", "mi_wf_dr_tpr_test_set"] =\
    mi_wf_dr_results.tpr_test_set.median()
results_stats.loc["standard_deviation", "mi_wf_dr_tpr_test_set"] =\
    mi_wf_dr_results.tpr_test_set.std()
results_stats.loc["variance", "mi_wf_dr_tpr_test_set"] =\
    mi_wf_dr_results.tpr_test_set.var()

#
results_stats.loc["mean", "full_feat_percent_accepted_test_set"] = \
    bf_dr_full_feat_results.percent_accepted_test_set.mean()
results_stats.loc["median", "full_feat_percent_accepted_test_set"] =\
    bf_dr_full_feat_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "full_feat_percent_accepted_test_set"] =\
    bf_dr_full_feat_results.percent_accepted_test_set.std()
results_stats.loc["variance", "full_feat_percent_accepted_test_set"] =\
    bf_dr_full_feat_results.percent_accepted_test_set.var()

#
results_stats.loc["mean", "ol_other_bf_dr_percent_accepted_test_set"] = \
    ol_other_bf_dr_results.percent_accepted_test_set.mean()
results_stats.loc["median", "ol_other_bf_dr_percent_accepted_test_set"] =\
    ol_other_bf_dr_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "ol_other_bf_dr_percent_accepted_test_set"] =\
    ol_other_bf_dr_results.percent_accepted_test_set.std()
results_stats.loc["variance", "ol_other_bf_dr_percent_accepted_test_set"] =\
    ol_other_bf_dr_results.percent_accepted_test_set.var()

results_stats.loc["mean", "ol_other_wf_dr_percent_accepted_test_set"] = \
    ol_other_wf_dr_results.percent_accepted_test_set.mean()
results_stats.loc["median", "ol_other_wf_dr_percent_accepted_test_set"] =\
    ol_other_wf_dr_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "ol_other_wf_dr_percent_accepted_test_set"] =\
    ol_other_wf_dr_results.percent_accepted_test_set.std()
results_stats.loc["variance", "ol_other_wf_dr_percent_accepted_test_set"] =\
    ol_other_wf_dr_results.percent_accepted_test_set.var()

#
results_stats.loc["mean", "ol_by_other_bf_dr_percent_accepted_test_set"] = \
    ol_by_other_bf_dr_results.percent_accepted_test_set.mean()
results_stats.loc["median", "ol_by_other_bf_dr_percent_accepted_test_set"] =\
    ol_by_other_bf_dr_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "ol_by_other_bf_dr_percent_accepted_test_set"] =\
    ol_by_other_bf_dr_results.percent_accepted_test_set.std()
results_stats.loc["variance", "ol_by_other_bf_dr_percent_accepted_test_set"] =\
    ol_by_other_bf_dr_results.percent_accepted_test_set.var()

results_stats.loc["mean", "ol_by_other_wf_dr_percent_accepted_test_set"] = \
    ol_by_other_wf_dr_results.percent_accepted_test_set.mean()
results_stats.loc["median", "ol_by_other_wf_dr_percent_accepted_test_set"] =\
    ol_by_other_wf_dr_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "ol_by_other_wf_dr_percent_accepted_test_set"] =\
    ol_by_other_wf_dr_results.percent_accepted_test_set.std()
results_stats.loc["variance", "ol_by_other_wf_dr_percent_accepted_test_set"] =\
    ol_by_other_wf_dr_results.percent_accepted_test_set.var()

#
results_stats.loc["mean", "mi_bf_dr_percent_accepted_test_set"] = \
    mi_bf_dr_results.percent_accepted_test_set.mean()
results_stats.loc["median", "mi_bf_dr_percent_accepted_test_set"] =\
    mi_bf_dr_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "mi_bf_dr_percent_accepted_test_set"] =\
    mi_bf_dr_results.percent_accepted_test_set.std()
results_stats.loc["variance", "mi_bf_dr_percent_accepted_test_set"] =\
    mi_bf_dr_results.percent_accepted_test_set.var()

results_stats.loc["mean", "mi_wf_dr_percent_accepted_test_set"] = \
    mi_wf_dr_results.percent_accepted_test_set.mean()
results_stats.loc["median", "mi_wf_dr_percent_accepted_test_set"] =\
    mi_wf_dr_results.percent_accepted_test_set.median()
results_stats.loc["standard_deviation", "mi_wf_dr_percent_accepted_test_set"] =\
    mi_wf_dr_results.percent_accepted_test_set.std()
results_stats.loc["variance", "mi_wf_dr_percent_accepted_test_set"] =\
    mi_wf_dr_results.percent_accepted_test_set.var()


#
results_stats.loc["mean", "full_feat_percent_accepted_overlap_set"] = \
    bf_dr_full_feat_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "full_feat_percent_accepted_overlap_set"] =\
    bf_dr_full_feat_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "full_feat_percent_accepted_overlap_set"] =\
    bf_dr_full_feat_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "full_feat_percent_accepted_overlap_set"] =\
    bf_dr_full_feat_results.percent_accepted_overlap_set.var()

#
results_stats.loc["mean", "ol_other_bf_dr_percent_accepted_overlap_set"] = \
    ol_other_bf_dr_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "ol_other_bf_dr_percent_accepted_overlap_set"] =\
    ol_other_bf_dr_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "ol_other_bf_dr_percent_accepted_overlap_set"] =\
    ol_other_bf_dr_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "ol_other_bf_dr_percent_accepted_overlap_set"] =\
    ol_other_bf_dr_results.percent_accepted_overlap_set.var()

results_stats.loc["mean", "ol_other_wf_dr_percent_accepted_overlap_set"] = \
    ol_other_wf_dr_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "ol_other_wf_dr_percent_accepted_overlap_set"] =\
    ol_other_wf_dr_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "ol_other_wf_dr_percent_accepted_overlap_set"] =\
    ol_other_wf_dr_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "ol_other_wf_dr_percent_accepted_overlap_set"] =\
    ol_other_wf_dr_results.percent_accepted_overlap_set.var()

#
results_stats.loc["mean", "ol_by_other_bf_dr_percent_accepted_overlap_set"] = \
    ol_by_other_bf_dr_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "ol_by_other_bf_dr_percent_accepted_overlap_set"] =\
    ol_by_other_bf_dr_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "ol_by_other_bf_dr_percent_accepted_overlap_set"] =\
    ol_by_other_bf_dr_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "ol_by_other_bf_dr_percent_accepted_overlap_set"] =\
    ol_by_other_bf_dr_results.percent_accepted_overlap_set.var()

results_stats.loc["mean", "ol_by_other_wf_dr_percent_accepted_overlap_set"] = \
    ol_by_other_wf_dr_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "ol_by_other_wf_dr_percent_accepted_overlap_set"] =\
    ol_by_other_wf_dr_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "ol_by_other_wf_dr_percent_accepted_overlap_set"] =\
    ol_by_other_wf_dr_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "ol_by_other_wf_dr_percent_accepted_overlap_set"] =\
    ol_by_other_wf_dr_results.percent_accepted_overlap_set.var()

#
results_stats.loc["mean", "mi_bf_dr_percent_accepted_overlap_set"] = \
    mi_bf_dr_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "mi_bf_dr_percent_accepted_overlap_set"] =\
    mi_bf_dr_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "mi_bf_dr_percent_accepted_overlap_set"] =\
    mi_bf_dr_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "mi_bf_dr_percent_accepted_overlap_set"] =\
    mi_bf_dr_results.percent_accepted_overlap_set.var()

results_stats.loc["mean", "mi_wf_dr_percent_accepted_overlap_set"] = \
    mi_wf_dr_results.percent_accepted_overlap_set.mean()
results_stats.loc["median", "mi_wf_dr_percent_accepted_overlap_set"] =\
    mi_wf_dr_results.percent_accepted_overlap_set.median()
results_stats.loc["standard_deviation", "mi_wf_dr_percent_accepted_overlap_set"] =\
    mi_wf_dr_results.percent_accepted_overlap_set.std()
results_stats.loc["variance", "mi_wf_dr_percent_accepted_overlap_set"] =\
    mi_wf_dr_results.percent_accepted_overlap_set.var()
rt = results_stats.transpose()
plt.show()

a = 1
