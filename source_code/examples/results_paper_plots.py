from pathlib import Path
import numpy as np
import pandas as pd
import os
# from metrics.confusion_matrix import ConfusionMatrix
# from metrics.roc_curve import RocCurve
import matplotlib.pyplot as plt
from joblib import dump, load
import seaborn as sns
import glob

root_path = Path(__file__).parent.parent.parent.parent
plt_save_path = os.path.join(root_path, "experiment_results\\paper_plots\\")
data_sets = ['touch', 'keystroke', 'mouse', 'gait', 'voice']
# data_sets = ['touch', 'keystroke', 'mouse', 'gait']
clf_types = ['svm', 'knn', 'rf']

touch_dir = "adv_attack_hmog"
keystroke_dir = "adv_attack_dsn_cv"
mouse_dir = "adv_attack_mouse"
gait_dir = "adv_attack_gait"
voice_dir = "adv_attack_voice"

hyp_at_f_name = {'svm': '*_hyp_at_prd_svm.csv', 'knn': '*_hyp_at_prd_knn.csv', 'rf': '*_hyp_at_prd_rf.csv'}
stat_at_f_name = {'svm': '*_stat_at_prd_svm.csv', 'knn': '*_stat_at_prd_knn.csv', 'rf': '*_stat_at_prd_rf.csv'}
kpp_at_f_name = {'svm': '*_kpp_at_prd_svm.csv', 'knn': '*_kpp_at_prd_knn.csv', 'rf': '*_kpp_at_prd_rf.csv'}
mk_at_f_name = {'svm': '*_mk_at_prd_svm.csv', 'knn': '*_mk_at_prd_knn.csv', 'rf': '*_mk_at_prd_rf.csv'}

exp_dir = {'touch': touch_dir, 'keystroke': keystroke_dir, 'mouse': mouse_dir, 'gait': gait_dir, 'voice': voice_dir}

exp_paths = {ds: os.path.join(root_path, f'experiment_results\\{exp_dir[ds]}\\')
             for ds in data_sets}

at_file_names = {'Hypervolume': hyp_at_f_name, 'Vanilla-s': stat_at_f_name,
                 'K-means++': kpp_at_f_name, 'MasterKey': mk_at_f_name}
at_types = ['Hypervolume', 'Vanilla-s', 'K-means++', 'MasterKey']

gr_list = ['gr1', 'gr2']

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

cluster_paths = {ds: os.path.join(exp_paths[ds], 'cluster_data\\') for ds in data_sets}

hyp_cls_scoring_paths = {ds: {gr: glob.glob(os.path.join(exp_paths[ds], f"*_hyp_cls_ranking.csv"))
                              for gr in gr_list} for ds in data_sets}

hyp_cls_scoring_files = {ds: {gr: pd.read_csv(hyp_cls_scoring_paths[ds][gr][0])
                              for gr in gr_list} for ds in data_sets}
cls_ol_rankings = pd.DataFrame(columns=['Biometric', 'group', 'cluster_number', 'mean_ol', 'cls_score'])

row = 0
for ds in data_sets:
    for gr in gr_list:
        for rank in range(6):
            cls_ol_rankings.loc[row, 'Biometric'] = ds
            cls_ol_rankings.loc[row, 'group'] = gr
            cls_ol_rankings.loc[row, 'mean_ol'] = hyp_cls_scoring_files[ds][gr].mean_ol.iloc[rank]
            cls_ol_rankings.loc[row, 'cluster_number'] = hyp_cls_scoring_files[ds][gr].cluster_number.iloc[rank]
            cls_ol_rankings.loc[row, 'cls_score'] = hyp_cls_scoring_files[ds][gr].cls_score.iloc[rank]
            row += 1

cluster_data_path = {ds: {gr: {cls: os.path.join(cluster_paths[ds], f"cls_group_{gr[-1]}_{cls}.csv")
                               for cls in range(gr_params[ds][gr])}
                          for gr in gr_list}
                     for ds in data_sets}

cls_data = {ds: {gr: {cls: pd.read_csv(cluster_data_path[ds][gr][cls])
                      for cls in range(gr_params[ds][gr])}
                 for gr in gr_list}
            for ds in data_sets}

at_prd_file_names = {ds: {at: {clf: glob.glob(os.path.join(exp_paths[ds], at_file_names[at][clf]))
                               for clf in clf_types}
                          for at in at_types}
                     for ds in data_sets}

at_prd_files = {ds: {at: {clf: {f.split("\\")[-1].split('_')[0]: pd.read_csv(f)
                                for f in at_prd_file_names[ds][at][clf]}
                          for clf in clf_types}
                     for at in at_types}
                for ds in data_sets}
gr_list = ['gr1', 'gr2']
hyp_cls_score = {ds: {gr: pd.read_csv(os.path.join(exp_paths[ds], f"{gr}_hyp_at_score.csv")) for gr in gr_list}
                 for ds in data_sets}
hyp_cls_info = {ds: {clf: {gr: at_prd_files[ds]['Hypervolume'][clf][gr].cluster_number.unique()
                           for gr in gr_list}
                     for clf in clf_types}
                for ds in data_sets}

at_cls_dat = {
    ds: {
        clf: {
            gr: {
                cls: at_prd_files[ds]['Hypervolume'][clf][gr]
                [at_prd_files[ds]['Hypervolume'][clf][gr].cluster_number == cls].drop('cluster_number', axis=1)
                for cls in hyp_cls_info[ds][clf][gr]}
            for gr in gr_list}
        for clf in clf_types}
    for ds in data_sets}

user_crk = {
    ds: {
        clf: {
            gr: {
                cls: {
                    tr: at_cls_dat[ds][clf][gr][cls].iloc[tr, :]
                    [at_cls_dat[ds][clf][gr][cls].iloc[tr, :] == 1].index.to_numpy()
                    for tr in range(len(at_cls_dat[ds][clf][gr][cls]))}
                for cls in hyp_cls_info[ds][clf][gr]}
            for gr in gr_list}
        for clf in clf_types}
    for ds in data_sets}

user_crk_comb = {
    ds: {clf: {gr: {cls: pd.DataFrame.from_dict(dict([(k, pd.Series(v))
                                                      for k, v in user_crk[ds][clf][gr][cls].items()]))
                    for cls in hyp_cls_info[ds][clf][gr]}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

user_crk_1 = {
    ds: {clf: {gr: {tr: {cls: user_crk[ds][clf][gr][cls][tr] for cls in hyp_cls_info[ds][clf][gr]}
                    for tr in range(3)}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

user_crk_comb_1 = {
    ds: {clf: {gr: {tr: pd.DataFrame.from_dict(dict([(k, pd.Series(v))
                                                     for k, v in user_crk_1[ds][clf][gr][tr].items()]))
                    for tr in range(3)}
               for gr in gr_list}
         for clf in clf_types}
    for ds in data_sets}

# Writing file to disk
hyp_data_save_path = os.path.join(plt_save_path, 'hyp_at_usr_analysis')
for ds in data_sets:
    for clf in clf_types:
        for gr in gr_list:
            for tr in range(3):
                user_crk_comb_1[ds][clf][gr][tr].columns.name = "cluster_number"
                user_crk_comb_1[ds][clf][gr][tr].to_csv(
                    os.path.join(hyp_data_save_path, f"{ds}//usr_crk_{gr}_{clf}_{tr}.csv"), index=True, mode='w+')

for ds in data_sets:
    for clf in clf_types:
        for gr in gr_list:
            at_prd_files[ds]['Hypervolume'][clf][gr] = \
                at_prd_files[ds]['Hypervolume'][clf][gr].drop('cluster_number', axis=1)
a = 1

per_usr_cracked_dict = {ds: {clf: {gr: pd.DataFrame(columns=at_types)
                                   for gr in gr_list}
                             for clf in clf_types}
                        for ds in data_sets}
row_range = 50
for ds in data_sets:
    for at in at_types:
        for clf in clf_types:
            for gr in gr_list:
                cracked_user = np.empty(0)
                users = at_prd_files[ds][at][clf][gr].columns.to_numpy()
                for row in range(row_range):
                    at_cracked_user = \
                        at_prd_files[ds][at][clf][gr].iloc[row, :][
                            at_prd_files[ds][at][clf][gr].iloc[row, :] == 1].index.to_numpy()
                    cracked_user = np.unique(np.append(cracked_user, at_cracked_user))
                    per_usr_cracked = np.round(len(cracked_user) / len(users), 5)
                    per_usr_cracked_dict[ds][clf][gr].loc[row, at] = per_usr_cracked

per_usr_cracked_dict_comb = {ds: {clf: {at: pd.DataFrame(columns=['try_num', 'per_crk', 'at_type']) for at in at_types}
                                  for clf in clf_types}
                             for ds in data_sets}
plt_df = {ds: {clf: pd.DataFrame(columns=['try_num', 'per_crk', 'at_type'])
               for clf in clf_types}
          for ds in data_sets}

for ds in data_sets:
    for clf in clf_types:
        for at in at_types:
            for row in range(row_range):
                per_usr_cracked_dict_comb[ds][clf][at].loc[row, 'try_num'] = row + 1
                val_1 = per_usr_cracked_dict[ds][clf]['gr1'].loc[row, at]
                val_2 = per_usr_cracked_dict[ds][clf]['gr2'].loc[row, at]
                val_3 = np.round(np.mean([val_1, val_2]), 5)
                per_usr_cracked_dict_comb[ds][clf][at].loc[row, "per_crk"] = val_3
                per_usr_cracked_dict_comb[ds][clf][at].loc[row, "at_type"] = at
            per_usr_cracked_dict_comb[ds][clf][at].try_num = \
                per_usr_cracked_dict_comb[ds][clf][at].try_num.astype("float64")
            per_usr_cracked_dict_comb[ds][clf][at]["per_crk"] = \
                per_usr_cracked_dict_comb[ds][clf][at]["per_crk"].astype("float64")
            per_usr_cracked_dict_comb[ds][clf][at]["at_type"] = \
                per_usr_cracked_dict_comb[ds][clf][at]["at_type"].astype("string")

        per_usr_cracked_dict_comb[ds][clf]['Hypervolume'] = \
            pd.concat([pd.DataFrame({'try_num': 0, 'per_crk': 0, 'at_type': 'Hypervolume'}, index=[0]),
                       per_usr_cracked_dict_comb[ds][clf]['Hypervolume']]).reset_index(drop=True)

        per_usr_cracked_dict_comb[ds][clf]['Vanilla-s'] = \
            pd.concat([pd.DataFrame({'try_num': 0, 'per_crk': 0, 'at_type': 'Vanilla-s'}, index=[0]),
                       per_usr_cracked_dict_comb[ds][clf]['Vanilla-s']]).reset_index(drop=True)

        per_usr_cracked_dict_comb[ds][clf]['K-means++'] = \
            pd.concat([pd.DataFrame({'try_num': 0, 'per_crk': 0, 'at_type': 'K-means++'}, index=[0]),
                       per_usr_cracked_dict_comb[ds][clf]['K-means++']]).reset_index(drop=True)

        per_usr_cracked_dict_comb[ds][clf]['MasterKey'] = \
            pd.concat([pd.DataFrame({'try_num': 0, 'per_crk': 0, 'at_type': 'MasterKey'}, index=[0]),
                       per_usr_cracked_dict_comb[ds][clf]['MasterKey']]).reset_index(drop=True)

        plt_df[ds][clf] = pd.concat([per_usr_cracked_dict_comb[ds][clf]['Hypervolume'],
                                     per_usr_cracked_dict_comb[ds][clf]['Vanilla-s'],
                                     per_usr_cracked_dict_comb[ds][clf]['K-means++'],
                                     per_usr_cracked_dict_comb[ds][clf]['MasterKey']])

sns.set_theme(context='poster', style="white", font_scale=1.75)
line_draw_list = [1, 5, 10, 20, 50]
fig_dict = {ds: {clf: plt.figure(figsize=(19.2, 10.8)) for clf in clf_types} for ds in data_sets}
ax_dict = {ds: {clf: fig_dict[ds][clf].add_subplot(111) for clf in clf_types} for ds in data_sets}

for ds in data_sets:
    for clf in clf_types:
        sns.lineplot(data=plt_df[ds][clf], x="try_num", y="per_crk", hue="at_type", ax=ax_dict[ds][clf])
        sns.scatterplot(data=plt_df[ds][clf].loc[line_draw_list, :], x="try_num", y="per_crk", hue="at_type",
                        ax=ax_dict[ds][clf], legend=None)
        ax_dict[ds][clf].legend(loc=(0.0, 1.01), ncol=4, columnspacing=1.0, handletextpad=0.4, handlelength=1.0,
                                frameon=False)
        ax_dict[ds][clf].set_ylabel("% Compromised")
        ax_dict[ds][clf].set_xlabel("Attempts to Bypass")
        ax_dict[ds][clf].set_ylim([-0.01, 1.01])
        for x_pos in line_draw_list:
            ax_dict[ds][clf].axvline(x=x_pos, c='gray', alpha=0.99, linestyle='dotted')
        fig_dict[ds][clf].tight_layout()
        fig_dict[ds][clf].savefig(os.path.join(plt_save_path, f"{ds}_{clf}_at_{row_range}_comp.pdf"))

fig_dict_p = {ds: {clf: {line: plt.figure(figsize=(19.2, 10.8)) for line in line_draw_list}
                   for clf in clf_types} for ds in data_sets}
ax_dict_p = {ds: {clf: {line: fig_dict_p[ds][clf][line].add_subplot(111) for line in line_draw_list}
                  for clf in clf_types} for ds in data_sets}

for ds in data_sets:
    for clf in clf_types:
        for attempt, idx in zip(line_draw_list, range(len(line_draw_list))):
            sp_list = line_draw_list[:idx+1]
            lp_list = [0]
            lp_list.extend(sp_list)
            sns.lineplot(data=plt_df[ds][clf].loc[lp_list, :], x="try_num", y="per_crk",
                         hue="at_type", ax=ax_dict_p[ds][clf][attempt])
            sns.scatterplot(data=plt_df[ds][clf].loc[sp_list, :],
                            x="try_num", y="per_crk", hue="at_type",
                            ax=ax_dict_p[ds][clf][attempt], legend=None)
            ax_dict_p[ds][clf][attempt].legend(loc=(0.0, 1.01), ncol=4, columnspacing=1.0, handletextpad=0.4, handlelength=1.0,
                                    frameon=False)
            ax_dict_p[ds][clf][attempt].set_ylabel("% Compromised")
            ax_dict_p[ds][clf][attempt].set_xlabel("Attempts to Bypass")
            ax_dict_p[ds][clf][attempt].set_ylim([-0.01, 1.01])
            for x_pos in line_draw_list:
                ax_dict_p[ds][clf][attempt].axvline(x=x_pos, c='gray', alpha=0.99, linestyle='dotted')
            fig_dict_p[ds][clf][attempt].tight_layout()
            fig_dict_p[ds][clf][attempt].savefig(os.path.join(plt_save_path, f"{ds}_{clf}_at_{row_range}_comp-{idx}.png"))

paper_table = pd.DataFrame(columns=['Dataset', 'Classifier', 'Try', 'Hypervolume', 'Vanilla-s', 'K-means++',
                                    'MasterKey'])

paper_table_f = {ds: {clf: pd.DataFrame(columns=['try', 'Hypervolume', 'Vanilla-s', 'K-means++', 'MasterKey'])
                      for clf in clf_types}
                 for ds in data_sets}
for ds in data_sets:
    for clf in clf_types:
        for row in range(row_range + 1):
            paper_table_f[ds][clf].loc[row, 'try'] = row
            paper_table_f[ds][clf].loc[row, 'Hypervolume'] = \
                plt_df[ds][clf][plt_df[ds][clf].at_type == "Hypervolume"].per_crk[row]

            paper_table_f[ds][clf].loc[row, 'Vanilla-s'] = \
                plt_df[ds][clf][plt_df[ds][clf].at_type == "Vanilla-s"].per_crk[row]

            paper_table_f[ds][clf].loc[row, 'K-means++'] = \
                plt_df[ds][clf][plt_df[ds][clf].at_type == "K-means++"].per_crk[row]
            paper_table_f[ds][clf].loc[row, 'MasterKey'] = \
                plt_df[ds][clf][plt_df[ds][clf].at_type == "MasterKey"].per_crk[row]

# row_list = [0, 4, 9, 14, 19, 24, 49]
if row_range == 10:
    row_list = [0, 2, 3, 4, 5, 6, 7, 8, 9]
elif row_range == 50:
    # row_list = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 50]
    row_list = range(51)
elif row_range == 100:
    row_list = [0, 4, 9, 24, 49, 74, 99]
row_c = 0
for ds in data_sets:
    for clf in clf_types:
        for trn in row_list:
            paper_table.loc[row_c, 'Dataset'] = ds
            paper_table.loc[row_c, 'Classifier'] = clf
            paper_table.loc[row_c, 'Try'] = trn
            paper_table.loc[row_c, 'Hypervolume'] = paper_table_f[ds][clf].loc[trn, 'Hypervolume']
            paper_table.loc[row_c, 'Vanilla-s'] = paper_table_f[ds][clf].loc[trn, 'Vanilla-s']
            paper_table.loc[row_c, 'K-means++'] = paper_table_f[ds][clf].loc[trn, 'K-means++']
            paper_table.loc[row_c, 'MasterKey'] = paper_table_f[ds][clf].loc[trn, 'MasterKey']
            row_c += 1
paper_table.to_csv(os.path.join(plt_save_path, f"results_{row_range}_sum.csv"), index=False, mode='w+')

