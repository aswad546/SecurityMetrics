import time
from pathlib import Path
import pandas as pd
import os
from source_code.dataset.biometric_dataset import BioDataSet
from source_code.dataset.low_variance_feature_removal import LowVarFeatRemoval
from source_code.utilities.hyp_at_exp_paper_svm import HypExp
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
root_path = Path(__file__).parent.parent.parent
data_path = os.path.join(root_path, "processed_data\\dsn_keystroke")
bootstrap_data_path = os.path.join(data_path, 'bs_data')
cluster_data_path = os.path.join(data_path, 'cluster_data')
hyp_vol_data_path = os.path.join(data_path, 'hyper_volume_data')
pos_user_per_dim_ol_path = os.path.join(hyp_vol_data_path, f"gr1_hyper_vol_size_overlap_per_dim_overlap_df.csv")

classifier_path = os.path.join(data_path, 'trained_clf')
data_metric_save_path = os.path.join(data_path, 'exp_results')
feature_paths = {f'gr{gr}': os.path.join(data_path, f'df_group_{gr}_gr_scl.csv') for gr in range(1, 3)}
data = {f'{gr}': pd.read_csv(path) for gr, path in feature_paths.items()}

num_samples = 100
num_cls = 6
rand_state = 42
train_base_classifiers = False
boot_strap_st_at = False
classifier_list = ['svm', 'knn', 'rf']

experiments = {clf: HypExp(pop_df=data['gr1'], attack_df=data['gr2'], pop_classifier_path=classifier_path,
                 pos_user_per_dim_ol_path=pos_user_per_dim_ol_path, active_gr='gr1',
                 results_save_path=data_metric_save_path, train_classifiers=train_base_classifiers,
                 attack_samples=num_samples, boot_strap_st_at=boot_strap_st_at, bs_data_path=bootstrap_data_path,
                 bs_mul=1, clf_type=clf,
                 hv_cut_off=0.04, gr2_per_dim_ol_path=hyp_vol_data_path,
                 cluster_data_path=cluster_data_path,
                 hyp_vol_data_path=hyp_vol_data_path,
                 hyp_at_u_data=None, num_cls=num_cls,
                 rand_state=rand_state) for clf in classifier_list}
_ = [experiments[clf].run_exp() for clf in classifier_list]

# Gathering classifier base performance
per_mes = ['Precision', 'AuRoc', 'EER']
col_index = pd.MultiIndex.from_product([['svm', 'knn', 'rf'], ['Precision', 'AuRoc', 'EER']],
                                       names=['clf', 'metric'])
training_performance_df = pd.DataFrame(columns=col_index)
user_gr_1 = data['gr1'].user.unique()
active_gr = "gr1"

for usr in user_gr_1:
    for clf in experiments.keys():
        training_performance_df.loc[usr, (clf, "Precision")] = round(experiments[clf].test_precision[usr], 5)
        training_performance_df.loc[usr, (clf, "AuRoc")] = round(experiments[clf].roc_dict[usr].auc_roc, 5)
        training_performance_df.loc[usr, (clf, "EER")] = round(experiments[clf].roc_dict[usr].eer, 5)
training_performance_df.index.name = 'user'
training_performance_df.to_csv(os.path.join(data_metric_save_path, f"{active_gr}_training_performance.csv"),
                               mode='w+')
