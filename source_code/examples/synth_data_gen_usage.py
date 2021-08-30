import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from analytics.fp_acceptance_percentage import FpAcceptance
from classifiers.svm_classifier import SvmClassifier
from dataset.biometric_dataset import BioDataSet
from dataset.min_max_scaling_operation import MinMaxScaling
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from metrics.confusion_matrix import ConfusionMatrix
from metrics.fcs import FCS
from metrics.roc_curve import RocCurve
from synth_data_gen.gauss_blob_generator import GaussBlob
from matplotlib.lines import Line2D
from analytics.dataoverlap_interval import OverLapInt
import os
from pathlib import Path
from sklearn.utils import shuffle
from dataset.outlier_removal import OutLierRemoval
import seaborn as sns

root_path = Path(__file__).parent.parent.parent.parent
hmog_in = os.path.join(root_path, 'raw_data\\hmog_dataset\\public_dataset')
hmog_out = os.path.join(root_path, 'processed_data\\hmog_touch\\df_example.csv')
scaled_data_path = os.path.join(root_path,
                                'experiment_results\\overlap_test\\df_ovr.csv')
data_metric_save_path = os.path.join(root_path,
                                     'experiment_results\\overlap_test\\')

rand_state = 42
features = 2
samples = 575
users = 10
train_data_size = 0.6
cv = 5

sns.set_theme(context="poster")
sns.set_style("whitegrid")

# Creating dictionaries for gathering figures
roc_fcs_fig_dict = dict()
roc_fcs_fig_dict_at = dict()
fcs_fig_dict_fc = dict()
fcs_fig_dict_at = dict()
db_fig_dict = dict()

# Creating dictionaries for gathering predictions
full_feat_test_set_pred_dict = dict()
full_feat_overlap_set_pred_dict = dict()
full_feat_test_set_pred_dict_prob = dict()
full_feat_overlap_set_pred_dict_prob = dict()
cm_val_full_feat_test_set_svm_dict = dict()
cm_val_full_feat_overlap_set_svm_dict = dict()
test_roc_dict = dict()
overlap_roc_dict = dict()

# Creating dataframes for results gathering
fp_accept_pers_svm_df = \
    pd.DataFrame(columns=["user", "cv_iter", "c_val", "rand_state", "fp_accept_pers", "fp_accept_pers_at"])

eer_df = pd.DataFrame(columns=["user", "cv_iter", "c_val", "rand_state", "eer_test", "eer_ol"])
auc_df = pd.DataFrame(columns=["user", "cv_iter", "c_val", "rand_state", "auc_test", "auc_ol"])


center_pos = [(0., 0.)]
centers_neg = [(2, 2), (5, 5), (1, 2), (2, 1), (3, 3), (1, 3), (0, 2), (2, 0), (3, 2)]

centers = center_pos + centers_neg
neg_labels = [f"neg_class_{i + 1}" for i in range(len(centers_neg))]
labels = ['pos_class'] + neg_labels
std_dev = [0.75, 0.75, 0.75, 0.75, 1.75, 0.75, 0.75, 0.75, 1.0, 0.75]
pos_user = labels[0]
df, cent = GaussBlob().generate_data(n_classes=users, n_features=features, n_samples=samples, centers=centers,
                                     random_state=rand_state, cluster_std=std_dev, class_names=labels)

df = MinMaxScaling().operate(df, (0, 1))

user_df_dict = dict()
user_df_dict = {user: df[df['user'] == user] for user in labels}

df_pos_class = df[df['user'] == 'pos_class']
df_neg_class = df[df['user'] != 'pos_class']

ds = BioDataSet(feature_data_frame=df, random_state=rand_state)
pos_user_data = ds.get_data_set(u="pos_class")
pos_user_data = shuffle(pos_user_data, random_state=rand_state)
df_pos_class_overlap, df_neg_class_u1_overlap = OverLapInt(df_pos_class, df_neg_class, std_dev=2).get_analytics()
new_df = df_pos_class_overlap.append(df_neg_class_u1_overlap)

xy = (new_df.Dim_00.median(), new_df.Dim_01.median())
xy_1 = [(new_df.Dim_00.median(), new_df.Dim_01.median())]
attacker_name = ["attack_points"]
attack_df, attack_cent = GaussBlob().generate_data(n_classes=1, n_features=features, n_samples=200, centers=xy_1,
                                                   random_state=rand_state + 1, cluster_std=[0.05],
                                                   class_names=attacker_name)
attack_df_l = attack_df.drop("user", axis=1)
attack_df_l = OutLierRemoval().operate(attack_df_l, 1.6)
attack_df_l['labels'] = 0
attack_df_l['user'] = "attack_points"
attack_df_l = attack_df_l[["user", "Dim_00", "Dim_01", "labels"]]

pos_val_for_overlap_data = pos_user_data[pos_user_data['user'] == pos_user].iloc[0:75, :]

full_feat_overlap_df = attack_df_l.append([pos_val_for_overlap_data])
full_feat_overlap_df = shuffle(full_feat_overlap_df, random_state=rand_state)
full_feat_overlap_df = full_feat_overlap_df.reset_index(drop=True)

# Deleting extracted samples from the data frame
pos_user_data = pd.concat([pos_user_data, pos_val_for_overlap_data]).drop_duplicates(keep=False)
pos_user_data = pos_user_data.drop_duplicates()

pos_user_data = pos_user_data.reset_index(drop=True)
divs = len(pos_user_data) // cv
row_drop = len(pos_user_data) - (divs * cv)
pos_user_data = pos_user_data.drop(pos_user_data.tail(row_drop).index)
ds_split = np.split(pos_user_data, cv)

at_df = df.append(attack_df_l.drop("labels", axis=1))

# Plotting population visualization
fig_pop_vis_1 = plt.figure(figsize=(19.2, 10.8))
ax_1 = fig_pop_vis_1.add_subplot(1, 1, 1)

sns.scatterplot(data=df.sample(frac=0.5, replace=False, random_state=rand_state),
                x="Dim_00", y="Dim_01", hue="user", style="user", ax=ax_1)

rad_1 = abs(new_df.Dim_00.mean() - new_df.Dim_00.max())
rad_2 = abs(new_df.Dim_00.mean() - new_df.Dim_00.min())
rad_3 = abs(new_df.Dim_01.mean() - new_df.Dim_01.max())
rad_4 = abs(new_df.Dim_01.mean() - new_df.Dim_01.min())
rad_list = [rad_1, rad_2, rad_3, rad_4]
# rad = max(rad_1, rad_2, rad_3, rad_4)
rad = sum(rad_list) / len(rad_list)
cir = plt.Circle(xy=xy, radius=rad, fill=False, edgecolor="red", lw=3)

ax_1.add_patch(cir)
ax_1.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0.)

fig_pop_vis_2 = plt.figure(figsize=(19.2, 10.8))
ax_2 = fig_pop_vis_2.add_subplot(1, 1, 1)
fig_pop_vis_1.tight_layout()

sns.scatterplot(data=at_df, x="Dim_00", y="Dim_01", hue="user", style="user", ax=ax_2)

rad = abs(new_df.Dim_00.median() - new_df.Dim_00.max())
cir_1 = plt.Circle(xy=xy, radius=rad, fill=False, edgecolor="red", lw=3)

ax_2.add_patch(cir_1)
ax_2.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0.)
fig_pop_vis_2.tight_layout()

h = .02  # step size in the mesh
c_val = 1.0  # SVM regularization parameter

for cv_iter in range(cv):
    print(f"Cross validation itretion number = {cv_iter + 1}")
    # Full feature classifier
    full_feat_clf_svm = SvmClassifier(pos_user=pos_user, random_state=rand_state, probability=True)
    full_feat_clf_svm.split_data(data_frame=pos_user_data, training_data_size=train_data_size, save_path=None)

    full_feat_test_set_svm_df = ds_split[cv_iter].copy()
    full_feat_test_set_svm = full_feat_test_set_svm_df.drop(['user', 'labels'], axis=1).values
    full_feat_test_labels_svm = full_feat_test_set_svm_df.labels.values

    ds_split_c = ds_split.copy()
    del ds_split_c[cv_iter]
    full_feat_train_set_svm = pd.DataFrame(columns=pos_user_data.columns)
    train_df = full_feat_train_set_svm.append(ds_split_c)
    full_feat_train_set_svm = train_df.drop(['user', 'labels'], axis=1)
    full_feat_train_labels_svm = train_df.labels.values

    full_feat_clf_svm.classifier.C = c_val
    full_feat_clf_svm.classifier.fit(X=full_feat_train_set_svm,
                                     y=full_feat_train_labels_svm)

    X = full_feat_train_set_svm.values
    y = full_feat_train_labels_svm

    full_feat_overlap_data_gr_2_p = full_feat_overlap_df.drop('user', axis=1)
    full_feat_overlap_set_values = full_feat_overlap_data_gr_2_p.drop('labels', axis=1).values
    full_feat_overlap_labels = full_feat_overlap_data_gr_2_p.labels.values

    print(f"Performing classification for user {pos_user}, c value {c_val}, random state {rand_state}")
    # Full features
    full_feat_test_set_pred = full_feat_clf_svm.classifier.predict(full_feat_test_set_svm)
    full_feat_overlap_set_pred = full_feat_clf_svm.classifier.predict(full_feat_overlap_set_values)
    # Full features probabilities
    full_feat_test_set_pred_prob = full_feat_clf_svm.classifier.predict_proba(full_feat_test_set_svm)
    full_feat_overlap_set_pred_prob = full_feat_clf_svm.classifier.predict_proba(full_feat_overlap_set_values)


    # Gathering predictions
    full_feat_test_set_pred_dict[cv_iter] = full_feat_test_set_pred
    full_feat_overlap_set_pred_dict[cv_iter] = full_feat_overlap_set_pred

    full_feat_test_set_pred_dict_prob[cv_iter] = full_feat_test_set_pred_prob
    full_feat_overlap_set_pred_dict_prob[cv_iter] = full_feat_overlap_set_pred_prob
    """
    False positive acceptance percentage example
    """
    # Full features test set
    fp_accept_full_feat_test_set_svm = \
        FpAcceptance(df=full_feat_test_set_svm_df.drop(['user'], axis=1), prediction=full_feat_test_set_pred)
    fp_accept_pers_full_feat_test_set_svm = round(fp_accept_full_feat_test_set_svm.get_analytics(), 2)
    print('%.2f' % fp_accept_pers_full_feat_test_set_svm,
          f"percent of false users accepted for full feature on test set for SVM classifier trained for "
          f"user = {pos_user}, c value = {c_val}, random state = {rand_state} ")
    fp_accept_pers_svm_df.loc[cv_iter, "cv_iter"] = cv_iter
    fp_accept_pers_svm_df.loc[cv_iter, "user"] = pos_user

    fp_accept_pers_svm_df.loc[cv_iter, "c_val"] = c_val
    fp_accept_pers_svm_df.loc[cv_iter, "rand_state"] = rand_state

    fp_accept_pers_svm_df.loc[cv_iter, "fp_accept_pers"] = fp_accept_pers_full_feat_test_set_svm

    # Full features overlap set
    fp_accept_full_feat_overlap_set_svm = \
        FpAcceptance(df=full_feat_overlap_df.drop(['user'], axis=1), prediction=full_feat_overlap_set_pred)
    fp_accept_pers_full_feat_overlap_set_svm = round(fp_accept_full_feat_overlap_set_svm.get_analytics(), 2)
    print('%.2f' % fp_accept_pers_full_feat_overlap_set_svm,
          f"percent of false users accepted for full feature on overlap set for SVM classifier trained for "
          f"user = {pos_user}, c value = {c_val}, random state = {rand_state} ")

    fp_accept_pers_svm_df.loc[cv_iter, "fp_accept_pers_at"] = fp_accept_pers_full_feat_overlap_set_svm

    """
    Confusion Matrix 
    """
    # Confusion matrix for full features for test set
    cm_full_feat_test_set_svm = ConfusionMatrix()
    cm_val_full_feat_test_set_svm = cm_full_feat_test_set_svm.get_metric(true_labels=full_feat_test_labels_svm,
                                                                         predicted_labels=full_feat_test_set_pred,
                                                                         output_path=None)
    cm_val_full_feat_test_set_svm_dict[cv_iter] = cm_val_full_feat_test_set_svm

    cm_full_feat_overlap_set_svm = ConfusionMatrix()
    cm_val_full_feat_overlap_set_svm = cm_full_feat_overlap_set_svm.get_metric(true_labels=full_feat_test_labels_svm,
                                                                               predicted_labels=full_feat_test_set_pred,
                                                                               output_path=None)
    cm_val_full_feat_overlap_set_svm_dict[cv_iter] = cm_val_full_feat_overlap_set_svm

    fig_roc_fcs = plt.figure(figsize=(19.2, 10.8))
    fig_roc_fcs_at = plt.figure(figsize=(19.2, 10.8))

    ax_roc = fig_roc_fcs.add_subplot(1, 2, 1)
    ax_roc_at = fig_roc_fcs_at.add_subplot(1, 2, 1)

    """
        ROC Curves
    """

    t_roc = RocCurve()
    test_roc = t_roc.get_metric(test_set_features=full_feat_test_set_svm,
                                test_set_labels=full_feat_test_labels_svm
                                , classifier=full_feat_clf_svm.classifier, ax=ax_roc)
    plt.close()
    plt.close()

    test_roc_dict[cv_iter] = t_roc

    ol_roc = RocCurve()
    overlap_roc = ol_roc.get_metric(test_set_features=full_feat_overlap_set_values,
                                    test_set_labels=full_feat_overlap_labels
                                    , classifier=full_feat_clf_svm.classifier, ax=ax_roc_at)
    plt.close()
    plt.close()
    overlap_roc_dict[cv_iter] = ol_roc

    ax_roc.set_title(f"EER = {round(t_roc.eer, 3)}, CV iteration # {cv_iter}")
    ax_roc_at.set_title(f"EER = {round(ol_roc.eer, 3)}, CV iteration # {cv_iter}")
    test_auc = round(test_roc.roc_auc, 3)
    overlap_auc = round(overlap_roc.roc_auc, 3)

    eer_df.loc[cv_iter, "cv_iter"] = cv_iter
    eer_df.loc[cv_iter, "user"] = pos_user
    eer_df.loc[cv_iter, "rand_state"] = rand_state
    eer_df.loc[cv_iter, "eer_test"] = t_roc.eer
    eer_df.loc[cv_iter, "eer_ol"] = ol_roc.eer
    eer_df.loc[cv_iter, "c_val"] = c_val

    auc_df.loc[cv_iter, "cv_iter"] = cv_iter
    auc_df.loc[cv_iter, "user"] = pos_user
    auc_df.loc[cv_iter, "rand_state"] = rand_state
    auc_df.loc[cv_iter, "auc_test"] = test_roc.roc_auc
    auc_df.loc[cv_iter, "auc_ol"] = overlap_roc.roc_auc
    auc_df.loc[cv_iter, "c_val"] = c_val
    """
        FCS
    """
    ax_fcs = fig_roc_fcs.add_subplot(1, 2, 2)
    ax_fcs_at = fig_roc_fcs_at.add_subplot(1, 2, 2)
    bins = 100
    fcs_svm_test_set = FCS(classifier_name='svc')
    fcs_plot = fcs_svm_test_set.get_metric(true_labels=full_feat_test_labels_svm,
                                           predicted_probs=full_feat_test_set_pred_prob,
                                           pred_labels=full_feat_clf_svm.predictions, bins=bins, ax=ax_fcs)

    fcs_svm_ol_set = FCS(classifier_name='svc')
    fcs_svm_ol_set.get_metric(true_labels=full_feat_overlap_labels,
                              predicted_probs=full_feat_overlap_set_pred_prob,
                              pred_labels=full_feat_clf_svm.predictions_ext_df, bins=bins, ax=ax_fcs_at)

    ax_fcs.set_title("")
    ax_fcs.legend(bbox_to_anchor=(0, 1.16), loc='upper left', borderaxespad=0.)

    ax_fcs_at.set_title("")
    ax_fcs_at.legend(bbox_to_anchor=(0, 1.16), loc='upper left', borderaxespad=0.)

    ax_roc.legend(bbox_to_anchor=(0, 1.16), loc='upper left', borderaxespad=0.)
    ax_roc_at.legend(bbox_to_anchor=(0, 1.16), loc='upper left', borderaxespad=0.)
    fig_roc_fcs_at.suptitle(f"Attack Set Metrics CV iteration # {cv_iter}")
    fig_roc_fcs.suptitle(f"Test Set Metrics CV iteration # {cv_iter}")
    fig_roc_fcs.tight_layout()
    fig_roc_fcs_at.tight_layout()

    roc_fcs_fig_dict[cv_iter] = fig_roc_fcs
    roc_fcs_fig_dict_at[cv_iter] = fig_roc_fcs_at

    db_fig = plt.figure(figsize=(19.2, 10.8))
    ax_db = db_fig.add_subplot(1, 1, 1)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # title for the plots
    titles = f'SVC with RBF kernel Decision Boundary cv iteration # {cv_iter}'

    Z = full_feat_clf_svm.classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    cmap = matplotlib.colors.ListedColormap(["blue", "red"])
    ax_db.contourf(xx, yy, Z, alpha=0.4)
    # Plot also the training points
    y_c = pd.DataFrame(y, columns=["dec"])
    y_c.loc[y_c.dec == 1, "sample_type"] = "pos_samples"
    y_c.loc[y_c.dec == 0, "sample_type"] = "neg_samples"
    sns.scatterplot(data=full_feat_train_set_svm, x="Dim_00", y="Dim_01", style=y_c.sample_type.values, hue=y_c.sample_type.values,
                    palette=["red", "blue"]
                    , alpha=0.7)

    # ax_db.set_xlim(xx.min(), xx.max())
    # ax_db.set_ylim(yy.min(), yy.max())
    # ax_db.set_xticks(())
    # ax_db.set_yticks(())
    ax_db.set_title(titles)
    ax_db.set_xlabel("Dim_00")
    ax_db.set_ylabel("Dim_01")
    db_fig.tight_layout()

    db_fig_dict[cv_iter] = db_fig

# df.to_csv(os.path.join(data_metric_save_path
#                        , f'pop_df.csv'), index=False, mode='w+')
# plt.show()