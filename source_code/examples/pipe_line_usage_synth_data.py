import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import cm
import pandas as pd
from dataset.min_max_scaling_operation import MinMaxScaling
from dataset.biometric_dataset import BioDataSet
from classifiers.random_forest_classifier import RandomForestClassifier
from classifiers.svm_classifier import SvmClassifier
from classifiers.knn_classifier import KnnClassifier
from metrics.confusion_matrix import ConfusionMatrix
from metrics.roc_curve import RocCurve
from metrics.gini_coefficient import GiniCoef
from metrics.fcs import FCS
from analytics.dataoverlap_interval import OverLapInt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from synth_data_gen.gauss_blob_generator import GaussBlob
from pathlib import Path
import os
import math

"""
Data and metrics saving path
"""
root_path = Path(__file__).parent.parent.parent.parent
data_metric_save_path = os.path.join(root_path, 'experiment_results\\neg_class_contained_center_edge2_03D_02U\\')
"""
Generating Isometric Gaussian data, in this example data is generated with three features
"""
rand_state_1 = 42
features = 3
users = 1
neg_users = 1
samples_pos = 1000
outlier_filter_x_std = 2
samples_neg = math.ceil(samples_pos / neg_users)

center_pos_class = [np.zeros(features)]
center_neg_class_1 = np.ones(features) * 1.5
# center_neg_class_2 = np.ones(features) * -1
# center_neg_class_3 = np.ones(features) * 0.9
# center_neg_class_4 = np.ones(features) * -12
# center_neg_class_5 = np.ones(features) * 0.7
# center_neg_class_6 = np.ones(features) * -2
# center_neg_class_7 = np.ones(features) * -0.8
# center_neg_class_8 = np.ones(features) * 0.9
# center_neg_class_9 = np.ones(features) * 1

centers_neg_class = []
centers_neg_class.extend((value for name, value in globals().items() if name.startswith('center_neg_class_')))

labels_pos = ['pos_class']
labels_neg = ['neg_class_u%02d' % i for i in range(neg_users)]

std_dev_pos = [1]
# std_dev_neg = [1.75, 1.5, 1, 1, 1, 1, 1, 1, 2]
std_dev_neg = [1]

std_dev_neg_df = pd.DataFrame(labels_neg)
std_dev_neg_df.columns = ['users']
std_dev_neg_df['std_dev'] = std_dev_neg
std_dev_neg_df_path = os.path.join(data_metric_save_path, 'neg_std_dev.csv')
std_dev_neg_df.to_csv(std_dev_neg_df_path, index=False, mode='w+')

std_dev_pos_df = pd.DataFrame(labels_pos)
std_dev_pos_df.columns = ['users']
std_dev_pos_df['std_dev'] = std_dev_pos
std_dev_pos_df_path = os.path.join(data_metric_save_path, 'pos_std_dev.csv')
std_dev_pos_df.to_csv(std_dev_pos_df_path, index=False, mode='w+')

df_pos, cen_pos = GaussBlob().generate_data(n_classes=users, n_features=features, n_samples=samples_pos,
                                            centers=center_pos_class, random_state=rand_state_1,
                                            cluster_std=std_dev_pos, class_names=labels_pos)

df_neg, cen_neg = GaussBlob().generate_data(n_classes=neg_users, n_features=features, n_samples=samples_neg,
                                            centers=centers_neg_class, random_state=rand_state_1,
                                            cluster_std=std_dev_neg, class_names=labels_neg)

"""
Outlier removal and overlap points extraction
"""

df_pos_class_overlap, df_neg_class_overlap = \
    OverLapInt(df1=df_pos, df2=df_neg, std_dev=outlier_filter_x_std).get_analytics()
overlap_df = df_pos_class_overlap.append(df_neg_class_overlap)
overlap_df_path = os.path.join(data_metric_save_path, 'overlap_df.csv')
overlap_df.to_csv(overlap_df_path, index=False, mode='w+')


df = pd.DataFrame()
df = df.append(df_pos)
df = df.append(df_neg)
df = df.reset_index(drop=True)


"""
Data scaling  in range of 0 to 1 and saving the data on disk for hyper-volume analysis
"""
blob_path = os.path.join(data_metric_save_path, 'blob.csv')
df = MinMaxScaling().operate(df, (0, 1), output_path=blob_path)
pos_cen_path = os.path.join(data_metric_save_path, 'pos_centers.csv')
neg_cen_path = os.path.join(data_metric_save_path, 'neg_centers.csv')
cen_pos.to_csv(pos_cen_path, index=False, mode='w+')
cen_neg.to_csv(neg_cen_path, index=False, mode='w+')

""""
Generating tagged data
"""
syn_data = BioDataSet(feature_data_frame=df)  # example of using data frame
users = syn_data.user_list
tagged_data = syn_data.get_data_set(u=users[0])

"""
Creating classifiers
"""
clf_rf = RandomForestClassifier(pos_user=users[0], random_state=rand_state_1)
clf_svm = SvmClassifier(pos_user=users[0], random_state=rand_state_1, cache_size=3000)
clf_knn = KnnClassifier(pos_user=users[0], random_state=rand_state_1)

"""
Splitting data into test and training sets
"""

clf_rf.split_data(data_frame=tagged_data, training_data_size=0.6, save_path=data_metric_save_path)
clf_svm.split_data(data_frame=tagged_data, training_data_size=0.6, save_path=data_metric_save_path)
clf_knn.split_data(data_frame=tagged_data, training_data_size=0.6, save_path=data_metric_save_path)

"""
RF Model tuning using random search CV
"""
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
cv = 10
scoring_metric = 'precision'
n_iter = 20

clf_rf.random_train_tune_parameters(pram_dist=grid_rf, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

"""
SVM model tuning using random search cv
"""
c_range = range(1, 21)
grid_svm = {'C': c_range}


clf_svm.random_train_tune_parameters(pram_dist=grid_svm, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

"""
KNN model tuning using random search cvz
"""
leaf_size = list(range(1, 100))
n_neighbors = list(range(1, 60))
p = [1, 2]

grid_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

clf_knn.random_train_tune_parameters(pram_dist=grid_knn, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

"""
Model Classification 
"""
predictions_rf = clf_rf.classify()
predictions_svm = clf_svm.classify()
predictions_knn = clf_knn.classify()
"""
Model evaluation Confusion Matrix
"""
rf_cm_path = os.path.join(data_metric_save_path, 'rf_cm.csv')
test_set_rf = clf_rf.test_data_frame.drop('labels', axis=1)
test_labels_rf = clf_rf.test_data_frame.labels.values
cm_rf = ConfusionMatrix()
matrix_rf = cm_rf.get_metric(true_labels=test_labels_rf, predicted_labels=predictions_rf, output_path=rf_cm_path)

svm_cm_path = os.path.join(data_metric_save_path, 'svm_cm.csv')
test_set_svm = clf_svm.test_data_frame.drop('labels', axis=1)
test_labels_svm = clf_svm.test_data_frame.labels.values
cm_svm = ConfusionMatrix()
matrix_svm = cm_svm.get_metric(true_labels=test_labels_svm, predicted_labels=predictions_svm, output_path=svm_cm_path)

knn_cm_path = os.path.join(data_metric_save_path, 'knn_cm.csv')
test_set_knn = clf_knn.test_data_frame.drop('labels', axis=1)
test_labels_knn = clf_knn.test_data_frame.labels.values
cm_knn = ConfusionMatrix()
matrix_knn = cm_knn.get_metric(true_labels=test_labels_knn, predicted_labels=predictions_knn, output_path=knn_cm_path)

""""
Model evaluation ROC curve
"""
ax_roc = plt.gca()
roc_rf = RocCurve().get_metric(test_set_features=test_set_rf.values, test_set_labels=test_labels_rf
                               , classifier=clf_rf.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, 'ROC_RF.png'), )


roc_svm = RocCurve().get_metric(test_set_features=test_set_svm.values, test_set_labels=test_labels_svm
                                , classifier=clf_svm.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, 'ROC_SVM.png'))

roc_knn = RocCurve().get_metric(test_set_features=test_set_knn.values, test_set_labels=test_labels_knn
                                , classifier=clf_knn.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, 'ROC_KNN.png'))

ax_roc.figure.set_figheight(12)
ax_roc.figure.set_figwidth(12)
ax_roc.figure.savefig((os.path.join(data_metric_save_path, 'ROC_SVM_RF_KNN.png')))


"""
    Model evaluation Gini Coef
"""

frr_rf = np.subtract(1, roc_rf.tpr)  # FRR = 1- TPR
gini_rf = GiniCoef(classifier_name='RF')
lc = gini_rf.get_metric(frr_rf)
gini_rf_num = gini_rf.gini_num(frr_rf)
gini_rf_graph = gini_rf.gini_graph(frr_rf)
plt.savefig(os.path.join(data_metric_save_path, 'GINI_RF.png'))


frr_svm = 1 - roc_svm.tpr
gini_svm = GiniCoef(classifier_name='svm')
lc = gini_svm.get_metric(frr_svm)
gini_svm_num = gini_svm.gini_num(frr_svm)
gini_svm_graph = gini_svm.gini_graph(frr_svm)
plt.savefig(os.path.join(data_metric_save_path, 'GINI_SVM.png'))


frr_knn = 1 - roc_knn.tpr
gini_knn = GiniCoef(classifier_name='knn')
lc = gini_knn.get_metric(frr_knn)
gini_knn_num = gini_knn.gini_num(frr_knn)
gini_knn_graph = gini_knn.gini_graph(frr_knn)
plt.savefig(os.path.join(data_metric_save_path, 'GINI_KNN.png'))


"""
    FCS
"""
fcs_rf = FCS(classifier_name='RF')
fcs_rf.get_metric(true_labels=test_labels_rf, predicted_probs=clf_rf.predictions_prob, pred_labels=clf_rf.predictions)
plt.savefig(os.path.join(data_metric_save_path, 'FCS_RF.png'))


fcs_svm = FCS(classifier_name='svc')
fcs_svm.get_metric(true_labels=test_labels_svm, predicted_probs=clf_svm.predictions_prob,
                   pred_labels=clf_svm.predictions)
plt.savefig(os.path.join(data_metric_save_path, 'FCS_SVM.png'))


fcs_knn = FCS(classifier_name='knn')
fcs_knn.get_metric(true_labels=test_labels_knn, predicted_probs=clf_knn.predictions_prob,
                   pred_labels=clf_knn.predictions)
plt.savefig(os.path.join(data_metric_save_path, 'FCS_KNN.png'))


"""
Visualizing Generated data for Data with three features
"""
if features <= 3:
    fig = plt.figure(figsize=[13, 13])
    ax = fig.add_subplot(111, projection='3d')

    col = pd.DataFrame(df['user'])
    col.loc[col['user'] == 'pos_class'] = 1
    col.loc[col['user'] == 'neg_class_u00'] = 0
    col.loc[col['user'] == 'neg_class_u01'] = 2
    col.loc[col['user'] == 'neg_class_u02'] = 3
    col.loc[col['user'] == 'neg_class_u03'] = 4
    col.loc[col['user'] == 'neg_class_u04'] = 5
    col.loc[col['user'] == 'neg_class_u05'] = 6
    col.loc[col['user'] == 'neg_class_u06'] = 7
    col.loc[col['user'] == 'neg_class_u07'] = 8
    col.loc[col['user'] == 'neg_class_u08'] = 9
    plt.title("Synthetic Data Visualization")

    colours = ListedColormap(colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'aqua', 'purple', 'olive'])

    ax.scatter(df['Dim_00'].to_numpy(), df['Dim_01'].to_numpy(), df['Dim_02'].to_numpy(),
               marker='o', c=col['user'].to_numpy(), s=25, edgecolor='k', cmap=colours)
    legend_elements = [Line2D([0], [0], marker='o', color='w', lw=4, label='Negative Class U1',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Positive Class',
                              markerfacecolor='g', markersize=10),

                       ]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', lw=4, label='Negative Class U1',
    #                           markerfacecolor='r', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Positive Class',
    #                           markerfacecolor='g', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U2',
    #                           markerfacecolor='b', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U3',
    #                           markerfacecolor='c', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U4',
    #                           markerfacecolor='m', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U5',
    #                           markerfacecolor='y', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U6',
    #                           markerfacecolor='k', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U7',
    #                           markerfacecolor='aqua', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U8',
    #                           markerfacecolor='purple', markersize=10),
    #                    Line2D([0], [0], marker='o', color='w', label='Negative Class U9',
    #                           markerfacecolor='olive', markersize=10)
    #
    #                    ]

    legend1 = ax.legend(handles=legend_elements, title="Classes")
    ax.add_artist(legend1)
    plt.savefig(os.path.join(data_metric_save_path, 'Data_Visul.png'))
plt.show()