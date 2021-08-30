from external_dataset_parsers import hmog_parser
from dataset.biometric_dataset import BioDataSet
from dataset.min_max_scaling_operation import MinMaxScaling
from dataset.standard_scaling_operation import StandardScaling
from dataset.dim_red_pca_operation import PcaDimRed
from classifiers.random_forest_classifier import RandomForestClassifier
from classifiers.knn_classifier import KnnClassifier
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
from dataset.outlier_removal import OutLierRemoval
import pandas as pd
from sklearn import inspection
from dataset.monaco_normalization import MonacoNormalize

root_path = Path(__file__).parent.parent.parent.parent
hmog_in = os.path.join(root_path, 'raw_data\\hmog_dataset\\public_dataset')
hmog_out = os.path.join(root_path, 'processed_data\\hmog_touch\\df_example.csv')
scaled_data_path = os.path.join(root_path,
                                'experiment_results\\touch_analytics_220962_outlier_removed\\df_scaled.csv')
pca_data_path = os.path.join(root_path, 'processed_data\\hmog_touch\\df_pca_example.csv')
std_scale_data_path = os.path.join(root_path, 'processed_data\\hmog_touch\\df_std_scl_example.csv')

data_metric_save_path = os.path.join(root_path,
                                     'experiment_results\\touch_analytics_342329_outlier_removed\\')
gr1_feature_path = os.path.join(root_path, 'processed_data\\hmog_touch\\overlap_analysis\\df_main_group_1.csv')
gr2_feature_path = os.path.join(root_path, 'processed_data\\hmog_touch\\overlap_analysis\\df_main_group_2.csv')

rand_state = 0
neg_sample_sources = 23
cv = 10
n_iter = 50
scoring_metric = 'precision'

df_group_1 = pd.read_csv(gr1_feature_path)
df_group_2 = pd.read_csv(gr2_feature_path)

''' 
External parser usage example
Read the dataset from raw, parse it, and write the feature vector a non scaled and scaled dataframe on the disk
'''
# df = hmog_parser.HMOGParser().raw_to_feature_vectors(raw_data_path=hmog_in)

"""
Removing features with low variance
"""
df = LowVarFeatRemoval().operate(data=df_group_1)

"""
Removing ourliers from dataset

"""

df = OutLierRemoval().operate(data=df, z_score_threshold=3)

"""
Applying min max operation and PCA on full data set

"""
min_max_tuple = (0, 1)
MinMaxScaling().operate(data=df, scaler_tuple=min_max_tuple, output_path=scaled_data_path)

df_pca = PcaDimRed()
df_pca.operate(df, output_path=pca_data_path)

"""
Biometrics module usage example

"""

# tb_data = BioDataSet(feature_data_path=hmog_out)  #example of reading dataframe stored on local disk

tb_data = BioDataSet(feature_data_frame=df, random_state=rand_state)  # example of using data frame

''' 
   get the user list from the dataset class object
'''

users = tb_data.user_list

''' 
   generate tagged data set for each user
'''
Data = dict()

for user in users:
    Data[user] = tb_data.get_data_set(user, neg_sample_sources=neg_sample_sources, neg_test_limit=True)

''' 
    Example usage of dataset operations
    min max scaling shown, similar format of command for using other datset operations the foramat of the api is
    DatasetOperation.operate(data, parameters)
'''

MinMaxData = dict()
for user in Data:
    MinMaxData[user] = MinMaxScaling().operate(Data[user], min_max_tuple)

''' 
   perform dataset dimension reduction
'''

red_data = dict()
for us in Data:
    red_data[us] = PcaDimRed().operate(Data[us], n_components=13)

''' 
    Classifier module usage example
   Initialize classifier object and split the data into training, testing and tuning data
'''
pos_user = 342329
mn = MonacoNormalize()
data_frame = mn.operate(pos_user=pos_user, data=Data[pos_user])
# data_frame.to_csv(os.path.join(data_metric_save_path, "user_pop_df.csv"), index=False, mode='w+')

nd = Data[pos_user].head(50)
nd = nd.append(Data[100669])
nd['labels'] = np.where(nd['user'] == pos_user, 1, nd['labels'])
nd['labels'] = np.where(nd['user'] != pos_user, 0, nd['labels'])

feature_names = list(data_frame.columns.drop(['user', 'labels']))

for feat in feature_names:
    lb = mn.bounds.loc[feat, 'lower_bound']
    ub = mn.bounds.loc[feat, 'upper_bound']
    nd[feat] = mn.monormalize(feature=nd[feat], lower_bound=lb, upper_bound=ub)

clf_rf = RandomForestClassifier(pos_user=pos_user, random_state=rand_state)
clf_rf.split_data(data_frame=data_frame, training_data_size=0.6, save_path=data_metric_save_path)

clf_svm = SvmClassifier(pos_user=pos_user, random_state=rand_state)
clf_svm.split_data(data_frame=data_frame, training_data_size=0.6, save_path=data_metric_save_path)

clf_knn = KnnClassifier(pos_user=pos_user, random_state=rand_state)
clf_knn.split_data(data_frame=data_frame, training_data_size=0.6, save_path=data_metric_save_path)

''' 
   Model parameter tuning using exhaustive search with cross validation
'''

# criterion = "entropy"
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', "log2"]
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

# clf_rf.tune_parametres(pram_grid=grid_rf, cv=cv, scoring_metric=scoring_metric)

''' 
   Model parameter tuning using random search with cross validation
'''

clf_rf.random_train_tune_parameters(pram_dist=grid_rf, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)
rf_important_features = pd.DataFrame(clf_rf.classifier.feature_importances_, index=feature_names)
rf_important_features = rf_important_features.sort_values(by=0)

rf_important_features_perm = inspection.permutation_importance(estimator=clf_rf.classifier,
                                                               X=clf_rf.test_data_frame.drop('labels', axis=1),
                                                               y=clf_rf.test_data_frame.labels.values,
                                                               n_jobs=-1, n_repeats=100, random_state=rand_state)
rf_important_features_perm_ = pd.DataFrame(rf_important_features_perm.importances_mean, index=feature_names)
rf_important_features_perm_ = rf_important_features_perm_.sort_values(by=0)
''' 
   SVM Model model tuning using random search cv
'''
c_range = np.logspace(start=1, stop=10000, num=50)
pram_grid = {'C': c_range}
scoring_metric = 'precision'

clf_svm.random_train_tune_parameters(pram_dist=pram_grid, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

svm_important_features_perm = inspection.permutation_importance(estimator=clf_svm.classifier,
                                                                X=clf_svm.test_data_frame.drop('labels', axis=1),
                                                                y=clf_svm.test_data_frame.labels.values,
                                                                n_jobs=-1, n_repeats=100, random_state=rand_state)
svm_important_features_perm_ = pd.DataFrame(svm_important_features_perm.importances_mean, index=feature_names)
svm_important_features_perm_ = svm_important_features_perm_.sort_values(by=0)

"""
KNN model tuning using random search cv
"""
leaf_size = list(range(1, 70))
n_neighbors = list(range(1, 50))
p = [1, 2]
grid_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

clf_knn.random_train_tune_parameters(pram_dist=grid_knn, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

knn_important_features_perm = inspection.permutation_importance(estimator=clf_knn.classifier,
                                                                X=clf_knn.test_data_frame.drop('labels', axis=1),
                                                                y=clf_knn.test_data_frame.labels.values,
                                                                n_jobs=-1, n_repeats=100, random_state=rand_state)
knn_important_features_perm_ = pd.DataFrame(knn_important_features_perm.importances_mean, index=feature_names)
knn_important_features_perm_ = knn_important_features_perm_.sort_values(by=0)

''' 
   Model Classification
'''

predictions_rf = clf_rf.classify()
predictions_svm = clf_svm.classify()
predictions_knn = clf_knn.classify()

"""
    Metrics module example
"""

"""
   Test set and labels extraction
"""
test_set_rf = clf_rf.test_data_frame.drop('labels', axis=1)
test_labels_rf = clf_rf.test_data_frame.labels.values

test_set_svm = clf_svm.test_data_frame.drop('labels', axis=1)
test_labels_svm = clf_svm.test_data_frame.labels.values

test_set_knn = clf_knn.test_data_frame.drop('labels', axis=1)
test_labels_knn = clf_knn.test_data_frame.labels.values

"""
    Confusion Matrix Curves
"""

rf_cm_path = os.path.join(data_metric_save_path, 'rf_cm.csv')
cm_rf = ConfusionMatrix()
matrix_rf = cm_rf.get_metric(true_labels=test_labels_rf, predicted_labels=predictions_rf, output_path=rf_cm_path)

svm_cm_path = os.path.join(data_metric_save_path, 'svm_cm.csv')
cm_svm = ConfusionMatrix()
matrix_svm = cm_svm.get_metric(true_labels=test_labels_svm, predicted_labels=predictions_svm, output_path=svm_cm_path)

knn_cm_path = os.path.join(data_metric_save_path, 'knn_cm.csv')
cm_knn = ConfusionMatrix()
matrix_knn = cm_knn.get_metric(true_labels=test_labels_knn, predicted_labels=predictions_knn, output_path=knn_cm_path)

"""
    ROC Curves
"""
ax_roc = plt.gca()

rf_roc = RocCurve()
roc_rf = rf_roc.get_metric(test_set_features=test_set_rf.values, test_set_labels=test_labels_rf
                               , classifier=clf_rf.classifier, ax=ax_roc)
print(f'EER for rf classifier = {rf_roc.eer}')
plt.savefig(os.path.join(data_metric_save_path, 'ROC_RF.png'), )

svm_roc = RocCurve()
roc_svm = svm_roc.get_metric(test_set_features=test_set_svm.values, test_set_labels=test_labels_svm
                                , classifier=clf_svm.classifier, ax=ax_roc)
print(f'EER for svm classifier = {svm_roc.eer}')
plt.savefig(os.path.join(data_metric_save_path, 'ROC_SVM.png'))

knn_roc = RocCurve()
roc_knn = knn_roc.get_metric(test_set_features=test_set_knn.values, test_set_labels=test_labels_knn
                                , classifier=clf_knn.classifier, ax=ax_roc)
print(f'EER for knn classifier = {knn_roc.eer}')
plt.savefig(os.path.join(data_metric_save_path, 'ROC_KNN.png'))

ax_roc.figure.set_figheight(12)
ax_roc.figure.set_figwidth(12)
ax_roc.figure.savefig((os.path.join(data_metric_save_path, 'ROC_SVM_RF_KNN.png')))

"""
    Gini Coef
"""

frr_rf = np.subtract(1, roc_rf.tpr)
gini_rf = GiniCoef(classifier_name='RF')
lc_rf = gini_rf.get_metric(frr_rf)
gini_rf_num = gini_rf.gini_num(frr_rf)
gini_rf_graph = gini_rf.gini_graph(frr_rf)
plt.savefig(os.path.join(data_metric_save_path, 'GINI_RF.png'))

frr_svm = 1 - roc_svm.tpr
gini_svm = GiniCoef(classifier_name='svm')
lc_svm = gini_svm.get_metric(frr_svm)
gini_svm_num = gini_svm.gini_num(frr_svm)
gini_svm_graph = gini_svm.gini_graph(frr_svm)
plt.savefig(os.path.join(data_metric_save_path, 'GINI_SVM.png'))

frr_knn = 1 - roc_knn.tpr
gini_knn = GiniCoef(classifier_name='knn')
lc_knn = gini_knn.get_metric(frr_knn)
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


nd_ul = nd.drop(['user', 'labels'], axis=1)

oth_prd = clf_rf.classifier.predict(nd_ul)
oth_prob = clf_rf.classifier.predict_proba(nd_ul)
cm_rf_oth = ConfusionMatrix()
matrix_rf_oth = cm_rf_oth.get_metric(true_labels=nd.labels.values, predicted_labels=oth_prd, output_path=None)

rf_roc_oth = RocCurve()
roc_oth = rf_roc_oth.get_metric(nd_ul.values,  test_set_labels=nd.labels.values, classifier=clf_rf.classifier)

fcs_rf_oth = FCS(classifier_name='RF')
fcs_rf_oth.get_metric(true_labels=nd.labels.values, predicted_probs=oth_prob, pred_labels=oth_prd)

plt.show()
