import os
from pathlib import Path

import numpy as np
import pandas as pd
from metrics.fcs import FCS
from sklearn import inspection
import matplotlib.pyplot as plt
from metrics.roc_curve import RocCurve
from metrics.gini_coefficient import GiniCoef
from dataset.biometric_dataset import BioDataSet
from external_dataset_parsers import hmog_parser
from dataset.dim_red_pca_operation import PcaDimRed
from classifiers.knn_classifier import KnnClassifier
from metrics.confusion_matrix import ConfusionMatrix
from dataset.monaco_normalization import MonacoNormalize
from dataset.min_max_scaling_operation import MinMaxScaling
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


"""
Parse and store raw touch data from HMOG
"""
def parse_hmog_touch_data(root_path):
    hmog_in = os.path.join(root_path, 'raw_data\\hmog_dataset\\public_dataset')
    hmog_out = os.path.join(root_path, 'processed_data\\hmog_touch\\df_10.csv')
    
    ''' 
    Read the dataset from raw, parse it, and write the feature vector as a dataframe
    '''
    df = hmog_parser.HMOGParser().raw_to_feature_vectors(hmog_in, hmog_out)
    return df

"""
Preprocess and return HMOG touch data
"""
def preprocess_hmog_touch_data(root_path, df=None):
    raw_bio_data = os.path.join(root_path, 'processed_data\\hmog_touch\\df_10.csv')

    tb_data = BioDataSet(feature_data_path=raw_bio_data)
    # tb_data = BioDataSet(feature_data_frame=df)
    ''' 
   get the user list from the dataset class object
    '''

    users = tb_data.user_list


    ''' 
    generate tagged data set for each user
    '''
    Data = dict()
    for user in users:
        Data[user] = tb_data.get_data_set(user, neg_sample_sources=6, neg_test_limit=True)
    ''' 
   perform min max scaling
    '''
    min_max_tuple = (0, 1)
    MinMaxData =dict()
    for user in Data:
        MinMaxData[user] = MinMaxScaling().operate(Data[user], min_max_tuple)
    ''' 
   perform dataset dimension reduction
    '''
    red_data = dict()
    for us in MinMaxData:
        red_data[us] = PcaDimRed().operate(MinMaxData[us], n_components=13)
    return Data


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    data_metric_save_path = os.path.join(root_path, 'experiment_results\\touch_analytics_342329_outlier_removed\\')
    if not os.path.exists(data_metric_save_path):
    # If not, create the directory
        os.makedirs(data_metric_save_path)
    # hmog_out = os.path.join(root_path, 'processed_data\\hmog_touch\\df_10.csv')
    # df = hmog_parser.HMOGParser().get_feature_vectors(hmog_out, limit=2)
    Data = preprocess_hmog_touch_data(root_path=root_path)
    pos_user = 240168
    mn = MonacoNormalize()
    df = pd.DataFrame(Data[pos_user])
    # # print(Data)
    data_frame = mn.operate(pos_user, df)
    data_frame.to_csv(os.path.join(data_metric_save_path, "user_pop_df.csv"), index=False, mode='w+')
    # print(data_frame.to_string())
    nd = Data[pos_user].head(50)
    nd = nd.append(Data[100669])
    nd['labels'] = np.where(nd['user'] == pos_user, 1, nd['labels'])
    nd['labels'] = np.where(nd['user'] != pos_user, 0, nd['labels'])
    
    feature_names = list(data_frame.columns.drop(['user', 'labels']))

    for feat in feature_names:
        lb = mn.bounds.loc[feat, 'lower_bound']
        ub = mn.bounds.loc[feat, 'upper_bound']
        nd[feat] = mn.monormalize(feature=nd[feat], lower_bound=lb, upper_bound=ub)

    clf_knn = KnnClassifier(pos_user=pos_user, random_state=0)
    clf_knn.split_data(data_frame=data_frame, training_data_size=0.6, save_path=data_metric_save_path)

    leaf_size = list(range(1, 70))
    n_neighbors = list(range(1, 50))
    p = [1, 2]
    grid_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    clf_knn.random_train_tune_parameters(pram_dist=grid_knn, cv=10, scoring_metric='precision', n_itr=50)

    knn_important_features_perm = inspection.permutation_importance(estimator=clf_knn.classifier,
                                                                    X=clf_knn.test_data_frame.drop('labels', axis=1),
                                                                    y=clf_knn.test_data_frame.labels.values,
                                                                    n_jobs=-1, n_repeats=100, random_state=0)
    knn_important_features_perm_ = pd.DataFrame(knn_important_features_perm.importances_mean, index=feature_names)
    knn_important_features_perm_ = knn_important_features_perm_.sort_values(by=0)

    predictions_knn = clf_knn.classify()
    print(predictions_knn)

    test_set_knn = clf_knn.test_data_frame.drop('labels', axis=1)
    test_labels_knn = clf_knn.test_data_frame.labels.values

    knn_cm_path = os.path.join(data_metric_save_path, 'knn_cm.csv')
    cm_knn = ConfusionMatrix()
    matrix_knn = cm_knn.get_metric(true_labels=test_labels_knn, predicted_labels=predictions_knn, output_path=knn_cm_path)

    ax_roc = plt.gca()

    knn_roc = RocCurve()
    roc_knn = knn_roc.get_metric(test_set_features=test_set_knn.values, test_set_labels=test_labels_knn
                                    , classifier=clf_knn.classifier, ax=ax_roc)
    print(f'EER for knn classifier = {knn_roc.eer}')
    plt.savefig(os.path.join(data_metric_save_path, 'ROC_KNN.png'))

    ax_roc.figure.set_figheight(12)
    ax_roc.figure.set_figwidth(12)
    ax_roc.figure.savefig((os.path.join(data_metric_save_path, 'ROC_Comparison_KNN.png')))

    frr_knn = 1 - roc_knn.tpr
    gini_knn = GiniCoef(classifier_name='knn')
    lc_knn = gini_knn.get_metric(frr_knn)
    gini_knn_num = gini_knn.gini_num(frr_knn)
    gini_knn_graph = gini_knn.gini_graph(frr_knn)
    plt.savefig(os.path.join(data_metric_save_path, 'GINI_KNN.png'))

    fcs_knn = FCS(classifier_name='knn')
    fcs_knn.get_metric(true_labels=test_labels_knn, predicted_probs=clf_knn.predictions_prob,
                    pred_labels=clf_knn.predictions)
    plt.savefig(os.path.join(data_metric_save_path, 'FCS_KNN.png'))


 