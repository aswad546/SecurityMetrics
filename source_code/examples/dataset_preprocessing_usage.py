from dataset.dim_red_pca_operation import PcaDimRed
from dataset.min_max_scaling_operation import MinMaxScaling
from dataset.standard_scaling_operation import StandardScaling
from dataset.biometric_dataset import BioDataSet
from external_dataset_parsers import hmog_parser

"""
Example of feature generation from feature data for touch biometric data and preforming feature scaling and feature 
reduction
"""

''' 
   Read the feature data from disk and generate features
'''

raw_bio_data = 'C:\\Users\\esi\\Documents\\WD\\data-sufficiency_uniqueness\\SecurityMetrics\\processed_data\\hmog_touch\\df_10.csv'

tb_data = BioDataSet(feature_data_path=raw_bio_data)

''' 
   Read the feature dataframe and generate features
'''

hmog_in = 'C:\\wd\\research\\data-sufficiency_uniqueness\\SecurityMetrics\\raw_data\\hmog_dataset\\public_dataset'
df = hmog_parser.HMOGParser().raw_to_feature_vectors(hmog_in)
tb_data = BioDataSet(feature_data_frame=df)

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
   perform min max scaling and standard scaling
'''
min_max_tuple = (0, 2)
MinMaxData =dict()
StandardScaleData = dict()
for user in Data:
    MinMaxData[user] = MinMaxScaling().operate(Data[user], min_max_tuple)
    StandardScaleData[user] = StandardScaling().operate(Data[users])

''' 
   perform dataset dimension reduction
'''

red_data = dict()
for us in Data:
    red_data[us] = PcaDimRed().operate(Data[us], n_components=13)
