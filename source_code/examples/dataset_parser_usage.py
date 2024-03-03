# -*- coding: utf-8 -*-
"""
Example Usage of dataset Parsers

"""


def hmog_touch_usage_example():
    from external_dataset_parsers import hmog_parser

    hmog_in = 'D:\\Waterloo Work\\SecurityMetrics\\raw_data\\hmog_dataset\\public_dataset'
    hmog_out = 'D:\\Waterloo Work\\SecurityMetrics\\processed_data\\hmog_touch\\df_10'

    ''' 
    Read the dataset from raw, parse it, and write the feature vector as a dataframe
    '''
    df = hmog_parser.HMOGParser().raw_to_feature_vectors(hmog_in, hmog_out)
    ''' 
    Read the parsed dataset from feature vector format as dataframe
    '''
    df = hmog_parser.HMOGParser().get_feature_vectors(hmog_out, limit=2)


def dsn_keystroke_usage_example():
    from external_dataset_parsers import dsn_keystroke_parser

    raw_data_path = 'D:\\Waterloo Work\\SecurityMetrics\\raw_data\\dsn_keystroke\\DSL-StrongPasswordData.csv'
    output_path = 'D:\\Waterloo Work\\SecurityMetrics\\processed_data\\dsn_keystroke\\df'

    ''' 
    Read the dataset from raw, parse it, and write the feature vector as a dataframe
    '''
    df = dsn_keystroke_parser.DSNParser().raw_to_feature_vectors(raw_data_path, output_path)
    ''' 
    Read the parsed dataset from feature vector format as dataframe
    '''
    df = dsn_keystroke_parser.DSNParser().get_feature_vectors(output_path, limit=2)


dsn_keystroke_usage_example()
