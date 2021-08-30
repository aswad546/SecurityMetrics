"""
MIT License

Copyright (c) 2021, Sohail Habib

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

------------------------------------------------------------------------------------------------------------------------

Biometric Data Class
=====================
 This class preprocess the data for classifiers, this class deals with touch features data
"""

import math
import random
import numpy as np
import sys
import pandas as pd
from sklearn import preprocessing
from dataset.dataset import DataSet
from sklearn.utils import shuffle


class BioDataSet(DataSet):
    """
    This class preprocess the data for classifiers, this class deals with touch features data.
    """

    def __init__(self, feature_data_frame=None, feature_data_path=None, attacker=None, random_state=None):
        """
        Initializer for Touch Biometrics dataset preprocessing this class expects pandas dataframe as input or a stored
        in csv
        """
        if feature_data_frame is not None:
            self.feature_df = feature_data_frame
        else:
            if feature_data_path is not None:
                self.feature_data_path = feature_data_path
                self.feature_df = pd.read_csv(self.feature_data_path)
            else:
                print("Please provide data frame or dataframe path")
                sys.exit(1)

        self.attacker = attacker
        self.user_list = list(self.feature_df.user.unique())
        self.exclude_user_list = list()
        if attacker is not None:
            self.exclude_user_list.append(attacker)
        self.random_state = random_state

        return

    @property
    def get_users_list(self):
        """
        Method to get list of unique users from data frame
        
        @rtype: list
        @return (list): Returns a list of users
        """
        return self.user_list

    def get_positive_class_data(self, u):
        """
        This method extracts positive class user data from data frame
        @param u: User name for positive user
        @return (Dataframe) : Returns dataframe containing extracted data for positive class user 
        """
        pos_class_user_data = self.feature_df[self.feature_df['user'] == u]
        pos_labels = np.ones(len(pos_class_user_data))
        pos_class_user_data.insert(len(pos_class_user_data.columns), "labels", pos_labels)

        return pos_class_user_data

    def get_negative_class_data(self, pos_user, neg_sample_sources=None, neg_test_limit=True):
        """
        This method extracts negative class user data from dataframe
        @param pos_user: name of positive class user 
        @param neg_sample_sources: number of users to use for negative samples
        @param neg_test_limit: This parameter limits the number of negative samples per user the negative samples per 
                                negative user are calculated by dividing the total number of positive samples by number 
                                of negative users.
        @return: returns dataframe containing extracted data for negative class users 
        """
        positive_user = pos_user
        negative_class_data_sources = len(self.user_list) - len(self.exclude_user_list) - 1
        neg_user_list = list()
        neg_class_user_data = pd.DataFrame()
        for user in self.user_list:
            if user == positive_user:
                continue
            elif user in self.exclude_user_list:
                continue
            else:
                neg_user_list.append(user)

        total_neg_users = len(neg_user_list)
        shuffle_user_list = neg_user_list

        if neg_sample_sources is not None:
            if neg_sample_sources > negative_class_data_sources:
                print("not enough users to exclude so all not excluded user's data would be used")
                shuffle_user_list = neg_user_list
            else:
                shuffle_user_list = shuffle(neg_user_list, random_state=self.random_state)
                del shuffle_user_list[0:(total_neg_users - neg_sample_sources)]

        if neg_test_limit is True:
            neg_samples_per_user = math.ceil(len(self.feature_df[self.feature_df['user'] == pos_user]) /
                                             len(shuffle_user_list))

            for user in shuffle_user_list:
                user_data = self.feature_df[self.feature_df['user'] == user]
                user_data = user_data.reset_index(drop=True)
                neg_class_user_data = neg_class_user_data.append(
                    user_data.iloc[0:min(neg_samples_per_user, len(user_data))][:])
        else:
            for user in shuffle_user_list:
                user_data = self.feature_df[self.feature_df['user'] == user]
                neg_class_user_data = neg_class_user_data.append(user_data)

        neg_labels = np.zeros(len(neg_class_user_data))
        neg_class_user_data.insert(len(neg_class_user_data.columns), "labels", neg_labels)

        return neg_class_user_data

    def scale_data(self, class_data_frame, min_max_scaler):
        """
        This method scales the dataframe between minimum and maximum value
        @param class_data_frame: Data frame containing features
        @param min_max_scaler:A tuple object containing minimum and maximum value (min, max)
        @return:returns scaled dataframe
        """
        min_max_scaler = preprocessing.MinMaxScaler(min_max_scaler, True)
        class_data_frame.loc[:, class_data_frame.columns != 'user'] = \
            min_max_scaler.fit_transform(class_data_frame.loc[:, class_data_frame.columns != 'user'])
        return class_data_frame

    def get_data_set(self, u, neg_sample_sources=None, neg_test_limit=True):
        """
        This method generats a dataframe which contains combined data positive and negative class and their labels
        @param u: positive user name
        @param neg_sample_sources:
        @param neg_test_limit:
        @return:
        """
        positive_class_data = self.get_positive_class_data(u)
        negative_class_data = self.get_negative_class_data(u, neg_sample_sources=neg_sample_sources,
                                                           neg_test_limit=neg_test_limit)
        DataSet = positive_class_data.append(negative_class_data, ignore_index=True)

        return DataSet
