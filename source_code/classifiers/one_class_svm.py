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

One class SVM Classifier
=====================
This class implements one class SVM classifier

"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from source_code.classifiers.classifier import Classifier
from source_code.utilities.classifier_utilities import predict
from source_code.utilities.classifier_utilities import random_train_best_parametres
from source_code.utilities.classifier_utilities import train_best_parametres


class OneClassSvmClassifier(Classifier):

    def __init__(self, pos_user, gamma="scale", cache_size=2000, nu=0.5, random_state=None):
        self.pos_user = pos_user
        self.classifier = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu, cache_size=cache_size)
        self.classifier.probability = False
        self.train_status = False
        self.training_data_frame = pd.DataFrame()
        self.pram_tuning_data_frame = pd.DataFrame()
        self.test_data_frame = pd.DataFrame()
        self.grid = None
        self.predictions = None
        self.predictions_prob = None
        self.random_state = random_state
        self.predictions_ext_df = None
        self.predictions_prob_ext_df = None
        self.loaded_clf = None
        return

    def split_data(self, data, training_data_size, save_path=None):
        data_frame = data[data['user'] == self.pos_user]
        training_data_size = training_data_size
        test_size = (1 - training_data_size)
        data_labels = data_frame['labels']
        data = data_frame.drop(['user', 'labels'], axis=1)
        train_data, test_data, train_labels, test_labels = \
            train_test_split(data.values, data_labels.values, test_size=test_size, random_state=self.random_state)
        self.training_data_frame = pd.DataFrame(train_data)
        self.training_data_frame['labels'] = train_labels
        self.test_data_frame = pd.DataFrame(test_data)
        self.test_data_frame['labels'] = test_labels

        if save_path is not None:
            train_file_name = f"{self.pos_user}_SVM_training_data.csv"
            train_output_path = os.path.join(save_path, train_file_name)
            self.training_data_frame.to_csv(train_output_path, index=False, mode='w+')
            test_file_name = f"{self.pos_user}_SVM_test_data.csv"
            test_output_path = os.path.join(save_path, test_file_name)
            self.test_data_frame.to_csv(test_output_path, index=False, mode='w+')

        return

    def train_tune_parameters(self, pram_grid, cv, scoring_metric=None):
        print("Optimizing and Training SVM Model")
        self.classifier, self.grid = \
            train_best_parametres(self.classifier, self.training_data_frame, pram_grid=pram_grid, cv=cv,
                                  scoring_metric=scoring_metric)
        print("Optimizing and Training SVM Model Complete")
        self.train_status = True
        return

    def random_train_tune_parameters(self, pram_dist, cv, scoring_metric, n_itr=10):
        print("Optimizing and Training One Class SVM Model")
        self.classifier, self.grid = \
            random_train_best_parametres(classifier=self.classifier, train_data_frame=self.training_data_frame,
                                         scoring_metric=scoring_metric, pram_dist=pram_dist, cv=cv, n_itr=n_itr,
                                         random_state=self.random_state)
        print("Optimizing and Training Once Class SVM Model Complete")
        self.train_status = True
        return

    def classify(self, df=None):
        if df is None:
            self.predictions = predict(classifier=self.classifier, test_data_frame=self.test_data_frame)
            return self.predictions
        else:
            self.predictions_ext_df = predict(classifier=self.classifier, test_data_frame=df)
            return self.predictions_ext_df


