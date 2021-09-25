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

Random Forest (RF) Classifier
=====================
This class implements Random Forest (RF) classifier

"""

import os
from source_code.classifiers.classifier import Classifier
import pandas as pd
from sklearn import ensemble
from source_code.utilities.classifier_utilities import get_test_train_sets
from source_code.utilities.classifier_utilities import train_best_parametres
from source_code.utilities.classifier_utilities import random_train_best_parametres
from source_code.utilities.classifier_utilities import predict
from source_code.utilities.classifier_utilities import predict_proba
import random


class RandomForestClassifier(Classifier):

    def __init__(self, pos_user, n_estimators=100, criterion='gini', n_jobs=-1, random_state=None):
        self.pos_user = str(pos_user)
        self.classifier = ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, n_jobs=n_jobs,
                                                          random_state=random_state)
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
        self.sys_random_state_init = random.getstate()
        self.sys_random_state_split = None
        return

    def split_data(self, data_frame, training_data_size, save_path=None):
        self.training_data_frame, self.test_data_frame = \
            get_test_train_sets(data_frame=data_frame, training_data_size=training_data_size,
                                random_state=self.random_state)
        if save_path is not None:
            train_file_name = self.pos_user + "_RF_training_data.csv"
            train_output_path = os.path.join(save_path, train_file_name)
            self.training_data_frame.to_csv(train_output_path, index=False, mode='w+')
            test_file_name = self.pos_user + "_RF_test_data.csv"
            test_output_path = os.path.join(save_path, test_file_name)
            self.test_data_frame.to_csv(test_output_path, index=False, mode='w+')
            self.sys_random_state_split = random.getstate()

        return

    def train_tune_parametres(self, pram_grid, cv, scoring_metric=None):
        print("Optimizing and Training RF Model")
        self.classifier, self.grid = \
            train_best_parametres(self.classifier, self.training_data_frame, pram_grid=pram_grid, cv=cv,
                                  scoring_metric=scoring_metric)
        print("Optimizing and Training RF Model Complete")
        return

    def random_train_tune_parameters(self, pram_dist, cv, scoring_metric, n_itr=10):
        print("Optimizing and Training RF Model")
        self.classifier, self.grid = \
            random_train_best_parametres(classifier=self.classifier, train_data_frame=self.training_data_frame,
                                         scoring_metric=scoring_metric, pram_dist=pram_dist, cv=cv, n_itr=n_itr,
                                         random_state=self.random_state)
        print("Optimizing and Training RF Model Complete")

        return

    def classify(self, df=None):
        if df is None:
            self.predictions = predict(classifier=self.classifier, test_data_frame=self.test_data_frame)
            self.predictions_prob = predict_proba(classifier=self.classifier, test_data_frame=self.test_data_frame)
            return self.predictions
        else:
            self.predictions_ext_df = predict(classifier=self.classifier, test_data_frame=df)
            self.predictions_prob_ext_df = predict_proba(classifier=self.classifier, test_data_frame=df)
            return self.predictions_ext_df
