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

Classifiers Base Class
=====================
This is the Abstract Base Class for Classifiers
This class provides interface for classifier training and parameter tuning
"""

import abc


class Classifier(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        Initializes the class object.



        """
        self.classifier = None
        self.training_data_frame = None
        self.pram_tuning_data_frame = None
        self.test_data_frame = None
        return

    @abc.abstractmethod
    def split_data(self, data_frame, training_data_size, save_path=None):
        """
        This methods takes in a dataframe and splits the data into three groups with their labels
        @param data_frame: Dataframe with tagged feature data
        @param training_data_size: Size of training data
        @param save_path: Optional path for saving training and test data
        @return(dataframe): Parameter turning, training and testing dataframes are returned
        """

        return

    @abc.abstractmethod
    def train_tune_parameters(self, pram_grid, cv, scoring_metric):
        """

        @param pram_grid: list of parameters to be tuned
        @param cv: number of k folds for cross validation
        @param scoring_metric: scoring metric to be used for cross validation
        @return: returns the best scores, parameters and estimator
        """
        return

    @abc.abstractmethod
    def random_train_tune_parameters(self, pram_dist, cv, scoring_metric, n_itr):
        """

        @param n_itr:
        @param pram_dist: list of parameters to be tuned
        @param cv: number of k folds for cross validation
        @param scoring_metric: scoring metric to be used for cross validation
        @pram n_itr: maximum iterations for random parameters tuning
        @return: returns the best estimator
        """
        return

    @abc.abstractmethod
    def classify(self):
        """
        This method classifies data
        @return: returns the classification
        """
        return
