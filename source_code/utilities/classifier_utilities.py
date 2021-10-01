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

Classifier utilities
===========================
Utility functions for classifiers module implemented here

"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import os
import pandas as pd
import sys
from joblib import dump, load


def get_test_train_sets(data_frame, training_data_size, random_state):
    """
    This function takes in a pandas data frame and split the data into training and testing data frames.
    @param data_frame:
    @param training_data_size:
    @param random_state:
    @return: training_data_frame, test_data_frame
    """

    data_frame = data_frame
    training_data_size = training_data_size
    test_size = (1 - training_data_size)
    rf_data_labels = data_frame['labels']
    rf_data = data_frame.drop(['user', 'labels'], axis=1)
    train_data, test_data, train_labels, test_labels = \
        train_test_split(rf_data.values, rf_data_labels.values.ravel(), test_size=test_size, random_state=random_state)
    training_data_frame = pd.DataFrame(train_data)
    training_data_frame.insert(len(training_data_frame.columns), 'labels', train_labels)
    test_data_frame = pd.DataFrame(test_data)
    test_data_frame.insert(len(test_data_frame.columns), 'labels', test_labels)
    return training_data_frame, test_data_frame


def predict(classifier, test_data_frame):
    x_test = pd.DataFrame()
    x_test = test_data_frame.drop('labels', axis=1)
    predictions = classifier.predict(x_test.values)

    return predictions


def train_best_parametres(classifier, train_data_frame, pram_grid, cv, scoring_metric=None, n_jobs=-1):
    grid = GridSearchCV(classifier, param_grid=pram_grid, cv=cv, scoring=scoring_metric, n_jobs=n_jobs)
    X = train_data_frame.drop('labels', axis=1)
    y = train_data_frame['labels'].values.ravel()
    grid.fit(X, y)
    classifier = grid.best_estimator_

    return classifier, grid


def random_train_best_parametres(classifier, train_data_frame, pram_dist, cv, scoring_metric, n_itr=10,
                                 random_state=None
                                 , n_jobs=-1):
    grid = RandomizedSearchCV(classifier, param_distributions=pram_dist, cv=cv, scoring=scoring_metric,
                              n_iter=n_itr, n_jobs=n_jobs, random_state=random_state)
    X = train_data_frame.drop('labels', axis=1)
    y = train_data_frame['labels'].values.ravel()
    grid.fit(X, y)
    classifier = grid.best_estimator_

    return classifier, grid


def predict(classifier, test_data_frame):
    x_test = test_data_frame
    x_test = test_data_frame.drop('labels', axis=1)
    predictions = classifier.predict(x_test.values)

    return predictions


def predict_proba(classifier, test_data_frame):
    x_test = pd.DataFrame()
    x_test = test_data_frame.drop('labels', axis=1)
    predictions_prob = classifier.predict_proba(x_test.values)

    return predictions_prob


def save_classifier(clf_obj, path):
    save_path = os.path.join(path, path)
    dump(clf_obj, save_path)
    print(f'saved classifier available ay path {save_path}')

    return


def load_classifier(f_name):
    if not os.path.isfile(f_name):
        print(f"file not found at {f_name}")
        sys.exit(1)
    loaded_clf = load(f_name)

    return loaded_clf

