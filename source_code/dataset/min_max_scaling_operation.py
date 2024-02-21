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

Min-max normalization
=====================
This class implements Min-max normalization

"""
from dataset.dataset_operation import DatasetOperation
import pandas as pd
from sklearn import preprocessing


class MinMaxScaling(DatasetOperation):

    def __init__(self):
        """
        Initializes the class object
        """
        self.min = 0
        self.max = 3
        self.min_max_scaler = (self.min, self.max)
        self.scaled_df = pd.DataFrame()
        self.min_max_scaler_obj = None
        return

    def operate(self, data, scaler_tuple, output_path=None):
        """
        Returns a numpy array after applying the operation

        Parameters:
        data(numpy.ndarray): A numpy array on which to perform an operation
        parameters(dictionary): A key value pair containing parameters for the operation

        @return (numpy.ndarray): scaled numpy array
        """
        self.min_max_scaler = self.set_min_max_scaler(scaler_tuple)
        self.scaled_df = data
        self.min_max_scaler_obj = preprocessing.MinMaxScaler(feature_range=self.min_max_scaler, copy=True)
        labels = None
        if 'labels' in self.scaled_df:
            labels = self.scaled_df['labels']
            self.scaled_df = self.scaled_df.drop(columns='labels')

        self.scaled_df.loc[:, self.scaled_df.columns != 'user'] = \
            self.min_max_scaler_obj.fit_transform(self.scaled_df.loc[:, self.scaled_df.columns != 'user'].values)

        if labels is not None:
            self.scaled_df.insert(len(self.scaled_df.columns), "labels", labels)

        if output_path is not None:
            self.scaled_df.to_csv(output_path, index=False, mode='w+')

        return self.scaled_df

    def set_scaler_min(self, min):
        self.min = int(min)
        self.min_max_scaler = (min, self.max)

    def set_scaler_max(self, max):
        self.max = int(max)
        self.min_max_scaler = (self.min, max)

    def set_min_max_scaler(self, scaler_tuple):
        self.min_max_scaler = scaler_tuple

        return self.min_max_scaler
