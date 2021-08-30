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

Standard scaling
=====================
This class implements standardize features by removing the mean and scaling to unit variance
z = (x - u) / s
"""
from dataset.dataset_operation import DatasetOperation
import pandas as pd
from sklearn import preprocessing


class StandardScaling(DatasetOperation):

    def __init__(self):
        """
        Initializes the class object


        """
        self.scaled_df = pd.DataFrame()
        return

    def operate(self, data, copy=None, with_mean=None, with_std=None, output_path=None):
        """
        Returns a numpy array after applying the operation

        Parameters:
        data(numpy.ndarray): A numpy array on which to perform an operation
        parameters(dictionary): A key value pair containing parameters for the operation

        @return (numpy.ndarray): scaled numpy array
        """
        self.scaled_df = data
        standard_scaler = preprocessing.StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        labels = None
        if 'labels' in self.scaled_df:
            labels = self.scaled_df['labels']
            self.scaled_df = self.scaled_df.drop(columns='labels')

        self.scaled_df.loc[:, self.scaled_df.columns != 'user'] = \
            standard_scaler.fit_transform(self.scaled_df.loc[:, self.scaled_df.columns != 'user'])

        if labels is not None:
            self.scaled_df.insert(len(self.scaled_df.columns), "labels", labels)

        if output_path is not None:
            self.scaled_df.to_csv(output_path, index=False, mode='w+')

        return self.scaled_df


