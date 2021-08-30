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

Principle Component Analysis (PCA) dimensionality reduction
=====================
This class implements dimension reduction using Principle Component Analysis (PCA)

"""

from dataset.dataset_operation import DatasetOperation
import pandas as pd
from scipy.stats import zscore
import numpy as np


class OutLierRemoval(DatasetOperation):
    """
    This class removes the outliers with z score greater than a threshold default threshold is 3
    """

    def __init__(self):
        """
        Initializes the class object

        """
        self.df = pd.DataFrame()
        self.z_score_threshold = 3
        self.z_sore = None
        self.df_out = pd.DataFrame()

        return

    def operate(self, data, z_score_threshold=3):
        self.df = data
        self.z_score_threshold = z_score_threshold
        self.z_sore = np.abs(zscore(self.df))
        outlier_filter = (self.z_sore <= z_score_threshold).all(axis=1)
        self.df_out = self.df[outlier_filter]

        return self.df_out
