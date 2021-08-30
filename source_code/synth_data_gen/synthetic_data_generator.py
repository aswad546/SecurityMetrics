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

Synthetic Data Generator Base Class
===========================
Abstract base class for synthetic data generator
"""

import abc


class SynthDataGen(object):
    """Abstract base class for all synthetic data generators."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        Initializes the class object


        """
        return

    @abc.abstractmethod
    def generate_data(self, n_classes=2, n_features=2, n_samples=200,output_path=None, random_state=None, params=None):
        """

        @param n_classes: Number of distinct classes in the dataset
        @param n_features: Number of features per dataset
        @param n_samples: Number of samples per feature
        @param output_path: path to save the synthetic dataframe
        @param random_state: Random state input for reproducing the dataset
        @param params: Any additional specific paramaters
        @return:
        """

        return