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

Dataset Base Class
=====================
Abstract Base Class for dataset
It takes in user feature data and process it for classifier on it
It expects the data as a Pandas Data frame
"""

import abc


class DataSet(object):
    """Abstract base class for all dataset generation."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        Initializes the class object


        """
        return

    @abc.abstractmethod
    def get_users_list(self):
        """
        Returns a list containing all users.

        @return (string): The list of users.
        """
        return

    @abc.abstractmethod
    def get_positive_class_data(self, u):
        """
        Returns a numpy array containing tagged data for positive class user

        Parameters:
        u(str): user name for positive user

        @return (Dataframe): A numpy array containing positive class user data
        """
        return

    @abc.abstractmethod
    def get_negative_class_data(self, u, neg_sample_source):
        """
        Returns a numpy array containing tagged data for negative class

        Parameters:
        u(str): user name for positive user
        neg_sample_source(int): Number of users to use for making negative class

        @return (Dataframe): A numpy array containing negative class user data
        """
        return


