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
External Parser Base Class
===========================
Abstract Base Class for External dataset Parsers
It supports datsets that are already in stored as Feature Vectors 
(using the routine get_feature_vectors) or transformation of 
raw data to Feature Vectors (using routine raw_to_feature_vectors)
Data is returned as pandas dataframe
If data is written to the disk, it is also written as a pandas DataFrame
"""

import abc


class ExternalParser(object):
    """Abstract base class for all external parsers."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """
        Initializes the class object
    
        
        """
        return

    @abc.abstractmethod
    def raw_to_feature_vectors(self, raw_data_path, output_path=None, limit=None):
        """
        Parses external datasets and converts it into an internal format
    
        Parameters:
        raw_data_path (String): Path to raw data
        output_path (String): Output folder path. Only reads and return 
        the loaded data if NULL
        limit (int): Limits the number of folders or users to read. 
                    None for unlimited
        
        Returns:
        A dictionary, where keys are users and values are are list of feature vectors    
        """
        return

    # @abc.abstractmethod
    # def get_feature_vectors (self, input_path, limit = None):
    #     """
    #     Reads the FeatureVectors and returns them
    #
    #     Parameters:
    #     input_path (String): Path to input data in the internal format (i.e., daraframe on disk)
    #
    #     Returns:
    #     A DatFrame
    #     """
    #     return
