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

Biometrics Base Class
=====================
This is the Abstract Base Class for Biometrics.

"""

import abc


class Biometric(object):
    """Abstract base class for all biometrics."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_feature_header(self):
        """
        Returns a list containing name of all features.
    
        @return (string): The list of features
        """
        return

    @abc.abstractmethod
    def raw_to_feature_vector(self, raw_data):
        """
        Returns a list raw data required to calculate features.
    
        Parameters:
        raw_data (list): raw_data that conforms to data in the raw_format 
        
    
        @return (list): Features as a list
        """

        return
