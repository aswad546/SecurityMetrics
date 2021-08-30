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

False Positive Acceptance Percentage
=====================
This class takes in data with predictions to calculate false positives
"""
from analytics.analytics import Analytics
from metrics.confusion_matrix import ConfusionMatrix
import sys


class FpAcceptance(Analytics):

    def __init__(self, df, prediction):
        """
        Initializes the class object
        """
        self.df = df
        if 'labels' in self.df:
            self.true_labels = self.df['labels']
        else:
            print("Passed data frame doesn't contain ture labels")
            sys.exit(1)
        self.df_zeros = self.df[self.df['labels'] == 0]

        self.prediction = prediction
        if len(self.df) == len(self.prediction):
            pass
        else:
            print("length of data frame and predictions does not match")
            sys.exit(1)
        self.cm = ConfusionMatrix()
        self.confusion_matrix = self.cm.get_metric(true_labels=self.true_labels, predicted_labels=self.prediction)
        self.percent_accepted = None

        return

    def get_analytics(self):
        self.percent_accepted = self.cm.fp / len(self.df_zeros)*100
        return self.percent_accepted
