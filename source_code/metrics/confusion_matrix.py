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

Confusion Matrix (CM)
=====================
This class implements confusion matrix

"""

from metrics.metric import Metrics
from sklearn.metrics import confusion_matrix
import pandas as pd


class ConfusionMatrix(Metrics):
    """
    This class generates confusion matrix and extracts information from the matrix
    """

    def __init__(self):
        """
        Class initialization
        """
        self.true_labels = None
        self.predicted_labels = None
        self.sample_weight = None
        self.normalize = None
        self.cm = None
        self.cm_generated = False
        self.tn = None
        self.fp = None
        self.fn = None
        self.tp = None
        self.tpr = None
        self.fpr = None

        return

    def get_metric(self, true_labels: object, predicted_labels: object, sample_weight: object = None, normalize: object = None, output_path: object = None) -> object:
        """

        @param output_path: Path to write the confusion matrix
        @param true_labels:  Ground truth (correct) target values.
        @param predicted_labels: Estimated targets as returned by a classifier.
        @param sample_weight: Sample weights.
        @param normalize: Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the
                          population. If None, confusion matrix will not be normalized.
        @return: confusion matrix self.cm
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.cm = confusion_matrix(y_true=self.true_labels, y_pred=self.predicted_labels,
                                   sample_weight=self.sample_weight, normalize=self.normalize)
        self.cm_generated = True
        self.tn = self.cm[0, 0]
        self.fp = self.cm[0, 1]
        self.fn = self.cm[1, 0]
        self.tp = self.cm[1, 1]
        self.tpr = self.tp / (self.tp + self.fn)
        self.fpr = self.fp / (self.tn + self.fp)

        if output_path is not None:
            cm_df = pd.DataFrame(self.cm)
            cm_df.to_csv(output_path, index=False, mode='w+')

        return self.cm

    def get_tn(self):
        """
        
        @return: true negative self.tn 
        """""
        if self.cm_generated:
            return self.tn
        else:
            return print("Generate confusion matrix using get_metric")

    def get_fp(self):
        """

        @return: False positive self.fp
        """
        if self.cm_generated:
            return self.fp
        else:
            return print("Generate confusion matrix using get_metric")

    def get_fn(self):
        """

        @return:  False negative self.fn
        """
        if self.cm_generated:
            return self.fn
        else:
            return print("Generate confusion matrix using get_metric")

    def get_tp(self):
        """

        @return: True positive self.tp
        """
        if self.cm_generated:
            return self.tp
        else:
            return print("Generate confusion matrix using get_metric")

    def get_tpr(self):
        """

        @return: True positive self.tp
        """
        if self.cm_generated:
            return self.tpr
        else:
            return print("Generate confusion matrix using get_metric")

    def get_fpr(self):
        """

        @return: False positive self.fp
        """
        if self.cm_generated:
            return self.fpr
        else:
            return print("Generate confusion matrix using get_metric")