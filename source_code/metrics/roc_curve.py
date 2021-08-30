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

Receiver Operating Characteristic curve (ROC curve)
=====================
This class implements ROC curve

"""
from metrics.metric import Metrics
from sklearn.metrics import plot_roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class RocCurve(Metrics):

    def __init__(self):
        self.roc_curve = None
        self.tpr = None
        self.fpr = None
        self.auc_roc = None
        self.eer_threshold = None
        self.thresholds = None
        self.eer = None
        pass
        return

    def get_metric(self, test_set_features, test_set_labels, classifier, ax=None, pos_label=1):
        self.roc_curve = plot_roc_curve(classifier, test_set_features, test_set_labels)

        predictions = classifier.predict(test_set_features)
        scores = classifier.predict_proba(test_set_features)
        fpr, tpr, self.thresholds = roc_curve(y_true=test_set_labels, y_score=scores[:, 1], pos_label=pos_label)
        self.eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        self.eer_threshold = interp1d(fpr, self.thresholds)(self.eer)
        self.tpr = self.roc_curve.tpr
        self.fpr = self.roc_curve.fpr
        self.auc_roc = self.roc_curve.roc_auc
        self.roc_curve.plot(ax=ax)
        plt.close('all')
        return self.roc_curve
