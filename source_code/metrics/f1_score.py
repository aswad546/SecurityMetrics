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

F1 Score
=====================
This class implements F1 score

"""
from metrics.metric import Metrics
from sklearn.metrics import f1_score


class F1Score(Metrics):
    """
    This class Calculates f1 score F1 = 2 * (precision * recall) / (precision + recall)
    """

    def __init__(self):
        self.f1_score = None
        return

    def get_metric(self, true_labels, predicted_labels):
        """

        @param true_labels:
        @param predicted_labels:
        @param average:
        @return:
        """
        self.f1_score = f1_score(y_true=true_labels, y_pred=predicted_labels)
        return self.f1_score

