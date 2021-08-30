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

Gini Coefficient (GC) based metric
=====================
This Gini Coefficient (GC) based metric proposed in the following paper

@inproceedings{eberz2017evaluating,
  title={Evaluating behavioral biometrics for continuous authentication: Challenges and metrics},
  author={Eberz, Simon and Rasmussen, Kasper B and Lenders, Vincent and Martinovic, Ivan},
  booktitle={Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security},
  pages={386--399},
  year={2017}
}
"""
from metrics.metric import Metrics
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc


class GiniCoef(Metrics):
    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        self.error_array = None
        self.x_lorenz = None
        self.gini_coef_num = None
        self.gini_coef_graph = None
        return

    def get_metric(self, error_array):
        self.error_array = np.copy(error_array)
        lc = self.lorenz_curve(self.error_array)

        return lc

    def lorenz_curve(self, x):
        x_lorenz = self.f(x)
        fig, ax_gini = plt.subplots(figsize=[12, 12])
        # s = "Gini-Coeff " + self.classifier_name + "=" + str((self.gini_num(x)))
        ## plot of Lorenz curve
        # plt.text(0, 1, s)
        ax_gini.plot(np.arange(x_lorenz.size) / (x_lorenz.size - 1), x_lorenz, label='Lorenz Curve')
        ## line plot of equality
        ax_gini.plot((np.arange(x_lorenz.size) / (x_lorenz.size - 1)), (np.arange(x_lorenz.size) / (x_lorenz.size - 1))
                     , label='line of Equality')
        plt.fill_between(np.arange(x_lorenz.size) / (x_lorenz.size - 1), np.arange(x_lorenz.size) / (x_lorenz.size - 1),
                         x_lorenz, alpha=0.5, facecolor='gray')
        plt.title(self.classifier_name + " Classifier Gini Coefficient =" + str(self.gini_num(x)))
        plt.legend()
        plt.xlabel('fraction of users')
        plt.ylabel('Cumulative error')

        self.x_lorenz = x_lorenz

        return self.x_lorenz

    def gini_num(self, arr):
        ## first sort
        sorted_arr = arr.copy()
        sorted_arr.sort()
        n = arr.size
        coef_ = 2. / n
        const_ = (n + 1.) / n
        weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])
        self.gini_coef_num = coef_ * weighted_sum / (sorted_arr.sum()) - const_
        return self.gini_coef_num

    def gini_graph(self, arr):
        x = self.f(arr)
        auc1 = auc(np.arange(x.size) / (x.size - 1), x)
        auc2 = auc(np.arange(2), np.arange(2))
        gini_coef = (auc2 - auc1) / auc2

        auc1_int = np.trapz(x=np.arange(x.size) / (x.size - 1), y=x)
        auc2_int = np.trapz(np.arange(2), np.arange(2))
        gini_coef_int = (auc2_int - auc1_int) / auc2_int
        self.gini_coef_graph = gini_coef

        return self.gini_coef_graph

    @staticmethod
    def f(X):
        x = np.copy(X)
        x.sort()
        x_lorenz = x.cumsum() / x.sum()
        x_lorenz = np.insert(x_lorenz, 0, 0)
        return x_lorenz
