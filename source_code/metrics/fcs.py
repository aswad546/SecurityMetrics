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

Frequency Count Score (FCS)
=====================
This FCS metric proposed in the following paper

@inproceedings{sugrim2019robust,
  title={Robust performance metrics for authentication systems},
  author={Sugrim, Shridatt and Liu, Can and McLean, Meghan and Lindqvist, Janne},
  booktitle={Network and Distributed Systems Security (NDSS) Symposium 2019},
  year={2019}
}
"""
import matplotlib.pyplot as plt
import pandas as pd

from metrics.metric import Metrics


class FCS(Metrics):

    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        self.true_labels = None
        self.pred_probs = None
        self.pred_labels = None
        self.plot = None
        self.axes = None

        return

    def get_metric(self, true_labels, predicted_probs, pred_labels, bins="auto", ax=None, alpha=0.75):
        self.true_labels = true_labels
        self.pred_probs = predicted_probs
        self.pred_labels = pred_labels
        self.axes = ax
        df = pd.DataFrame(self.pred_probs)
        df.columns = ['prob_0', 'prob_1']
        df['pred_labels'] = pred_labels
        df['true_labels'] = true_labels
        df.loc[df['true_labels'] == 1, 'user_type'] = "positive_user"
        df.loc[df['true_labels'] == 0, 'user_type'] = "negative_user"

        bins = bins
        if self.axes is None:
            # sns.set_theme("whitegrid")
            self.plot = plt.figure(figsize=[12, 12])
            self.axes = self.plot.add_subplot(1, 1, 1)
            # sns.histplot(x="prob_1", data=df, hue="user_type", bins=bins, ax=self.axes, legend=True)

            self.axes.hist(df[df['true_labels'] == 1].prob_1.values, bins=bins, label="Positive User",
                           edgecolor='black', linewidth=1.2, alpha=alpha)
            self.axes.hist(df[df['true_labels'] == 0].prob_1.values, bins=bins, label="Negative User",
                           edgecolor='black', linewidth=1.2, alpha=alpha)

            self.axes.set_title('Frequency Count Score for ' + self.classifier_name)
            self.axes.set_xlabel('scores')
            self.axes.set_ylabel('Frequency Count')
            plt.close('all')
            return self.plot
        else:
            pass
            # sns.histplot(x="prob_1", data=df, hue="user_type", bins=bins, ax=self.axes, legend=True)

            self.axes.hist(df[df['true_labels'] == 1].prob_1.values, bins=bins, label="Positive User",
                           edgecolor='black', linewidth=1.2, alpha=alpha)
            self.axes.hist(df[df['true_labels'] == 0].prob_1.values, bins=bins, label="Negative User",
                           edgecolor='black', linewidth=1.2, alpha=alpha)

            self.axes.set_title('Frequency Count Score for ' + self.classifier_name)
            self.axes.set_xlabel('scores')
            self.axes.set_ylabel('Frequency Count')
            plt.close('all')
        return


