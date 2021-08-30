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

Gaussian Blob Data Generator
===========================
This class implements gaussian blob data generator

"""
from synth_data_gen.synthetic_data_generator import SynthDataGen
from sklearn.datasets import make_blobs
import pandas as pd
import sys
import numpy as np
import os


class GaussBlob(SynthDataGen):
    def __init__(self):
        self.n_classes = 0
        self.n_features = 0
        self.n_samples = 0
        self.random_state = None
        self.centers = None
        self.cluster_std = 1
        self.shuffle = None
        self.class_names = None

        return

    def generate_data(self, n_classes=2, n_features=2, n_samples=200, centers=None, cluster_std=1, class_names=None
                      , output_path=None, random_state=None, shuffle=None) -> object:
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state
        self.centers = centers
        if len(self.centers) != self.n_classes:
            print("number of centers need to be equal to n_classes")
            sys.exit(1)
        self.cluster_std = cluster_std
        # Checking if the standard deviation is given an an integer or a list if list then if it is the correct length
        if type(cluster_std) != int and type(cluster_std) == list and len(cluster_std) == n_classes:
            pass
        elif type(cluster_std) == int:
            pass
        else:
            print("cluster_std needs to be list of length equal to n_classes or an int")
            sys.exit(1)

        self.shuffle = shuffle
        if class_names is None:
            self.class_names = range(0, self.n_classes)
        else:
            self.class_names = class_names
            if self.n_classes != len(self.class_names):
                print('Class names should be equal to the number of classes')
                sys.exit(1)

        df = pd.DataFrame()
        clus_centers = pd.DataFrame()
        for user in range(0, n_classes):
            """
            Depending on the type of standard deviation given int or list select the appropriate statement for getting
            the data frame
            """
            if type(self.cluster_std) == list:
                if self.random_state is not None:
                    data, cen = self.get_data(center=self.centers[user], user_label=self.class_names[user],
                                              cluster_std=self.cluster_std[user],
                                              random_state=(self.random_state + user))

                else:
                    data, cen = self.get_data(center=self.centers[user], user_label=self.class_names[user],
                                              cluster_std=self.cluster_std[user],
                                              random_state=self.random_state)
                df = df.append(data)
                clus_centers = clus_centers.append(cen)
            else:
                data, cen = self.get_data(center=self.centers[user], user_label=self.class_names[user],
                                          cluster_std=self.cluster_std)
                df = df.append(data)
                clus_centers = clus_centers.append(cen)
        df = df.reset_index(drop=True)
        if output_path is not None:
            if not os.path.isdir(output_path):
                print("Enter valid directory path")
                sys.exit(1)
            df_path = os.path.join(output_path, 'blob.csv')
            cen_path = os.path.join(output_path, 'center.csv')
            df.to_csv(df_path, index=False, mode='w+')
            clus_centers.to_csv(cen_path, index=False, mode='w+')
        return df, clus_centers

    def get_data(self, center=None, user_label=None, cluster_std=1, random_state=None):
        center = np.asarray(center)
        center = center.reshape(1, -1)
        data, labels, centers = make_blobs(n_features=self.n_features, centers=center, random_state=random_state,
                                           n_samples=self.n_samples, cluster_std=cluster_std, shuffle=False,
                                           return_centers=True)
        df = pd.DataFrame(data=data)
        centers = pd.DataFrame(centers)
        """
        Generating column names currently implimented to two digit numbers, if more digits needed change Dim_%02d to
        Dim_%0xd where x the number of digits 
        """
        col_name = ['Dim_%02d' % i for i in range(len(df.columns))]
        df.columns = col_name
        df.insert(0, "user", labels)
        if user_label is not None:
            df['user'] = user_label
        return df, centers
