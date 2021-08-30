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
Mouse Data Parser
=====================
A class that parses Mouse movement data.
@inproceedings{eberz2018your,
  title={When your fitness tracker betrays you: Quantifying the predictability of biometric features across contexts},
  author={Eberz, Simon and Lovisotto, Giulio and Patane, Andrea and Kwiatkowska, Marta and Lenders, Vincent and
  Martinovic, Ivan},
  booktitle={2018 IEEE Symposium on Security and Privacy (SP)},
  pages={889--905},
  year={2018},
  organization={IEEE}
}
"""
import os
import collections
from io import StringIO
import pandas as pd
from biometrics import mouse_biometric
from utilities import io_utilities as io
from external_dataset_parsers.external_parser import ExternalParser


class MouseParser(ExternalParser):

    def __init__(self):
        return

    def raw_to_feature_vectors(self, raw_data_path, choice, output_path=None, limit=None):
        mb = mouse_biometric.MouseBiometric()
        dataset_path = raw_data_path
        dirs = io.get_directory_list(dataset_path)
        print("Users found: " + str(len(dirs)))

        if limit is None:
            users = dirs

        else:
            users = dirs[:min(len(dirs), limit)]
            print("Generating features for: ", len(users),  "users")


        csv_str = StringIO()
        csv_str.write('user, ')
        csv_str.write('%s' % ', '.join(map(str, mb.get_feature_header()[1:])) + '\n')

        for u in users:
            mouseup_counter = int(0)
            print("Generating features for " + str(u))
            user_data_path = os.path.join(dataset_path, u)
            if choice == 1:
                specified_data_path = os.path.join(user_data_path, "mouse.csv")
            else:
                specified_data_path = os.path.join(user_data_path, "trackpad.csv")

            with open(specified_data_path) as f:
                lines = f.readlines()

            segments = list()
            for l in lines:
                tokens = l.strip('\n').split(',')
                if mouseup_counter == 1:
                    segments.append(tokens)

                if mouseup_counter == 2:
                    mouseup_counter = 0
                    """Send to biometrics here"""
                    fv = mb.extract_features(segments, u)
                    segments.clear()
                    if fv is not None:
                        csv_str.write(u + ',' + str(fv[1:])[1:-1] + '\n') # Commenting this out fixes the problem
                if tokens[1] == "mouseleftup":
                    mouseup_counter += 1
        csv_str.seek(0)
        df = pd.read_csv(csv_str, header=0, index_col=False)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        if output_path is not None:
            df.to_csv(output_path, index=False, mode='w+')
            print("Feature Generation Complete files are available at path", output_path)
        else:
            print("Feature Generation Complete")
        return df

    def get_feature_vectors(self, input_path, limit=None):
        df = pd.read_csv(input_path, header=0, index_col=False)
        if limit is not None:
            ctr = collections.Counter(df['user'])
            top_users_ctr = ctr.most_common(min(len(ctr), limit))
            users = list()
            for user in top_users_ctr:
                users.append(user[0])
            df = df[df['user'].isin(users)]
        return df





