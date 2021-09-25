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

Masterkey attack
=====================
This class implements masterkey attack from paper Using global knowledge of users' typing traits to attack
keystroke biometrics templates

@inproceedings{serwadda2011using,
  title={Using global knowledge of users' typing traits to attack keystroke biometrics templates},
  author={Serwadda, Abdul and Phoha, Vir V and Kiremire, Ankunda},
  booktitle={Proceedings of the Thirteenth ACM Multimedia Workshop on Multimedia and Security},
  pages={51--60},
  year={2011}
}

"""
from source_code.adversaries.adversarial_attacks import Attacks
import pandas as pd
import numpy as np


class MkAttack(Attacks):

    def __init__(self, data, required_attack_samples):
        """

        @param required_attack_samples: Expects an integer for number of attack samples to generate
        @param data: Expects a Pandas dataframe
        """
        self.attack_df = data
        self.attack_samples = required_attack_samples
        self.attack_df_mk = None

    def generate_attack(self):
        if 'user' in self.attack_df.columns:
            centroid = self.attack_df.drop('user', axis=1).mean().values.reshape(1, -1)
            # Using numpy arrays for more efficient usage
            feat_list = self.attack_df.columns.drop('user').to_list()
        else:
            centroid = self.attack_df.mean()
            # Using numpy arrays for more efficient usage
            feat_list = self.attack_df.columns.drop('user').to_list()

        # Generating attack set
        pos_list = np.round(np.arange(0, 5, (5 / (self.attack_samples / 2))), 3)
        neg_list = np.round(pos_list * -1, 3)
        std_list = list()
        for idx in range(len(pos_list)):
            std_list.append(pos_list[idx])
            std_list.append(neg_list[idx])

        self.attack_df_mk = pd.DataFrame(columns=feat_list)

        for row_num, std in zip(range(len(std_list)), std_list):
            self.attack_df_mk.loc[row_num, :] = \
                centroid + (std * centroid.reshape(1, -1))

        return self.attack_df_mk
