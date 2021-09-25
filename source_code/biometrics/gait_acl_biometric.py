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

Gait Biometric
===============
Gait Biometric class implementation.

Refrence Paper:
@inproceedings{thang2012gait,
  title={Gait identification using accelerometer on mobile phone},
  author={Thang, Hoang Minh and Viet, Vo Quang and Thuc, Nguyen Dinh and Choi, Deokjai},
  booktitle={2012 International Conference on Control, Automation and Information Sciences (ICCAIS)},
  pages={344--348},
  year={2012},
  organization={IEEE}
}

Dataset used for testing:
Zou, Q., Wang, Y., Zhao, Y., Wang, Q., and Li, Q. Deep Learning-Based Gait Recognition Using Smartphones in the Wild.
IEEE Transactions on Information Forensics and Security, 2020.

"""

from source_code.biometrics.biometric import Biometric
from scipy import fftpack
import numpy as np
import pandas as pd
import dask as dd
from pathlib import Path


class GaitAclBiometric(Biometric):

    def get_feature_header(self):
        """Returns a list containing name of all features."""
        feats = ['users']
        fft_feats = [f"{num}_fft_coef" for num in range(40)]
        feats.extend(fft_feats)

        return feats

    def raw_to_feature_vector(self, raw_data, freq=50):
        """

        @param raw_data: Raw data input as numpy array
        @param freq: Sampling frequency for the dataset, default value of 50 because of
         reference dataset sampling frequency
        @return: feature dataframe
        """

        samples = len(raw_data)
        sampling_time_step = 1 / freq
        fft_coef = fftpack.rfft(raw_data)
        fft_freq = fftpack.rfftfreq(samples, sampling_time_step)

        return fft_coef[:40], fft_freq[:40]
