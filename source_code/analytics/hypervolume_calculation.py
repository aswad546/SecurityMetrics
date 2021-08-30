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

=======================================================================================================================
Hypervolume Calculation
=====================
This class takes in the path of the data files and file name to calculate hypervolumes using R script
"""
import os

from analytics.analytics import Analytics
import subprocess
from pathlib import Path


class CalcHyperVolume(Analytics):

    def __init__(self, dir_name, file_name):
        """
        Initializes the class object
        """
        root_path = Path(__file__).parent.parent.parent
        # Enter path for Rscript shell here
        self.command = 'C:/Program Files/R/R-4.0.2/bin/Rscript'
        self.arg = f"--vanilla"
        # Enter path for R script for calculating hypervolumes
        self.path2script = os.path.join(root_path, 'source_code\\analytics\\hyper_vol_usage.R')
        # Changing path to R format
        self.dir_name = dir_name.replace('\\', '/')
        self.file_name = file_name

        return

    def get_analytics(self):
        retcode = subprocess.call([self.command, self.arg, self.path2script, self.dir_name, self.file_name], shell=True)
        return retcode
