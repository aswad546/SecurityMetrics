[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
# Project Architecture
## Overview
This project evaluates biometrics for vulnerability against statistical attacks. 
A new statistical attack and defence are proposed and evaluated. The framework provided here
enables quick prototyping of machine learning pipelines.  
The proposed attack is successful against a variety of biometrics and require the fewest attempts for successful attack.
Code for statistical attack is not released yes as the paper is under submission.
Email me if you are interested in code
for statistical attacks.

## Architecture


          -------------------------------------------
          |       External        |    Synthetic    |
          |     Dataset Parsers   |  Data Generator | 
          |   -----------------------------------   |
          |   |              Biometrics         |   |
          |   |                                 |   |
          -------------------------------------------
          |                 Dataset                 |
          | Dataset Generation | Dataset Operations |
          |   -----------------------------------   |
          |   |     Operations and Analysis     |   |
          |   |        Dim. Red., scaling       |   |
          |   -----------------------------------   |
          |   |    Adversarial Data Generation  |   |
          |   |      Vanilla Stat. , MasterKey  |   |
          |   |             K-Means++           |   |
          -------------------------------------------
          |                Classifiers              |
          |                                         |
          -------------------------------------------
          |                 Metrics                 |
          |                                         |
          -------------------------------------------
          
A high level of components is provided below:

### External Dataset Parser
This module interfaces with external datasets. 
For example usage of this module refer to  dataset_parser_usage.py in examples folder
### Synthetic Data Generator
Synthetic data generators generate synthetic data for ground truth for metrics.
For example usage of this module refer to  pipe_line_usage_synth_data.py and synth_data_gen_usage.py in examples folder
### Biometrics
If the external datasets are in raw format, Biometrics translate them to feature vectors.  
For example usage of this module refer to  dataset_parser_usage.py in examples folder
### Dataset
Internal dataset representation. This layer also provides interface to various dependent variables 
including cross-validation parameters, ratio of training and test set, the number of different classes (other users) 
to consider when creating negative training sets.
For example usage of this module refer to pipe_line_usage.py and pipe_line_usage_synth_data.py in examples folder

### Classifier
An interface over classifiers from sklearn. This contains simple distance-based measures, one-class classifiers, 
and other popular classifiers including SVM, KNN, RF.
For example usage of this module refer to pipe_line_usage.py and pipe_line_usage_synth_data.py in examples folder

### Metrics
Implementation of different metrics
For example usage of this module refer to pipe_line_usage.py and pipe_line_usage_synth_data.py in examples folder

#How to run
## Installation

Run following commands to install from the source 
```
git clone https://github.com/sohailhabib/SecurityMetrics.git
cd SecurityMetrics
cd source_code
python setup.py install
```
## Examples
Refer to example folder for details