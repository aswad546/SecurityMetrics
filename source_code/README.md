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

# Datasets
Datasets can be downloaded from the links in the subsections. 
## Touch
Download link:

http://www.cs.wm.edu/~qyang/hmog.html

Paper reference:

@misc{sitova2015hmog_dataset,
  title={HMOG: New behavioral biometric features for continuous authentication of smartphone users},
  author={Sitov{\'a}, Zde{\v{n}}ka and {\v{S}}ed{\v{e}}nka, Jaroslav and Yang, Qing and Peng, Ge and Zhou, Gang and Gasti, Paolo and Balagani, Kiran S},
  url={http://www.cs.wm.edu/~qyang/hmog.html},
}

## Keystroke Dynamics
Download link:

https://www.cs.cmu.edu/~keystroke/

Paper reference:

@inproceedings{killourhy2009comparing,
  title={Comparing anomaly-detection algorithms for keystroke dynamics},
  author={Killourhy, Kevin S and Maxion, Roy A},
  booktitle={2009 IEEE/IFIP International Conference on Dependable Systems \& Networks},
  pages={125--134},
  year={2009},
  organization={IEEE}
}

## Mouse Movement
Download link:

https://ora.ox.ac.uk/objects/uuid:0175c157-2c9b-47d0-aa77-febaf07fca71

Paper reference:

@inproceedings{eberz2018your,
  title={When your fitness tracker betrays you: Quantifying the predictability of biometric features across contexts},
  author={Eberz, Simon and Lovisotto, Giulio and Patane, Andrea and Kwiatkowska, Marta and Lenders, Vincent and Martinovic, Ivan},
  booktitle={2018 IEEE Symposium on Security and Privacy (SP)},
  pages={889--905},
  year={2018},
  organization={IEEE}
}

## Gait
Download link:

https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_

Paper reference:

@article{zou2020deep,
  title={Deep learning-based gait recognition using smartphones in the wild},
  author={Zou, Qin and Wang, Yanling and Wang, Qian and Zhao, Yi and Li, Qingquan},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={15},
  pages={3197--3212},
  year={2020},
  publisher={IEEE}
}

## Voice
Download link:

https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

Paper reference:

@article{chung2018voxceleb2,
  title={Voxceleb2: Deep speaker recognition},
  author={Chung, Joon Son and Nagrani, Arsha and Zisserman, Andrew},
  journal={Interspeech},
  year={2018}
}