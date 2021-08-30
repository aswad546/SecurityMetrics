import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def make_hyp_cube(min_max=(0, 1), num_samples=1000, dimensions=2):
    low = min_max[0]
    high = min_max[1]
    samples = num_samples
    dims = dimensions
    cube = pd.DataFrame(np.random.uniform(low=low, high=high, size=(samples, dims)))
    col_name = ['Dim_%02d' % i for i in cube.columns]
    cube.columns = col_name
    cube['sum'] = cube.sum(axis=1)
    cube = cube.sort_values(by='sum')
    cube = cube.drop(columns='sum')
    cube = cube.reset_index(drop=True)

    return cube


def make_hyp_rect(maximums=[1, 1], mins=[0, 0], num_samples=1000):
    dimensions = len(maximums)
    mins = mins
    samples = num_samples
    maximums = maximums
    if len(mins) != len(maximums):
        print("Length of minumum and maximum lists should be same")
        sys.exit(1)
    hyper_rect = pd.DataFrame()
    for max, min in zip(maximums, mins):
        array = pd.DataFrame(np.random.uniform(low=min, high=max, size=(samples)))
        hyper_rect = pd.concat([hyper_rect, array], axis=1)
    col_name = ['Dim_%02d' % i for i in range(len(hyper_rect.columns))]
    hyper_rect.columns = col_name

    return hyper_rect




maximums_r1 = [1, 1]
mins_r1 = [0, 0]
maximums_r2 = [1, 0.5]
mins_r2 = [0, 0]

dims = 2
samples = 10000
rec1 = make_hyp_rect(maximums=maximums_r1, mins=mins_r1, num_samples=samples)
rec2 = make_hyp_rect(maximums=maximums_r2, mins=mins_r2, num_samples=samples)

rec1.insert(0, 'id', 'rec1')
rec2.insert(0, 'id', 'rec2')
df = rec1.append(rec2)
df.to_csv(
    "C:\\Users\\esi\\Documents\\WD\\data-sufficiency_uniqueness\\SecurityMetrics\\processed_data\\hmog_touch\\rec.csv",
    index=False, mode='w+')
plt.scatter(rec1.Dim_00.values, rec1.Dim_01.values, label='rec1', marker='s')
plt.scatter(rec2.Dim_00.values, rec2.Dim_01.values, label='rec2', marker='^')
plt.legend()
plt.show()
m = 1
