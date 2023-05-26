import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt, ceil

# /content/model_params_ConvNet1.pickle
# Opening file for reading in binary mode
with open('svm_model.pickle', 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type

# Showing loaded data from file
for i, j in d.items():
    print(i + ':', j.shape)
