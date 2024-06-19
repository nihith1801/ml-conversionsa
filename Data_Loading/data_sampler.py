import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from loader import load_data, augment_data, load_model_and_labels, model_paths

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Define function to handle over-sampling
def over_sample_data(X, y):
    ros = RandomOverSampler()
    X_res, y_res = ros.fit_resample(X.reshape(len(X), -1), y)
    X_res = X_res.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return X_res, y_res

# Define function to handle under-sampling
def under_sample_data(X, y):
    rus = RandomUnderSampler()
    X_res, y_res = rus.fit_resample(X.reshape(len(X), -1), y)
    X_res = X_res.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return X_res, y_res

# Define function to handle combined over- and under-sampling
def combine_sample_data(X, y):
    smote_enn = SMOTEENN()
    X_res, y_res = smote_enn.fit_resample(X.reshape(len(X), -1), y)
    X_res = X_res.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return X_res, y_res
