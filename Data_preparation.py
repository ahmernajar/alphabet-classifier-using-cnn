
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ahmernajar
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('A_Z Handwritten Data.csv')
X = dataset.iloc[: , 1:].values
y = dataset.iloc[: , :1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


X_test = X_test / 255.0
X_test = X_test.reshape(-1,28,28,1)

X_train = X_train / 255.0
X_train = X_train.reshape(-1,28,28,1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()