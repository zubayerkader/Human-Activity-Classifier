import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier


def main():

    # # Action Classifier
    data = pd.read_csv('ml_input_data.csv')
    X = data.loc[:, data.columns != 'action']
    X = X.loc[:, X.columns != 'speed']
    y = data['action']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, stratify = y)

    model = make_pipeline(
        # StandardScaler(),
        KNeighborsClassifier(4) # 0.8875 0.825
        # SVC(kernel='poly', degree=3, C=1) # 0.975 0.825
        # RandomForestClassifier(50, max_depth=20)
        # AdaBoostClassifier(n_estimators=50)
    )

    model.fit(X_train, y_train)
    print('activity_classifier_train', model.score(X_train, y_train))
    print('activity_classifier_valid', model.score(X_valid, y_valid))

    ###Speed Regressor
    data = pd.read_csv('regressor_input_data.csv')
    data = data[(data['action'] != 'upstairs')].copy()
    data = data[(data['action'] != 'downstairs')].copy()
    data = data[(data['action'] != 'running')].copy()

    X = data.loc[:, data.columns != 'action']
    X = X.loc[:, X.columns != 'speed']
    y = data['speed']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2) # , random_state = 0, stratify = y

    model = make_pipeline(
        # StandardScaler(),
        MinMaxScaler(),
        KNeighborsRegressor(3) # 0.8875 0.825
        # RandomForestRegressor(50, max_depth=20)
        # AdaBoostRegressor(n_estimators=300) #1500
        # GradientBoostingRegressor (n_estimators=300, max_depth = 70)
        # SVC(kernel='poly', degree=3, C=1) # 0.975 0.825
    )

    model.fit(X_train, y_train)
    print('speed_regressor_train', model.score(X_train, y_train))
    print('speed_regressor_valid', model.score(X_valid, y_valid))

    predictions = model.predict(X_valid)

    predictions = pd.DataFrame(model.predict(X_valid), columns=['predictions'])
    y_valid_df = y_valid.reset_index(drop=True)
    comparison = pd.concat([y_valid_df, predictions], axis=1)
    comparison['difference'] =  comparison['speed'] - comparison['predictions']
    print(comparison)
