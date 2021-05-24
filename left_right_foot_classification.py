import numpy as np
import pandas as pd
from scipy import signal
from math import sqrt
from pylab import *
import re
import os
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from re import search
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression


#TAKEN form GReg Baker, SFU prof, lectures note
OUTPUT_TEMPLATE_CLASSIFIER = (
    'Bayesian classifier: {bayes:.3g}\n'
    'kNN classifier:      {knn:.3g}\n'
    'SVM classifier:      {svm:.3g}\n'
)

OUTPUT_TEMPLATE_REGRESSION = (
    'Linear regression:     {lin_reg:.3g}\n'
    'Polynomial regression: {pol_reg:.3g}\n'
)


def ML_classifier(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    bayes_model = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3)
    )
    svc_model = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear')
    )


    models = [bayes_model, knn_model, svc_model]

    for i, m in enumerate(models):
        m.fit(X_train, y_train)

    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes=bayes_model.score(X_test, y_test),
        knn=knn_model.score(X_test, y_test),
        svm=svc_model.score(X_test, y_test),
    ))


def Acceleration_Euclidean_Distance(df):

    return sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

def plot_accelaration(df, x_axis, output_name):
    plt.figure()
    plt.plot(df[x_axis], df['acceleration'])
    plt.title('Total Linear Acceleration')
    plt.xlabel(x_axis)
    plt.show()
    #plt.savefig(output_name + '_acc.png')
    #plt.close()


def plot_data(df):
    # df.plot(x = "time", y = features[:3],  title = "G-Force")
    df.plot(title = "Linear Acceleration")
    # df.plot(x = "time", y = features[6:9], ,title ="Angular Velocity")
    plt.show()

def Fourier_transform(walk_data, data):

    data_FT = walk_data.apply(np.fft.fft, axis=0)
    data_FT = data_FT.apply(np.fft.fftshift, axis=0)
    data_FT = data_FT.abs()


    #Determine the sampling frequency
    sampling_frequency = round(len(data)/data.at[len(data)-1, 'time'])

    # now create the frequency axis: it runs from 0 to the sampling rate /2
    data_FT['freq'] = np.linspace(-sampling_frequency, sampling_frequency, num=len(data))

    return data_FT


def analyse_data(foot_type, stairs):
    if stairs:
        folder = str(Path('./filtered_data/upstairs/*.csv'))
    else:
        folder = str(Path('./filtered_data/walk/*.csv'))

    data_files = glob.glob(folder)

    main_df = pd.DataFrame()

    for filename in data_files:
        if search(foot_type, filename):
            df = pd.read_csv(filename, index_col=None, header=0)
            walk_data = pd.DataFrame(columns=['acceleration'])
            walk_data['acceleration'] = df.apply(Acceleration_Euclidean_Distance, axis=1)



            data_FT = Fourier_transform(walk_data, df)
            # ignore low freq noise
            data_FT = data_FT[data_FT['freq'] > 0.5]


            # Get high frquency value only4
            x = argrelextrema(data_FT.acceleration.values, np.greater) #Calculate the relative extrema of data.
            local_max = data_FT.acceleration.values[x]
            value = 0.5* local_max.max()
            local_max = local_max[local_max > value ]
            main_df = main_df.append(data_FT[data_FT['acceleration'].isin(local_max)])

    return main_df


def main():

    # left leg and right leg on flat ground analysis
    stairs = False
    ground_left =analyse_data('left', stairs)
    ground_right = analyse_data('right', stairs)

    x1 = ground_right['freq']
    x2 = ground_left['freq']

    plt.plot(x1, ground_right.acceleration, 'r.', label='right leg')
    plt.plot(x2, ground_left.acceleration,'g.', label='left leg')
    plt.title('Fig 3: Left and Right leg Characteristic Frequencies')
    plt.legend()
    plt.ylabel('acceleration')
    plt.xlabel('freq')

    ground_right['label'] = 'right'
    ground_left['label'] = 'left'
    flat_ground_data = ground_right.append(ground_left)

    print("Left leg vs right leg classification:")
    ML_classifier(flat_ground_data[['freq', 'acceleration']].values, flat_ground_data['label'].values)
    # plt.show()

    # left leg and right leg on stairs
    stairs = True
    stairs_left= analyse_data('left', stairs)
    stairs_right = analyse_data('right', stairs)

    x1 = (stairs_right['freq'])
    x2 = (stairs_left['freq'])

    plt.plot(x1, stairs_right.acceleration, 'b.', label='right leg')
    plt.plot(x2, stairs_left.acceleration, 'g.', label='left leg')

    plt.title('Fig 3: Left and Right leg Characteristic Frequencies')
    plt.legend()
    plt.xlabel('freq')
    plt.ylabel('acceleration')

    stairs_right['label'] = 'right'
    stairs_left['label'] = 'left'
    stairs_data = stairs_right.append(stairs_left)

    # # print("Left leg vs right leg classification stairs:")
    # # ML_classifier(stairs_data[['freq', 'acceleration']].values, stairs_data['label'].values)
    # plt.show()

    # # print("Left leg vs right leg classification Final:")
    # final_df = stairs_right.append(flat_ground_data)


    # # ML_classifier(final_df[['freq', 'acceleration']].values, final_df['label'].values)



if __name__ == '__main__':
    main()
