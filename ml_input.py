import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
currentdir = os.getcwd()
walking = os.path.join(currentdir, 'filtered_data/walk')
upstairs = os.path.join(currentdir, 'filtered_data/upstairs')
downstairs = os.path.join(currentdir, 'filtered_data/downstairs')
running = os.path.join(currentdir, 'filtered_data/run')
# print (currentdir)
x = pd.DataFrame()
regressor_data = pd.DataFrame()

def Acceleration_Euclidean_Distance(df):
    return np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

def Fourier_transform(walk_data, data):

    data_FT = walk_data.apply(np.fft.fft, axis=0)
    data_FT = data_FT.apply(np.fft.fftshift, axis=0)
    data_FT = data_FT.abs()

    #Determine the sampling frequency
    sampling_frequency = round(len(data)/data.at[len(data)-1, 'time'])

    # now create the frequency axis: it runs from 0 to the sampling rate /2
    data_FT['freq'] = np.linspace(-sampling_frequency, sampling_frequency, num=len(data))

    return data_FT

def get_subject_and_foot(filename):
    base  = os.path.basename(filename)
    base = base.split(".")
    subject = base[0].split('_')
    subject, foot, secs, file_number = subject[2], subject[1], subject[3], subject[4]

    return (subject, foot, secs, file_number)

def allData(dirname, action):
    global x
    global regressor_data
    for filename in os.listdir(dirname):
        # print(x)
        # print ("!!!!!!!!!!!!!!!!!!",filename)
        file = os.path.join(dirname, filename)
        data = pd.read_csv(file)
        # print (data)
        noise_filtered = data[["time","ax","ay","az"]].copy()
        noise_filtered['prev_time'] = noise_filtered['time'].shift(1)
        noise_filtered = noise_filtered.dropna()
        noise_filtered['del_t'] = noise_filtered['time'] - noise_filtered['prev_time']
        noise_filtered['vx'] = noise_filtered['ax']*noise_filtered['del_t']
        noise_filtered['vy'] = noise_filtered['ay']*noise_filtered['del_t']
        noise_filtered['vz'] = noise_filtered['az']*noise_filtered['del_t']
        noise_filtered['px'] = noise_filtered['vx']*noise_filtered['del_t']
        noise_filtered['py'] = noise_filtered['vy']*noise_filtered['del_t']
        noise_filtered['pz'] = noise_filtered['vz']*noise_filtered['del_t']

        speed = None
        if (action in ['walking', 'jogging']):
            subject, foot, secs, file_number = get_subject_and_foot(filename)
            speed = 19.5/float(secs) # m/s


        reg_df = noise_filtered.copy()
        sample = pd.DataFrame(Acceleration_Euclidean_Distance(reg_df), columns=['euclidean'])
        reg_df = Fourier_transform(sample, reg_df)
        reg_data = noise_filtered.copy()
        reg_data = pd.concat([reg_data, reg_df], axis=1)
        # print(noise_filtered)

        noise_filtered = noise_filtered.drop(['time', 'prev_time', 'del_t'], axis = 1).copy()
        noise_filtered = noise_filtered.drop(noise_filtered.index[0])
        noise_filtered = noise_filtered.values.flatten()
        noise_filtered = pd.DataFrame(noise_filtered).transpose()
        action_col = pd.DataFrame([[action, speed]], columns=['action', 'speed'])
        noise_filtered = pd.concat([action_col, noise_filtered], axis=1)

        reg_data = reg_data.drop(['time', 'prev_time', 'del_t'], axis = 1).copy()
        reg_data = reg_data.drop(reg_data.index[0])
        reg_data = reg_data.values.flatten()
        reg_data = pd.DataFrame(reg_data).transpose()
        action_col = pd.DataFrame([[action, speed]], columns=['action', 'speed'])
        reg_data = pd.concat([action_col, reg_data], axis=1)


        regressor_data = regressor_data.append(reg_data)
        x = x.append(noise_filtered)


def createMlData():
    allData(walking, 'walking')
    allData(upstairs, 'upstairs')
    allData(downstairs, 'downstairs')
    allData(running, 'running')
    # print(x.shape)
    x.to_csv('ml_input_data.csv', index=False)
    regressor_data.to_csv('regressor_input_data.csv', index=False)
    # print(x)

if __name__ == '__main__':
    createMlData()
