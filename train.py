from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import cv2

early_stopper = EarlyStopping(patience=5)

def get_data():
    batch_size = 64
    input_shape = (3072,)

    data = pd.read_csv('datafile.csv',sep=',')
    x = data.imagefile
    y = data.data
    xx=[]
    for i in range(len(x)):
        xx.append(cv2.imread(i))
    x_train = xx[:800]
    x_test = xx[800:]

    y_train = y[:800]
    y_test = y[800:]

