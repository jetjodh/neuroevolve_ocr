from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import cv2

early_stopper = EarlyStopping(patience=5)

def get_data():
    batch_size = 64
    input_shape = (50,50)

    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())

    data = pd.read_csv('datafile.csv',sep=',')
    x = data.imagefile
    y = data.data
    xx=[]
    for i in x:
        image = cv2.imread(i)
        new_image = cv2.resize(image,(50,50))
        xx.append(new_image)
    x_train = xx[:800]
    x_test = xx[800:]

    y_train = y[:800]
    y_test = y[800:]

    nb_classes = len(valid_words)
    print(nb_classes)
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return(nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def compile_model(network, nb_classes, input_shape):
    """
    Compile a sequential model.

    Args:
    network (dict): the parameters of the network

    Returns:
    a compiled network

    """
    nb_layers = network['nb_layers']
    layer = network['layer']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    for i in range(nb_layers):
        if i == 0:
            model.add(Conv2D(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(layer(nb_neurons, activation=activation))
        
        model.add(Dropout(0.2))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
    """
    nb_classes, batch_size, input_shape, x_train, x_test,\
    y_train, y_test = get_data()

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=true, mode='auto') 

    model = model.compile(network, nb_classes, input_shape)

    model_fit(x_train, y_train, batch_size=batch_size, epochs=10000, verbose=0,
                validation_data=(x_test, y_test), callbacks=[early_stopper, checkpoint])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]