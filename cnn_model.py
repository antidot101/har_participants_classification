import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten


warnings.filterwarnings("ignore")


def load_data():

    df1 = pd.read_csv('01-0-resampled.csv')
    df2 = pd.read_csv('02-0-resampled.csv')
    print('............... Datasets loaded.')
    return df1, df2


def resample_data(df1, df2, t='S'):

    df1.dt = pd.to_datetime(df1.dt)
    df2.dt = pd.to_datetime(df2.dt)
    df1 = df1.set_index('dt', drop=True)
    df2 = df2.set_index('dt', drop=True)
    df1 = df1.resample(t).mean()
    df2 = df2.resample(t).mean()
    print('............... Resampling completed.')
    return  df1, df2


def plot_acc(hist):

    accuracy = hist.history['accuracy']
    val_accuracy = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(len(accuracy))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
    ax[0].plot(epochs, accuracy, label='Training accuracy')
    ax[0].plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()


def get_preprocessed_data(df1, df2):

    df1['target'] = 0
    df2['target'] = 1
    df = pd.concat([df1, df2])
    X = df.iloc[:, :3].values
    y = df.target.values

    """
    Для применения CNN входные данные нужно разбить на равные временные отрезки (окна). 
    Каждое такое окно будет размерность (m x n), 
    где m - количество временных меток, n = 3 - количество признаков (x, y, z).
    Теперь в качестве объектов входных данных будут эти окна.
    Разделим весь набор данных на 5000 равных окон и создадим новый массив. 
    """
    Xs = np.array_split(X, 5000)
    ys = np.array_split(y, 5000)
    Xs = np.array(Xs)
    ys = np.array(ys)

    """ 
    Теперь каждому окну нужно присвоить метку класса 0 или 1 (взамен меток для каждой временной точки измерения).
    Чтобы разметить все окна, можно просто перемножить метки каждой точки, т.е. окну с временными метками 0 будет присвоена 
    метка класса 0, а окно с метками 1 будет размечено как 1.
    P.S. В одном единственном окне вследствие конкатенации будут обе метки 0 и 1. 
         Пусть класс этого окна будет 0 (результат перемножения).
    """
    df_y = pd.DataFrame(ys)
    labels = 1
    for i in range(df_y.shape[1]):
        labels *= df_y.iloc[:, i]
    df_y['label'] = labels

    df_y.drop(df_y.columns[0:-1], axis='columns', inplace=True)
    ys = df_y.values
    ys = ys.reshape(-1)
    print('New data dims: X: {}, y: {}'.format(Xs.shape, ys.shape))

    # Для масштабирования признаков преобразуем массив в двумерный
    Xs_rs= Xs.reshape(Xs.shape[0], Xs.shape[1] * Xs.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(Xs_rs, ys, test_size=0.2)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    X_train = X_train_std.reshape(4000, 1065, 3)
    X_test = X_test_std.reshape(1000, 1065, 3)
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    print('............... Data preprocessing completed.')
    print()
    return X_train, X_test, y_train, y_test


def cnn_modeling(x_train_std, x_test_std, y_train, y_test):

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(1065, 3)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train_std, y_train, batch_size=256, epochs=50, verbose=1, validation_split=0.2)

    loss, accuracy = model.evaluate(x_test_std, y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    plot_acc(history)


if __name__ == '__main__':
    pass