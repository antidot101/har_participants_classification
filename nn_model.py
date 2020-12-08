import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

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


def get_preprocessed_data(df1, df2):

    df1['target'] = 0
    df2['target'] = 1
    df = pd.concat([df1, df2])

    X = df.iloc[:, :3].values
    y = df.target.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values
    print('............... Data preprocessing completed.')
    print()
    return X_train_std, X_test_std, y_train, y_test


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


def nn_modeling(x_train_std, x_test_std, y_train, y_test):

    model=Sequential()
    model.add(Dense(256, input_shape=(3, ), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_std, y_train, batch_size=256, epochs=20, validation_split=0.3)

    loss, accuracy = model.evaluate(x_test_std, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    plot_acc(history)


if __name__ == '__main__':
    df1, df2 = load_data()
    df1, df2 = resample_data(df1, df2, t='100L')
    x_train_std, x_test_std, y_train, y_test = get_preprocessed_data(df1, df2)
    nn_modeling(x_train_std, x_test_std, y_train, y_test)