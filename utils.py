import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
import joblib
import matplotlib.pyplot as plt


def plot_history(history):
    # summarize history for accuracy
    legend = ['train']
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
        legend.append('test')

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.show()

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.show()


def load_model(_dir, prefix='model'):
    desc_file = os.path.join(_dir,  "{}_{}.json".format(prefix, "spec"))
    weights_file = os.path.join(_dir, "{}_{}.h5".format(prefix, "weights"))
    scaler_file = os.path.join(_dir, "{}_{}.bin".format(prefix, "scaler"))

    model = None
    with open(desc_file, "r") as spec_file:
        model = model_from_json(spec_file.read())
    model.load_weights(weights_file)
    scaler = joblib.load(scaler_file)
    return (model, scaler)


def save_model(model, scaler, _dir, prefix='model'):
    if not os.path.exists(_dir):
        os.mkdir(_dir)

    desc_file = os.path.join(_dir,  "{}_{}.json".format(prefix, "spec"))
    weights_file = os.path.join(_dir, "{}_{}.h5".format(prefix, "weights"))
    scaler_file = os.path.join(_dir, "{}_{}.bin".format(prefix, "scaler"))

    model_json = model.to_json()
    with open(desc_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_file)
    joblib.dump(scaler, scaler_file, compress=True)


def read_data(fileName="customer-data.csv"):
    data = pd.read_csv(fileName)
    X = data.iloc[:, 3:13].values
    y = data.iloc[:, 13].values

    country_encoder = LabelEncoder()
    X[:, 1] = country_encoder.fit_transform(X[:, 1])

    gender_encoder = LabelEncoder()
    X[:, 2] = gender_encoder.fit_transform(X[:, 2])

    country_dummy_encoding = OneHotEncoder(categorical_features=[1])
    X = country_dummy_encoding.fit_transform(X).toarray()
    X = X[:, 1:]
    return X, y


def split_and_normalize(X, y, scaler):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def accuracy(actual, predicted):
    tn, _, _, tp = confusion_matrix(actual, predicted).ravel()
    return (tn+tp)/len(actual)
