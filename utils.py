import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
