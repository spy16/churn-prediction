from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from utils import read_data, split_and_normalize, accuracy, save_model


def build_classifier(optimizer='adam', loss='binary_crossentropy'):
    model = Sequential()

    # Setup hidden layers with dropout
    model.add(Dense(6, activation='relu',
                    kernel_initializer='uniform', input_dim=11))
    model.add(Dropout(rate=0.2))
    model.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    # Compile the model with given optimizer and loss function
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    scaler = StandardScaler()
    X, y = read_data()
    X_train, X_test, y_train, y_test = split_and_normalize(X, y, scaler)

    classifier = KerasClassifier(
        build_fn=build_classifier, batch_size=10, epochs=100)
    classifier.fit(X_train, y_train)
    save_model(classifier, scaler, "models", prefix="v3")

    accuracies = cross_val_score(
        estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
    mean = accuracies.mean()
    variance = accuracies.std()

    print("k-Fold Cross Validation: Max={:f} Min={:f} [Mean={:f} Variance={:f}]".format(
        accuracies.max(), accuracies.min(), mean, variance
    ))

    y_pred_train = classifier.predict(X_train) > 0.5
    y_pred_test = classifier.predict(X_test) > 0.5

    acc_train = accuracy(y_train, y_pred_train) * 100
    acc_test = accuracy(y_test, y_pred_test) * 100

    print("Accuracy (train): {}%".format(str(acc_train)))
    print("Accuracy (test) : {}%".format(str(acc_test)))
