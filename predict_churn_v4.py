from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

from utils import read_data, split_and_normalize, accuracy


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

    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10, n_jobs=-1)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    print("Best Accuracy: {:f}".format(best_accuracy))