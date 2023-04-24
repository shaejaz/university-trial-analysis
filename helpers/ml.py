from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


def create_basic_lstm_model(input_shape):
    """
    Creates a basic LSTM model
    """
    model = models.Sequential()
    model.add(layers.LSTM(10, input_shape=input_shape))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_lstm_dropout_model(input_shape):
    """
    Creates a keras model with two LSTM layers with dropout layers between
    """
    model = models.Sequential()
    model.add(layers.LSTM(16, input_shape=input_shape, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(10))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_logistic_regression_kfold_cross_validation(df, n_folds=5):
    subject_kfold = KFold(n_splits=n_folds, shuffle=True)

    logistic_regression_scores = []

    for fold, (train_subjects, test_subjects) in enumerate(subject_kfold.split(df['Subject'].unique())):
        print('Fold {}'.format(fold))
        print('Train Subjects: {}'.format(train_subjects))
        print('Test Subjects: {}'.format(test_subjects))
        
        # Get data where the subject is in the train or test subjects
        train_data = df[df['Subject'].isin(train_subjects)]
        test_data = df[df['Subject'].isin(test_subjects)]
        
        X_train = train_data.drop(['Subject', 'Label'], axis=1)
        y_train = train_data['Label']
        X_test = test_data.drop(['Subject', 'Label'], axis=1)
        y_test = test_data['Label']
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)

        print('Accuracy: {}'.format(score))
        print('\n')

        logistic_regression_scores.append(score)

    return logistic_regression_scores