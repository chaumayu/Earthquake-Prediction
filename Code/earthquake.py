import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

def read_features(X_train_path,y_train_path,test_path):
    X_train = pd.read_csv(X_train_path)
    print 'X_train ==>', X_train.shape
    y_train = pd.read_csv(y_train_path)
    print 'y_train ==>', y_train.shape
    test = pd.read_csv(test_path)
    print 'test ==>', test.shape

    return X_train, y_train, test

def feature_scaling(X_train, test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    print 'X_train_scaled ==>', X_train_scaled.shape

    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
    print 'test_scaled ==>', test_scaled.shape

    return X_train_scaled, test_scaled


def main():
    X_train_path = 'Data/X_train.csv'
    y_train_path = 'Data/y_train.csv'
    test_path = 'Data/X_test.csv'

    X_train, y_train, test = read_features(X_train_path,y_train_path,test_path)

    X_train_scaled, test_scaled = feature_scaling(X_train, test)

    prediction = 0.0
    scores = []
    n_fold = 5
    kf = KFold(n_splits=n_fold)
    for fold_n, (train_index, test_index) in enumerate(kf.split(X_train_scaled)):
        print('Fold', fold_n)
        train_features, test_features = X_train_scaled.iloc[train_index], X_train_scaled.iloc[test_index]
        train_labels, test_labels = y_train.iloc[train_index], y_train.iloc[test_index]

        print 'Training Features Shape:', train_features.shape
        print 'Training Labels Shape:', train_labels.shape
        print 'Testing Features Shape:', test_features.shape
        print 'Testing Labels Shape:', test_labels.shape

        cat_model = CatBoostRegressor(iterations=20000, learning_rate = 0.01,
                                loss_function='MAE', eval_metric='MAE', use_best_model=True,
                                random_state=0, early_stopping_rounds=200)

        cat_model.fit(train_features, train_labels, eval_set=(test_features, test_labels))

        print cat_model.best_score_
        print cat_model.best_iteration_

        y_pred_test = cat_model.predict(test_features)
        score = mean_absolute_error(test_labels, y_pred_test)
        scores.append(score)

        y_pred = cat_model.predict(test)
        prediction += y_pred

    print "MAE ===>", scores
    print "Mean MAE ===>", np.mean(np.asarray(scores))

    prediction /= n_fold

    submission = pd.read_csv('Data/sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = prediction
    submission.to_csv('Results/submission-cat24.csv')

if __name__ == '__main__':
    main()


'''
CatBoostRegressor(iterations=20000, learning_rate = 0.01,
                        loss_function='MAE', eval_metric='MAE', use_best_model=True,
                        random_state=0, early_stopping_rounds=200)
Mean MAE ===> 2.249960277113364
'''
