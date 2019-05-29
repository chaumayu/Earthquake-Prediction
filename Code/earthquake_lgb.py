import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

import lightgbm as lgb

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

        params = {'num_leaves': 128,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3
         }

        train_data = lgb.Dataset(train_features, label=train_labels)
        valid_data = lgb.Dataset(test_features, label=test_labels)

        lgb_model = lgb.train(params, train_data, num_boost_round=10000, # change 20 to 2000
                    valid_sets = [valid_data], verbose_eval=500,
                    early_stopping_rounds = 200) # change 10 to 200

        test_predict = lgb_model.predict(test_features, num_iteration=lgb_model.best_iteration)
        score = mean_absolute_error(test_labels, test_predict)
        scores.append(score)

        test_predict = lgb_model.predict(test_scaled, num_iteration=lgb_model.best_iteration)
        prediction += test_predict

    print "MAE ===>", scores
    print "Mean MAE ===>", np.mean(np.asarray(scores))

    prediction /= n_fold

    submission = pd.read_csv('Data/sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = prediction
    submission.to_csv('Results/submission-lgb24.csv')

if __name__ == '__main__':
    main()


'''
MAE ===> [2.2694456968595507, 2.470310913106242, 2.4711382278967426, 1.5728970265293865, 2.5065051788746997]
Mean MAE ===> 2.2580594086533248
'''
