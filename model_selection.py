import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt

warnings.filterwarnings("ignore")


"""
В качестве исходных данных используются ресемплированные датасеты с частотой дискретизации 10 Гц.
Рассматриваются следующие классификаторы:
- линейные (логистическая регрессия, метод опорных векторов);
- ансамблевые (случайный лес, градиентный бустинг).
Для настройки гиперпараметров моделей используется сеточный поиск с k-блочной кросс-валидацией и оценкой по accuracy.
Для линейных классификаторов проведена нормализация значений признаков методом стандартизации.
Тренировочные и тестовые данные разделены в соотношении 70/30.
"""


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

# df_train1 = df1[df1.index < df1.index[0] + td(days=2)]
# df_train2 = df2[df2.index < df2.index[0] + td(days=2)]
# df_train1['target'], df_train2['target'] = 0, 1

# df_train = pd.concat([df_train1, df_train2])

# df_test1 = df1[df1.index >= df1.index[0] + td(days=2)]
# df_test2 = df2[df2.index >= df2.index[0] + td(days=2)]
# df_test1['target'], df_test2['target'] = 0, 1

# df_test = pd.concat([df_test1, df_test2])

# X_train = df_train.iloc[:,:3].values
# y_train = df_train.iloc[:,3].values
# X_test = df_test.iloc[:,:3].values
# y_test = df_test.iloc[:,3].values

# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)


def get_splitted_data(df1, df2):

    df1['target'] = 0
    df2['target'] = 1
    df = pd.concat([df1, df2])
    X = df.iloc[:, :3].values
    y = df.target.values
    return train_test_split(X, y, test_size=0.3)


def get_scaled_data(x_train, x_test):

    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)
    return x_train_std, x_test_std


def grid_search(x_train, y_train, x_test, y_test, model, pg, cv=5):

    start_time = dt.now()
    grid = GridSearchCV(estimator=model, param_grid=pg, cv=cv, scoring='accuracy')
    grid.fit(x_train, y_train)
    end_time = dt.now() - start_time
    print('Time: ', end_time)
    print('Best params: ', grid.best_params_)
    print('Best score: %.3f' % grid.best_score_)

    best_estimator = grid.best_estimator_
    y_test_pred = best_estimator.predict(x_test)
    print('-------------- Classification report -------------')
    print(classification_report(y_test, y_test_pred))
    print('---------------- Confusion matrix ----------------')
    print(confusion_matrix(y_test, y_test_pred))
    print('----------------\n\nAccuracy test data: %.3f' % accuracy_score(y_test, y_test_pred))


def get_log_regr_best(x_train, y_train, x_test, y_test):

    pg = {'C': [10**i for i in range(-3, 4, 1)], 'penalty': ['l1', 'l2']}
    kf = KFold(n_splits=5, shuffle=True)
    return grid_search(x_train, y_train, x_test, y_test, LogisticRegression(), pg, cv=kf)


def get_svm_best(x_train, y_train, x_test, y_test):

    pg = {'C': [10**i for i in range(-3, 4, 1)], 'kernel': ('linear', 'rbf')}
    kf = KFold(n_splits=5, shuffle=True)
    return grid_search(x_train, y_train, x_test, y_test, SVC(), pg, cv=kf)


def get_tree_best(x_train, y_train, x_test, y_test):

    pg = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 4, 8]}
    kf = KFold(n_splits=5, shuffle=True)
    return grid_search(x_train, y_train, x_test, y_test, DecisionTreeClassifier(), pg, cv=kf)


def get_forrest_best(x_train, y_train, x_test, y_test):

    pg = {'n_estimators': [100, 200, 300], 'max_depth': [2, 5, 15]}
    kf = KFold(n_splits=5, shuffle=True)
    return grid_search(x_train, y_train, x_test, y_test, RandomForestClassifier(), pg, cv=kf)


def get_gb_best(x_train, y_train, x_test, y_test):

    kf = KFold(n_splits=5, shuffle=True)
    gb_clfs = [(n_est, GradientBoostingClassifier(n_estimators=n_est)) for n_est in [100, 300, 500]]
    best_estimator = cv_acc(gb_clfs, x_train, y_train, kf)
    best_estimator.fit(x_train, y_train)
    y_test_pred = best_estimator.predict(x_test)
    print('\n--------------- Classification report -------------')
    print(classification_report(y_test, y_test_pred))
    print('----------------- Confusion matrix -----------------')
    print(confusion_matrix(y_test, y_test_pred))
    print('----------------------------------------------------'
          '\n\nAccuracy test data: %.3f' % accuracy_score(y_test, y_test_pred))


def cv_acc(clfs, X, y, cv):

    acc_scores = []
    for n_est, clf in clfs:
        start_time = dt.now()
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, scoring='accuracy')
        end_time = dt.now() - start_time
        print("n_estimators: %f: accuracy = %0.8f | time elapsed: " % (n_est, scores.mean()), end_time)
        acc_scores.append(scores.mean())
    best_estimator = clfs[np.argmax(acc_scores)]
    return best_estimator[1]


if __name__ == '__main__':
    df1, df2 = load_data()
    df1, df2 = resample_data(df1, df2, t='S')
    x_train, x_test, y_train, y_test = get_splitted_data(df1, df2)
    x_train_std, x_test_std = get_scaled_data(x_train, x_test)

    # get_log_regr_best(x_train_std, y_train, x_test_std, y_test)
    # get_tree_best(x_train, y_train, x_test, y_test)
    get_forrest_best(x_train, y_train, x_test, y_test)
