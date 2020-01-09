import numpy as np


def harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)


def statistic_features():
    pass


def compute_meta_features(clf, X_train, y_train, X_test, cv, *args, **kwargs):
    train_meta_factors = []
    for train_index, test_index in cv.split(X_train):
        clf.fit(X_train[train_index], y_train[train_index])
        mf = clf.predict_proba(X_train[test_index])
        train_meta_factors += mf.tolist()
    train_meta_factors = np.array(train_meta_factors)

    clf.fit(X_train, y_train)
    test_meta_factors = clf.predict_proba(X_test)

    return train_meta_factors, test_meta_factors


def compute_meta_features_models(clfs, X_train, y_train, X_test, cv, *args, **kwargs):
    train_mfs, test_mfs = [], []
    for clf in clfs:
        train_mf, test_mf = compute_meta_features(clf, X_train, y_train, X_test, cv)
        train_mfs.append(train_mf)
        test_mfs.append(test_mf)

    train_mfs = np.array(train_mfs).T
    test_mfs = np.array(test_mfs).T
    return train_mfs, test_mfs
