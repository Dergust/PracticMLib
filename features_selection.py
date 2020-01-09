import numpy as np

from collections import OrderedDict
from itertools import combinations
from sklearn.model_selection import cross_val_score


class FullSearch:
    def __init__(self, X, y, model, scorer, score_proba_pred=True,
                 cv=None, X_test=None, y_test=None,
                 feature_names=None, verbose=False):

        if (cv is None) & ((X_test is None) | (y_test is None)):
            raise  Exception("cv or (X_test, y_test) must be initialized")

        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        self.model = model
        self.scorer = scorer
        self.cv = cv

        if feature_names is None:
            feature_names = list(range(self.n_features))
        assert len(feature_names) == self.n_features
        self.feature_names = feature_names
        self.fname2ind = {name: i for i, name in enumerate(self.feature_names)}
        self.ind2fname = {i: name for i, name in enumerate(self.feature_names)}

        self.X_test = X_test
        self.y_test = y_test
        if not ((self.X_test is None) | (self.y_test is None)):
            assert self.X_test.shape[0] == self.y_test.shape[0]
            assert self.X.test_shape[1] == self.n_features

        self.verbose = verbose
        self.proba_pred = score_proba_pred
        self.log_results = dict()

        print("Attention! Combinations count is {}".format(self._combinations_count()))

    def _combinations_count(self):
        return 2 ** self.n_features

    def run(self):
        best_score = np.inf
        best_subset = []

        for n_features in range(1, self.n_features + 1):
            for indices in combinations(range(self.n_features), n_features):
                if self.cv is None:
                    self.model.fit(self.X, self.y)
                    if self.proba_pred:
                        y_pred = self.model.predict_proba(self.X_test)
                    else:
                        y_pred = self.model.predict(self.X_test)
                    score = [self.scorer(self.y_test, y_pred)]

                else:
                    score = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=self.scorer, n_jobs=1)
                    score = [score.mean(), score.std()]
                self.log_results[indices] = score

                if self.verbose:
                    print(indices, score)

                score = score[0]

                if score < best_score:
                    best_score = score
                    best_subset = indices
        feature_names = [self.ind2fname[i] for i in best_subset]

        return OrderedDict([('feature_indices', best_subset),
                            ('feature_names', feature_names),
                            ('best_value', best_score)])
