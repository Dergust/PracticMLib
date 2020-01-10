import numpy as np
import copy
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
        #self.fname2ind = {name: i for i, name in enumerate(self.feature_names)}
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

    def _get_score(self, indices):
        if self.cv is None:
            self.model.fit(self.X[:, indices], self.y)
            if self.proba_pred:
                y_pred = self.model.predict_proba(self.X_test[:, indices])
            else:
                y_pred = self.model.predict(self.X_test[:, indices])
            return [self.scorer(self.y_test, y_pred)]
        else:
            score = cross_val_score(self.model, self.X[:, indices], self.y, cv=self.cv, scoring=self.scorer, n_jobs=1)
            return [score.mean(), score.std()]

    def run(self):
        best_score = -np.inf
        best_subset = []

        for n_features in range(1, self.n_features + 1):
            for indices in combinations(range(self.n_features), n_features):
                score = self._get_score(indices)
                self.log_results[indices] = score

                if self.verbose:
                    print(indices, score)

                score = score[0]

                if score > best_score:
                    best_score = score
                    best_subset = indices
        feature_names = [self.ind2fname[i] for i in best_subset]

        return OrderedDict([('feature_indices', best_subset),
                            ('feature_names', feature_names),
                            ('best_value', best_score)])


class GreedySearch:
    def __init__(self, X, y, model, scorer, score_proba_pred=True,
                 cv=None, X_test=None, y_test=None, feature_names=None, forward=True, verbose=False):

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
        self.forward = forward
        self.all_features = set(range(self.n_features))

        self.X_test = X_test
        self.y_test = y_test
        if not ((self.X_test is None) | (self.y_test is None)):
            assert self.X_test.shape[0] == self.y_test.shape[0]
            assert self.X.test_shape[1] == self.n_features

        self.verbose = verbose
        self.proba_pred = score_proba_pred
        self.log_results = dict()

    def run(self):
        if self.forward:
            self.curr_value = -np.inf
            self.curr_features = set()
            while self._forward_step():
                pass
        else:
            self.curr_value = -np.inf
            self.curr_features = set(range(self.n_features))
            while self._backward_step():
                pass

        best_subset = sorted(list(self.curr_features))
        feature_names = [self.feature_names[i] for i in best_subset]
        return OrderedDict([('feature_indices', best_subset),
                            ('feature_names', feature_names),
                            ('best_value', self.curr_value)])

    def _get_score(self, indices):
        if self.cv is None:
            self.model.fit(self.X[:, indices], self.y)
            if self.proba_pred:
                y_pred = self.model.predict_proba(self.X_test[:, indices])
            else:
                y_pred = self.model.predict(self.X_test[:, indices])
            return [self.scorer(self.y_test, y_pred)]
        else:
            score = cross_val_score(self.model, self.X[:, indices], self.y, cv=self.cv, scoring=self.scorer, n_jobs=1)
            return [score.mean(), score.std()]

    def _forward_step(self):
        self.not_considered = self.all_features - self.curr_features
        best_feature = -1
        best_value = -np.inf
        for n_feature in sorted(list(self.not_considered)):
            new_features = copy.deepcopy(self.curr_features)
            new_features.add(n_feature)
            new_features = np.array(sorted(list(new_features)))

            value = self._get_score(new_features)[0]
            self._print('F_step: new feature set {} gives value = {}'.format(new_features, value))
            if value > best_value:
                best_value = value
                best_feature = n_feature
        if best_value > self.curr_value:
            self.curr_value = best_value
            self._print('F_step: feature {} added to feature set {}\n'.format(
                best_feature, sorted(list(self.curr_features))))
            self.curr_features.add(best_feature)
            return True
        else:
            self._print('F_step: maximum reached for feature set {}; current value = {}\n'.format(
                self.curr_features, self.curr_value))
            return False

    def _backward_step(self):
        best_feature = -1
        best_value = -np.inf
        for n_feature in sorted(list(self.curr_features)):
            new_features = copy.deepcopy(self.curr_features)
            new_features.remove(n_feature)
            new_features = np.array(sorted(list(new_features)))
            if len(new_features) == 0:
                break
            value = self._get_score(new_features)[0]
            self._print('B_step: new feature set {} gives value = {}'.format(new_features, value))
            if value > best_value:
                best_value = value
                best_feature = n_feature
        if best_value > self.curr_value:
            self.curr_value = best_value
            self._print('B_step: feature {} removed from feature set {}\n'.format(
                best_feature, sorted(list(self.curr_features))))
            self.curr_features.remove(best_feature)
            return True
        else:
            self._print('B_step: maximum reached for feature set {}; current value = {}\n'.format(
                self.curr_features, self.curr_value))
            return False

    def _print(self, msg):
        if self.verbose:
            print(msg)
