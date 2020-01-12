from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score


def full_search_test():
    data = datasets.load_breast_cancer()
    X, y = data.data[:, :12], data.target
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    model = LogisticRegression(C=1, solver='liblinear')
    scorer = make_scorer(roc_auc_score)
    full_search = FullSearch(X, y, model=model, scorer=scorer, cv=cv, verbose=True)
    print(full_search.run())

def greedy_search_test():
    data = datasets.load_breast_cancer()
    X, y = data.data[:, :], data.target
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    model = LogisticRegression(C=1, solver='liblinear')
    scorer = make_scorer(roc_auc_score)
    greedy_search = GreedySearch(X, y, model=model, scorer=scorer, cv=cv,
                                 frozen_indices=[1], forward=False, verbose=True)
    print(greedy_search.run())

def clear_test():
    data = datasets.load_breast_cancer()
    X, y = data.data[:, :], data.target
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    model = LogisticRegression(C=1, solver='liblinear')
    scorer = make_scorer(roc_auc_score)
    score = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    print(score.mean(), score.std())

if __name__ == "__main__":
    greedy_search_test()
