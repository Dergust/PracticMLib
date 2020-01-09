from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from features_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score


def full_search_test():
    data = datasets.load_breast_cancer()
    X, y = data.data[:, :10], data.target
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    model = LogisticRegression(C=1, solver='liblinear')
    scorer = make_scorer(roc_auc_score)
    full_search = FullSearch(X, y, model=model, scorer=scorer, cv=cv, verbose=True)
    print(full_search.run())

if __name__ == "__main__":
    full_search_test()
