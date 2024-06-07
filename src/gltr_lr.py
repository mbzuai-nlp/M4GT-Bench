import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import evaluate

# -------------------------------------------------------------------
# Subtask A train and test
# -------------------------------------------------------------------
def train(X, y, random_state=0, max_iter=1000):
    clf = LogisticRegression(random_state=random_state, max_iter=max_iter).fit(X, y)
    print(f"Trained {clf.n_iter_} iterations.")
    return clf

def test(X_test, y_test, clf):
    metric = evaluate.load("bstrai/classification_report")
    preds = clf.predict(X_test)
    # prob = clf.predict_proba(X_test)
    results = metric.compute(predictions=preds, references=y_test)
    return results

def get_gltr_features(data):
    X = list(data["GLTR_features"])
    y = list(data["label"])
    return np.array(X), np.array(y)
