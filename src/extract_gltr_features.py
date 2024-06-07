import sys
sys.path.append("./GLTR")

import time
import numpy as np
import pandas as pd
from backend.api import LM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import List, Any, Tuple, Dict

# -------------------------------------------------------------------
# Extract GLTR features
# -------------------------------------------------------------------
def count_top_k(ranks: List[int]) -> List[int]:
    """ranks: the rank of actual tokens, list of rank. 
    We categories rank into top10, top100, top1000 and other.
    output: count the number of tokens falling into these four categories."""
    top_k_count = {10: 0, 100: 0, 1000: 0, 'other': 0}
    for i in ranks:
        # the rank starts by 0 instead of 1 in payload
        if i < 10:
            top_k_count[10] += 1
        elif i < 100:
            top_k_count[100] += 1
        elif i < 1000:
            top_k_count[1000] += 1
        else:
            top_k_count['other'] + 1
    return list(top_k_count.values())


def gltr_payload_to_features(payload: Dict[str, List[Any]]) -> List[float]:
    """payload the what returned from GLTR LM().check_probabilities().
    We keep 14 features, topk count distribution, and fracp with 10 bins distribution."""
    bpe = payload['bpe_strings']
    N = len(bpe) - 1 # the number of tokens
    real_topk = payload['real_topk']
    pred_topk = payload['pred_topk']
    # print(len(bpe), len(real_topk), len(pred_topk))
    assert(len(real_topk) == N)
    assert(len(pred_topk) == N)

    # top1 prob of predicted tokens
    max_probs = [item[0][1] for item in pred_topk]
    ranks = [item[0] for item in real_topk]
    probs = [item[1] for item in real_topk]
    fracp = [probs[i]/max_probs[i] for i in range(N)]

    hist, edges = np.histogram(fracp, bins=10, range=(0, 1.0), density=False)
    feature = count_top_k(ranks) + list(hist)
    return feature


def get_gltr_feature_in_batch(text_list: List[str], savepath: str="features.json")-> List[List[float]]:
    lm = LM()
    features = {}
    start = time.time()
    for i, text in enumerate(text_list):
        try:
            # text = text[:1000] # maximum token length=1024 for GPT2
            payload = lm.check_probabilities(text, topk=1)
            features[i] = gltr_payload_to_features(payload)
        except:
            print(f"exception for {i}")
            features[i] = []

        if i % 500 == 0:
            print(i)
            pd.Series(features).to_json(savepath)
            
    end = time.time()
    print("{:.2f} Seconds for a check with GPT-2".format(end - start))
    pd.Series(features).to_json(savepath)
    return list(features.values())  # shape=N*14

    
def eval_binary(model, X_test, y_test, pos_label=1, average="binary"):
    """pos_label: postive label is machine text here, label is 1, human text is 0"""
    y_pred = model.predict(X_test)
    precision, recall, F1, support = precision_recall_fscore_support(
        y_test, y_pred, pos_label = pos_label, average = average)
    # accuracy
    accuracy = model.score(X_test, y_test)
    # precison
    # pre = precision_score(y_test, y_pred, pos_label = pos_label, average = average)
    # recall
    # rec = recall_score(y_test, y_pred, pos_label = pos_label, average = average)
    # probs = model.predict_proba(X_test)
    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics