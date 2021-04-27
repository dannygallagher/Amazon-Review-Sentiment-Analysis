from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


# return f1, accuracy, prec, recall
def get_metrics(true_labels, pred_labels):
    f1 = f1_score(true_labels, pred_labels, average='macro')
    acc = accuracy_score(true_labels, pred_labels, average='macro')
    rec = recall_score(true_labels, pred_labels, average='macro')
    prec = precision_score(true_labels, pred_labels, average='macro')
    return f1, acc, rec, prec
