from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


# return f1, accuracy, prec, recall
def get_metrics(true_labels, pred_labels):
    f1 = f1_score(true_labels, pred_labels, average='macro')
    acc = accuracy_score(true_labels, pred_labels, average='macro')
    rec = recall_score(true_labels, pred_labels, average='macro')
    prec = precision_score(true_labels, pred_labels, average='macro')
    return f1, acc, rec, prec

def metric_test(model,  device, test_iter, choose_best_epoch = True):
    if choose_best_epoch:
        model.load_state_dict(torch.load('RNN-train.pt'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        predictions = []
        labels = []
        for idx, batch in enumerate(test_iter):
            text = batch.text[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            text, target = text.to(device), target.to(device)
            labels.append(labels)
            outputs = model(text)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            #check if this is a label index or a labe;
            predictions.append(predicted)
            correct += (predicted == target).sum().item()
    f1, acc, rec, prec = get_metrics(labels, predictions)
    return f1, acc, rec, prec
