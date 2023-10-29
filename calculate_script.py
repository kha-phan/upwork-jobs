from sklearn.metrics import precision_score, recall_score, f1_score

# it will return precision, recall and f1-score
def calculate_metric(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f'Precision: {precision} | Recall: {recall} | F1-score: {f1}')
