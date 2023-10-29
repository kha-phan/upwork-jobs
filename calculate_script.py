""" Get labels """
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def get_labels(model, test_data, column):
    # https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset
    api = KaggleApi()
    api.authenticate()  # Make sure to authenticate with your Kaggle credentials

    # Download the dataset
    api.dataset_download_files('saurabhshahane/urdu-news-dataset', path='./data', unzip=True)

    # Load the dataset
    data = pd.read_csv('data/urdu-news-dataset.csv')  # Adjust the filename if needed

    # Get the true labels
    true_labels = data[column]  # Ex: Headline, News Text..

    # Apply the model to new or test data to get predicted labels
    predicted_labels = model.predict(test_data)
    return true_labels, predicted_labels


""" Calculate metric """
from sklearn.metrics import precision_score, recall_score, f1_score

# it will return precision, recall and f1-score
def calculate_metric(model, test_data, column):
    """
    :param model: your model used training
    :param test_data: your test data
    :param column: the column that you want to get labels
    :return:
    """
    true_labels, predicted_labels = get_labels(model, test_data, column)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f'Precision: {precision} | Recall: {recall} | F1-score: {f1}')
