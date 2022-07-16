import numpy as np
from sys import argv
import pickle
import os

from utils import stem_headlines
from utils import create_dictionary
from utils import load_news_dataset
from utils import bag_of_words
from utils import refine_to_one_vs_rest_dataset


def fit_naive_bayes_model(matrix, labels, multinomial=True):
    total_headlines_in_label_one = np.sum(labels)
    phi_y = np.sum(labels) / len(labels)

    # Initialize phi_x with shape (2, |vocabulary|)
    phi_x = np.zeros((2, matrix.shape[1]))
    
    phi_x[0] = matrix[labels == 0].sum(axis=0)
    phi_x[1] = matrix[labels == 1].sum(axis=0)

    # Laplace smoothing (for multinomial event)
    if multinomial:
        phi_x += 1
        phi_x = phi_x / phi_x.sum(axis=1, keepdims=True)
    else:
        # Laplace smoothing (for bernoulli event)
        phi_x += 1
        phi_x[0] = phi_x[0] / (len(labels) - total_headlines_in_label_one + 3)
        phi_x[1] = phi_x[1] / (total_headlines_in_label_one + 3)

    return phi_x, phi_y


# fit() takes in stemmed_normalized_headlines and raw_labels
def fit(headlines, raw_labels, multinomial=True):
    # maintaining all binary classifiers and dumping later
    models = []

    # Creating dictionary of words
    dictionary = create_dictionary(headlines)
    print(f"Total dictionary words: {len(dictionary)}")

    if headlines.ndim == 1:
        headlines = np.expand_dims(headlines, -1)

    # Feature Extraction
    X = bag_of_words(headlines, dictionary, multinomial)

    classification_labels = ['0', '1', '2']

    for class_label in classification_labels:
        # refine dataset to fit for one-vs-rest
        headlines, Y = refine_to_one_vs_rest_dataset(headlines, raw_labels, class_label)

        # fit model
        model = fit_naive_bayes_model(X, Y, multinomial)
        models.append(model)

    # Pickling models
    pickle_out_model = open("data_files/pickled_files_after_fitting/multinomial_models.pickle", "wb")
    pickle.dump(models, pickle_out_model)
    pickle_out_model.close()

    # Pickling dictionary
    pickle_out_dictionary = open("data_files/pickled_files_after_fitting/model_dictionary.pickle", "wb")
    pickle.dump(dictionary, pickle_out_dictionary)
    pickle_out_model.close()
    print('##############################')
    print('Dumping Models and dictionary')
    print('##############################')
    print("Model fitted and dumped two files: multinomial_models.pickle, model_dictionary.pickle")
    print("\n")


def main(training_data_file_path=None):
    current_dir = os.getcwd()
    # Training data file path
    if training_data_file_path is None:
        training_data_file_path = current_dir + r'/data_files/training_data/first_training_data_updated_balanced.tsv'


    print('##############################')
    print('Loading Training Data...')
    print('##############################')
    
    # Loading news data
    headlines, raw_labels = load_news_dataset(training_data_file_path, only_headlines_and_labels=True)
    
    print('##############################')
    print('Performing Data Preprocessing')
    print('##############################')
    # Stemming and normalizing
    stemmed_normalized_headlines = stem_headlines(headlines)

    # pickle_out_stemmed_normalized_headlines = open("pickled_files_after_fitting/stemmed_normalized_headlines_training.pickle", "wb")
    # pickle.dump(stemmed_normalized_headlines, pickle_out_stemmed_normalized_headlines)
    # pickle_out_stemmed_normalized_headlines.close()

    # pickle_in_stemmed_normalized_headlines = open("pickled_files_after_fitting/stemmed_normalized_headlines_training.pickle", "rb")
    # stemmed_normalized_headlines = pickle.load(pickle_in_stemmed_normalized_headlines)

    fit(np.array(stemmed_normalized_headlines), np.array(raw_labels), multinomial=True)


if __name__ == "__main__":
    if len(argv) == 2:
        main(argv[1])
    if len(argv) == 1:
        main()
