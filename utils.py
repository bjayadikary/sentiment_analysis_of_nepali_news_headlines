import numpy as np
import csv
import re
import my_stemmer_updated
import time


def load_news_headlines(scraped_newsdata_filename):
    headlines = []
    with open(scraped_newsdata_filename, 'r', encoding='utf8') as tsv_file:
        filereader = csv.reader(tsv_file, delimiter='\t')
        for row in filereader:
            headlines.append(row[0])

    return headlines


def load_news_dataset(training_data_file_path, all_cols=False, only_headlines_and_labels=False, only_headlines=False):
    headlines = []
    labels = []
    all_newsdata = []
    with open(training_data_file_path, 'r', encoding='utf8') as tsv_file:
        filereader = csv.reader(tsv_file, delimiter='\t')

        if all_cols: # For loading all columns of a news row
            for row in filereader:
                all_newsdata.append([row[0], row[1], row[2], row[3], row[4], row[5]])
            return all_newsdata

        elif only_headlines_and_labels:  # This is for training_data that has only headlines and labels
            for row in filereader:
                headlines.append(row[0])
                # labels.append(int(row[1]))
                labels.append(row[1])
            return headlines, labels

        elif only_headlines:  # This is to for any dataset to extract only the headlines
            for row in filereader:
                headlines.append(row[0])
            return headlines

        else:
            for row in filereader:  # This is for labelled dataset that has headlines in col0 and labels in col4
                headlines.append(row[0])
                labels.append(int(row[4]))
            return headlines, labels


def clean_rows_before_proceeding_for_prediction(unprocessed_headlines):
    cleaned_headlines = []
    for headline in unprocessed_headlines:
        # firstly, eliminating rows with headlines in English
        if re.search("^[A-Za-z0-9]", headline) is not None:
            continue
        else:
            # Secondly,eliminating rows with headlines having wordcounts less than 4
            word_list = get_words(headline)
            if len(word_list) >= 4:
                cleaned_headlines.append(headline)
            else:
                continue
    return cleaned_headlines


def clean_rows_before_proceeding_for_prediction_updated(unprocessed_headlines):
    cleaned_headlines = []
    leftovers = []
    cleandata_idxs = []
    english_headlines_count = 0
    wordcount_less_than_4_headlines_count = 0

    for i, headline in enumerate(unprocessed_headlines):
        leftover_flag = False

        # firstly, eliminating rows with headlines in English
        if re.search("[A-Za-z]", headline) is None:

            # Secondly,eliminating rows with headlines having wordcounts less than 4
            word_list = get_words(headline)
            if len(word_list) >= 4:
                cleaned_headlines.append(headline)
                cleandata_idxs.append(i)
            else:
                example2 = headline
                wordcount_less_than_4_headlines_count += 1
                leftover_flag = True
        else:
            example1 = headline
            english_headlines_count += 1
            leftover_flag = True

        if leftover_flag:
            leftovers.append(headline)

    time.sleep(5)
    print(f"Eliminated {english_headlines_count} news in English")
    time.sleep(5)
    print(f"Eliminated {wordcount_less_than_4_headlines_count} news having headlines with wordcounts < 4")
    return cleaned_headlines, leftovers, cleandata_idxs


# updated for handling inputs as list or numpy array, rather than only string
def get_words(inputs):
    if type(inputs) == list or np.ndarray:
        sent_string = ''.join(inputs)
        return [word for word in sent_string.split()]
    else:
        return [word for word in inputs.split()]


# stems multiple headlines
def stem_headlines(headlines):
    stemmed_headlines = []
    for i, headline in enumerate(headlines):
        result_from_my_stemmer = my_stemmer_updated.stem_it(headline)
        if i % 10 == 0 and i !=0:
            print(f"{i} Headlines Stemming Complete")
        stemmed_headlines.append([result_from_my_stemmer])

    print(f"{np.shape(headlines)[0]} Headlines Stemming Complete")
    print(f"Stemming Completed.")

    return stemmed_headlines


def create_dictionary(headlines):
    temp_counter = {}
    vocabs = {}
    i = 0
    for headline in headlines:
        words = get_words(headline)
        for word in words:
            if word not in temp_counter:
                temp_counter[word] = 1
            else:
                temp_counter[word] += 1

            # only considering word into dictionary if it's occurence exceeds given threshold
            if temp_counter[word] == 1:
                vocabs[word] = i
                i += 1
    return vocabs


def bag_of_words(headlines, word_dictionary, multinomial):
    N, V = len(headlines), len(word_dictionary)
    data = np.zeros((N, V))
    for i, headline in enumerate(headlines):
        for word in get_words(headline):
            if word in word_dictionary:
                if multinomial:
                    data[i, word_dictionary[word]] += 1 # Multinomial event model
                else:
                    data[i, word_dictionary[word]] = 1 # Bernoulli event model
    return data
    
    
def refine_to_one_vs_rest_dataset(headlines, labels, one_label=None):
    one_vs_rest_labels = []
    if one_label is not None:
        for label in labels:
            one_vs_rest_labels.append(1 if label == one_label else 0)
    else:
        raise TypeError("Missing positional argument: one_label")

    return headlines, np.array(one_vs_rest_labels)

