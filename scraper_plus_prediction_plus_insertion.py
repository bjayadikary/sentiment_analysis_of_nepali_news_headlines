from selenium.webdriver import Chrome

import os
import time
import datetime
import csv
import numpy as np
import pickle
import mysql.connector
from mysql.connector import connection

import utils
from utils import load_news_dataset
from utils import clean_rows_before_proceeding_for_prediction
from utils import stem_headlines
from utils import bag_of_words


def url_queue(host):
    news_categories = ['popular', 'national', 'literature', 'health', 'technology', 'economics', 'international',
                       'entertainment', 'sports']
    # news_categories = ['popular', 'national']
    queue = [host + '/' + str(category) for category in news_categories]

    return queue


def file_to_write(file_path):
    # Getting current datetime for concatenating it with tsv_file_name
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%y-%b-%d-%I%p')  # (shortYear-shortMonthText-Day-Hour(00-12)AM/PM)
    tsv_file_name = file_path + 'news_data_' + formatted_time + '.tsv'

    # Making ready the file to write data into it
    tsv_file = open(tsv_file_name, 'w', newline='')

    # Returning the file object
    return tsv_file, tsv_file_name


def scrape(driver, url_list, tsv_file):
    time.sleep(4)
    print("Scraping in progress from hamropatro.com/news")

    # Looping to get data from each categorized tab
    i = 0
    for url in url_list:
        # Fetching url of a category and loading it to driver
        driver.get(url)

        # Splitting url to fetch category
        category = url.split('/')[4]

        # Fetching specific elements from DOM
        news_card_elements = driver.find_elements_by_xpath(
            "//div[@class='news-list-wrapper']/div[@class='item newsCard']")

        for newsCardElement in news_card_elements:
            news_info_element = newsCardElement.find_element_by_xpath("div[@class='newsInfo']")

            # Extracting newsTitle
            news_title_element = news_info_element.find_element_by_xpath("h2/a[@class='newsTitle']")
            news_title_element_text = news_title_element.text

            # Extracting newsSummary
            news_summary_element = news_info_element.find_element_by_xpath("div/div/div[@class='desc newsSummary']")
            news_summary_element_text = news_summary_element.text

            # Extracting newsSource and newsUploadTime
            news_source_plus_time_element = news_info_element.find_element_by_xpath("div[@class='source newsSource']")
            news_source_plus_time_element_text = news_source_plus_time_element.text

            news_source_text, news_upt_text = news_source_plus_time_element_text.split('\n')[0].split(' . ')

            # Extracting newsLink
            news_link_element = news_info_element.find_element_by_xpath("div[@class='source newsSource']/a")
            news_link_element_href = news_link_element.get_attribute('href')

            # Writing to tsv file
            # header: [newsTitle, newsSummary, newsSource, newsUploadTime, newsLink, category]

            # Getting csv_filewriter object
            writer = csv.writer(tsv_file, delimiter='\t')
            writer.writerow(
                [news_title_element_text, news_summary_element_text, news_source_text, news_upt_text,
                 news_link_element_href, category])

            i += 1
            if i % 50 == 0:
                print(f"Scraped {i} rows")

    time.sleep(4)
    print(f"Scraping complete with {i} rows of data")


def predict(models, dictionary, stemmed_normalized_headlines):
    # maintaining list to store likelihood for each category: log_likelihoods shape = (3, M)
    log_likelihoods = []

    print(f"{len(dictionary)} words in dictionary maintained while training")

    # Feature Extraction : # X_matrix shape = (M, V)
    X_matrix = bag_of_words(stemmed_normalized_headlines, dictionary, multinomial=True)
    print(f"Shape of Feature Matrix: {np.shape(X_matrix)} ")

    for model in models:  # Each model shape: ((2, V), (1,))
        log_likelihood = predict_from_naive_bayes_model(model, X_matrix)  # (M, 2)
        log_likelihoods.append(log_likelihood[:, 1])  # only concatenating probability P(y=1|x) and not the P(y=0|x)

    # predictions from the model
    predictions = np.argmax(log_likelihoods, axis=0)

    return predictions


def predict_from_naive_bayes_model(model, matrix):
    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=-2)

    phi_x, phi_y = model
    log_likelihood = np.sum(matrix[:, None] * np.log(phi_x[None]), axis=-1)
    log_likelihood[:, 0] += np.log(1 - phi_y)
    log_likelihood[:, 1] += np.log(phi_y)

    return np.array(log_likelihood)


def main():
    # Getting the current directory path
    current_dir = os.getcwd()

    ################
    # Scraping part
    ################
    try:
        # Creating webdriver object
        driver = Chrome(executable_path=current_dir + '/chromedriver_linux64/chromedriver')

        # News hoster-the site we used to scrape
        resource_host = 'https://www.hamropatro.com/news'

        # Getting url list to scrape
        url_list = url_queue(resource_host)

        # Getting file object
        file_path = current_dir + '/data_files/scraped_newsdata/'
        scraper_tsv_file, scraper_tsv_file_path = file_to_write(file_path)

        # Initiate scraper
        print("----------------------")
        print("Initiating Scraper...")
        print("----------------------")
        scrape(driver, url_list, scraper_tsv_file)

    except Exception as e:
        print(e)

    finally:
        try:
            driver.quit()
            scraper_tsv_file.close()
        except Exception as e:
            print(e)

    ####################
    # Preprocessing part
    ####################
    # below file taken only for test
    # scraper_tsv_file_path = r'/home/jay/projectWorks/Inception/data_scraping/data/news_data_22-Mar-27-03PM.tsv'

    unprocessed_newsdata = load_news_dataset(scraper_tsv_file_path, all_cols=True)
    unprocessed_headlines = np.array(unprocessed_newsdata)[:, 0]

    time.sleep(4)
    print(f"\n")
    print("----------------------------")
    print("Proceeding Data Cleaning...")
    print("----------------------------")
    # Eliminated headlines in English and headlines having wordcounts less than 4
    cleaned_headlines, leftovers, cleandata_idxs = utils.clean_rows_before_proceeding_for_prediction_updated(
        unprocessed_headlines)
    time.sleep(4)
    print(f"{np.shape(cleaned_headlines)[0]} rows of data remaining after cleaning")

    # print(f"{np.shape(leftovers)[0]} rows are leftovers")

    # Stemming and normalizing the preprocessed_headlines
    time.sleep(4)
    print(f"\n")
    print("--------------------------")
    print(f"Proceeding Stemming.....")
    print("--------------------------")

    stemmed_normalized_headlines = stem_headlines(cleaned_headlines)

    # dumping
    # pickle_out_stemmed_normalized_headlines = open('stemmed_normalized_headlines.pickle', 'wb')
    # pickle.dump(stemmed_normalized_headlines, pickle_out_stemmed_normalized_headlines)

    # loading
    # pickle_in_stemmed_normalized_headlines = open('stemmed_normalized_headlines.pickle', 'rb')
    # stemmed_normalized_headlines = pickle.load(pickle_in_stemmed_normalized_headlines)

    ###################################
    # Model Loading and Prediction part
    ###################################

    time.sleep(4)
    print(f"\n")
    print("------------------------------")
    print("Model Loading and Predictions")
    print("------------------------------")

    # loading models and dictionary from dumped pickle file
    model_dictionary_pickle = open(current_dir+"/data_files/pickled_files_after_fitting/model_dictionary.pickle", "rb")
    models_pickle = open(current_dir+"/data_files/pickled_files_after_fitting/multinomial_models.pickle", "rb")

    dictionary = pickle.load(model_dictionary_pickle)
    models = pickle.load(models_pickle)

    # Model predictions on stemmed_normalized_headlines
    predictions = predict(models, dictionary, stemmed_normalized_headlines)

    time.sleep(5)
    print("\n")
    print("The predictions are:")
    print(predictions)

    negatives_count = 0
    neutrals_count = 0
    positives_count = 0
    for x in predictions:
        if x == 0:
            negatives_count += 1
        elif x == 1:
            neutrals_count += 1
        elif x == 2:
            positives_count += 1
        else:
            continue

    print(f"Total Negatives: {negatives_count}")
    print(f"Total Neutrals: {neutrals_count}")
    print(f"Total Positives: {positives_count}")

    ############################
    # Database connectivity part
    ############################
    time.sleep(5)
    print(f"\n")
    print("--------------------------")
    print("Database Connectivity...")
    print("--------------------------")

    # Making ready news data list for inserting into database
    cleaned_newsdata_for_db = np.array(unprocessed_newsdata)[cleandata_idxs]

    # Expanding dimension of predictions list
    predictions = np.expand_dims(predictions, axis=-1)

    # Appending labels to each newsdata row
    newsdata_for_db = np.append(cleaned_newsdata_for_db, predictions, axis=1)
    newsdata_for_db_only_positive = [newsdata for newsdata in newsdata_for_db if newsdata[6] == '2']

    with open(current_dir+"/data_files/newsdata_for_database/newsdata_for_db"+ scraper_tsv_file_path[-19:], 'w') as file:
        filewriter = csv.writer(file, delimiter='\t')
        for row in newsdata_for_db:
            filewriter.writerow(row)

    try:
        try:
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="root",
                database="comfortdb"

            )
        except mysql.connector.Error as e:
            if e.errno == 1049:
                print("Oops! Seems like database doesn't exist. Create database 'comfortdb' first.")

        if mydb.is_connected():
            time.sleep(4)
            print("Established Connection to database")
            mycursor = mydb.cursor()

            # First, fetching previous newsdata sn to delete any previous data from database before inserting new ones
            sn_fetch_sqlquery = "SELECT sn FROM news_ground_news"
            mycursor.execute(sn_fetch_sqlquery)
            previous_newsdata_sn_tuple = tuple(mycursor.fetchall())

            # Inserting a row at a time
            for a_row in newsdata_for_db_only_positive:
                insertion_sqlquery = "INSERT INTO news_ground_news(newstopic, summary, newssource, uploadtime, link, category, plabel)" \
                                     "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                value = tuple(a_row)
                mycursor.execute(insertion_sqlquery, value)

            print(f"--Inserted {len(newsdata_for_db_only_positive)}  rows of newsdata to database")
            deletion_sqlquery = "DELETE FROM news_ground_news where sn=%s"
            mycursor.executemany(deletion_sqlquery, previous_newsdata_sn_tuple)
            time.sleep(4)
            print(f"--Deleted {len(previous_newsdata_sn_tuple)} rows of previous newsdata from database")

        try:
            mydb.commit()
            time.sleep(4)
            print("--Successfully committed newsdata to database")
        except:
            time.sleep(4)
            print("Commit to database unsuccessful")

    except mysql.connector.Error as error:
        print(error)

    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()
            time.sleep(4)
            print("Connection to database is closed")
            print("\n")


if __name__ == "__main__":
    main()
