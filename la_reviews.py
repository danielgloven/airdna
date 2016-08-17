import pandas as pd
from collections import Counter
import numpy as np
import glob2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

'''
This script will perform a sentiment analysis on Airbnb reviews.

It takes in a .csv of Airbnb reviews. The reviews are tokenized, stemmed, stripped of punctuation, and vectorized.

The reviews are fed in to a model (either a Naive Bayes classifier or SVM). The model is trained on TripAdvisor reviews and associated ratings (on a scale of 1 through 5, 5 being most positive). The TripAdvisor reviews are processed like the Airbnb reviews.

Input: List of Airbnb reviews
Output: List of predicted ratings (on a scale of 1 through 5)

'''


def load_dat(filename):
    '''
    Load an individual TripAdvisor .dat file

    Input: .dat file
    Output: Array of reviews, array of associated ratings
    '''
    content = []
    rating = []
    f = open(filename, "r")
    for line in f:
        if "<Content>" in line:
            content.append(line.strip().replace('<Content>',''))
        if "<Overall>" in line:
            rating.append(line.strip().replace('<Overall>',''))
    return content,rating


def load_training(folder):
    '''
    Load a folder of TripAdvisor .dat files
    '''
    train_df = pd.DataFrame()
    files_in_directory = glob2.glob(folder+'/*.dat')
    num_files = len(files_in_directory)
    for idx,g in enumerate(sorted(files_in_directory),1):
        print '[{} of {}] {}'.format(idx,num_files,g)
        content,rating = load_dat(g)
        temp_df = pd.concat([pd.Series(content),pd.Series(rating)],axis=1)
        # print temp_df
        train_df  = train_df.append(temp_df,ignore_index=True)
    train_df.columns = ['review','rating']
    return train_df

def process_text(collection):
    '''
    Process a corpus of documents (remove stopwords, punc, stem, etc.)
    '''
    porter = PorterStemmer()
    X = []
    stop_words = set(stopwords.words('english'))
    for document in collection:
        try:
            text = document.encode('ascii',errors='ignore')
        except:
            text = document.decode('utf8',errors='ignore')
        text = word_tokenize(text.lower())
        text = [word for word in text if word not in stop_words]
        text = [w for w in text if w.isalpha()]
        text = [porter.stem(w) for w in text]
        # print '*******************'
        text = ' '.join(text)
        # print text
        X.append(text)
    return X


if __name__ == '__main__':

    print 'Loading data ...'
    # train_df = load_training('./Review_Texts')
    # train_df.to_csv('./tripadvisor_reviews.csv')
    train_df = pd.read_csv('tripadvisor_reviews.csv')[:1000]
    train_labels = train_df['rating']

    df = pd.read_csv('la_reviews.csv')
    df.dropna(inplace=True)

    collection = train_df['review']

    print 'Processing Text ...'
    X_train = process_text(collection)

    '''Training Vectorizer'''
    print 'TFIDF Training ...'
    tfidf = TfidfVectorizer(stop_words='english',decode_error='ignore')
    tfidf.fit(pd.Series(X_train))
    train_vectors = tfidf.transform(X_train)

    test_collection = df['review_text'][:10000]

    print 'Processing Text ...'
    X = process_text(test_collection)


    '''Test Vectorizer'''
    print 'TFIDF Test ...'
    # tfidf = TfidfVectorizer(stop_words='english',max_features=train_vectors.shape[1])
    # tfidf.fit(X)
    test_vectors = tfidf.transform(X)

    '''SVM/NB'''
    print 'Fitting SVM/NB ...'
    # classifier = SVC()
    classifier = MultinomialNB()
    classifier.fit(train_vectors, train_labels)
    print 'Predicting SVM ...'
    prediction_rbf = classifier.predict(test_vectors)

    pred_df = pd.DataFrame(prediction_rbf,columns=['rating'])

    #Count by month
    # print df.groupby('review_date').count()['id']
