#!/usr/bin/python

import csv
import pickle
import os
import re


def preprocess(document):
        document[0] = unicode(document[0], errors='ignore')
        document[0] = document[0].lower()
        document[0] = re.sub(r'http\S*', '', document[0])  # urls
        document[0] = re.sub(r'@\S*', '', document[0])  # @ tag
        document[0] = re.sub(r'#\S*', '', document[0])  # hash tags
        document[0] = re.sub(r'(\w)\1{2,}', r'\1\1', document[0])  # e.g. looovvveee -> loovvee
        document[0] = re.sub(r'[,\./\?!:;\|\$%\^\*\+=`~\\\(\)\[\]\{\}<>]', ' ', document[0])  # punctuations and chars
        document[0] = re.sub(r'[a-zA-Z0-9]*[&][a-zA-Z0-9]*', '', document[0])  # html entities e.g. &lt; means "<"
        document[0] = document[0].strip()  # remove the spaces at the beginning and end of the string


def main():

    """try:
        import numpy
        print "numpy is install"
    except ImportError:
        print "numpy is not installed"


    try:
        import scipy
        print "scipy is installed"
    except ImportError:
        print "scipy is not installed"


    try:
        import pandas
        print "pandas is installed"
    except ImportError:
        print "pandas is not installed"

    try:
        import nltk
        print "nltk installed"
    except ImportError:
        print "nltk not installed"""""

    # if data was stored previously, just load it
    if os.path.isfile('trainingdata.pickle') and os.path.isfile('testdata.pickle'):
        trainingdata_f = open('trainingdata.pickle', 'r')
        trainingdata = pickle.load(trainingdata_f)

        testdata_f = open('testdata.pickle', 'r')
        testdata = pickle.load(testdata_f)

        trainingdata_f.close()
        testdata_f.close()
    # get tweets and their sentiment labels from training dataset and test dataset
    else:
        f = open('trainingdata.csv', 'r')
        f_csv = csv.reader(f)

        trainingdata = []

        count = 0

        for row in f_csv:
            count += 1
            if count <= 800000:
                trainingdata.append([row[5], 'neg'])
            else:
                trainingdata.append([row[5], 'pos'])

        f.close()

        # get tweets and their sentiment labels from test dataset
        f = open('testdata.csv', 'r')
        f_csv = csv.reader(f)

        testdata = []

        for row in f_csv:
            if row[0] != '2':  # ignore neutral test data
                if row[0] == '0':
                    testdata.append([row[5], 'neg'])
                else:
                    testdata.append([row[5], 'pos'])

        f.close()

        # store training data
        save_documents = open('trainingdata.pickle', 'w')
        pickle.dump(trainingdata, save_documents)
        save_documents.close()

        # store test data
        save_documents = open('testdata.pickle', 'w')
        pickle.dump(testdata, save_documents)
        save_documents.close()

    # if preprocessed data was stored previously, just load it
    if os.path.isfile('preptrainingdata4k.pickle') and os.path.isfile('preptestdata.pickle'):
        preptrainingdata_f = open('preptrainingdata4k.pickle', 'r')
        preptrainingdata = pickle.load(preptrainingdata_f)

        preptestdata_f = open('preptestdata.pickle', 'r')
        preptestdata = pickle.load(preptestdata_f)

        preptrainingdata_f.close()
        preptestdata_f.close()
    # preprocess training and test data and store them
    else:
        # take 2000 negative tweets and 2000 positive tweets for training
        sampletraining = trainingdata[:2000] + trainingdata[1598000:]

        # preprocessing step
        for row in sampletraining+testdata:
            preprocess(row)
            # print row

        preptrainingdata = sampletraining
        preptestdata = testdata

        # store preprocessed training data
        save_documents = open('preptrainingdata4k.pickle', 'w')
        pickle.dump(preptrainingdata, save_documents)
        save_documents.close()

        # store preprocessed test data
        save_documents = open('preptestdata.pickle', 'w')
        pickle.dump(preptestdata, save_documents)
        save_documents.close()


if __name__ == "__main__":
    main()
