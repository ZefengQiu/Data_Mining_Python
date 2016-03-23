#!/usr/bin/python

import csv
import pickle
import os
from preproc_fea_extraction import Preprocessor, FeatureExtractor


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

    # if preprocessed data was stored previously, just load it
    if os.path.isfile('preptrainingdata4k.pickle') and os.path.isfile('preptestdata.pickle'):
        preptrainingdata_f = open('preptrainingdata4k.pickle', 'r')
        preptrainingdata = pickle.load(preptrainingdata_f)

        preptestdata_f = open('preptestdata.pickle', 'r')
        preptestdata = pickle.load(preptestdata_f)

        preptrainingdata_f.close()
        preptestdata_f.close()

    else:
        # preprocess training and test data and store them
        f = open('origintrainingdata.csv', 'r')
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
        f = open('origintestdata.csv', 'r')
        f_csv = csv.reader(f)

        testdata = []

        for row in f_csv:
            if row[0] != '2':  # ignore neutral test data
                if row[0] == '0':
                    testdata.append([row[5], 'neg'])
                else:
                    testdata.append([row[5], 'pos'])

        f.close()

        # take 2000 negative tweets and 2000 positive tweets for training
        sampletraining = trainingdata[:2000] + trainingdata[1598000:]

        # preprocessing step
        preprocessor = Preprocessor()

        for row in sampletraining+testdata:
            row[0] = preprocessor.preprocess(row[0])

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

    # feature extraction and feature set construction
    fea_extractor = FeatureExtractor()
    features = []

    for row in preptrainingdata+preptestdata:
        features.extend(fea_extractor.getfeavector(row[0]))

    print features


    # for row in preptrainingdata+preptestdata:
    #     print row


if __name__ == "__main__":
    main()
