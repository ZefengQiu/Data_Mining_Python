#!/usr/bin/python

import csv
import pickle
import os
import random
from preproc_fea_extraction import Preprocessor, FeatureExtractor


def main():

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

    if os.path.isfile('trainingfeaset4k.pickle') \
            and os.path.isfile('testfeaset.pickle')\
            and os.path.isfile('word_features.pickle'):
        trainingfeaset_f = open('trainingfeaset4k.pickle', 'r')
        trainingfeaset = pickle.load(trainingfeaset_f)

        testfeaset_f = open('testfeaset.pickle', 'r')
        testfeaset = pickle.load(testfeaset_f)

        trainingfeaset_f.close()
        testfeaset_f.close()

    else:
        # feature extraction and feature set construction and store them
        fea_extractor = FeatureExtractor()
        all_words = []

        for row in preptrainingdata+preptestdata:
            all_words.extend(fea_extractor.getfeavector(row[0]))

        word_features = fea_extractor.getfeatures(all_words, 5000)

        trainingfeaset = [(fea_extractor.construct_feaset(row[0], word_features), row[1]) for row in preptrainingdata]
        testfeaset = [(fea_extractor.construct_feaset(row[0], word_features), row[1]) for row in preptestdata]

        random.shuffle(trainingfeaset)
        random.shuffle(testfeaset)

        save_documents = open('word_features.pickle', 'w')
        pickle.dump(word_features, save_documents)
        save_documents.close()

        save_documents = open('trainingfeaset4k.pickle', 'w')
        pickle.dump(trainingfeaset, save_documents)
        save_documents.close()

        save_documents = open('testfeaset.pickle', 'w')
        pickle.dump(testfeaset, save_documents)
        save_documents.close()

    # train classifiers and classify test data


if __name__ == "__main__":
    main()
