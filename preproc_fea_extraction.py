#!/usr/bin/python

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def preprocess(document):
        document = unicode(document, errors='ignore')
        document = document.lower()
        document = re.sub(r'http\S*', '', document)  # urls
        document = re.sub(r'@\S*', '', document)  # @ tag
        document = re.sub(r'#\S*', '', document)  # hash tags
        document = re.sub(r'(\w)\1{2,}', r'\1\1', document)  # e.g. looovvveee -> loovvee

        return document


class FeatureExtractor:
    nltk_stop_words = set(stopwords.words('english'))  # default stop words list
    porterstemmer = PorterStemmer()  # default stemmer
    wordnetlemmatizer = WordNetLemmatizer()  # default lemmatizer

    def __init__(self, stop_words=None, stemmer=None, lemmatizer=None):
        if stop_words is None:
            self.stop_words = FeatureExtractor.nltk_stop_words
        else:
            self.stop_words = stop_words

        if stemmer is None:
            self.stemmer = FeatureExtractor.porterstemmer
        else:
            self.stemmer = stemmer

        if lemmatizer is None:
            self.lemmatizer = FeatureExtractor.wordnetlemmatizer
        else:
            self.lemmatizer = lemmatizer

    def getfeavector(self, document):

        words = word_tokenize(document)

        # remove special chars. e.g. punctuation
        words = [word for word in words
                 if re.match(r'^["\'&/,\.\?!:;\|\$%\^\*\+=`~\\\(\)\[\]\{\}<>_\-]+$', word) is None]

        # get rid of words that contain numbers
        words = [word for word in words
                 if re.match(r'.*\d+.*', word) is None]

        # remove stop words
        words = [word for word in words if word not in self.stop_words]

        # apply lemmatizing
        words = [self.lemmatizer.lemmatize(word) for word in words]

        # apply stemming
        words = [self.stemmer.stem(word) for word in words]

        print words

        return words

    def construct_feaset(self):
        pass
