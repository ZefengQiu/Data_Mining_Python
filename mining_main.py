#!/usr/bin/python


def main():

    try:
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
        print "nltk not installed"


if __name__ == "__main__":
    main()