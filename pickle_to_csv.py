# -*- coding: utf-8 -*-
import pickle
import pandas as pd


if __name__ == '__main__':

    '''
    with open("/Users/d.e.magno/Datasets/tickers.pickle", 'rb') as f:
        file = pickle.load(f)
    '''

    df = pd.read_pickle("/Users/d.e.magno/Datasets/tickers.pickle")
    #df.to_csv('tickers.csv')
    print(df)
