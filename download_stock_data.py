#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Get Stock Data
"""

import time
import pandas as pd
import yfinance as yf 


if __name__ == '__main__':

    
    # Path to file
    # TODO: make directory if directory doesn't exist
    f_file = "/Users/d.e.magno/Datasets/raw_stocks_new.csv"

    # TODO: need to check which is already downloaded
    stock_file = pd.read_csv('/Users/d.e.magno/Datasets/tickers/generic.csv')
    stock_list = stock_file.Ticker

    start_timeA = time.time() 
    for stock in stock_list:
        try:
            start_timeB = time.time()
            print("Downloading {}...".format(stock))
            yf.Ticker(stock).history(period="max").to_csv(
                f_file.format(stock))
            time.sleep(10)
            end_timeB = time.time()
            print("Time elapsed:", end_timeB - start_timeB)
            print()
        except Exception as ex:
            pass
        except KeyboardInterrupt as ex:
            break

    print("Finished.")

    end_timeA = time.time()
    print("Total time elapsed:", end_timeA - start_timeA)


