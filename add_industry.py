import os 
import pandas as pd 

def remove_extension(filename: str) -> str:
    """Remove extension to each filename.
    """
    return os.path.splitext(filename)[0]

def add_csv(filename: str) -> str:
    """Add '.csv' extension to each 
    ticker name.
    """
    return filename + '.csv'


if __name__ == '__main__':

    # Import path
    stocks_path = "/Users/d.e.magno/Datasets/raw_stocks_new"

    # Destination path
    dest_path = "/Users/d.e.magno/Datasets/stocks_with_industry_new"

    # Add to list tickers with downloaded stocks
    downloaded_tickers_csv = os.listdir(stocks_path)
    
    # Remove extensions from tickers filename
    downloaded_tickers = list(map(remove_extension,
                                  downloaded_tickers_csv))

    # Import tickers with corresponding industry data
    df_tickers_industry = pd.read_csv("/Users/d.e.magno/Datasets/tickers/ticker-with-industry.csv",
                                      encoding="ISO-8859-1").dropna()

    # Convert the above dataframe to dict for easier lookup
    tickers_industry_dict = dict(zip(df_tickers_industry.Ticker,
                                     df_tickers_industry['Category Name']))

    # Add industry in the 'Industry' column to each ticker dataframe
    for i in downloaded_tickers:
        for j in tickers_industry_dict:
            if i == j:
                print("Adding industry in {}".format(i))
                df = pd.read_csv(os.path.join(stocks_path, add_csv(i)))
                df['Industry'] = tickers_industry_dict[i]
                df.to_csv(os.path.join(dest_path, add_csv(i)))
                print("Done.")
                print()

    print("Finished.")



# TODO: File manager to move and delete files (stocks)
# TODO: Automate data pipeline with just one script