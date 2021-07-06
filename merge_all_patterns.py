import pandas as pd
import subprocess


if __name__ == '__main__':

    # TODO:
    # Import the new all_patterns
    df1 = pd.read_csv('new_patterns.csv').drop('Unnamed: 0', axis=1)

    # Import merged_patterns
    df0 = pd.read_csv('old_patterns.csv').drop('Unnamed: 0', axis=1)

    # Merge all_patterns
    all_patterns = pd.concat([df0, df1]).reset_index(drop=True)

    # Export merged_patterns to csv
    all_patterns.to_csv('old_patterns.csv')



    