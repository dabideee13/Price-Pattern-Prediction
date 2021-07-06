# -*- coding: utf-8 -*-
"""
Price Pattern Detection Tools
"""

import warnings
import time
import datetime
from collections import defaultdict
from typing import Optional, Tuple
import statistics as stat

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from dfply import *

import matplotlib.pyplot as plt
import seaborn as sns 

warnings.filterwarnings('ignore')
sns.set()
plt.style.use('seaborn-whitegrid')


def get_max_min(prices: pd.DataFrame,
                smoothing: int = 3,
                window_range: int = 10) -> pd.Series:
    """Returns the integer index values with price,
    for each min/max point.

    Args:
        prices (pd.DataFrame): stock or any historical price data
        smoothing (int, optional): [description]. Defaults to 3.
        window_range (int, optional): [description]. Defaults to 10.

    Returns:
        pd.Series: extremum values, a function of price, could either
                   be minimum or maximum 
    """
    smooth_prices = prices['Close'].rolling(
        window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values,
                              np.greater)[0]
    local_min = argrelextrema(smooth_prices.values,
                              np.less)[0]

    price_local_max_dt = list()
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(
                prices.iloc[
                    i - window_range: i + window_range][
                        'Close'].idxmax())

    price_local_min_dt = list()
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(
                prices.iloc[
                    i - window_range: i + window_range][
                        'Close'].idxmin())

    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    
    max_min.index.name = 'index_date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.index_date.duplicated()]

    p = prices.reset_index()
    max_min['day_num'] = p[p.index.isin(max_min.index_date)].index.values
    max_min = max_min.set_index('day_num')['Close']

    return max_min

def find_patterns(max_min: pd.Series) -> Tuple[defaultdict,
                                               defaultdict]:
    """Detects the 'Double Top' pattern from the extreme values.

    Args:
        max_min (pd.Series): [description]

    Returns:
        Tuple[defaultdict, defaultdict]: [description]
    """
    patterns = defaultdict(list)
    features = defaultdict(list)

    for i in range(5, len(max_min)):
        window = max_min.iloc[i-5:i]
        if window.index[-1] - window.index[0] > 100:
            continue

        a, b, c, d, e = window.iloc[0:5]
        neck_range = np.mean([a, c, e]) * 0.02

        # TODO: add distance (or number of candles) between features
        if (all(diff <= neck_range for diff in (abs(a-c), abs(c-e), abs(a-e)))) and \
            (all(low < b and low < d for low in (a, c, e))) and \
            (abs(b-d) <= np.mean([b,d]) * 0.02):
            patterns['DT'].append((window.index[0], window.index[-1]))
            features['features'].append((a, b, c, d, e))

    return patterns, features

def find_patterns_IHS(max_min: pd.Series) -> Tuple[defaultdict,
                                                   defaultdict]:
    """Detects the defined pattern from the extreme values.

    Args:
        max_min (pd.Series): extreme values extracted from historical
                             price data, could either be min or max

    Returns:
        Tuple[defaultdict, defaultdict]: pattern, features
    """
    patterns = defaultdict(list)
    features = defaultdict(list)

    # Window range is 5 units.
    # specific for inverted head-and-shoulders pattern
    for i in range(5, len(max_min)):
        window = max_min.iloc[i-5:i]

        # Pattern must play out in less than n units.
        # TODO: Why is this 100?
        if window.index[-1] - window.index[0] > 100:
            continue

        a, b, c, d, e = window.iloc[0:5]

        # Inverted head-and-shoulders criteria
        if a < b and c < a and c < e and c < d and e < d and abs(b - d) <= np.mean([b, d]) * 0.02:
            patterns['IHS'].append((window.index[0], window.index[-1]))
            features['features'].append((a, b, c, d, e))

    return patterns, features

def convert_to_timestamp(data: pd.Series,
                         format_: str = "%Y-%m-%d") -> pd.Series:
    """Converts a column of date (in strings) from a dataframe
    to timestamps.

    Args:
        data (pd.Series): [description]
        format_ (str): [description]

    Returns:
        pd.Series: [description]
    """
    
    return data.apply(
        lambda x: time.mktime(
            datetime.datetime.strptime(x, format_).timetuple()))

def plot_minimax_patterns(prices, max_min, patterns,
                          window, ema):
    incr = str((prices.index[1] - prices.index[0]))

    if len(patterns) == 0:
        pass
    else:
        num_pat = len([x for x in patterns.items()][0][1])
        f, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes = axes.flatten()
        prices_ = prices.reset_index()['Close']
        axes[0].plot(prices_)
        axes[0].scatter(max_min.index, max_min, s=100, alpha=.3,
                        color='orange')
        axes[1].plot(prices_)
        
        for name, end_day_nums in patterns.items():
            for i, tup in enumerate(end_day_nums):
                sd = tup[0]
                ed = tup[1]
                axes[1].scatter(max_min.loc[sd:ed].index,
                                max_min.loc[sd:ed].values,
                                s=200, alpha=.3)
                plt.yticks([])
        plt.tight_layout()
        plt.title('{}: EMA {}, Window {} ({} patterns)'.format(incr, ema, window, num_pat))

# TODO: add timeframe in the dataset
def get_results(prices: pd.DataFrame, 
                max_min: pd.Series, 
                patterns: defaultdict, 
                features: defaultdict, 
                ema_: int, 
                window_: int) -> pd.DataFrame:
    """[summary]

    Args:
        prices (pd.DataFrame): [description]
        max_min (pd.Series): [description]
        patterns (defaultdict): [description]
        features (defaultdict): [description]
        ema_ (int): [description]
        window_ (int): [description]

    Returns:
        pd.DataFrame: [description]
    """
    incr = str((prices.index[1] - prices.index[0]))

    # fw_list = [1, 12, 24, 36]
    fw_list = [1, 2, 3]
    results = list()
    if len(patterns.items()) > 0:
        end_dates = [v for k, v in patterns.items()][0]
        features = [v for k, v in features.items()][0]
        for date, feat in zip(end_dates, features):
            param_res = {'increment': incr,
                         'ema': ema_,
                         'window': window_,
                         'date': date}
            for i, f in enumerate(feat):
                param_res['f{}'.format(i + 1)] = f
            for x in fw_list:
                returns = (
                    prices['Close'].pct_change(x).shift(-x).reset_index(
                        drop=True).dropna())
                try:
                    param_res['fw_ret_{}'.format(x)] = returns.loc[date[1]]
                except Exception as e:
                    param_res['fw_ret_{}'.format(x)] = e
            results.append(param_res)

    else:
        param_res = {'increment': incr,
                     'ema': ema_,
                     'window': window_,
                     'date': None}
        for x in fw_list:
            param_res['fw_ret_{}'.format(x)] = None
        results.append(param_res)
    return pd.DataFrame(results)

@dfpipe
def screener(prices: pd.DataFrame, 
             ema_list: Optional = None, 
             window_list: Optional = None, 
             plot: bool = False, 
             results: bool = True):
    """[summary]

    Args:
        prices (pd.DataFrame): [description]
        ema_list (Optional, optional): [description]. Defaults to None.
        window_list (Optional, optional): [description]. Defaults to None.
        plot (bool, optional): [description]. Defaults to False.
        results (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    all_results = pd.DataFrame()

    if ema_list is None:
        ema_list = [3, 10, 20, 30]

    if window_list is None:
        window_list = [3, 10, 20, 30]

    for ema_ in ema_list:
        for window_ in window_list:
            max_min = get_max_min(prices, smoothing=ema_,
                                    window_range=window_)
            patterns, features = find_patterns(max_min)

            if plot == True:
                plot_minimax_patterns(
                    prices, max_min, patterns, window_, ema_)

            if results == True:
                all_results = pd.concat(
                    [all_results, get_results(
                        prices, max_min, patterns,
                        features, ema_, window_)], axis=0)

    if results == True:
        return all_results.reset_index(
            drop=True).dropna().reset_index().drop(
                ['index'], axis=1)

def get_index_range(df_patterns_: pd.DataFrame,
                    features_: pd.Series) -> tuple:
    """Given the features,
    determine the index range."""

    if df_patterns_ is None:
        return 
    if len(df_patterns_) == 0:
        return

    to_check = list()
    checker = list()
    for i, f in enumerate(features_):
        vars = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        indexer = df_patterns_.index[
                    df_patterns_[
                        vars[i]] == features_.values[i]].values
        
        # TODO: handle for all multiple indices
        # TODO: to_check changed to append, instead of assignment
        if len(indexer) > 1:
            to_check.append(indexer)
        
        if len(indexer) == 1:
            checker.append(int(indexer))
    
    # TODO: fix
    # TODO: and len(checker) > 1
    # TOOD: else
        
    if len(checker) < 5:
        double_val = list(set(to_check).intersection(set([stat.mode(checker)])))[0]
        checker.append(double_val)

    try:
        check_val = stat.mode(checker)
        if all(idx == check_val for idx in checker):
            return df_patterns_.date[checker[0]]
        elif not all(idx == check_val for idx in checker):
            return df_patterns_.date[checker[0]]
        else: pass
    except:
        pass

def get_index(df_: pd.DataFrame,
              index_range: tuple,
              features_: pd.Series) -> list:
    
    def get_an_index(indices: list,
                     index_range: tuple) -> int:
        for idx in indices:
            if idx in range(index_range[0],
                            index_range[1] + 1):
                return idx
            else: pass
            
    def disintegrate(*features_):
        return [f for feat_ in features_
                  for f in feat_]

    f_indices = list()
    
    for feat in disintegrate(features_):
        indices = df_['Close'][df_['Close'] == feat].index.values
        
        if len(indices) == 1:
            f_indices.append(int(indices))
        if len(indices) > 1:
            f_indices.append(get_an_index(indices, index_range))
        else: pass
    
    return sorted(f_indices)

def max_or_min(df: pd.DataFrame,
               df_patterns_: pd.DataFrame,
               features: pd.Series,
               label_window: int = 10) -> int:
    """A function that labels the given features.
    1 for down, 2 for neutral, and 3 for up.

    Args:
        df (pd.DataFrame): [description]
        features (np.ndarray): [description]
        label_window (int, optional): [description]. Defaults to 10.

    Returns:
        int: [description]
    """
    if df_patterns_ is None:
        return
    if len(df_patterns_) == 0:
        return

    indices = get_index(df, 
                  get_index_range(df_patterns_, features),
                  features) 
    
    look_ahead = df[
        indices[4] + 1: indices[4] + 1 + label_window]['Close'].values
    
    # TODO: account for indices
    if look_ahead.max() > features.max(): return 3, indices
    elif look_ahead.min() < features.min(): return 1, indices
    else: return 2, indices

@dfpipe
def add_label(df_patterns_: pd.DataFrame,
              df: pd.DataFrame) -> pd.DataFrame:
    """Add 'label' column in the results dataframe.

    Args:
        df (pd.DataFrame): [description]
        results (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    if len(df_patterns_) == 0:
        return
    
    if df_patterns_ is None:
        return

    # Assuming that f1, f2, f3, f4, f5 are placed
    # in the dataframe properly and in this order.
    f1_index = df_patterns_.columns.get_loc('f1')
    f5_index = df_patterns_.columns.get_loc('f5')
    
    # Initialize 'label' column
    df_patterns_['label'] = 0

    # TODO:
    #df_patterns_['indices'] = 0
    to_append = ['idx1', 'idx2', 'idx3', 'idx4', 'idx5']

    # TODO: initialize dataframe
    df1 = pd.DataFrame()
    
    # TODO:
    for each in range(len(df_patterns_)):
        minimax = max_or_min(df, df_patterns_,
            df_patterns_.loc[each][f1_index:f5_index + 1])
        df_patterns_['label'][each] = minimax[0]

        # TODO:
        if each == 0:
            df1[to_append] = pd.DataFrame([minimax[1]])
            df_patterns_ = pd.concat([df_patterns_, df1], axis=1)
        else: 
            df_patterns_.loc[each, to_append] = minimax[1]
    
    return df_patterns_

@dfpipe
def copy_(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of dataframe with different
    ID.

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    return df.copy(deep=True)

def import_(path: str) -> pd.DataFrame:
    """Import data which automatically converts
    string-tuple as tuple.

    It is assumed that there's always an
    'Unnamed' column right after the index column.
    This 'Unnamed' column was created due to 
    exporting data to csv file.

    Args:
        path (str): path to file and name of file

    Returns:
        pd.DataFrame: new dataframe with cleaned data
    """
    
    def eval_(df: pd.DataFrame) -> pd.Series:
        """Converts string data to non-string. If
        data was supposedly a tuple, but was
        implicitly converted to string during
        import, this converts it to its 
        'supposed' data type, in this case, tuple.

        Args:
            df (pd.DataFrame): [description]

        Returns:
            pd.Series: [description]
        """
        return df['tmp'].apply(eval)
    
    # TODO: generalize
    return (pd.read_csv(path, index_col=0) >>
              rename(tmp=X.date) >>
              mutate(date=eval_(X)) >>
              drop(X.tmp))

@dfpipe
def get_target(df: pd.DataFrame) -> pd.DataFrame:
    return df.label

@dfpipe
def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """It selects the features part of the whole
    dataframe. This assumes that the features
    are adjacent to each other and in order
    from f1 to f5.
    """
    f1_index = df.columns.get_loc('f1')
    f5_index = df.columns.get_loc('f5')
    return df.iloc[:, 
            f1_index: f5_index + 1]

def pick(to_pick: set) -> int:
    elem_list = list()
    idx_list = list()
    for i, elem in enumerate(to_pick):
        elem_list.append(elem)
        idx_list.append(i)
    idx = np.random.randint(idx_list[0], idx_list[-1]+1)
    return elem_list[idx]


if __name__ == '__main__':

    # TODO: Validate with the usual data, bitcoin

    # Data
    #df = pd.read_csv('BTC-USD.csv')
    #df2 = pd.read_csv('BTC-USD.csv')
    df = pd.read_csv('/Users/d.e.magno/Datasets/stocks/AMZN.csv')

    # Detect patterns
    '''
    patterns2 = (df2 >>
                  screener >>
                  add_label(df2))
    '''
    try:
        patterns = (df >>
                    screener >>
                    add_label(df))
    except:
        print('Skipped.')
        pass

    '''
    if patterns2 is None:
        pd.DataFrame().to_csv('no_data.csv')
    if patterns2 is not None and len(patterns2) != 0:
        patterns2.to_csv('detected_patterns_data.csv')
    '''

    try:
        if patterns is not None:
            patterns.to_csv('detected_patterns_data.csv')
    except:
        pass


# TODO: Loosen finding pattern parameters

# TODO: Add argparse feature
# TODO: Finalize pipeline
# TODO: Understand fw_ret meaning
# TODO: Understand what is the purpose of EMA

