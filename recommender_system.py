import random
import math
from collections import defaultdict
import pandas as pd
import numpy as np

def load_item_data(location, index):
    """Loads user-rating data from csv file

    Parameters
    ----------
    location : str
        Location of the csv file
    index : str
        Column to be used as the index


    Returns
    -------
    pandas DataFrame
        A DataFrame of the csv file with the index set to specified column name
    """
    return pd.read_csv(location, sep=";", quotechar="\"", escapechar="\\").set_index(index)

def get_item_titles(item_ids, item_list, item_title_column_name):
    """Retrieves the titles of items

    Parameters
    ----------
    item_ids : str or array
        List of item ids or single item id
    item_list : DataFrame
        DataFrame of item ids mapped to item titles
    item_title_column_name : str
        Column name of the item title column


    Returns
    -------
    item_list : pandas Series
        A list of item titles corresponding to item ids
    """
    if type(item_ids) is str:
        return item_list[item_list.index == item_ids][item_title_column_name]
    return item_list[item_list.index.isin(item_ids)][item_title_column_name]

def user_id_to_series(user, ratings, user_column_name, rating_column_name):
    """Builds a pandas Series of a single user from a pandas DataFrame of
    multiple users and ratings of items

    Parameters
    ----------
    user : str
        User id
    ratings : DataFrame
        DataFrame containing all users and their corresponding ratings
    user_column_name : str
        Column name of the user id column
    rating_column_name : str
        Column name of the rating column


    Returns
    -------
    user_series : pandas Series
        A pandas series containing item ids mapped to ratings for a single user
    """
    user_rows = ratings[ratings[user_column_name] == user]
    user_series = pd.Series()
    for item_id, row in user_rows.iterrows():
        user_series[item_id] = row[rating_column_name]
    return user_series

def restructure_data(ratings, user_column_name, rating_column_name, means=False):
    """Parses a pandas DataFrame into dictionaries for faster processing of data

    Parameters
    ----------
    ratings : 
        DataFrame containing all users and their corresponding ratings
    user_column_name : str
        Column name of the user id column
    rating_column_name : str
        Column name of the rating column
    means : bool
        Whether to return a dict of all the users' ratings means. Used for adjusted
        cosine similarity


    Returns
    -------
    items : defaultdict
        Each item mapped to each user that rated it
    user_ratings : defaultdict
        Each user mapped to each item he/she rated and the rating
    means : dict
        Each user mapped to his/her rating means. Used only for adjusted
        cosine similarity
    """
    items = defaultdict(list)
    user_ratings = defaultdict(dict)
    for item, idx_row in ratings.iterrows():
        items[item].append(idx_row[user_column_name])
        user_ratings[idx_row[user_column_name]][item] = idx_row[rating_column_name]
    if means == True:
        user_means = {}
        for k, v in user_ratings.iteritems():
            user_means[k] = np.mean(v.values())
        return (items, user_ratings, user_means)
    return (items, user_ratings)

def cosine_similarity(vec_1, vec_2):
    """Calculates the cosine similarity between two vectors

    Parameters
    ----------
    vec_1 : array
        One dimensional array containing item_1s ratings
    vec_2 : array
        One dimensional array containing item_2s ratings. Must be in same user
        order as vec_1 

    Returns
    -------
    Float
        Value between 0 and 1 corresponding to the similarity between two items
    """
    num, den1, den2 = 0.0, 0.0, 0.0
    for i, _ in enumerate(vec_1):
        v1, v2 = vec_1[i], vec_2[i]
        num += v1 * v2
        den1 += v1**2
        den2 += v2**2
    return num/(math.sqrt(den1)*math.sqrt(den2))

def adjusted_cosine_similarity(user_averages, vec_1, vec_2):
    """Calculates the adjusted cosine similarity between two vectors

    Parameters
    ----------
    user_averages : array
        Contains the means of the users that rated both item_1 and item_2
    vec_1 : array
        One dimensional array containing item_1s ratings. Must be in same user
        order as user_averages
    vec_2 : array
        One dimensional array containing item_2s ratings. Must be in same user
        order as user_averages

    Returns
    -------
    Float
        Value between -1 and 1 corresponding to the similarity between two items
    """
    num, den1, den2 = 0.0, 0.0, 0.0
    for i, _ in enumerate(vec_1):
        ua = user_averages[i]
        val1 = vec_1[i] - ua
        val2 = vec_2[i] - ua
        num += val1 * val2
        den1 += val1**2
        den2 += val2**2
    if den1 == 0 or den2 == 0:
        return 0
    return num/(math.sqrt(den1)*math.sqrt(den2))

def train_test_split(users_ratings, test_size=0.2, random_state=None):
    """Splits data into training and testing datasets.

    Parameters
    ----------
    users_ratings : dict
        Each user mapped to each item he/she rated and the rating
    test_size : float
        Value between 0 and 1 that corresponds to the percentage of each user's
        ratings to set to None
    random_state: int
        Seed of the random number generator to use when shuffling data

    Returns
    -------
    X_train : defaultdict
        Each user mapped to each item he/she rated and the actual rating
    X_test : defaultdict
        Each user mapped to a percentage of items all with a None value
    y_test : defaultdict
        x_test dict but with each item mapped to the actual rating
    """
    X_train, X_test, y_test = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    if random_state != None:
        random.seed(random_state)
    for user, items in users_ratings.iteritems():
        item_keys = items.keys()
        random.shuffle(item_keys)
        test_n = int(len(items) * test_size)
        idx = 0
        for key in item_keys:
            if idx < test_n:
                X_test[user].update({key: None})
                y_test[user].update({key: items[key]})
            else:
                X_train[user].update({key: items[key]})
            idx += 1
    return (X_train, X_test, y_test)

def mean_absolute_error(y_test, y_pred):
    """Calculates the mean absolute error value of two vectors

    Parameters
    ----------
    y_test : dict
        Each user mapped to a percentage of items with the actual ratings
    y_pred : dict
        Each user mapped to a percentage of items with the predicted ratings
        based on the model

    Returns
    -------
    Float
        Error value corresponding to the two arrays. Lower is better.
    """
    total, n = 0.0, 0
    for user, items in y_pred.iteritems():
        for item, rating in items.iteritems():
            if not rating == None:
                n += 1
                total += abs(y_test[user][item] - y_pred[user][item])
    return None if n == 0 else total/n

def split_k_fold(users_ratings, kf, items_to_omit=4):
    """Splits data into training and testing folds by indices

    Parameters
    ----------
    users_ratings : dict
        Each user mapped to each item he/she rated and the rating
    kf : 2 dimensional array
        First array contains indices of training data
        Second array contains indices of test data
    items_to_omit : int
        Number of items to flip to None in each test user's sample

    Returns
    -------
    X_train : defaultdict
        Fold of users mapped to each item he/she rated and the actual rating
    X_test : defaultdict
        Fold of users mapped to all items he/she rated, with some values
        set to None, other values set to the real values
    y_test : defaultdict
        X_test fold of users with each item that had been set to None now set
        to the actual rating
    """
    X_train, X_test, y_test = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    keys = users_ratings.keys()
    for i in kf[0]:
        X_train[keys[i]] = users_ratings[keys[i]]
    for j in kf[1]:
        idx = 0
        for k, v in users_ratings[keys[j]].iteritems():
            if idx < items_to_omit:
                X_test[keys[j]][k] = None
                y_test[keys[j]][k] = v
                idx += 1
            else:
                X_test[keys[j]][k] = v
    return (X_train, X_test, y_test)
