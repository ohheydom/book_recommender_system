import random
import math
from collections import defaultdict
import pandas as pd
import numpy as np

# All data is located here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

# Load user-rating data from csv file
def load_item_data(location, index):
    return pd.read_csv(location, sep=";", quotechar="\"", escapechar="\\").set_index(index)

# Get item titles from a list of item ids from a DataFrame of item titles mapped to item ids
def get_item_titles(item_titles, item_list, item_column_name):
    if type(item_titles) is str:
        return item_list[item_list.index == item_titles][item_column_name]
    return item_list[item_list.index.isin(item_titles)][item_column_name]

# Given a user and a DataFrame of user-rating data, build a Series containing the user's ratings and the respective item ids
def user_id_to_series(user, ratings, user_column_name, rating_column_name):
    user_rows = ratings[ratings[user_column_name] == user]
    user_series = pd.Series()
    for item_id, row in user_rows.iterrows():
        user_series[item_id] = row[rating_column_name]
    return user_series

# restructure_data returns a tuple containing a dict of items along with the users that rated it, and a dict of users and all their ratings, and if means=True, a dict of the means of users' ratings (for adjusted cosine similarity)
def restructure_data(ratings, user_column_name, rating_column_name, means=False):
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

# Calculate the cosine similarity between two items given two vectors of ratings
def cosine_similarity(vec_1, vec_2):
    num, den1, den2 = 0.0, 0.0, 0.0
    for i, _ in enumerate(vec_1):
        v1, v2 = vec_1[i], vec_2[i]
        num += v1 * v2
        den1 += v1**2
        den2 += v2**2
    return num/(math.sqrt(den1)*math.sqrt(den2))

# Calculate the adjusted cosine similarity between two items given two vectors of ratings and the two users' averages
def adjusted_cosine_similarity(user_averages, vec_1, vec_2):
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

# Given a dict of users, train_test_split will split the users and their ratings according to the test size. It returns a tuple of dicts (X_train, X_test, y_test) where X_test contains users with a percentage of their ratings set to None. y_test is a dict of users and their actual rating values for their corresponding items. 
# {user : {'b1':'R', 'b2':'R'}, user2 : {'b1':'R', 'b2':R}}
def train_test_split(dict_of_users, test_size=0.2, random_state=None):
    X_train, X_test, y_test = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    if random_state != None:
        random.seed(random_state)
    for user, items in dict_of_users.iteritems():
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

# mean_absolute_error takes in a dict of users mapped to items and their respective ratings and the estimated values according to the model. It returns the mean absolute error value. Lower is better.
def mean_absolute_error(y_test, y_pred):
    total, n = 0.0, 0
    for user, items in y_pred.iteritems():
        for item, rating in items.iteritems():
            if not rating == None:
                n += 1
                total += abs(y_test[user][item] - y_pred[user][item])
    return None if n == 0 else total/n

# split_k_fold takes in an input of a dict of users and their ratings of items as well as a 2 dimensional array, first item contains the training indices, second contains the test indices. Items to omit will be how many items in the test set to convert to None. This returns a tuple of X_train, X_test, and y_test label data.
def split_k_fold(ratings, kf, items_to_omit=4):
    X_train, X_test, y_test = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    keys = ratings.keys()
    for i in kf[0]:
        X_train[keys[i]] = ratings[keys[i]]
    for j in kf[1]:
        idx = 0
        for k, v in ratings[keys[j]].iteritems():
            if idx < items_to_omit:
                X_test[keys[j]][k] = None
                y_test[keys[j]][k] = v
                idx += 1
            else:
                X_test[keys[j]][k] = v
    return (X_train, X_test, y_test)
