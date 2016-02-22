import random
import math
from collections import defaultdict
import pandas as pd
import numpy as np

''' All data is located here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
Download the files from there before running the following program.'''

# Load book data from csv file
def load_rating_data(location):
    book_list = pd.read_csv(location, sep=";", quotechar="\"", escapechar="\\")
    return book_list.set_index(['ISBN'])

def load_book_data(location):
    book_titles = pd.read_csv(location, sep=";", quotechar="\"", escapechar="\\")
    return book_titles.set_index(['ISBN'])

def get_book_titles(book_titles, book_list):
    if type(book_titles) is str:
        return book_list[book_list.index == book_titles]['Book-Title']
    return book_list[book_list.index.isin(book_titles)]['Book-Title']

def user_id_to_series(user, ratings):
    user_rows = ratings[ratings.index == user]
    user_series = pd.Series()
    for _, row in user_rows.iterrows():
        user_series[row['ISBN']] = row['Book-Rating']
    return user_series

# restructure_data returns a tuple containing a dict of books along with the users that rated it, and a dict of users and all their ratings, and if means=True, a dict of user's ratings means (for adjusted cosine similarity)
def restructure_data(ratings, means=False):
    books = defaultdict(list)
    user_ratings = defaultdict(dict)
    for book, idx_row in ratings.iterrows():
        books[book].append(idx_row['User-ID'])
        user_ratings[idx_row['User-ID']][book] = idx_row['Book-Rating']
    if means == True:
        user_means = {}
        for k, v in user_ratings.iteritems():
            user_means[k] = np.mean(v.values())
        return (books, user_ratings, user_means)
    return (books, user_ratings)

def cosine_similarity(vec_1, vec_2):
    num, den1, den2 = 0.0, 0.0, 0.0
    for i, _ in enumerate(vec_1):
        v1, v2 = vec_1[i], vec_2[i]
        num += v1 * v2
        den1 += v1**2
        den2 += v2**2
    return num/(math.sqrt(den1)*math.sqrt(den2))


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

# Given a dict of users, this will split the users and their ratings according to the test size. It returns a tuple of dicts (X_train, X_test, y_test) where X_test contains users with two of their ratings set to None. y_test is a dict of users and their two actual rating values for their corresponding books. 
# {user : {'b1':'R', 'b2':'R'}, user2 : {'b1':'R', 'b2':R}}
def train_test_split(dict_of_users, test_size=0.2, random_state=None):
    X_train, X_test, y_test = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    if random_state != None:
        random.seed(random_state)
    for user, books in dict_of_users.iteritems():
        book_keys = books.keys()
        random.shuffle(book_keys)
        test_n = int(len(books) * test_size) + 1
        idx = 0
        for key in book_keys:
            if idx < test_n:
                X_test[user].update({key: None})
                y_test[user].update({key: books[key]})
            else:
                X_train[user].update({key: books[key]})
            idx += 1
    return (X_train, X_test, y_test)

# mean_absolute_error takes in a dict of users mapped to books and their respective ratings and the estimated values according to the model. It returns the mean absolute errorvalue. Lower is better.
def mean_absolute_error(y_test, y_pred):
    total, n = 0.0, 0
    for user, books in y_pred.iteritems():
        for book, rating in books.iteritems():
            if not rating == None:
                n += 1
                total += abs(y_test[user][book] - y_pred[user][book])
    return total/n
