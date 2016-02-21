import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Personalized Item Based Collaborative Filtering
class PersonalizedCF(object):
    def __init__(self, book_list=None, similar_items=defaultdict(dict), threshold=0.5, X_train={}):
        self.book_list_ = book_list
        self.book_comparisons_ = defaultdict(dict) #pd.DataFrame() # DataFrame of all book_comparisons
        self.similar_items_ = similar_items # Dict of items mapped to their similar items and cosine similarity values
        self.threshold_ = threshold # Between 0 and 1. Items where the cosine similarity is greater than or equal to the threshold will be considered similar
        self.X_train_ = X_train

    # fit receives an input of the DataFrame of ratings and the minimum number of comparisons required and creates a DataFrame of all the comparisons and the dict of similar items. If loading in similar_items and X_train when creating the PersonalizedCF object, this step is unnecessary.
    def fit(self, books, ratings, min_comparisons=4):
        self.X_train_ = ratings
        self.i_to_i_s(books, ratings, min_comparisons)

    # i_to_i_s takes in a hash of all the books mapped to users, a hash where users are mapped to their ratings, and minimum comparisons allowed. These hashes can be created with the restructure_data() function on the original rating dataset.
    def i_to_i_s(self, books, users, min_comparisons):
        for book, users_arr in books.iteritems():
            nh = {}
            items = []
            for user in users_arr:
                nh[user] = users[user]
                items.append(users[user].keys())
            self.e_v_f_d(nh, book, np.unique(list(itertools.chain(*items))), min_comparisons)

    # e_v_f_d takes in a dict of users and their ratings,a book title string, an array of book titles to compare, and the minimum comparisons allowed
    def e_v_f_d(self, nh, book_title, items, min_comparisons):
        items = items[items != book_title]
        for i in items:
            v1, v2 = [], []
            for u, v in nh.iteritems():
                if (i in v) == False or (book_title in v) == False:
                    continue
                v1.append(v[book_title])
                v2.append(v[i])
            if len(v1) >= min_comparisons:
                val = cosine_similarity([v1], [v2])[0][0]
                self.book_comparisons_[book_title][i] = val
                if val >= self.threshold_:
                    self.similar_items_[book_title][i] = val


    # transform_rating converts ratings into either -1, 0, or 1 to allow for a wider range of values when computing cosine similarity
    def transform_rating(self, val):
        return val
        if val > 5:
            return 1
        elif val < 5:
            return -1
        return 0

    # predict_item calculates a value for a single item given a Series of a user's ratings for individual books, and the item to rate. To create the proper series, you can use the book_recommender_system library's user_id_to_series.
    def predict_item(self, user, item):
        if not item in self.similar_items_:
            return None
        adder = 0.0
        denom = 0.0
        for book, similarity in self.similar_items_[item].iteritems():
            if book in user:
                adder += user[book] * similarity
                denom += similarity
        if denom == 0:
            return None
        return adder/denom

    # predict calculates the values of books_to_predict based on the users current ratings and the model. It takes an input of a DataFrame with users as the index and columns of books with values representing the users ratings. The second argument is a multidimensional array of isbn strings representing the values you want to predict for each user. The ith array would contain a number of isbn values to predict for the user in the ith row of the user DataFrame.
    def predict(self, users):
        predictions = defaultdict(dict)
        for user, books in users.iteritems():
            for book, _ in books.iteritems():
                adder, denom = 0.0, 0.0
                if not book in self.similar_items_:
                    predictions[user][book] = None
                    continue
                for book2, similarity in self.similar_items_[book].iteritems():
                    if book2 in self.X_train_[user]:
                        adder += self.X_train_[user][book2] * similarity
                        denom += similarity
                if denom == 0:
                    predictions[user][book] = None
                    continue
                predictions[user][book] = adder/denom
        return predictions

    # top_n takes in a Series of a user containing isbns and their corresponding ratings and returns the n most similar items to all the items that a user has liked. To create the proper series, you can use the book_recommender_system library's user_id_to_series.
    def top_n(self, user, n):
        sim_items = []
        for book, rating in user.iteritems():
            for k in self.similar_items_[book].keys():
                if not k in user:
                    sim_items.append(k)

        sim_items = np.unique(sim_items)
        n_sim_items = len(sim_items)
        n = n_sim_items if n_sim_items < n else n
        return sim_items[:n]
