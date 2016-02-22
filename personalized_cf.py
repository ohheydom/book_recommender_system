import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
import book_recommender_system as brs

# Personalized Item Based Collaborative Filtering
class PersonalizedCF(object):
    def __init__(self, book_list=None, similar_items=defaultdict(dict), threshold=0.5, similarity='cosine', X_train={}):
        self.book_list_ = book_list
        self.book_comparisons_ = defaultdict(dict) #pd.DataFrame() # DataFrame of all book_comparisons
        self.similar_items_ = similar_items # Dict of items mapped to their similar items and cosine similarity values
        self.threshold_ = threshold # Between 0 and 1. Items where the cosine similarity is greater than or equal to the threshold will be considered similar
        self.X_train_ = X_train # Training Data, only necessary if not fitting and you want to predict
        self.similarity_ = similarity  # Similarity function, either 'cosine' or 'adjusted_cosine'

    # fit(self, books, ratings, min_comparison, means)  receives an input of the two dictionaries:
    # books is a dictionary of books mapped to the user ids of users who have rated them
    # ratings is a dictionary of users mapped to their ratings of books
    # min_comparisons establishes how many comparisons are required before performing a similarity function on the data
    # means is a dict of user means. Used only for adjusted_cosine similarity
    # fit compares the items and sets the book_comparison_ and similar_items_ instance variables
    # If loading in similar_items and X_train when creating the PersonalizedCF object, this step is unnecessary.
    # The three dictionary arguments can be created with the restructure_data() function from the book_recommender_library
    def fit(self, books, ratings, min_comparisons=4, means={}):
        self.X_train_ = ratings
        self.means_ = means
        self.compare_items(books, ratings, min_comparisons)

    # compare_items(books, users, min_comparisons) takes in a hash of all the books mapped to users, a hash where users are mapped to their ratings, and minimum comparisons allowed. It loops through the items and passes down the comparison to the calculate_similarity function
    def compare_items(self, books, users, min_comparisons):
        for book, users_arr in books.iteritems():
            temp_users = {}
            items = []
            for user in users_arr:
                temp_users[user] = users[user]
                items.append(users[user].keys())
            if self.similarity_ == 'adjusted-cosine':
                self.calculate_similarity_adjusted_cosine(temp_users, book, np.unique(list(itertools.chain(*items))), min_comparisons)
            else:
                self.calculate_similarity(temp_users, book, np.unique(list(itertools.chain(*items))), min_comparisons)

    # calculate_similarity(users, book_title, items, min_comparisons) takes in a dict of users and their ratings, a book title string, an array of book titles to compare, and the minimum comparisons allowed and stores the entire dictionary of computed comparisons (via cosine similarity) in the instance variable book_comparisons_. The method also saves the most similar items according to the given threshold in the instance variable similar_items_.
    def calculate_similarity(self, users, book_title, items, min_comparisons):
        if len(items) == 0:
            return
        items = items[items != book_title]
        for i in items:
            v1, v2 = [], []
            for u, v in users.iteritems():
                if (i in v) == False or (book_title in v) == False:
                    continue
                v1.append(v[book_title])
                v2.append(v[i])
            if len(v1) >= min_comparisons:
                val = brs.cosine_similarity(v1, v2)
                self.book_comparisons_[book_title][i] = val
                if val >= self.threshold_:
                    self.similar_items_[book_title][i] = val

    # calculate_similarity_adjusted_cosine(users, book_title, items, min_comparisons) takes in a dict of users and their ratings, a book title string, an array of book titles to compare, and the minimum comparisons allowed and stores the entire dictionary of computed comparisons (via adjusted cosine similarity) in the instance variable book_comparisons_. The method also saves the most similar items according to the given threshold in the instance variable similar_items_.
    def calculate_similarity_adjusted_cosine(self, users, book_title, items, min_comparisons):
        if len(items) == 0:
            return
        items = items[items != book_title]
        for i in items:
            v1, v2, ua = [], [], []
            for u, v in users.iteritems():
                if (i in v) == False or (book_title in v) == False:
                    continue
                v1.append(v[book_title])
                v2.append(v[i])
                ua.append(self.means_[u])
            if len(v1) >= min_comparisons:
                val = brs.adjusted_cosine_similarity(ua, v1, v2)
                self.book_comparisons_[book_title][i] = val
                if val >= self.threshold_:
                    self.similar_items_[book_title][i] = val

    # predict_item(user, item) calculates a value for a single item given a Series of a user's ratings for individual books and the item to rate. To create the proper series, you can use the book_recommender_system library's user_id_to_series.
    def predict_item(self, user, item):
        if not item in self.similar_items_:
            return None
        adder, denom = 0.0, 0.0
        for book, similarity in self.similar_items_[item].iteritems():
            if book in user:
                adder += user[book] * similarity
                denom += similarity
        if denom == 0:
            return None
        return adder/denom

    # predict(users) takes an input of a dictionary with users mapped to item ids and empty ratings. Based on the fitted model, predict will calculate values for the empty ratings.
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

    def k_fold_predict(self, users):
        predictions = defaultdict(dict)
        for user, books in users.iteritems():
            for book, v in books.iteritems():
                if v == None:
                    total, denom = 0.0, 0.0
                    if not book in self.similar_items_:
                        predictions[user][book] = None
                        continue
                    for book2, similarity in self.similar_items_[book].iteritems():
                        if book2 in users[user] and users[user][book2] != None:
                            total += users[user][book2] * similarity
                            denom += similarity
                    if denom == 0:
                        predictions[user][book] = None
                        continue
                    predictions[user][book] = total/denom
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
