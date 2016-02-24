import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
import recommender_system as rs

# Personalized Item Based Collaborative Filtering
class PersonalizedCF(object):
    def __init__(self, item_list=None, threshold=0.5, similarity='cosine', X_train={}):
        self.item_list_ = item_list
        self.item_comparisons_ = defaultdict(dict) # DataFrame of all item_comparisons
        self.similar_items_ = defaultdict(dict) # Dict of items mapped to their similar items and cosine similarity values
        self.threshold_ = threshold # Between 0 and 1. Items where the cosine similarity is greater than or equal to the threshold will be considered similar
        self.X_train_ = X_train # Training Data, only necessary if not fitting and you want to predict
        self.similarity_ = similarity  # Similarity function, either 'cosine' or 'adjusted_cosine'

    # fit(self, items, ratings, min_comparison, means)  receives an input of the two dictionaries:
    # items is a dictionary of items mapped to the user ids of users who have rated them
    # ratings is a dictionary of users mapped to their ratings of items
    # min_comparisons establishes how many comparisons are required before performing a similarity function on the data
    # means is a dict of user means. Used only for adjusted_cosine similarity
    # fit compares the items and sets the item_comparison_ and similar_items_ instance variables
    # If loading in similar_items and X_train when creating the PersonalizedCF object, this step is unnecessary.
    # The three dictionary arguments can be created with the restructure_data() function from the item_recommender_library
    def fit(self, items, ratings, min_comparisons=4, means={}):
        self.X_train_ = ratings
        self.means_ = means
        self.compare_items(items, ratings, min_comparisons)

    # compare_items(items, users, min_comparisons) takes in a dictionary of all the items mapped to users, a dictionary where users are mapped to their ratings, and minimum comparisons allowed. It loops through the items and passes down the comparison to the calculate_similarity function
    def compare_items(self, items, users, min_comparisons):
        for item, users_arr in items.iteritems():
            temp_users = {}
            items = []
            for user in users_arr:
                temp_users[user] = users[user]
                items.append(users[user].keys())
            if self.similarity_ == 'adjusted-cosine':
                self.calculate_similarity_adjusted_cosine(temp_users, item, np.unique(list(itertools.chain(*items))), min_comparisons)
            else:
                self.calculate_similarity(temp_users, item, np.unique(list(itertools.chain(*items))), min_comparisons)

    # calculate_similarity(users, item_title, items, min_comparisons) takes in a dict of users and their ratings, an item title string, an array of item titles to compare, and the minimum comparisons allowed and stores the entire dictionary of computed comparisons (via cosine similarity) in the instance variable item_comparisons_. The method also saves the most similar items according to the given threshold in the instance variable similar_items_.
    def calculate_similarity(self, users, item_title, items, min_comparisons):
        if len(items) == 0:
            return
        items = items[items != item_title]
        for i in items:
            v1, v2 = [], []
            for u, v in users.iteritems():
                if (i in v) == False or (item_title in v) == False:
                    continue
                v1.append(v[item_title])
                v2.append(v[i])
            if len(v1) >= min_comparisons:
                val = rs.cosine_similarity(v1, v2)
                self.item_comparisons_[item_title][i] = val
                if val >= self.threshold_:
                    self.similar_items_[item_title][i] = val

    # calculate_similarity_adjusted_cosine(users, item_title, items, min_comparisons) takes in a dict of users and their ratings, an item title string, an array of item titles to compare, and the minimum comparisons allowed and stores the entire dictionary of computed comparisons (via adjusted cosine similarity) in the instance variable item_comparisons_. The method also saves the most similar items according to the given threshold in the instance variable similar_items_.
    def calculate_similarity_adjusted_cosine(self, users, item_title, items, min_comparisons):
        if len(items) == 0:
            return
        items = items[items != item_title]
        for i in items:
            v1, v2, ua = [], [], []
            for u, v in users.iteritems():
                if (i in v) == False or (item_title in v) == False:
                    continue
                v1.append(v[item_title])
                v2.append(v[i])
                ua.append(self.means_[u])
            if len(v1) >= min_comparisons:
                val = rs.adjusted_cosine_similarity(ua, v1, v2)
                self.item_comparisons_[item_title][i] = val
                if val >= self.threshold_:
                    self.similar_items_[item_title][i] = val

    # predict_item(user, item) calculates a value for a single item given a Series of a user's ratings for individual items and the item to rate. To create the proper series, you can use the recommender_system library's user_id_to_series.
    def predict_item(self, user, item):
        if not item in self.similar_items_:
            return None
        adder, denom = 0.0, 0.0
        for item, similarity in self.similar_items_[item].iteritems():
            if item in user:
                adder += user[item] * similarity
                denom += similarity
        if denom == 0:
            return None
        return adder/denom

    # predict(users) takes an input of a dictionary with users mapped to item ids and empty ratings. Based on the fitted model, predict will calculate values for the empty ratings.
    def predict(self, users):
        predictions = defaultdict(dict)
        for user, items in users.iteritems():
            for item, _ in items.iteritems():
                total, denom = 0.0, 0.0
                if not item in self.similar_items_:
                    predictions[user][item] = None
                    continue
                for item2, similarity in self.similar_items_[item].iteritems():
                    if item2 in self.X_train_[user]:
                        total += self.X_train_[user][item2] * similarity
                        denom += similarity
                if denom == 0:
                    predictions[user][item] = None
                    continue
                predictions[user][item] = total/denom
        return predictions

    # predict(users) takes an input of a dictionary with users mapped to item ids and ratings and n item ids mapped to None. Based on the fitted model, predict will calculate values for the empty ratings.
    def k_fold_predict(self, users):
        predictions = defaultdict(dict)
        for user, items in users.iteritems():
            for item, v in items.iteritems():
                if v == None:
                    total, denom = 0.0, 0.0
                    if not item in self.similar_items_:
                        predictions[user][item] = None
                        continue
                    for item2, similarity in self.similar_items_[item].iteritems():
                        if item2 in users[user] and users[user][item2] != None:
                            total += users[user][item2] * similarity
                            denom += similarity
                    if denom == 0:
                        predictions[user][item] = None
                        continue
                    predictions[user][item] = total/denom
        return predictions

    # top_n takes in a panda Series of a user containing item ids and their corresponding ratings and returns the n most similar items to all the items that a user has rated favorably. To create the proper series, you can use the recommender_system library's user_id_to_series.
    def top_n(self, user, n):
        sim_items = []
        for item, rating in user.iteritems():
            for k in self.similar_items_[item].keys():
                if not k in user:
                    sim_items.append(k)

        sim_items = np.unique(sim_items)
        n_sim_items = len(sim_items)
        n = n_sim_items if n_sim_items < n else n
        return sim_items[:n]
