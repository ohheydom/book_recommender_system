import itertools
from collections import defaultdict
import numpy as np
import recommender_system as rs

class PersonalizedCF(object):
    """Personalized Item Based Collaborative Filtering
    Weighs ratings of items to make recommendations based on similar items. 
    Uses an item based collaborative filtering method and either cosine 
    similarity or adjusted cosine similarity to determine similar items.

    Parameters
    ----------
    threshold : float
        Similarity value at which to consider items similar
        If using cosine similarity, value between 0 and 1
        If using adjusted cosine similarity, value between -1 and 1
    similarity : str
        Which similarity function to use
        'cosine' - Cosine Similiarity
        'adjusted-cosine' - Adjusted Cosine Similarity. Utilizes users means
        to average out ratings

    Attributes
    ----------
    X_train_ : dict
        Each user mapped to each item he/she rated and the actual rating
    item_comparisons_ : defaultdict
        All items mapped to items and their similarity values
    similar_items_ :  defaultdict
        All items mapped to their similar items and similarity values
    """

    def __init__(self, threshold=0.5, similarity='cosine'):
        self.item_comparisons_ = defaultdict(dict)
        self.similar_items_ = defaultdict(dict)
        self.threshold_ = threshold
        self.similarity = similarity

    def fit(self, items, users_ratings, min_comparisons=4, means={}):
        """Fits the model using the training data(users_ratings)

        Parameters
        ----------
        items : dict
            Each item mapped to each user that rated it
        users_ratings : dict
            Each user mapped to each item he/she rated and the rating. Used
            as training data
        min_comparisons : int
            Minimum number of comparisons between 2 items before model will
            calculate similarity value
        means : dict
            Each user mapped to his/her rating means. Used only for adjusted
            cosine similarity

        Returns
        -------
        self : object
            returns self
        """
        self.X_train_ = users_ratings
        self.means_ = means
        self.compare_items(items, users_ratings, min_comparisons)
        return self

    def compare_items(self, items, users_ratings, min_comparisons):
        """Iterates through each item and compares it to items that have been
        rated by all the users that rated the item

        Parameters
        ----------
        items : dict
            Each item mapped to each user that rated it
        users_ratings : dict
            Each user mapped to each item he/she rated and the rating
        min_comparisons : int
            Minimum number of comparisons between 2 items before model will
            calculate similarity value

        Returns
        -------
        self : object
            returns self
        """
        for item, users_arr in items.iteritems():
            temp_users = {}
            items = []
            for user in users_arr:
                temp_users[user] = users_ratings[user]
                items.append(users_ratings[user].keys())
            if self.similarity == 'adjusted-cosine':
                self.calculate_similarity_adjusted_cosine(temp_users, item, np.unique(list(itertools.chain(*items))), min_comparisons)
            else:
                self.calculate_similarity(temp_users, item, np.unique(list(itertools.chain(*items))), min_comparisons)
        return self

    def calculate_similarity(self, users_ratings, item_id, items, min_comparisons):
        """Calculates the cosine similarities of all comparable items to the
        given item and saves the values into item_comparisons_. Also saves 
        values into similar_items_ if items are similar according to threshold.

        Parameters
        ----------
        users_ratings : dict
            Each user mapped to each item he/she rated and the rating
        item_id : str
            Item to calculate similarity values with all other comparable items
        items : dict
            Each item mapped to each user that rated it
        min_comparisons : int
            Minimum number of comparisons between 2 items before model will
            calculate similarity value

        Returns
        -------
        self : object
            returns self
        """
        if len(items) == 0:
            return
        items = items[items != item_id]
        for i in items:
            v1, v2 = [], []
            for u, v in users_ratings.iteritems():
                if (i in v) == False or (item_id in v) == False:
                    continue
                v1.append(v[item_id])
                v2.append(v[i])
            if len(v1) >= min_comparisons:
                val = rs.cosine_similarity(v1, v2)
                self.item_comparisons_[item_id][i] = val
                if val >= self.threshold_:
                    self.similar_items_[item_id][i] = val
        return self

    def calculate_similarity_adjusted_cosine(self, users_ratings, item_id, items, min_comparisons):
        """Calculates the adjusted cosine similarities of all comparable items
        to the given item and saves the values into item_comparisons_. Also saves
        values into similar_items_ if items are similar according to threshold.

        Parameters
        ----------
        users_ratings : dict
            Each user mapped to each item he/she rated and the rating
        item_id : str
            Item to calculate similarity values with all other comparable items
        items : dict
            Each item mapped to each user that rated it
        min_comparisons : int
            Minimum number of comparisons between 2 items before model will
            calculate similarity value

        Returns
        -------
        self : object
            returns self
        """
        if len(items) == 0:
            return
        items = items[items != item_id]
        for i in items:
            v1, v2, ua = [], [], []
            for u, v in users_ratings.iteritems():
                if (i in v) == False or (item_id in v) == False:
                    continue
                v1.append(v[item_id])
                v2.append(v[i])
                ua.append(self.means_[u])
            if len(v1) >= min_comparisons:
                val = rs.adjusted_cosine_similarity(ua, v1, v2)
                self.item_comparisons_[item_id][i] = val
                if val >= self.threshold_:
                    self.similar_items_[item_id][i] = val
        return self

    def predict_item(self, user_series, item):
        """Predicts the value that a user would rate an item

        Parameters
        ----------
        user_series : pandas Series
            A pandas series containing item ids mapped to ratings for a single user
        item : str
            Item to predict

        Returns
        -------
        float
            Predicted rating. Returns None if item rating is non calculable
        """
        if not item in self.similar_items_:
            return None
        total, denom = 0.0, 0.0
        for item, similarity in self.similar_items_[item].iteritems():
            if item in user_series:
                total += user_series[item] * similarity
                denom += similarity
        return None if denom == 0 else total/denom

    def predict(self, users_ratings):
        """Predicts the values that users would rate items. Used when testing
        split only includes users with items mapped to None

        Parameters
        ----------
        users_ratings : 
            Each user mapped to items with None values

        Returns
        -------
        predictions : defaultdict
            Users mapped to items from the users_ratings dict and their 
            corresponding predicted ratings, if calculable. If not calculable,
            returns None for specific item/item comparison
        """
        predictions = defaultdict(dict)
        for user, items in users_ratings.iteritems():
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

    def k_fold_predict(self, users_ratings):
        """Predicts the values that users would rate items. Used when testing
        split includes all users' items, including the rated items and the
        items that were switched to None for testing purposes.

        Parameters
        ----------
        users_ratings : dict
            Each user mapped to each item he/she rated, with some mapped to the
            actual rating and others mapped to None

        Returns
        -------
        predictions : defaultdict
            Users mapped to items from the users_ratings dict that held None values
            and their corresponding predicted ratings, if calculable. If not
            calculable, returns None for specific item/item comparison
        """
        predictions = defaultdict(dict)
        for user, items in users_ratings.iteritems():
            for item, v in items.iteritems():
                if v == None:
                    total, denom = 0.0, 0.0
                    if not item in self.similar_items_:
                        predictions[user][item] = None
                        continue
                    for item2, similarity in self.similar_items_[item].iteritems():
                        if item2 in users_ratings[user] and users_ratings[user][item2] != None:
                            total += users_ratings[user][item2] * similarity
                            denom += similarity
                    if denom == 0:
                        predictions[user][item] = None
                        continue
                    predictions[user][item] = total/denom
        return predictions

    def top_n(self, user_series, n):
        """Provides top n most similar items to a user's highly rated items

        Parameters
        ----------
        user_series : pandas Series
            A pandas series containing item ids mapped to ratings for a single user
        n : int
            Number of top n similar items to return

        Returns
        -------
        sim_items : array
            n items that compare favorably to the given user's highly rated items
        """
        sim_items = []
        for item, rating in user_series.iteritems():
            for k in self.similar_items_[item].keys():
                if not k in user_series:
                    sim_items.append(k)
        sim_items = np.unique(sim_items)
        n_sim_items = len(sim_items)
        n = n_sim_items if n_sim_items < n else n
        return sim_items[:n]
