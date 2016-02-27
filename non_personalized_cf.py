import recommender_system as rs
import numpy as np
import pandas as pd

class NonPersonalizedCF(object):
    """Non-Personalized Collaborative Filtering
    Finds the top n items with the highest ratings and will recommend these
    items to a specific user depending on what he/she has rated. Does not take
    into account the specifics of user ratings, only that the user has rated 
    the item.

    Parameters
    ----------
    users_ratings : dict
        Each user mapped to each item he/she rated and the rating
    item_list : DataFrame
        DataFrame of item ids mapped to item titles
    """
    def __init__(self, users_ratings, item_list=pd.DataFrame()):
        self.users_ratings = users_ratings
        self.item_list = item_list

    def recommend_items(self, user_series, top_items, item_title_column_name):
        """Returns an array of recommended items titles based on the top rated items
        that a user has not rated

        Parameters
        ----------
        user_series : pandas Series
            A pandas series containing item ids mapped to ratings for a single user
        top_items : array
            A list of the top rated items
        item_title_column_name : str
            Column name of the item title column

        Returns
        -------
        Array
            A list of item titles that the user has not rated
        """
        l = list(set(top_items) - set(user_series.index.values))
        if len(self.item_list) == 0:
            return l
        return rs.get_item_titles(l, self.item_list, item_title_column_name)

    def highest_rated_items(self, n=50, min_rating=8, max_rating=10, rating_column_name='Rating'):
        """Returns an array of of n highest rated items

        Parameters
        ----------
        n : int
            Number of highest rated items to return
        min_rating : int
            Lowest rating to consider an item to be top rated
        max_rating : int
            Highest rating to consider an item to be top rated
        rating_column_name : str
            Column name of the rating column

        Returns
        -------
        Array
            An array of n top rated item ids
        """
        b = self.users_ratings[self.users_ratings[rating_column_name].isin(np.arange(min_rating, max_rating+1))]
        return b.index.value_counts()[:n].axes[0]
