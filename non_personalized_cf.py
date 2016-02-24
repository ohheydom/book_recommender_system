import recommender_system as rs
import numpy as np
import pandas as pd

# Non-Personalized Item Based Collaborative Filtering
# Finds the top n items with the highest ratings and will recommend these items to a specific user depending on what he/she has rated.
# Does not take into account the specifics of user ratings, only that the user has rated the item.
class NonPersonalizedCF(object):
    def __init__(self, ratings, item_list=pd.DataFrame()):
        self.ratings_ = ratings
        self.item_list_ = item_list

    # Returns a DataFrame of recommended items based on the top rated items that the user has not rated
    def recommend_items(self, user, top_items, item_title_column_name):
        l = list(set(top_items) - set(user.index.values))
        print l
        if len(self.item_list_) == 0:
            return l
        return rs.get_item_titles(l, self.item_list_, item_title_column_name)

    # Returns a Pandas index of the top n rated items
    def highest_rated_items(self, n=50, min_rating=8, max_rating=10, rating_column_name='Rating'):
        b = self.ratings_[self.ratings_[rating_column_name].isin(np.arange(min_rating, max_rating+1))]
        return b.index.value_counts()[:n].axes[0]
