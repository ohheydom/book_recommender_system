import book_recommender_system as brs
import numpy as np
import pandas as pd

# Non-Personalized Item Based Collaborative Filtering
# Finds the top n books with the highest ratings and will recommend these books to a specific user depending on what he/she has rated.
# Does not take into account the specifics of user ratings, only that the user has rated and read the book.
class NonPersonalizedCF(object):
    def __init__(self, ratings, book_list=pd.DataFrame()):
        self.ratings_ = ratings
        self.book_list_ = book_list

    # Returns a DataFrame of recommended books based on the top rated books that the user has not seen
    def recommend_books(self, user, top_books):
        l = list(set(top_books) - set(user['ISBN'].values))
        if len(self.book_list_) == 0:
            return l
        return brs.get_book_titles(l, self.book_list_)

    # Returns a Pandas index of the top n rated books
    def highest_rated_books(self, n=50, min_rating=8, max_rating=10):
        b = self.ratings_[self.ratings_['Book-Rating'].isin(np.arange(min_rating, max_rating+1))]
        return b.stack().value_counts()[:n].axes[0][3:]
