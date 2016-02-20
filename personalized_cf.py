import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Personalized Item Based Collaborative Filtering
class PersonalizedCF(object):
    def __init__(self, book_list=None, similar_items=None, model='dict', threshold=0.5):
        self.book_list_ = book_list
        self.book_comparisons_ = pd.DataFrame() # DataFrame of all book_comparisons
        self.similar_items_ = similar_items # Dict of items mapped to their similar items and cosine similarity values
        self.model_ = model # Which model to use when fitting the data
        self.threshold_ = threshold # Between 0 and 1. Items where the cosine similarity is greater than or equal to the threshold will be considered similar

    # fit receives an input of the DataFrame of ratings and the minimum number of comparisons required and creates a DataFrame of all the comparisons and the dict of similar items. If loading in similar items when creating the PersonalizedCF object, this step is unnecessary.
    def fit(self, ratings, min_comparisons=4):
        if self.model_ == 'dict':
            dict_books, dict_ratings = self.restructure_data(ratings)
            self.i_to_i_s(dict_books, dict_ratings, min_comparisons)
        elif self.model_ == 'data1':
            self.item_to_item_similarity_1(ratings, min_comparisons)
        else:
            self.item_to_item_similarity_2(ratings, min_comparisons)
        self.create_dict_of_similar_items(self.threshold_)

    # item_to_item_similarity takes in a DataFrame of ratings and minimum comparisons allowed
    def item_to_item_similarity_1(self, ratings, min_comparisons):
        books = pd.unique(ratings.index.ravel())
        self.book_comparisons_ = pd.DataFrame(index=books)
        for idx_book in books:
            tempMat = pd.DataFrame()
            book_raters = ratings[ratings.index == idx_book]
            for idx_book, row_user in book_raters.iterrows():
                tempDF = pd.DataFrame(index=[row_user['User-ID']])
                for book, row_user2 in ratings[ratings['User-ID'] == row_user['User-ID']].iterrows():
                    tempDF[book] = row_user2['Book-Rating']
                tempMat = tempMat.append(tempDF)
            if len(tempMat) > 1:
                self.extract_vectors_from_dataframe(tempMat, idx_book, min_comparisons)
        self.book_comparisons_ = self.book_comparisons_.dropna(how='all')

    # item_to_item_similarity takes in a DataFrame of ratings and minimum comparisons allowed
    def item_to_item_similarity_2(self, ratings, min_comparisons):
        books = pd.unique(ratings.index.ravel())
        self.book_comparisons_ = pd.DataFrame(index=books)
        for book in books:
            users = ratings[ratings.index == book]['User-ID'].values
            tempMat = pd.DataFrame(index=users)
            for idx_book, row_user in ratings[ratings['User-ID'].isin(users)].iterrows():
                tempMat.loc[row_user['User-ID'], idx_book] = row_user['Book-Rating']
            self.extract_vectors_from_dataframe(tempMat, book, min_comparisons)
        self.book_comparisons_ = self.book_comparisons_.dropna(how='all')

    # extract_vectors_from_dataframe takes in a DataFrame of users and their ratings, a book title string, and the minimum comparisons allowed
    def extract_vectors_from_dataframe(self, x, book_title, min_comparisons):
        for name, row in x.iteritems(): 
            x_vec, y_vec = [], []
            if name == book_title:
                continue
            for val, rating in row.iteritems():
                if np.isnan(rating) == False:
                    x_vec.append(self.transform_rating(x[book_title][val]))
                    y_vec.append(self.transform_rating(rating))
            if len(x_vec) > min_comparisons:
                self.book_comparisons_.loc[book_title, name] = cosine_similarity([x_vec], [y_vec])

    # restructure_data returns a tuple containing a dict of books along with the users that rated it, and a dict of users and all their ratings
    def restructure_data(self, ratings):
        books = defaultdict(list)
        user_ratings = defaultdict(dict)
        for book, idx_row in ratings.iterrows():
            books[book].append(idx_row['User-ID'])
            user_ratings[idx_row['User-ID']].update({book: self.transform_rating(idx_row['Book-Rating'])})
        return (books, user_ratings)

    # i_to_i_s takes in a hash of all the books mapped to users, a hash where users are mapped to their ratings, and minimum comparisons allowed. These hashes can be created with the restructure_data() function on the original rating dataset.
    def i_to_i_s(self, books, users, min_comparisons):
        self.book_comparisons_ = pd.DataFrame(index=books.keys())
        for book, users_arr in books.iteritems():
            nh = {}
            items = []
            for user in users_arr:
                nh[user] = users[user]
                items.append(users[user].keys())
            self.e_v_f_d(nh, book, np.unique(list(itertools.chain(*items))), min_comparisons)
        self.book_comparisons_ = self.book_comparisons_.dropna(how='all')

    # e_v_f_d takes in a dict of users and their ratings,a book title string, an array of book titles to compare, and the minimum comparisons allowed
    def e_v_f_d(self, nh, book_title, items, min_comparisons):
        items = items[items != book_title]
        for i in items:
            v1, v2 = [], []
            for u, v in nh.iteritems():
                if (i in v) == False:
                    continue
                v1.append(v[book_title])
                v2.append(v[i])
            if len(v1) > min_comparisons:
                self.book_comparisons_.loc[book_title, i] = cosine_similarity([v1], [v2])

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
    def predict(self, users, books_to_predict):
        for user, books in users.iterrows():
            for isbn, rating in books.iteritems():
                {}
        return {}

    # create_dict_of_similar_users takes in a threshold for similarity. It returns a dict with isbns mapped to other isbns and their similarities according to the book calculations done with i_to_i_s
    def create_dict_of_similar_items(self, threshold):
        sim = defaultdict(dict)
        for row in self.book_comparisons_.iterrows():
            for col in row[1].iteritems():
                if col[1] >= threshold:
                    sim[row[0]].update({col[0]: col[1]})
        self.similar_items_ = sim

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
