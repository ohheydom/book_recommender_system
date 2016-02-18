import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import book_recommender_system as brs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Personalized Item Based Collaborative Filtering

def item_to_item_similarity_1(ratings):
    books = pd.unique(ratings['ISBN'].ravel())
    book_comparisons = pd.DataFrame(index=books)
    for idx_book in books:
        tempMat = pd.DataFrame()
        book_raters = ratings[ratings['ISBN'] == idx_book]
        for idx_user, row_user in book_raters.iterrows():
            tempDF = pd.DataFrame(index=[idx_user])
            for user, row_user2 in ratings[ratings.index == idx_user].iterrows():
                tempDF[row_user2['ISBN']] = row_user2['Book-Rating']
            tempMat = tempMat.append(tempDF)
        if len(tempMat) > 1:
            extract_vectors_from_dataframe(tempMat, idx_book, book_comparisons)
    return book_comparisons.dropna(how='all')

def item_to_item_similarity_2(ratings):
    books = pd.unique(ratings['ISBN'].ravel())
    book_comparisons = pd.DataFrame(index=books)
    for book in books:
        users = ratings[ratings['ISBN'] == book].index.values
        tempMat = pd.DataFrame(index=users)
        for idx_user, row_user in ratings[ratings.index.isin(users)].iterrows():
            tempMat.loc[idx_user, row_user['ISBN']] = row_user['Book-Rating']
        extract_vectors_from_dataframe(tempMat, book, book_comparisons)
    return book_comparisons.dropna(how='all')

def prediction_generator(user, book, book_matrix):
    return nil

def create_dict_of_similar_items(book_matrix, threshold):
    sim = {}
    for row in book_matrix.iterrows():
        for col in row[1].iteritems():
            if col[1] >= threshold:
                sim[row[0]] = {col[0]: col[1]}

    return sim

def extract_vectors_from_dataframe(x, book_title, book_comparisons):
    for name, row in x.iteritems(): 
        x_vec, y_vec = [], []
        if name == book_title:
            continue
        for val, rating in row.iteritems():
            if np.isnan(rating) == False:
                x_vec.append(transform_rating(x[book_title][val]))
                y_vec.append(transform_rating(rating))
        if len(x_vec) > 1:
            book_comparisons.loc[book_title, name] = cosine_similarity([x_vec], [y_vec])

def transform_rating(val):
    return val
    if val > 5:
        return 1
    elif val < 5:
        return -1
    return 0

# restructure_data returns a dict of books along with the users that rated it, and a dict of users and all their ratings
def restructure_data(ratings):
    books = defaultdict(list)
    user_ratings = defaultdict(dict)
    for user, idx_row in ratings.iterrows():
        books[idx_row['ISBN']].append(user)
        user_ratings[user].update({idx_row['ISBN']: idx_row['Book-Rating']})
    return (books, user_ratings)

def i_to_i_s(books, users):
    book_comparisons = pd.DataFrame(index=books.keys())
    for book, users_arr in books.iteritems():
        nh = {}
        items = []
        for user in users_arr:
            nh[user] = users[user]
            items.append(users[user].keys())
        e_v_f_d(book_comparisons, nh, book, np.unique(list(itertools.chain(*items))))
    return book_comparisons.dropna(how='all')


def e_v_f_d(book_comparisons, nh, book_title, items):
    for i in items:
        if i == book_title:
            continue
        v1, v2 = [], []
        for u, v in nh.iteritems():
            if (i in v) == False:
                continue
            v1.append(v[book_title])
            v2.append(v[i])
        if len(v1) > 1:
            book_comparisons.loc[book_title, i] = cosine_similarity([v1], [v2])

# Load data
rated_books = brs.load_rating_data('BX-Book-Ratings.csv')[:30000]
#rated_books = brs.load_rating_data('testing_data/book_ratings.csv')

# Preprocess

# Unfortunately, the amount of 0s in the dataset was heavily skewing the data.
# Perhaps users had simply rated books 0 that they hadn't read yet. We can use this data in another way, which we'll get to later.
# This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

# The following methods remove all the users who only voted for one item and books with only one rating
rated_books = rated_books.groupby([rated_books['ISBN']]).filter(lambda x: len(x) > 1)
rated_books = rated_books.groupby([rated_books.index]).filter(lambda x: len(x) > 1)

# DataFrame calculations
book_matrix = item_to_item_similarity_2(rated_books)
print book_matrix

# Dict calculations
#books, user_ratings = restructure_data(rated_books)
#print i_to_i_s(books, user_ratings)


# Graphs

## Find top 20 users with the most ratings
#rater_list = rated_books.index.value_counts()[:20]
#
## Top rater ids and their respective series objects
#top_raters = np.asarray(rater_list.axes).tolist()[0][0:2]
#top_rater_books = rated_books.loc[rated_books.index == top_raters[0]]
#second_rater_books = rated_books.loc[rated_books.index == top_raters[1]]
#
## Books that both raters rate
#books_in_common = pd.merge(top_rater_books, second_rater_books, how='inner', on=['ISBN']).sort_values('Book-Rating_x')
#
## Counts of all the ratings from the top two raters
#top_rater_counts = top_rater_books['Book-Rating'].value_counts().sort_index()
#second_rater_counts = second_rater_books['Book-Rating'].value_counts().sort_index()
#
## Let's look at a bar graph to see what kind of overall ratings the highest rater and the second highest rater give
#width = 0.4
#ind = np.arange(11)
#p1 = plt.bar(np.asarray(top_rater_counts.axes).tolist()[0], top_rater_counts.values.tolist(), width=width, color='#002d72')
#p2 = plt.bar((np.asarray(second_rater_counts.axes) + 0.4).tolist()[0], second_rater_counts.values.tolist(), width=width, color='#ff5910')
#
#plt.xticks(ind + width, ind)
#plt.xlabel('Ratings')
#plt.ylabel('Number of books rated')
#plt.legend(('Highest Rater', 'Second Highest Rater'), loc='upper left')
#plt.show()
#
## Now let's build a scatter plot to see how the users compare
#
## First, we'll create a nested dictionary with first user's ratings as the key, second user's ratings as 
## a nested key, and the value will be an integer of the number of times that specific comparison occurs
#counter = defaultdict(lambda : defaultdict(lambda : 0))
#counter[0][0] = 0
#for _, x_rated, y_rated in np.asarray(books_in_common):
#    counter[x_rated][y_rated] += 1
#
## Create a tuple of all the values to allow us to graph the object
#sizes = []
#for k,v in counter.iteritems():
#    for k1, v1 in v.iteritems():
#        sizes.append((k, k1, v1))
#sizes = zip(*sizes)
#
## Create the scatter plot with areas of circles corresponding to the number of times the comparison occurs
#area = np.pi * (10 * np.array(sizes[2]))
#colors = np.random.rand(len(area))
#plt.scatter(sizes[0], sizes[1], s=area, c=colors, alpha=0.5)
#plt.xlabel('Rater 1 Ratings')
#plt.ylabel('Rater 2 Ratings')
#plt.show()
