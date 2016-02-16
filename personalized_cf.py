from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import book_recommender_system as brs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Personalized Item Based Collaborative Filtering
# Creates features for rankings of top 50 most ranked books, not necessarily most popular
# Each sample will look like the following
# UserId Book-1-Rating Book-2-Rating Book-3-Rating ... Book-n-Rating
# I can then cluster users according to how similarly their features line up.

def recommend_books(user, top_books, book_titles):
    return nil

def item_to_item_similarity(books, users):
    book_comparisons = {}
    for idx_book, row_books in books[:100].iterrows():
        tempMat = pd.DataFrame()
        for idx_user, row_user in users[users['ISBN'] == idx_book].iterrows():
            tempDF = pd.DataFrame(index=[idx_user])
            for user, row_user2 in users[users.index == idx_user].iterrows():
                tempDF[row_user2['ISBN']] = row_user2['Book-Rating']
            tempMat = tempMat.append(tempDF)
        if len(tempMat) > 0:
            print extract_vectors_from_dataframe(tempMat, idx_book)
    return 1

def extract_vectors_from_dataframe(x, book_title):
    new_vec = []
    for name, row in x.iteritems(): 
        temp_vec = []
        if name == book_title:
            continue
        for val, rating in row.iteritems():
            if np.isnan(x[book_title][val]) == False and np.isnan(rating) == False:
                temp_vec.append([x[book_title][val], rating])
        if len(temp_vec) > 1:
            new_vec.append(np.array(temp_vec).T)
    return new_vec

# Load data
#rated_books = brs.load_rating_data('BX-Book-Ratings.csv')
#book_data = brs.load_book_data('BX-Books.csv')

rated_books = brs.load_rating_data('testing_data/book_ratings.csv')
book_data = brs.load_book_data('testing_data/books.csv')

# Preprocess
# Unfortunately, the amount of 0s in the dataset was heavily skewing the data.
# Perhaps users had simply rated books 0 that they hadn't read yet. We can use this data in another way, which we'll get to later.
# This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0].drop([11676])

item_to_item_similarity(book_data, rated_books)

# Find top 20 users with the most ratings
rater_list = rated_books.index.value_counts()[:20]

# Top rater ids and their respective series objects
top_raters = np.asarray(rater_list.axes).tolist()[0][0:2]
top_rater_books = rated_books.loc[rated_books.index == top_raters[0]]
second_rater_books = rated_books.loc[rated_books.index == top_raters[1]]

# Books that both raters rate
books_in_common = pd.merge(top_rater_books, second_rater_books, how='inner', on=['ISBN']).sort_values('Book-Rating_x')

# Counts of all the ratings from the top two raters
top_rater_counts = top_rater_books['Book-Rating'].value_counts().sort_index()
second_rater_counts = second_rater_books['Book-Rating'].value_counts().sort_index()

# Let's look at a bar graph to see what kind of overall ratings the highest rater and the second highest rater give
width = 0.4
ind = np.arange(11)
p1 = plt.bar(np.asarray(top_rater_counts.axes).tolist()[0], top_rater_counts.values.tolist(), width=width, color='#002d72')
p2 = plt.bar((np.asarray(second_rater_counts.axes) + 0.4).tolist()[0], second_rater_counts.values.tolist(), width=width, color='#ff5910')

plt.xticks(ind + width, ind)
plt.xlabel('Ratings')
plt.ylabel('Number of books rated')
plt.legend(('Highest Rater', 'Second Highest Rater'), loc='upper left')
plt.show()

# Now let's build a scatter plot to see how the users compare

# First, we'll create a nested dictionary with first user's ratings as the key, second user's ratings as 
# a nested key, and the value will be an integer of the number of times that specific comparison occurs
counter = defaultdict(lambda : defaultdict(lambda : 0))
counter[0][0] = 0
for _, x_rated, y_rated in np.asarray(books_in_common):
    counter[x_rated][y_rated] += 1

# Create a tuple of all the values to allow us to graph the object
sizes = []
for k,v in counter.iteritems():
    for k1, v1 in v.iteritems():
        sizes.append((k, k1, v1))
sizes = zip(*sizes)

# Create the scatter plot with areas of circles corresponding to the number of times the comparison occurs
area = np.pi * (10 * np.array(sizes[2]))
colors = np.random.rand(len(area))
plt.scatter(sizes[0], sizes[1], s=area, c=colors, alpha=0.5)
plt.xlabel('Rater 1 Ratings')
plt.ylabel('Rater 2 Ratings')
plt.show()
