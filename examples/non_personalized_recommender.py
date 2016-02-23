import sys
sys.path.append("..")
import personalized_cf as pcf
import non_personalized_cf as npcf
import book_recommender_system as brs
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.cross_validation import KFold

# Load data
rated_books = brs.load_rating_data('../book_data/BX-Book-Ratings.csv')
all_books = brs.load_book_data('../book_data/BX-Books.csv')

# Preprocess

# Set variables
min_book_ratings = 4
min_user_ratings = 3

# Unfortunately, the amount of 0s in the dataset was heavily skewing the data. This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

# The following function keeps only the books with greater than min_book_ratings
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)

# The following function keeps only the users who rated min_user_ratings or greater books
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)

# Find the top 10 users with the most ratings 
rater_list = rated_books['User-ID'].value_counts()[:10]

# Find the highested rated books
ncf = npcf.NonPersonalizedCF(rated_books, all_books)
top_books = ncf.highest_rated_books(n=50, min_rating=8, max_rating=10)

# Select the user with the most ratings
user = rated_books[rated_books['User-ID'] == np.asarray(rater_list.axes).tolist()[0][0]]

# Print the top books that the user hasn't rated
print ncf.recommend_books(user, top_books)
