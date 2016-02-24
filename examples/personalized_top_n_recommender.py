import sys
sys.path.append("..")
import personalized_cf as pcf
import recommender_system as rs

# Load data
rated_books = rs.load_item_data('../book_data/BX-Book-Ratings.csv', 'ISBN')

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

# Top n
user = rs.user_id_to_series(276680, rated_books, 'User-ID', 'Book-Rating')
cf = pcf.PersonalizedCF()
book_users, user_ratings, user_means = rs.restructure_data(rated_books, 'User-ID', 'Book-Rating', True)
cf.fit(items=book_users, ratings=user_ratings, min_comparisons=4, means=user_means)
print cf.top_n(user, 50)
