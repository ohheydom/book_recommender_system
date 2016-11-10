import sys
sys.path.append("..")
import personalized_cf as pcf
import recommender_system as rs


# Load data
rated_books = rs.load_item_data('../book_data/BX-Book-Ratings.csv', 'ISBN')

# Preprocess

# Set variables
min_book_ratings = 2
min_user_ratings = 3

# Unfortunately, the amount of 0s in the dataset was heavily skewing the data.
# This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

# The following function keeps only the books with greater than min_book_ratings
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)

# The following function keeps only the users who rated min_user_ratings or greater books
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)

# Train Test Split
min_comparisons = 2
book_users, user_ratings, user_means = rs.restructure_data(rated_books,
                                                           'User-ID',
                                                           'Book-Rating',
                                                           True)
X_train, X_test, y_test =  rs.train_test_split(user_ratings, test_size=0.2,
                                               random_state=0)
cf = pcf.PersonalizedCF(similarity='cosine', threshold=0.5)
cf.fit(items=book_users, users_ratings=X_train, min_comparisons=min_comparisons)
y_pred = cf.predict(X_test)
print "Mean Absolute Error: %f" % rs.mean_absolute_error(y_test, y_pred)

