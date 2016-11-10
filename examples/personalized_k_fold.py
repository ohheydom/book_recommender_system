import sys
sys.path.append("..")
import personalized_cf as pcf
import recommender_system as rs
from sklearn.cross_validation import KFold


# Load data
rated_books = rs.load_item_data('../book_data/BX-Book-Ratings.csv', 'ISBN')

# Preprocess

# Set variables
min_book_ratings = 4
min_user_ratings = 3

# Unfortunately, the amount of 0s in the dataset was heavily skewing the data.
# This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

# The following function keeps only the books with greater than min_book_ratings
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)

# The following function keeps only the users who rated min_user_ratings or greater books
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)

# K-Fold Cross Validation
n_folds = 10
total_error = 0.0
books_to_omit = 2
min_comparisons = 2
book_users, user_ratings, user_means = rs.restructure_data(rated_books,
                                                           'User-ID',
                                                           'Book-Rating',
                                                           True)
kf = KFold(len(user_ratings), n_folds=n_folds, random_state=5)

for train_index, test_index in kf:
    X_train, X_test, y_test = rs.split_k_fold(user_ratings,
                                              [train_index, test_index],
                                              books_to_omit)
    cf = pcf.PersonalizedCF(similarity='adjusted-cosine', threshold=0.5)
    cf.fit(items=book_users, users_ratings=X_train,
           min_comparisons=min_comparisons,
           means=user_means)
    y_pred = cf.k_fold_predict(X_test)
    mae = rs.mean_absolute_error(y_test, y_pred)
    total_error += mae
    print "Current Mean Absolute Error: ", mae
print "Overall Mean Absolute Error: ", total_error/n_folds

