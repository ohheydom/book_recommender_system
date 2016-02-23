import sys
sys.path.append("..")
import personalized_cf as pcf
import book_recommender_system as brs
from sklearn.cross_validation import KFold

# Load data
rated_books = brs.load_rating_data('../book_data/BX-Book-Ratings.csv')

# Preprocess

# Set variables
min_book_ratings = 4
min_user_ratings = 3

## Unfortunately, the amount of 0s in the dataset was heavily skewing the data. This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

### The following function keeps only the books with greater than min_book_ratings
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)

### The following function keeps only the users who rated min_user_ratings or greater books
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)


# Personalized Collaborative Filtering

# Train Test Split

min_comparisons = 4
book_users, user_ratings, user_means = brs.restructure_data(rated_books, True)
X_train, X_test, y_test =  brs.train_test_split(user_ratings, test_size=0.2, random_state=0)
cf = pcf.PersonalizedCF(similarity='adjusted-cosine')
cf.fit(books=book_users, ratings=X_train, min_comparisons=min_comparisons, means=user_means)
y_pred = cf.predict(X_test)
print brs.mean_absolute_error(y_test, y_pred)
