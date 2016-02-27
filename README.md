# Recommender System

The Recommender System uses an item-item based Collaborative Filtering Model on any dataset that contains users and rated items. The Recommender System will build a model that predicts item ratings and recommend items based on similarity scores.

The recommender system can utilize both personalized and non-personalized models to suggest items.

### Personalized

Weighs ratings of items to make recommendations based on similar items. Uses an item based collaborative filtering method and either cosine similarity or adjusted cosine similarity to determine similar items.

### Non-Personalized

Considers top n highest rated items in the entire dataset and makes recommendations of these items to a user according to what he/she hasn't yet rated.

## Usage

Here's an example that uses the Book Crossing Dataset which is available [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

### Setup

Please download the Book Crossing Dataset from the above website. Load the files using the helper methods available in the recommender_system library.

### Load, Preprocess, and Convert DataFrame to python Dictionaries

```python
import personalized_cf as pcf
import recommender_system as rs

# Load
rated_books = rs.load_item_data('../book_data/BX-Book-Ratings.csv', 'ISBN', 'User-ID')

# Preprocess
min_book_ratings = 2
min_user_ratings = 3
rated_books = rated_books[rated_books['Book-Rating'] != 0]
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)

# Convert to two or three dictionary files. 
# `user_means` dictionary is only required if using the adjusted cosine similarity function.
book_users, user_ratings, user_means = rs.restructure_data(rated_books, 'User-ID', 'Book-Rating', True)
```

### Fit the data

```python
# Number of times two items must be compared before a similarity value can be calculated
min_comparisons = 2
cf = pcf.PersonalizedCF(similarity='adjusted-cosine', threshold=0.5)
cf.fit(items=book_users, users_ratings=user_ratings, min_comparisons=min_comparisons, means=user_means)
```

### Predict

```python
# Parse a user (as a string value) from the original pandas DataFrame to a pandas Series
user = rs.user_id_to_series(USER_ID, rated_books, 'User-ID', 'Book-Rating')
cf.predict_item(user, 'ITEM_ID')
```

Please see the examples folder for more examples.

## Testing

### K-Fold Cross Validation

```python
from sklearn.cross_validation import KFold

total_errors = 0.0
books_to_omit = 2
min_comparisons = 2
n_folds = 10
kf = KFold(len(user_ratings), n_folds=n_folds, random_state=0)

for train_index, test_index in kf:
    X_train, X_test, y_test = rs.split_k_fold(user_ratings, [train_index, test_index], books_to_omit)
    cf = pcf.PersonalizedCF(similarity='adjusted-cosine', threshold=0.5)
    cf.fit(items=book_users, users_ratings=X_train, min_comparisons=min_comparisons, means=user_means)
    y_pred = cf.k_fold_predict(X_test)
    total_errors += rs.mean_absolute_error(y_test, y_pred)

print "Adjusted Cosine: ", total_errors/n_folds
```

### Train_Test_Split

```python
X_train, X_test, y_test =  rs.train_test_split(user_ratings, test_size=0.2, random_state=0)
cf = pcf.PersonalizedCF(similarity='adjusted-cosine')
cf.fit(items=book_users, users_ratings=X_train, min_comparisons=min_comparisons, means=user_means)
y_pred = cf.predict(X_test)
print rs.mean_absolute_error(y_test, y_pred)
```
