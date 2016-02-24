# Book Recommender System

The Book Recommender System uses an item-item based Collaborative Filtering Model on the Book-Crossing Dataset to predict item ratings and recommend items based on similarity scores. It utilizes the Book-Crossing dataset, which is available [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

The recommender system can utilize both personalized and non-personalized models to suggest items.

### Personalized

Weighs ratings of books to make recommendations based on similar items. Uses an item based collaborative filtering method and either cosine similarity or adjusted cosine similarity to determine similar items.

### Non-Personalized

Considers top n highest rated books in the entire dataset and makes recommendations of these books to a user according to what he/she hasn't yet rated.

## Setup

Please download the Book Crossing Dataset from the above website. Load the files using the helper methods available in book_recommender_system.

## Usage

### Load, Preprocess, and Convert DataFrame to python Dictionaries

```python
import personalized_cf as pcf
import non_personalized_cf as npcf
import book_recommender_system as brs

# Load
rated_books = brs.load_rating_data('book_data/BX-Book-Ratings.csv')

# Preprocess
min_book_ratings = 2
min_user_ratings = 3
rated_books = rated_books[rated_books['Book-Rating'] != 0]
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)

# Convert to two or three dictionary files. 
# `user_means` dictionary is only required if using the adjusted cosine similarity function.
book_users, user_ratings, user_means = brs.restructure_data(rated_books, means=True)
```

### Fit the data

```python
# Number of times two items must be compared before a similarity value can be calculated
min_comparisons = 2
cf = pcf.PersonalizedCF(similarity='adjusted-cosine', threshold=0.5)
cf.fit(books=book_users, ratings=user_ratings, min_comparisons=min_comparisons, means=user_means)
```

### Predict

```python
# Parse a user (as an integer value) from the original pandas DataFrame to a pandas Series
user = brs.user_id_to_series(USER_ID, rated_books)
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
    X_train, X_test, y_test = brs.split_k_fold(user_ratings, [train_index, test_index], books_to_omit)
    cf = pcf.PersonalizedCF(similarity='adjusted-cosine', threshold=0.5)
    cf.fit(books=book_users, ratings=X_train, min_comparisons=min_comparisons, means=user_means)
    y_pred = cf.k_fold_predict(X_test)
    mae = brs.mean_absolute_error(y_test, y_pred)
    total_errors += mae

    print "Adjusted cosine: ", mae

print "Adjusted Cosine: ", total_errors/n_folds
```

### Train_Test_Split

```python
X_train, X_test, y_test =  brs.train_test_split(user_ratings, test_size=0.2, random_state=0)
cf = pcf.PersonalizedCF(similarity='adjusted-cosine')
cf.fit(books=book_users, ratings=X_train, min_comparisons=min_comparisons, means=user_means)
y_pred = cf.predict(X_test)
print brs.mean_absolute_error(y_test, y_pred)
```
