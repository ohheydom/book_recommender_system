## Recommending Data from Book-Crossing Dataset

The Book-Crossing dataset is available [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

This is an item-item Collaborative Filtering Model. It uses both personalized and non personalized collaborative filtering.

### Personalized
Weighs ratings of books to make recommendations based on similar items. Uses an item based collaborative filtering method and either cosine similarity or adjusted cosine similarity to determine similar items.

### Non-Personalized
Considers top n highest rated books and makes recommendations of these books to a user according to what he/she hasn't yet rated.

### Setup

Please download the Book Crossing Dataset from the above website. Put all three files into the same directory as the python files. 

### Usage

For the Non-personalized Recommender System:

```python non_personalized_cf.py```

Or for the Personalized Recommender System:

An example can be found in main.py

```python main.py```

#### Using the book recommender library:

Given the book crossing dataset, first we need to load, preprocess, and convert to a dictionary:

```
import personalized_cf as pcf
import non_personalized_cf as npcf
import book_recommender_system as brs
from sklearn.cross_validation import KFold

rated_books = brs.load_rating_data('book_data/BX-Book-Ratings.csv')
min_book_ratings = 2
min_user_ratings = 3
rated_books = rated_books[rated_books['Book-Rating'] != 0]
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)
book_users, user_ratings, user_means = brs.restructure_data(rated_books, means=True)
```

To perform K-Fold Cross Validation on the dataset:
```
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
    total_error_adj_cos += mae

    print "Adjusted cosine: ", mae

print "Adjusted Cosine: ", total_error/n_folds
```

Performing tests on a train_test_split is even easier:
```
X_train, X_test, y_test =  brs.train_test_split(user_ratings, test_size=0.2, random_state=0)
cf = pcf.PersonalizedCF(similarity='adjusted-cosine')
cf.fit(books=book_users, ratings=X_train, min_comparisons=min_comparisons, means=user_means)
y_pred = cf.predict(X_test)
print brs.mean_absolute_error(y_test, y_pred)
```
