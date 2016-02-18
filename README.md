## Clustering Data from Book-Crossing Dataset

The Book-Crossing dataset is available [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

This is a basic clustering model of books based on ratings that utilizes collaborative filtering. It uses both personalized and non personalized collaborative filtering.

### Personalized
Weighs ratings of books to make recommendations based on similar items. Uses an item based collaborative filtering method and a cosine similarity score to determine similar items.

### Non-Personalized
Considers top n highest rated books and makes recommendations of these books to a user according to what he/she hasn't yet rated.

### Setup

Please download the Book Crossing Dataset from the above website. Put all three files into the same directory as the python files. 

### Usage

For the Non-personalized Recommender System:

```python non_personalized_cf.py```

Or for the Personalized Recommender System:

```python personalized_cf.py```

### Todo

* Build the item based personalized recommender system
* Plot and discuss the 0 values and the significance. Perhaps people haven't read these books.
  * If creating a recommender service, remove these from being put on lists
  * If nobody is reading these books, that perhaps means nobody is buying them either 
