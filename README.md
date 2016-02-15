## Clustering Data from Book-Crossing Dataset

The Book-Crossing dataset is available [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

This is a basic clustering model of books based on ratings that utilizes collaborative filtering. It uses both personalized and non personalized collaborative filtering.

### Personalized
Weighs ratings of books to make recommendations based on similar users.

### Non-Personalized
Considers top 50 highest rated books and makes recommendations of these books to users who haven't read them.

### Setup

Please download the dataset from the above website. Put all three files into the same directory as ```book_crossing_clusters.py```. Then run

```python book_crossing_clusters.py```

### Todo

* Build a recommender system (If there is a lot of agreement between two users, recommend other 8s, 9s, or 10s to each other)
* Plot and discuss the 0 values and the significance. Perhaps people haven't read these books.
  * If creating a recommender service, remove these from being put on lists
  * If nobody is reading these books, that perhaps means nobody is buying them either 
