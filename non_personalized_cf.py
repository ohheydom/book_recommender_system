import book_recommender_system as brs
import numpy as np

# Not-Personalized Collaborative Filtering
# Finds the top n books with the highest ratings and will recommend these books to a specific user depending on what he/she has rated.
# Does not take into account the specifics of user ratings, only that the user has rated and read the book.

# Returns a DataFrame of recommended books based on the top rated books that the user has not seen
def recommend_books(user, top_books, book_titles):
    l = list(set(top_books) - set(user['ISBN'].values))
    return brs.map_isbns_to_book_titles(book_titles, l)

# Returns a Pandas index of the top n rated books
def highest_rated_books(rated_books, n):
    b = rated_books[rated_books['Book-Rating'].isin([8,9,10])]
    return b.stack().value_counts()[:n].axes[0][3:]

# Load data
rated_books = brs.load_rating_data('BX-Book-Ratings.csv')
book_titles = brs.load_book_data('BX-Books.csv')

# Remove ratings of 0
rated_books = rated_books[rated_books['Book-Rating'] != 0]

# Let's use the top rater and print out a list of the most popular books that he/she hasn't read yet.
rater_list = rated_books.index.value_counts()[:10]
user = rated_books[rated_books.index == np.asarray(rater_list.axes).tolist()[0][0]]
top_books = highest_rated_books(rated_books, 100)
print recommend_books(user, top_books, book_titles)['Book-Title']
