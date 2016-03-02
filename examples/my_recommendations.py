import sys
sys.path.append("..")
import personalized_cf as pcf
import recommender_system as rs
import pandas as pd

# Load data
rated_books = rs.load_item_data('../book_data/BX-Book-Ratings.csv', 'ISBN', 'User-ID')
all_books = rs.load_item_data('../book_data/BX-Books.csv', 'ISBN')

# Preprocess

# Set variables
min_book_ratings = 8
min_user_ratings = 8

# Unfortunately, the amount of 0s in the dataset was heavily skewing the data. This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

# The following function keeps only the books with greater than min_book_ratings
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_book_ratings)

# The following function keeps only the users who rated min_user_ratings or greater books
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_ratings)

# My ratings
user = pd.Series()
user['0743424425'] = 7 # The Shining
user['0451139712'] = 6 # The Stand
user['0451168089'] = 7 # Salem's Lot
user['0316693707'] = 8 # Kiss The Girls
user['0316693642'] = 8 # Along Came A Spider
user['0553264850'] = 8 # Red Dragon
user['0451160444'] = 7 # Christine
user['0451160444'] = 8 # Needful Things

# Top n
cf = pcf.PersonalizedCF()
book_users, user_ratings, user_means = rs.restructure_data(rated_books, 'User-ID', 'Book-Rating', True)
cf.fit(items=book_users, users_ratings=user_ratings, min_comparisons=4, means=user_means)
recommendations = cf.top_n(user, 20)
print rs.get_item_titles(recommendations, all_books, 'Book-Title')

#What would I think of Carrie??
print "Predicted rating for Carrie : ", cf.predict_item(user, '0671039725')

""" Responses
ISBN                                                Book-Title
0140067477                                      The Tao of Pooh
0451177096                                    Dolores Claiborne
0451184963                                             Insomnia
0451167317                                        The Dark Half
0451163524    The Drawing of the Three (The Dark Tower, Book 2)
0451168615                                        Skeleton Crew
0451176464                                        Gerald's Game
0451169522                                               Misery
0451172817                                       Needful Things
0671039725                                               Carrie
0670858692                                          Rose Madder
0451161343                                              Thinner
0451167805                                          Firestarter
0451160444                                            Christine
0451161351                                                 Cujo
0451170385                                   Four Past Midnight

Predicted rating for Carrie: 7.50
"""
