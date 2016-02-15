import book_crossing_clusters
# Not Personalized Collaborative Filtering
# This finds the top 50 books with the highest ratings and will recommend these books to users who have not read them.
# Does not take into account user ratings

# Load data
book_list = book_crossing_clusters.load_book_data('BX-Book-Ratings.csv')

# Highest rated 50 books
b = book_list[book_list['Book-Rating'].isin([8,9,10])]
print b.stack().value_counts()[:50]
