import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# All data is located here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
# Load data
book_list = pd.read_csv('BX-Book-Ratings.csv', sep=";", quotechar="\"", escapechar="\\")
book_list = book_list.set_index(['User-ID'])

# Find top 20 users with the most ratings 11676
user_list = book_list.index.value_counts()[:20]

# Top user ids and their respective series objects
top_user = np.asarray(user_list.axes).tolist()[0][0]
top_user_books = book_list.loc[book_list.index == top_user]
second_user = np.asarray(user_list.axes).tolist()[0][2] #Using the third user because the second user voted almost all 0s
second_user_books = book_list.loc[book_list.index == second_user]

# Books that both users rated
books_in_common = pd.merge(top_user_books, second_user_books, how='inner', on=['ISBN'])

# Let's look at a bar graph to see what kind of ratings the top user gives
top_rater = book_list.loc[book_list.index == top_user]['Book-Rating'].value_counts().sort_index()
plt.bar(np.asarray(top_rater.axes).tolist()[0], top_rater.values.tolist())
plt.show()

# Now let's see a bar graph of the ratings the third user gives
second_rater = book_list.loc[book_list.index == second_user]['Book-Rating'].value_counts().sort_index()
plt.bar(np.asarray(second_rater.axes).tolist()[0], second_rater.values.tolist())
plt.show()

# Now let's build a scatter plot to see where they agree.
plt.scatter(books_in_common['Book-Rating_x'], books_in_common['Book-Rating_y'], s=20)
plt.show()
