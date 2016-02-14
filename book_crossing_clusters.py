import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# All data is located here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
# Load data
book_list = pd.read_csv('BX-Book-Ratings.csv', sep=";", quotechar="\"", escapechar="\\")
book_list = book_list.set_index(['User-ID'])

# Find top 20 users with the most ratings
user_list = book_list.index.value_counts()[:20]

# Top user ids and their respective series objects
top_user = np.asarray(user_list.axes).tolist()[0][0]
top_user_books = book_list.loc[book_list.index == top_user]
second_user = np.asarray(user_list.axes).tolist()[0][2] #Using the third user because the second user voted almost all 0s. I guess he hates books.
second_user_books = book_list.loc[book_list.index == second_user]

# Books that both users rate
books_in_common = pd.merge(top_user_books, second_user_books, how='inner', on=['ISBN']).sort_values('Book-Rating_x')

# Let's look at a bar graph to see what kind of ratings the highest rater and the second highest rater give
top_rater_counts = top_user_books['Book-Rating'].value_counts().sort_index()
second_rater_counts = second_user_books['Book-Rating'].value_counts().sort_index()

width = 0.4
ind = np.arange(11)
p1 = plt.bar(np.asarray(top_rater_counts.axes).tolist()[0], top_rater_counts.values.tolist(), width=width, color='#002d72')
p2 = plt.bar((np.asarray(second_rater_counts.axes) + 0.4).tolist()[0], second_rater_counts.values.tolist(), width=width, color='#ff5910')

plt.xticks(ind + width, ind)
plt.xlabel('Ratings')
plt.ylabel('Number of books rated')
plt.legend(('Highest Rater', 'Second Highest Rater'))
plt.show()

# Now let's build a scatter plot to see where they agree.
plt.scatter(books_in_common['Book-Rating_x'], books_in_common['Book-Rating_y'], s=20)
plt.show()
