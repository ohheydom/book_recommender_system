from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


''' All data is located here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
Download the files from there before running the following program.'''

# Load data
book_list = pd.read_csv('BX-Book-Ratings.csv', sep=";", quotechar="\"", escapechar="\\")
book_list = book_list.set_index(['User-ID'])

# Preprocess
# Unfortunately, the amount of 0s in the dataset was heavily skewing the data.
# Perhaps users had simply rated books 0 that they hadn't read yet. We can use this data in another way, which we'll get to later.

# This removes all 0 values, which gives us about a third of the data to utilize
book_list = book_list[book_list['Book-Rating'] != 0]

# Find top 20 users with the most ratings
rater_list = book_list.index.value_counts()[:20]

# Top rater ids and their respective series objects
top_raters = np.asarray(rater_list.axes).tolist()[0][0:2]
top_rater_books = book_list.loc[book_list.index == top_raters[0]]
second_rater_books = book_list.loc[book_list.index == top_raters[1]]

# Books that both raters rate
books_in_common = pd.merge(top_rater_books, second_rater_books, how='inner', on=['ISBN']).sort_values('Book-Rating_x')

# Counts of all the ratings from the top two raters
top_rater_counts = top_rater_books['Book-Rating'].value_counts().sort_index()
second_rater_counts = second_rater_books['Book-Rating'].value_counts().sort_index()

# Let's look at a bar graph to see what kind of overall ratings the highest rater and the second highest rater give
width = 0.4
ind = np.arange(11)
p1 = plt.bar(np.asarray(top_rater_counts.axes).tolist()[0], top_rater_counts.values.tolist(), width=width, color='#002d72')
p2 = plt.bar((np.asarray(second_rater_counts.axes) + 0.4).tolist()[0], second_rater_counts.values.tolist(), width=width, color='#ff5910')

plt.xticks(ind + width, ind)
plt.xlabel('Ratings')
plt.ylabel('Number of books rated')
plt.legend(('Highest Rater', 'Second Highest Rater'), loc='upper left')
plt.show()

# Now let's build a scatter plot to see how the users compare

# First, we'll create a nested dictionary with first user's ratings as the key, second user's ratings as 
# a nested key, and the value will be an integer of the number of times that specific comparison occurs
counter = defaultdict(lambda : defaultdict(lambda : 0))
counter[0][0] = 0
for _, x_rated, y_rated in np.asarray(books_in_common):
    counter[x_rated][y_rated] += 1

# Create a tuple of all the values to allow us to graph the object
sizes = []
for k,v in counter.iteritems():
    for k1, v1 in v.iteritems():
        sizes.append((k, k1, v1))
sizes = zip(*sizes)

# Create the scatter plot with areas of circles corresponding to the number of times the comparison occurs
area = np.pi * (10 * np.array(sizes[2]))
colors = np.random.rand(len(area))
plt.scatter(sizes[0], sizes[1], s=area, c=colors, alpha=0.5)
plt.xlabel('Rater 1 Ratings')
plt.ylabel('Rater 2 Ratings')
plt.show()
