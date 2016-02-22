import personalized_cf as pcf
import non_personalized_cf as npcf
import book_recommender_system as brs
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

# Load data
#all_books = brs.load_book_data('book_data/BX-Books.csv')
rated_books = brs.load_rating_data('book_data/BX-Book-Ratings.csv')
#rated_books = brs.load_rating_data('book_data/book_ratings.csv')
#all_books = brs.load_book_data('book_data/books.csv')

# Preprocess

# Set variables
min_ratings = 4
min_user_votes = 3
min_comparisons = 2

# Unfortunately, the amount of 0s in the dataset was heavily skewing the data. This removes all 0 values, which gives us about a third of the data to utilize
rated_books = rated_books[rated_books['Book-Rating'] != 0]

## The following function keeps only the books with greater than min_ratings
rated_books = rated_books.groupby(rated_books.index).filter(lambda x: len(x) >= min_ratings)

## Remove all ratings where a user voted on 2 or less books
rated_books = rated_books.groupby(rated_books['User-ID']).filter(lambda x: len(x) >= min_user_votes)


# Personalized Collaborative Filtering

#saved_similar_items = pickle.load( open( "similar_items.p", "rb" ) )
book_list, rating_list, user_means = brs.restructure_data(rated_books, True)
X_train, X_test, y_test =  brs.train_test_split(rating_list, test_size=0.1, random_state=33)
cf = pcf.PersonalizedCF(similarity='cosine')
cf.fit(books=book_list, ratings=X_train, min_comparisons=min_comparisons, means=user_means)
pred = cf.predict(X_test)
print brs.mean_absolute_error(y_test, pred)
#pickle.dump(cf.similar_items_, open('similar_items.p', 'wb'))

#saved_similar_items = pickle.load( open( "similar_items.p", "rb" ) )
#cf = pcf.PersonalizedCF(ratings=rated_books, similar_items=saved_similar_items)
#user = brs.user_id_to_series(276680, rated_books)
#pred = cf.predict_item(user, "0688163165")
#print pred
#print cf.top_n(user, 50)


# Non-Personalized Collaborative Filtering

# Let's use the top rater and print out a list of the most popular books that he/she hasn't read yet.
#rater_list = rated_books['User-ID'].value_counts()[:10]
#ncf = npcf.NonPersonalizedCF(rated_books, all_books)
#user = rated_books[rated_books['User-ID'] == np.asarray(rater_list.axes).tolist()[0][0]]
#top_books = ncf.highest_rated_books(n=1500)
#print ncf.recommend_books(user, top_books)

# Graphs

## Find top 20 users with the most ratings
#rater_list = rated_books.index.value_counts()[:20]
#
## Top rater ids and their respective series objects
#top_raters = np.asarray(rater_list.axes).tolist()[0][0:2]
#top_rater_books = rated_books.loc[rated_books.index == top_raters[0]]
#second_rater_books = rated_books.loc[rated_books.index == top_raters[1]]
#
## Books that both raters rate
#books_in_common = pd.merge(top_rater_books, second_rater_books, how='inner', on=['ISBN']).sort_values('Book-Rating_x')
#
## Counts of all the ratings from the top two raters
#top_rater_counts = top_rater_books['Book-Rating'].value_counts().sort_index()
#second_rater_counts = second_rater_books['Book-Rating'].value_counts().sort_index()
#
## Let's look at a bar graph to see what kind of overall ratings the highest rater and the second highest rater give
#width = 0.4
#ind = np.arange(11)
#p1 = plt.bar(np.asarray(top_rater_counts.axes).tolist()[0], top_rater_counts.values.tolist(), width=width, color='#002d72')
#p2 = plt.bar((np.asarray(second_rater_counts.axes) + 0.4).tolist()[0], second_rater_counts.values.tolist(), width=width, color='#ff5910')
#
#plt.xticks(ind + width, ind)
#plt.xlabel('Ratings')
#plt.ylabel('Number of books rated')
#plt.legend(('Highest Rater', 'Second Highest Rater'), loc='upper left')
#plt.show()
#
## Now let's build a scatter plot to see how the users compare
#
## First, we'll create a nested dictionary with first user's ratings as the key, second user's ratings as 
## a nested key, and the value will be an integer of the number of times that specific comparison occurs
#counter = defaultdict(lambda : defaultdict(lambda : 0))
#counter[0][0] = 0
#for _, x_rated, y_rated in np.asarray(books_in_common):
#    counter[x_rated][y_rated] += 1
#
## Create a tuple of all the values to allow us to graph the object
#sizes = []
#for k,v in counter.iteritems():
#    for k1, v1 in v.iteritems():
#        sizes.append((k, k1, v1))
#sizes = zip(*sizes)
#
## Create the scatter plot with areas of circles corresponding to the number of times the comparison occurs
#area = np.pi * (10 * np.array(sizes[2]))
#colors = np.random.rand(len(area))
#plt.scatter(sizes[0], sizes[1], s=area, c=colors, alpha=0.5)
#plt.xlabel('Rater 1 Ratings')
#plt.ylabel('Rater 2 Ratings')
#plt.show()
