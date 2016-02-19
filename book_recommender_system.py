import pandas as pd

''' All data is located here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
Download the files from there before running the following program.'''

# Load book data from csv file
def load_rating_data(location):
    book_list = pd.read_csv(location, sep=";", quotechar="\"", escapechar="\\")
    return book_list.set_index(['User-ID'])

def load_book_data(location):
    book_titles = pd.read_csv(location, sep=";", quotechar="\"", escapechar="\\")
    return book_titles.set_index(['ISBN'])

def get_book_titles(book_titles, book_list):
    if type(book_titles) is str:
        return book_list[book_list.index == book_titles]['Book-Title']
    return book_list[book_list.index.isin(book_titles)]['Book-Title']

def user_id_to_series(user, ratings):
    user_rows = ratings[ratings.index == user]
    user_series = pd.Series()
    for _, row in user_rows.iterrows():
        user_series[row['ISBN']] = row['Book-Rating']
    return user_series
