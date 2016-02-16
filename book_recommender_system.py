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

def map_isbns_to_book_titles(book_titles, isbns):
    return book_titles[book_titles.index.isin(isbns)]
