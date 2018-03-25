import os
import urllib.request
import zipfile
from _commons import warn, error, create_dir_path
import numpy as np

URL_PREFIX = 'http://files.grouplens.org/datasets/movielens/'
VARIANTS = ['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m']
FILENAME_SUFFIX = '.zip'

DATA_DIR = 'data'
DATASET_NAME = 'movielens'
ENCODING = 'utf-8'
STRING_ENCODING = 'ISO-8859-1'

"""
u.data -- The full u data set, 100000 ratings by 943 users on 1682 items. Each user has rated at least 20 movies. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of user id | item id | rating | timestamp. The time stamps are unix seconds since 1/1/1970 UTC

u.info -- The number of users, items, and ratings in the u data set.

u.item -- Information about the items (movies); this is a tab separated list of movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western | The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not; movies can be in several genres at once. The movie ids are the ones used in the u.data data set.

u.genre -- A list of the genres.

u.user -- Demographic information about the users; this is a tab separated list of user id | age | gender | occupation | zip code The user ids are the ones used in the u.data data set.

u.occupation -- A list of the occupations.
"""


class MovieLens:
    def __init__(self, variant='ml-100k',
                 pos_rating_threshold=4,
                 data_augmentation_mode='binary_unknown'):

        np.random.seed(0)
        self.DATA_AUGMENTATION_MODES = ['binary', 'binary_unknown', 'original']
        self.variant = variant
        self.UNKNOWN_RATING_VAL = 0
        self.POSITIVE_RATING_VAL = 1
        self.NEGATIVE_RATING_VAL = -1
        self.data_augmentation_mode = data_augmentation_mode
        self.pos_rating_threshold = pos_rating_threshold  # positive rating threshold

        self._maybe_download_and_extract()
        self.num_users, self.num_items, self.num_ratings = self._get_num_users_items()
        self.genre_names = self._get_genre_names()
        self.num_genres = len(self.genre_names)
        self.orig_R, self.implicit_ratings = self._get_rating_matrix()
        self.R, self.R_mask = self._augment_R(mode=data_augmentation_mode)
        self.item_titles, self.item_genres = self._get_item_info()

        self.user_indexes = np.array(range(self.R.shape[0]))  # order of selection of ratings for batches in next epoch
        np.random.shuffle(self.user_indexes)  # iterate through the rating matrix randomly when selecting next batch

        print('Statistics about self.R:')
        self.get_statistics(self.R)

    def _augment_R(self, mode):
        """
        mode == 'binary'
            R[R < self.pos_rating_threshold] = 0
            R[R >= self.pos_rating_threshold] = 1
        mode == 'binary_unknown':
            Unknown ratings => self.unknown_rating_val
            Positive ratings = ratings >= pos_rating_threshold => 1
            Negative ratings => 0
        mode == 'original':
            pass
        :return: Augmented rating matrix.
        """
        R = np.copy(self.orig_R)
        unknown_rating = 0
        if mode == 'binary':
            R[R < self.pos_rating_threshold] = self.UNKNOWN_RATING_VAL
            R[R >= self.pos_rating_threshold] = self.POSITIVE_RATING_VAL
            print("Binarized rating matrix. Ratings < {} turned to {}.".format(self.pos_rating_threshold,
                                                                               self.UNKNOWN_RATING_VAL))
        elif mode == 'binary_unknown':
            unknown_rating = self.UNKNOWN_RATING_VAL
            R[R == 0] = 999
            R[R < self.pos_rating_threshold] = self.NEGATIVE_RATING_VAL
            R[R == 999] = self.UNKNOWN_RATING_VAL
            R[R >= self.pos_rating_threshold] = self.POSITIVE_RATING_VAL
            print("Positive ratings (>={}) turned to {}, negative to {}, unknown to {}"
                  .format(self.pos_rating_threshold, self.POSITIVE_RATING_VAL, self.NEGATIVE_RATING_VAL,
                          self.UNKNOWN_RATING_VAL))
        elif mode == 'original':
            pass
        else:
            error("ERROR: _augment_R(mode): mode = '{}' is not recognized!".format(mode))
            print("R will not be modified!")

        R_mask = R != unknown_rating
        return R, R_mask

    def add_random_ratings(self, num_to_each_user=10):
        """
        Adds N random ratings to every user in np.copy(self.R).
        :param num_to_each_user: Number of random (positive=1 or negative=-1)ratings to be added to each user.
        :return: Copy of self.R with added ratings.
        """
        no_items = self.R.shape[1]
        no_users = self.R.shape[0]
        R = np.copy(self.R)
        for u in range(no_users):
            ids = np.random.randint(no_items, size=num_to_each_user)
            new_ratings = np.random.randint(2, size=num_to_each_user) * 2 - np.ones(shape=(num_to_each_user,),
                                                                                    dtype=int)
            R[u][ids] = new_ratings
            # print('ids:', ids)
            # print('ratings:', ratings)
            # print('R[u]:', self.R[u])
        return R

    def get_statistics(self, R):
        total_rats = R.size
        no_rats = len(R[R != self.UNKNOWN_RATING_VAL])
        no_pos_rats = len(R[R == self.POSITIVE_RATING_VAL])
        no_neg_rats = len(R[R == self.NEGATIVE_RATING_VAL])

        user_pos_rats = np.zeros(shape=(R.shape[0],), dtype=int)
        user_neg_rats = np.zeros(shape=(R.shape[0],), dtype=int)
        for u in range(R.shape[0]):
            user = R[u]
            user_pos_rats[u] = len(user[user == self.POSITIVE_RATING_VAL])
            user_neg_rats[u] = len(user[user == self.NEGATIVE_RATING_VAL])

        user_pos_rats_avg = np.average(user_pos_rats)
        user_neg_rats_avg = np.average(user_neg_rats)
        user_pos_rats_std = np.std(user_pos_rats)
        user_neg_rats_std = np.std(user_neg_rats)

        print('Total number of ratings:  ', total_rats)
        print('Known ratings:            ', no_rats)
        print('Known positive ratings:   ', no_pos_rats)
        print('Known negative ratings:   ', no_neg_rats)
        print('Ratio of known ratings:   ', no_rats / total_rats)
        print('Ratio of positive ratings:', no_pos_rats / total_rats)
        print('Ratio of negative ratings:', no_neg_rats / total_rats)
        print('Avg number of positive ratings per user: {} +- {}'.format(user_pos_rats_avg, user_pos_rats_std))
        print('Avg number of negative ratings per user: {} +- {}'.format(user_neg_rats_avg, user_neg_rats_std))

    def _maybe_download_and_extract(self):
        if self.variant not in VARIANTS:
            error(
                'ERROR: maybe_download_and_extract(): Provided variant {} is not in {}!'.format(self.variant, VARIANTS))

        filename = self.variant + FILENAME_SUFFIX
        data_dir = os.path.join(DATA_DIR, DATASET_NAME)
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            create_dir_path(data_dir)
            url = URL_PREFIX + filename
            print("Downloading {} from {}".format(filename, url))
            urllib.request.urlretrieve(url, filepath)
            print("Successfully downloaded {} from {}".format(filename, url))

            print("Extracting {}".format(filename))
            zip_ref = zipfile.ZipFile(filepath, 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()
            print("Successfully extracted {}".format(filename))
        else:
            print("{} is already downloaded.".format(filepath))

    def _get_num_users_items(self):
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, 'u.info')
        with open(filepath, encoding=ENCODING) as f:
            num_users = int(f.readline().split(' ')[0])
            num_items = int(f.readline().split(' ')[0])
            num_ratings = int(f.readline().split(' ')[0])

        return num_users, num_items, num_ratings

    def _get_rating_matrix(self):
        r = np.zeros(shape=(self.num_users, self.num_items), dtype=float)
        implicit_ratings = np.zeros(shape=(self.num_ratings, 4), dtype=int)

        filename = 'u.data'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)
        line_idx = 0
        with open(filepath, encoding=ENCODING) as f:
            for line in f:
                chunks = line.split('\t')
                user_id = int(chunks[0]) - 1  # IDs are numbered from 1
                item_id = int(chunks[1]) - 1
                rating = int(chunks[2])
                timestamp = int(chunks[3])

                implicit_ratings[line_idx] = np.array([user_id, item_id, rating, timestamp])
                r[user_id, item_id] = rating
                line_idx += 1

        print("Created a rating matrix of shape={} and dtype={} from {}.".format(r.shape, r.dtype, filename))
        return r, implicit_ratings

    def _get_item_info(self):
        genres = np.zeros(shape=(self.num_items, self.num_genres), dtype=float)
        titles = np.empty(shape=(self.num_items,), dtype=object)

        filename = 'u.item'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)

        with open(filepath, encoding=STRING_ENCODING) as f:
            for line in f:
                chunks = line.split('|')
                movie_id = int(chunks[0]) - 1  # IDs are numbered from 1
                title = chunks[1]
                titles[movie_id] = title
                # ignore release dates and url
                for i in range(len(chunks) - 5):
                    genres[movie_id, i] = int(chunks[i + 5])

        print("Created a genre matrix of shape={} and dtype={} from {}.".format(genres.shape, genres.dtype, filename))
        print("Created a titles matrix of shape={} and dtype={} from {}.".format(titles.shape, titles.dtype, filename))
        return titles, genres

    def _get_genre_names(self):
        filename = 'u.genre'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)
        with open(filepath, encoding=ENCODING) as f:
            genres = [x.strip().split('|')[0] for x in f.readlines()]
            genres = [x for x in genres if len(x) > 0]
            # print(genres)
        return genres
