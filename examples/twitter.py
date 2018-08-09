import h5py
import time
import os
import logging
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import pandas as pd


log = logging.getLogger("implicit")


def read_data(path='C:/Users/zhang/OneDrive/GitHub/mmtd/user_song_5.csv'):
    data = pd.read_csv(path, header=0)
    # 'tweet_userName', 'tweet_trackId', 'track_name', 'playing_count'

    data['user_id'] = data['tweet_userName'].astype("category").cat.codes
    data['track_id'] = data['tweet_trackId'].astype("category").cat.codes

    return data


def get_twitter():
    try:
        data = read_data()
    except:
        data = read_data(path='D:/OneDrive/GitHub/mmtd/user_song_5.csv')
    users = list(np.sort(data.user_id.unique()))
    tracks = list(np.sort(data.track_id.unique()))
    plays = [float(i) for i in list(data.playing_count)]

    rows = data.user_id.astype(int)
    cols = data.track_id.astype(int)

    data_sparse = csr_matrix((plays, (rows, cols)), shape=(len(users), len(tracks)))

    return np.array(tracks), np.array(users), data_sparse.T


def track_id_to_name():
    data = read_data()
    item_lookup = data[['track_id', 'tweet_trackId', 'track_title', 'artist_name']].drop_duplicates()
    item_lookup['track_id'] = item_lookup.track_id.astype(str)

    return item_lookup

