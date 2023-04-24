"""
@Description: Example: Classifying Sentiment of Restaurant Reviews
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-24 15:00:30
"""

import collections
import numpy as np
import pandas as pd
import re
from argparse import Namespace

args = Namespace(
    raw_train_dataset_csv='data/yelp/raw_train.csv',
    raw_test_dataset_csv='data/yelp/raw_test.csv',
    proportion_subset_of_train=.1,
    train_proportion=.7,
    val_proportion=.5,
    test_proportion=.15,
    output_munged_scv='data/yelp/reviews_with_split_lite.csv'
)

train_reviews = pd.read_csv(args.raw_train_dataset_csv, 
                            header=None,
                            names=['rating', 'review'])
by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())