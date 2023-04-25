"""
@Description: Example: Classifying Sentiment of Restaurant Reviews
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-24 15:00:30
"""

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import string
import json
from collections import Counter
from numpy import ndarray
from pandas import DataFrame


class ReviewDataSet(Dataset):
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df['split'] == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df['split'] == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df['split'] == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.val_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv: str):
        """Load dataset and make a new vectorizer from scratch.

        Args:
            review_csv (str): location of the dataset

        Returns:
            an instance of ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewDataSet.from_dataframe(review_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv: str, vectorizer_filepath: str):
        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return ReviewVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self) -> object:
        """返回分词器

        Returns:
            object: 使用的分词器
        """
        return self._vectorizer

    def set_split(self, split: str = 'train'):
        """使用dataframe中的一列来选择数据集的分割

        Args:
            split (str, optional): one of 'train', 'val', or 'test'. Defaults to 'train'.
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict(split)

    def __len__(self):
        return self._target_size

    def __getitem__(self, index: int) -> dict:
        """the primary entry point method for Pytorch datasets

        Args:
            index (int): 索引

        Returns:
            dict: _description_
        """
        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorize(row['review'])
        rating_index = self._vectorizer.rating_vocab.lookup_token(
            row['rating'])
        return {'x_data': review_vector,
                'y_target': rating_index}

    def get_num_batches(self, batch_size: int) -> int:
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int): _description_

        Returns:
            int: number of batches in the dataset
        """
        # the length returned by `len(self)` depends on the nature of
        # the object self and how its `__len__()` method is implemented.
        return len(self) // batch_size


class Vocabulary(object):
    def __init__(self,
                 token_to_index: dict = None,
                 add_unk: bool = True,
                 unk_token: str = "<UNK>"):
        if token_to_index is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token: str) -> int:
        """Update mapping dicts based on the token

        Args:
            token (str): the item to add into the Vocabulary

        Returns:
            int: the integer corresponding to the token 
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token: str) -> int:
        """Retrieve the index associated with the token or the UNK index if token isn't present

        Returns:
            _type_: _description_
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        if index not in self._idx_to_token:
            raise KeyError('The index (%d) is not in the Vocabulary' % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)


class ReviewVectorizer(object):
    """The Vectorizer which coordinates the Vocabularies and puts them to use
    """

    def __init__(self, review_vocab, rating_vocab):
        """

        Args:
            review_vocab (_type_): maps words to integers
            rating_vocab (_type_): maps class labels to integers
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review: str) -> ndarray:
        """Create a collapsed one-hot vector for the review

        Args:
            review (str): the review

        Returns:
            ndarray: the collapsed one-hot encoding
        """
        ohe = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                # 排除一些标点符号
                ohe[self.review_vocab.lookup_token(token)] = 1
        return ohe

    @classmethod
    def from_dataframe(cls, review_df: DataFrame, cutoff: int = 25):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            review_df (DataFrame): the review dataset
            cutoff (int, optional): the parameter for frequency-based filtering. Defaults to 25.

        Returns:
            _type_: an instance of the ReviewVectorizer
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # add rating
        for rating in sorted(set(review_df['rating'])):
            rating_vocab.add_token(rating)

        word_counts = Counter()

        for review in review_df['review']:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents: dict):
        """Instantiate a ReviewVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary

        Returns:
            an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


class ReviewClassifier(nn.Module):
    def __init__(self, num_features):
        super(ReviewClassifier, self).__init__()
        # fully connection layer, fc
        self.fc1 = nn.Linear(in_features=num_features,
                             out_features=1)

    def forward(self, x_in, apply_sigmoid: bool = False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out
