"""
@Description: Example: Surname Classification with an MLP
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-28 15:01:05
"""


from torch.utils.data import Dataset
from numpy import ndarray
from pandas import DataFrame
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Type


class SurnameDataset(Dataset):
    def __getitem__(self, index) -> dict:
        row = self._target_df_.iloc[index]
        surname_vector = self._vectorizer.vectorize(row.surname)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(
            row.nationality)
        return {'x_surname': surname_vector,
                'y_nationality': nationality_index}


class Vocabulary:
    pass


class SurnameVectorizer(object):
    def __init__(self, surname_vocab, nationality_vocab):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname: str) -> ndarray:
        """vectorize the provided surname

        Args:
            surname (str): the surname

        Returns:
            ndarray: a collapsed one-hot encoding
        """
        vocab = self.surname_vocab
        one_hot = np.zeros(len(vocab), dtype=np.float32)
        for token in surname:
            one_hot[vocab.lookup_token(token)] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, surname_df: DataFrame) -> Type['SurnameVectorizer']:
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)

        for _, row in surname_df.iterrows():
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)
        return cls(surname_vocab, nationality_vocab)


from torch import Tensor


class SurnameClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in: Tensor, apply_softmax: bool = False) -> Tensor:
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector)
        return prediction_vector
