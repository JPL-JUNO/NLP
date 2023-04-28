"""
@Description: Example: Classifying Sentiment of Restaurant Reviews
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-24 15:00:30
"""

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torch
import os
import re
import pandas as pd
import numpy as np
import string
import json
from collections import Counter
from numpy import ndarray
from pandas import DataFrame
from argparse import Namespace
from tqdm.notebook import tqdm


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
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

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
        self._target_df, self._target_size = self._lookup_dict[split]

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
                 token_to_idx: dict = None,
                 add_unk: bool = True,
                 unk_token: str = "<UNK>"):
        if token_to_idx is None:
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
    """a simple perceptron-based classifier
    """

    def __init__(self, num_features: int):
        """_summary_

        Args:
            num_features (int): the size of the input feature vector
        """
        super(ReviewClassifier, self).__init__()
        # fully connection layer, fc
        self.fc1 = nn.Linear(in_features=num_features,
                             out_features=1)

    def forward(self, x_in: Tensor, apply_sigmoid: bool = False) -> torch.Tensor:
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
            apply_sigmoid (bool, optional): a flag for the sigmoid activation
                should be false if used with the cross-entropy losses
                Defaults to False.

        Returns:
            _type_: the resulting tensor. tensor.shape should be (batch, )
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


args = Namespace(
    # Data and path information
    frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='data/yelp/reviews_with_split_lite.csv',
    save_dir='model_storage/ch3/yelp/',
    vectorizer_file='vectorizer.json',
    # No model hyperparameters
    # Training parameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=10,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=False,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)
    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    print('Expanded filepaths')
    print('\t{}'.format(args.vectorizer_file))
    print('\t{}'.format(args.model_state_file))

if not torch.cuda.is_available():
    args.cuda = False
print('Using CUDA: {}'.format(args.cuda))
args.device = torch.device("cuda" if args.cuda else 'cpu')

set_seed_everywhere(args.seed, args.cuda)


def make_train_state(args) -> dict:
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


def compute_accuracy(y_pred: Tensor, y_target: Tensor) -> float:
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > .5).cpu().long()
    # item()是转成标量？
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def update_train_state(args, model, train_state):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False
    elif train_state['epoch_index'] >= 1:
        loss_tml, loss_t = train_state['val_loss'][-2:]
        if loss_t >= train_state['early_stopping_best_val']:
            train_state['early_stopping_step'] += 1
        else:
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
            train_state['early_stopping_step'] = 0
        train_state['stop_early'] = train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


train_state = make_train_state(args)

# 初始化
if args.reload_from_files:
    print('Loading dataset and vectorizer')
    dataset = ReviewDataSet.load_dataset_and_load_vectorizer(args.review_csv,
                                                             args.vectorizer_file)
else:
    print("Loading dataset and creating vectorizer")
    dataset = ReviewDataSet.load_dataset_and_make_vectorizer(args.review_csv)
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()
# 模型
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

# loss and optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min', factor=.5,
                                                 patience=1)
epoch_bar = tqdm(desc='training routine',
                 total=2 * args.num_epochs,
                 position=0)

dataset.set_split('train')
train_bar = tqdm(desc='split=train',
                 total=dataset.get_num_batches(args.batch_size),
                 position=1,
                 leave=True)

dataset.set_split('val')
val_bar = tqdm(desc='split=val',
               total=dataset.get_num_batches(args.batch_size),
               position=1,
               leave=True)

for epoch_idx in range(args.num_epochs):
    train_state['epoch_index'] = epoch_idx

    # iterate over training dataset
    # setup: batch generator, set loss and acc to 0,
    # set train mode on
    dataset.set_split('train')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = .0
    running_acc = .0
    classifier.train()

    for batch_idx, batch_dict in enumerate(batch_generator):
        # the training routine is 5 steps

        # step 1: zero the gradients
        optimizer.zero_grad()
        # step 2: compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())
        # step 3: compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_idx + 1)
        # step 4: use loss to produce gradients
        loss.backward()
        # step 5: use optimizer to take gradient step
        optimizer.step()
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc)

        train_bar.set_postfix(loss=running_loss,
                              acc=running_acc,
                              epoch=epoch_idx)
        train_bar.update()

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    dataset.set_split('val')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_idx, batch_dict in enumerate(batch_generator):
        # step 1: compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # step 2: compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_idx + 1)

        # step 3 compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'].float())
        running_acc += (acc_batch - running_acc) / (batch_idx + 1)

        val_bar.set_postfix(loss=running_loss,
                            acc=running_acc,
                            epoch=epoch_idx)
        val_bar.update()

    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)

    train_state = update_train_state(args=args, model=classifier,
                                     train_state=train_state)
    scheduler.step(train_state['val_loss'][-1])
    train_bar.n = 0
    val_bar.n = 0
    epoch_bar.update()
    if train_state['stop_early']:
        break

    train_bar.n = 0
    val_bar.n = 0
    epoch_bar.update()


# compute the loss and accuracy on the test set using the best available model
classifier.load_state_dict(torch.load(train_state['model_filename']))
classifier = classifier.to(args.device)

dataset.set_split('test')
batch_generator = generate_batches(dataset,
                                   batch_size=args.batch_size,
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_idx, batch_dict in enumerate(batch_generator):
    y_pred = classifier(x_in=batch_dict['x_data'].float())

    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_t = loss.item()
    # average mean
    running_loss += (loss_t - running_loss) / (batch_idx + 1)
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_idx + 1)

train_state['test_acc'] = running_loss
train_state['test_acc'] = running_acc

print("Test loss: {:.3f}".format(train_state['test_loss']))
print('Test Accuracy: {:.2f}'.format(train_state['test_acc']))

# inference


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def predict_rating(review: str, classifier: ReviewClassifier,
                   vectorizer: ReviewVectorizer, decision_threshold: float = .5):
    review = preprocess_text(review)
    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))
    probability_value = F.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)


test_review = "this is a pretty awesome book"

classifier = classifier.cpu()
prediction = predict_rating(test_review,
                            classifier=classifier,
                            vectorizer=vectorizer,
                            decision_threshold=.5)
print('{} -> {}'.format(test_review, prediction))

# interpret ability
print(classifier.fc1.weight.shape)

fc1_weights = classifier.fc1.weight.detach()[0]
_, indices = torch.sort(fc1_weights, dim=0, descending=True)
indices = indices.numpy().tolist()

# Top 20 positive words
print('Influential words in Positive Reviews')
print('-------------------------------------')
for i in range(20):
    print(vectorizer.review_vocab.lookup_index(indices[i]))
print('\n')
# Top 20 negative words
print('Influential words in Negative Reviews')
print('-------------------------------------')
indices.reverse()
for i in range(20):
    print(vectorizer.review_vocab.lookup_index(indices[i]))
print('\n')
