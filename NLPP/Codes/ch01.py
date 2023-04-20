"""
@Description: Introduction
@Author(s): Stephen CUI
@Time: 2023-04-20 14:48:35
"""


from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ['Time flies flies like an arrow.',
          'Fruit flies like banana.']

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,
            cbar=False,
            xticklabels=one_hot_vectorizer.vocabulary_,
            yticklabels=['Sentence 1', 'Sentence 2'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
fig = plt.figure()
sns.heatmap(tfidf, annot=True, cbar=False,
            xticklabels=tfidf_vectorizer.vocabulary_,
            yticklabels=['Sentence 1', 'Sentence 2'])
