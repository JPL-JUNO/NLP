"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@Time: 2023-04-22 15:07:27
"""

sentence = """Thomas Jefferson began building Monticello at the age of 26."""

sentence.split()
str.split(sentence)

import numpy as np
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
', '.join(vocab)
num_tokens = len(token_sequence)
vocab_size = len(vocab)
ohe = np.zeros((num_tokens, vocab_size), int)
for i, word in enumerate(token_sequence):
    ohe[i, vocab.index(word)] = 1

import pandas as pd
pd.DataFrame(ohe, columns=vocab)

sentence_bow = {}
for token in sentence.split():
    sentence_bow[token] = 1
sorted(sentence_bow.items())


sentences = """Thomas Jefferson began building Monticello at the age of 26.\n"""
sentences += """Construction was done mostly by local masons and carpenters.\n"""
sentences += """He moved into the South Pavilion in 1770.\n"""
sentences += """Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."""
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
    corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])
v1.dot(v2)

# 如果不想降低 Pipeline 的处理速度，就不要这样在向量内部使用迭代处理
sum([x1 * x2 for x1, x2 in zip(v1, v2)])


df = df.T
df['sent0'].dot(df['sent1']), df['sent0'].dot(
    df['sent2']), df['sent0'].dot(df['sent3'])

[(k, v) for (k, v) in (df['sent0'] & df['sent3']).items() if v]
