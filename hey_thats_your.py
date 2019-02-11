from itertools import islice
import re
import json
import string
import warnings
import time

import gensim
from gensim.models import Word2Vec

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams

import numpy as np
from sklearn.decomposition import PCA

from keras_dec import DeepEmbeddingClustering
import keras

# download our needed packages
nltk.download('punkt')
nltk.download('stopwords')

file = 'tweets.json'

#how many tweet wanted?
wanted_tweet = 1000

# file is so big to our memory and we use the islice
def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]

open(file, 'r', encoding='utf8')
with open(file, 'r', encoding='utf8') as sample:
    lines_to_wanted = next_n_lines(sample, wanted_tweet)


# write to file our wanted tweets
with open("myfile.json", "w", encoding='utf8') as f:
    for item in lines_to_wanted:
        f.write("%s\n" % item)


# starting to cleaning our data

# example to seperate our sentences to contents with only word_tokenize
#tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
#print(word_tokenize(tweet))

# we add more regex to seperating
# you can add more regex to your need
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-zA-Z0-9ğüşöçİĞÜŞÖÇ]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-zA-Z0-9ğüşöçİĞÜŞÖÇ][a-zA-Z0-9ğüşöçİĞÜŞÖÇ'\-_]+[a-zA-Z0-9ğüşöçİĞÜŞÖÇ])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
print(preprocess(tweet))

# testing for turkish words
print(preprocess('ğaynın göğüs gerDİM'))

# add punctuation and turkish words to checking elimination to words
punctuation = list(string.punctuation)
stop = stopwords.words('turkish') + punctuation + ['rt', 'via']

sentences = []
fname = 'myfile.json'
with open(fname, 'r', encoding='utf8') as f:
    #count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms
        terms_all =  [term for term in preprocess(tweet['text']) if term not in stop]
        sentences.append(terms_all)

# more terms
terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]

terms_bigram = bigrams(terms_stop)

terms_single = set(terms_all)
# Count hashtags only
terms_hash = [term for term in preprocess(tweet['text'])
              if term.startswith('#')]
# Count terms only (no hashtags, no mentions)
terms_only = [term for term in preprocess(tweet['text'])
              if term not in stop and
              not term.startswith(('#', '@'))]


# our tweets converted to np.array
sentences = np.array(sentences)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# our words turned to vectors
model = Word2Vec(sentences, min_count=1)

#example similarity of vectors
#model.similar_by_word ('hava', topn = 5)

X = model[model.wv.vocab]

# make pca and decreasing components
pca = PCA(n_components=15)
result = pca.fit_transform(X)

# result our new vector np.array
#result.shape
#type(result)

'''Keras implementation of deep embedder to improve clustering, inspired by:
"Unsupervised Deep Embedding for Clustering Analysis" (Xie et al, ICML 2016)
Definition can accept somewhat custom neural networks. Defaults are from paper.
'''

tic = time.clock()
c = DeepEmbeddingClustering(n_clusters=15, input_dim=15, batch_size=128)
c.initialize(result, finetune_iters=100000, layerwise_pretrain_iters=50000)
tic2 = time.clock()
c.cluster(result, iter_max=1000)
toc = time.clock()
print('cluster+initialize is : ',toc - tic)
print('cluster is : ',toc - tic2)

t_sec = round(toc-tic)
(t_min, t_sec) = divmod(t_sec,60)
(t_hour,t_min) = divmod(t_min,60)
print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
