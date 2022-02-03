from collections import defaultdict

import seaborn as sns
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')
stop=set(stopwords.words('english'))
num_words=['1.', '2.', '3.', '4.', '5.']

gab = pd.read_csv('counterspeech_project-NLP/data/gab.csv')

gab_words = gab['text'].str.split().map(lambda x: len(x))
print(gab_words.max())
print(gab_words.min())
gab_words.hist(bins=500, range=(0,1000))
plt.show()

reddit = pd.read_csv('counterspeech_project-NLP/data/reddit.csv')

reddit_words = reddit['text'].str.split().map(lambda x: len(x))
print(reddit_words.max())
print(reddit_words.min())
reddit_words.hist(bins=500, range=(0,1000))
plt.show()

conan = pd.read_json('counterspeech_project-NLP/data/CONAN.json')
print(conan.head(3))

word_set = reddit['text']

corpus=[]
new= word_set.str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

dic=defaultdict(int)
for word in corpus:
    if word in stop and word not in num_words:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:15] 
x,y=zip(*top)
plt.bar(x,y)
plt.show()

counter=Counter(corpus)
most=counter.most_common()

x, y= [], []
for word,count in most[:100]:
    if (word not in stop and word not in num_words):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
plt.show()

def _get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                    for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:50]

top_n_bigrams=_get_top_ngram(word_set,2)[:20]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
plt.show()